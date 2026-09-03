# 四足 crawl1 策略真机部署
# quad_leg_crawl1 policy: 50维 obs -> 8维 action
# 观测顺序: base_ang_vel(3), projected_gravity(3), velocity_command(3),
# joint_pos_rel(8), joint_vel_rel(8), last_action(8), crawl_phase(2),
# crawl_leg_phase(8), desired_contacts(4), gait_params(3)
# 动作: 8个关节位置残差，JointPositionAction(use_default_offset=True)

# 先 dry-run 检查 50维 obs / IMU / policy 输出：
# D:\Conda\envs\env_isaaclab\python.exe -B quad_leg_xu\quad_leg_crawl1.py --keyboard --duration_s 10
# 首次上电建议只跑 RR 小输出：
# D:\Conda\envs\env_isaaclab\python.exe -B quad_leg_xu\quad_leg_crawl1.py --keyboard --output_scale 0.05 --default_mode motor_zero --active_joints rr --target_rate_limit_deg_s 8 --default_rate_limit_deg_s 3 --default_reached_threshold_deg 1.0 --default_reached_hold_s 0.5 --default_timeout_s 15 --warmup_s 1 --tau_limit 6
from __future__ import annotations

import os
import sys
import ctypes
import argparse
import math

# 针对 Ubuntu22.04 环境的 libusb 兼容性补丁
if sys.platform == "linux" and "CONDA_PREFIX" in os.environ:
    conda_lib_path = os.path.join(os.environ["CONDA_PREFIX"], "lib", "libusb-1.0.so")
    if os.path.exists(conda_lib_path):
        try:
            ctypes.CDLL(conda_lib_path, mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            print(f"Warning: Failed to preload conda libusb: {e}")

import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

if sys.platform == "win32":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    dll_dir = os.path.join(base_dir, "dlls")
    if not os.path.isdir(dll_dir):
        dll_dir = os.path.join(root_dir, "dlls")
    os.add_dll_directory(base_dir)
    os.add_dll_directory(root_dir)
    os.add_dll_directory(dll_dir)
    _dm_dll_handles = []
    for dll_name in (
        "libwinpthread-1.dll",
        "libgcc_s_seh-1.dll",
        "libstdc++-6.dll",
        "libusb-1.0.dll",
        "libdm_device.dll",
    ):
        _dm_dll_handles.append(ctypes.CDLL(os.path.join(dll_dir, dll_name)))

_dmcan_user_site = None
if sys.platform == "win32":
    # env_isaaclab 有 torch 但通常没有 dmcan/pyusb；本机 Python313 user-site 已有这两个纯 Python 包。
    # 只在导入 dmcan 时临时加入，导入后马上移除，避免 Python313 的 numpy 污染 Python311 的 torch。
    _candidate_user_site = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python313", "site-packages")
    if os.path.isdir(os.path.join(_candidate_user_site, "dmcan")) and _candidate_user_site not in sys.path:
        sys.path.insert(0, _candidate_user_site)
        _dmcan_user_site = _candidate_user_site

from dmcan import DmCanContext, dmcan_channel_can_info, dmcan_device_type, usb_rx_frame

if _dmcan_user_site is not None and _dmcan_user_site in sys.path:
    sys.path.remove(_dmcan_user_site)


# 默认传实验目录，resolve_policy_path() 会自动选择最新 run 下的 exported/policy.pt。
# DEFAULT_POLICY = r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_xu\quad_leg_crawl1"
DEFAULT_POLICY = r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_xu\quad_leg_crawl2\2026-07-09_09-39-57\exported\policy.pt"
# DEFAULT_POLICY = r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_xu\quad_leg_crawl3\2026-07-09_11-30-51\exported\policy.pt"
# DEFAULT_POLICY = r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_xu\quad_leg_crawl4\2026-07-09_12-47-52\exported\policy.pt"



COMMAND_DEADBAND_DEFAULT = 0.025


def resolve_policy_path(policy_arg: str) -> Path:
    """Resolve a policy file or the newest exported policy under an experiment directory."""
    path = Path(policy_arg)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(
            path.glob("*/exported/policy.pt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return path

def _parse_yaml_scalar_float(line: str) -> Optional[float]:
    value = line.split(":", 1)[1].strip()
    if not value or value in {"null", "None"} or value.startswith("!!python/"):
        return None
    try:
        return float(value.split()[0])
    except ValueError:
        return None


def load_train_alignment(policy_path: Path) -> dict:
    """Read the policy run's env.yaml fields that can be mirrored in deployment."""
    run_dir = policy_path.parent.parent if policy_path.parent.name == "exported" else policy_path.parent
    env_yaml = run_dir / "params" / "env.yaml"
    info = {"env_yaml": env_yaml if env_yaml.exists() else None}
    if not env_yaml.exists():
        return info

    lines = env_yaml.read_text(encoding="utf-8-sig").splitlines()
    in_actions = False
    in_joint_pos_action = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if line.startswith("sim:"):
            continue
        if stripped.startswith("dt:") and "sim_dt" not in info:
            info["sim_dt"] = _parse_yaml_scalar_float(line)
        elif line.startswith("decimation:"):
            info["decimation"] = _parse_yaml_scalar_float(line)
        elif stripped.startswith("stiffness:") and "stiffness" not in info:
            info["stiffness"] = _parse_yaml_scalar_float(line)
        elif stripped.startswith("damping:") and "damping" not in info:
            info["damping"] = _parse_yaml_scalar_float(line)
        elif stripped.startswith("effort_limit:") and "effort_limit" not in info:
            info["effort_limit"] = _parse_yaml_scalar_float(line)
        elif stripped.startswith("saturation_effort:") and "saturation_effort" not in info:
            info["saturation_effort"] = _parse_yaml_scalar_float(line)
        elif stripped.startswith("velocity_limit:") and "velocity_limit" not in info:
            info["velocity_limit"] = _parse_yaml_scalar_float(line)
        elif stripped.startswith("lin_vel_x:") and "command_x" not in info:
            parsed_command_x = _parse_yaml_scalar_float(line)
            if parsed_command_x is not None:
                info["command_x"] = parsed_command_x
        elif stripped.startswith("frequency_hz:") and "crawl_frequency_hz" not in info:
            info["crawl_frequency_hz"] = _parse_yaml_scalar_float(line)
        elif stripped.startswith("duty_factor:") and "crawl_duty_factor" not in info:
            info["crawl_duty_factor"] = _parse_yaml_scalar_float(line)
        elif stripped.startswith("swing_height:") and "crawl_swing_height" not in info:
            info["crawl_swing_height"] = _parse_yaml_scalar_float(line)

        if line.startswith("actions:"):
            in_actions = True
            in_joint_pos_action = False
            continue
        if in_actions and line and not line.startswith(" "):
            in_actions = False
            in_joint_pos_action = False
        if in_actions and line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            in_joint_pos_action = stripped == "joint_pos:"
            continue
        if in_joint_pos_action:
            if stripped.startswith("scale:"):
                info["action_scale_rad"] = _parse_yaml_scalar_float(line)
            elif stripped.startswith("max_joint_speed:"):
                info["max_joint_speed_rad_s"] = _parse_yaml_scalar_float(line)

    sim_dt = info.get("sim_dt")
    decimation = info.get("decimation")
    if sim_dt and decimation:
        info["env_step_dt"] = sim_dt * decimation
        info["policy_rate_hz"] = 1.0 / info["env_step_dt"]
    return info


class DM_Motor_Type(IntEnum):
    DM3507 = 0
    DM4310 = 1
    DM4310_48V = 2
    DM4340 = 3
    DM4340_48V = 4
    DM6006 = 5
    DM6248 = 6
    DM8006 = 7
    DM8009 = 8
    DM10010L = 9
    DM10010 = 10
    DMH3510 = 11
    DMH6215 = 12
    DMS3519 = 13
    DMG6220 = 14
    Num_Of_Motor = 15


class Control_Mode(IntEnum):
    MIT_MODE = 0x000
    POS_VEL_MODE = 0x100
    VEL_MODE = 0x200
    POS_FORCE_MODE = 0x300


class Control_Mode_Code(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    POS_FORCE = 4


control_mode_to_code = {
    Control_Mode.MIT_MODE: Control_Mode_Code.MIT,
    Control_Mode.POS_VEL_MODE: Control_Mode_Code.POS_VEL,
    Control_Mode.VEL_MODE: Control_Mode_Code.VEL,
    Control_Mode.POS_FORCE_MODE: Control_Mode_Code.POS_FORCE,
}


@dataclass
class DmActData:
    motorType: DM_Motor_Type
    mode: Control_Mode
    can_id: int
    mst_id: int
    channel: int = 0


class DM_REG(IntEnum):
    UV_Value = 0
    KT_Value = 1
    OT_Value = 2
    OC_Value = 3
    ACC = 4
    DEC = 5
    MAX_SPD = 6
    MST_ID = 7
    ESC_ID = 8
    TIMEOUT = 9
    CTRL_MODE = 10
    Damp = 11
    Inertia = 12
    hw_ver = 13
    sw_ver = 14
    SN = 15
    NPP = 16
    Rs = 17
    LS = 18
    Flux = 19
    Gr = 20
    PMAX = 21
    VMAX = 22
    TMAX = 23
    I_BW = 24
    KP_ASR = 25
    KI_ASR = 26
    KP_APR = 27
    KI_APR = 28
    OV_Value = 29
    GREF = 30
    Deta = 31
    V_BW = 32
    IQ_c1 = 33
    VL_c1 = 34
    can_br = 35
    sub_ver = 36
    u_off = 50
    v_off = 51
    k1 = 52
    k2 = 53
    m_off = 54
    dir = 55
    p_m = 80
    xout = 81


limit_param = [
    [12.566, 50, 5],   # DM3507        
    [12.5, 30, 10],   # DM4310         
    [12.5, 50, 10],   # DM4310_48V
    [12.5, 10, 28],   # DM4340         
    [12.5, 20, 28],   # DM4340_48V    
    [12.5, 45, 12],   # DM6006         
    [12.566, 20, 120],   # DM6248      
    [12.5, 45, 20],   # DM8006         
    [12.5, 45, 54],   # DM8009       
    [12.5, 25, 200],  # DM10010L       
    [12.5, 20, 200],  # DM10010        
    [12.5, 280, 1],   # DMH3510        
    [12.5, 45, 10],   # DMH6215
    [12.5, 2000, 2],    # DMS3519        
    [12.5, 45, 10]    # DMG6220         
]


class ValueUnion:
    def __init__(self):
        self.floatValue = 0.0
        self.uint32Value = 0


class ValueType:
    def __init__(self):
        self.value = ValueUnion()
        self.isFloat = False


class Motor:
    def __init__(
        self,
        motor_type: DM_Motor_Type,
        ctrl_mode: Control_Mode,
        can_id: int,
        master_id: int,
        channel: int = 0,
    ):
        self.Motor_Type = DM_Motor_Type(motor_type)
        self.mode = Control_Mode(ctrl_mode)
        self.Can_id = int(can_id)
        self.Master_id = int(master_id)
        self.channel = int(channel)
        self.limit_param = list(limit_param[self.Motor_Type.value])
        self.param_map: Dict[int, ValueType] = {}
        self.last_time_ = time.monotonic()
        self.delta_time_ = 0.0
        self.state_q = 0.0
        self.state_dq = 0.0
        self.state_tau = 0.0
        self.state_err = 0

    def updateTimeInterval(self) -> float:
        now = time.monotonic()
        self.delta_time_ = now - self.last_time_
        self.last_time_ = now
        return self.delta_time_

    def getTimeInterval(self):
        return self.delta_time_

    def receive_data(self, q: float, dq: float, tau: float, err: int = 0):
        self.state_q = q
        self.state_dq = dq
        self.state_tau = tau
        self.state_err = err

    def set_param(self, key: int, value):
        v = ValueType()
        if isinstance(value, IntEnum):
            value = int(value)
        if isinstance(value, int):
            v.value.uint32Value = value
            v.isFloat = False
        elif isinstance(value, float):
            v.value.floatValue = value
            v.isFloat = True
        else:
            raise TypeError(f"Unsupported param type: {type(value)!r}")
        self.param_map[int(key)] = v

    def get_param_as_float(self, key: int) -> float:
        v = self.param_map.get(int(key))
        if v is not None and v.isFloat:
            return v.value.floatValue
        return 0.0

    def get_param_as_uint32(self, key: int) -> int:
        v = self.param_map.get(int(key))
        if v is not None and not v.isFloat:
            return v.value.uint32Value
        return 0

    def is_have_param(self, key: int) -> bool:
        return int(key) in self.param_map

    def GetMotorType(self):
        return self.Motor_Type

    def GetMotorMode(self):
        return self.mode

    def get_limit_param(self):
        return self.limit_param

    def GetMasterId(self):
        return self.Master_id

    def GetCanId(self):
        return self.Can_id

    def GetChannel(self):
        return self.channel

    def Get_err(self):
        return self.state_err

    def Get_Err(self):
        return self.state_err

    def Get_Position(self):
        return self.state_q

    def Get_Velocity(self):
        return self.state_dq

    def Get_tau(self):
        return self.state_tau

    def set_mode(self, value: Control_Mode):
        self.mode = Control_Mode(value)


class Motor_Control:
    def __init__(
        self,
        nom_baud: int,
        dat_baud: int,
        sn: str = "",
        data_ptr: Optional[Iterable[DmActData]] = None,
        *,
        device_index: int = 0,
        device_type: Optional[dmcan_device_type] = None,
        canfd: bool = True,
        brs: bool = True,
        auto_enable: bool = True,
        set_baudrate: bool = True,
        can_sp: float = 0.75,
        canfd_sp: float = 0.75,
    ):
        self.data_ptr_ = list(data_ptr or [])
        self.motors: Dict[Tuple[int, int], Motor] = {}
        self._motors_by_id: Dict[int, Motor] = {}
        self.read_write_save = threading.Event()
        self.read_write_save.clear()
        self._lock = threading.RLock()
        self._param_event = threading.Event()
        self.nom_baud = int(nom_baud)
        self.dat_baud = int(dat_baud)
        self.sn = sn
        self.canfd = bool(canfd)
        self.brs = bool(brs)
        self._closed = False

        for act_data in self.data_ptr_:
            self.addMotor(
                Motor(
                    act_data.motorType,
                    act_data.mode,
                    act_data.can_id,
                    act_data.mst_id,
                    getattr(act_data, "channel", 0),
                )
            )

        self.context = DmCanContext()
        dev_count = self.context.find_devices(device_type)
        if dev_count <= 0:
            self.context.destroy()
            raise RuntimeError("No dmcan device found")
        if device_index >= dev_count:
            self.context.destroy()
            raise IndexError(f"device_index {device_index} out of range, found {dev_count} device(s)")

        self.device = self.context.get_device(device_index)
        if not self.device.open():
            self.context.destroy()
            raise RuntimeError(f"Failed to open dmcan device index {device_index}")

        channels = self._registered_channels()
        if not channels:
            channels = {0}
        for ch in channels:
            if set_baudrate:
                self._set_channel_baudrate(ch, self.nom_baud, self.dat_baud, can_sp, canfd_sp)
            self.device.enable_channel(ch, True)
        # 注册接收回调函数
        self.device.hook_recv_callback(self.canframeCallback)
        time.sleep(0.05)

        if auto_enable:
            self.enable_all()
        print("**********Motor_Control init success**********")

    @staticmethod
    def is_in_ranges(number: int) -> bool:
        number = int(number)
        return (7 <= number <= 10) or (13 <= number <= 16) or (35 <= number <= 36)

    @staticmethod
    def float_to_uint32(value: float) -> int:
        return int(value)

    @staticmethod
    def uint32_to_float(value: int) -> float:
        return float(value)

    @staticmethod
    def uint8_to_float(data: List[int]) -> float:
        if len(data) != 4:
            raise ValueError("data must be a list of 4 bytes")
        return struct.unpack("<f", bytes(data))[0]

    @staticmethod
    def _float_to_uint(value: float, min_value: float, max_value: float, bits: int) -> int:
        value = max(min_value, min(max_value, value))
        return int((value - min_value) / (max_value - min_value) * ((1 << bits) - 1))

    @staticmethod
    def _uint_to_float(value: int, min_value: float, max_value: float, bits: int) -> float:
        return (float(value) / ((1 << bits) - 1)) * (max_value - min_value) + min_value

    def _registered_channels(self) -> set[int]:
        return {motor.GetChannel() for motor in set(self.motors.values())}

    def _set_channel_baudrate(self, channel: int, nom_baud: int, dat_baud: int, can_sp: float, canfd_sp: float):
        info = dmcan_channel_can_info()
        info.channel = channel
        info.canfd = self.canfd
        info.can_baudrate = nom_baud
        info.canfd_baudrate = dat_baud
        info.can_sp = can_sp
        info.canfd_sp = canfd_sp
        self.device.set_channel_baudrate(channel, info)

    def _send(self, channel: int, can_id: int, payload, *, canfd: Optional[bool] = None, brs: Optional[bool] = None) -> bool:
        data = bytes(payload)
        return self.device.send_can(
            int(channel),
            int(can_id),
            len(data),
            data,
            self.canfd if canfd is None else canfd,
            False,
            False,
            self.brs if brs is None else brs,
        )

    def getMotor(self, *args) -> Optional[Motor]:
        if len(args) == 1:
            motor_id = int(args[0])
            motor = self._motors_by_id.get(motor_id)
        elif len(args) == 2:
            motor = self.motors.get((int(args[0]), int(args[1])))
        else:
            raise TypeError("getMotor expects id or (channel, id)")

        if motor is None:
            print(f"[Error] In getMotor, no motor with args {args} is registered.", file=sys.stderr)
        return motor

    def getUSBHw(self):
        return self.device

    def getDevice(self):
        return self.device

    def addMotor(self, DM_Motor: Motor):
        can_key = (DM_Motor.GetChannel(), DM_Motor.GetCanId())
        mst_key = (DM_Motor.GetChannel(), DM_Motor.GetMasterId())
        self.motors[can_key] = DM_Motor
        self.motors[mst_key] = DM_Motor
        self._motors_by_id.setdefault(DM_Motor.GetCanId(), DM_Motor)
        self._motors_by_id.setdefault(DM_Motor.GetMasterId(), DM_Motor)

    def getMotorsByChannel(self, ch: int) -> Dict[int, Motor]:
        channel = int(ch)
        return {motor_id: motor for (motor_ch, motor_id), motor in self.motors.items() if motor_ch == channel}

    def _unique_motors(self) -> List[Motor]:
        seen = set()
        motors = []
        for motor in self.motors.values():
            key = (motor.GetChannel(), motor.GetCanId())
            if key not in seen:
                seen.add(key)
                motors.append(motor)
        return motors

    def enable_all(self):
        for motor in self._unique_motors():
            code = control_mode_to_code[motor.GetMotorMode()]
            self.switchControlMode(motor, code)
            time.sleep(0.002)
        for motor in self._unique_motors():
            for _ in range(5):
                self.control_cmd(motor.GetCanId() + motor.GetMotorMode(), 0xFC, motor.GetChannel())
                time.sleep(0.002)

    def disable_all(self):
        for motor in self._unique_motors():
            for _ in range(5):
                self.control_cmd(motor.GetCanId() + motor.GetMotorMode(), 0xFD, motor.GetChannel())
                time.sleep(0.002)

    def read_motor_param(self, DM_Motor: Motor, RID: int, timeout: float = 1.0):
        self.read_write_save.set()
        self._param_event.clear()
        can_id = DM_Motor.GetCanId()
        payload = bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, 0x33, int(RID), 0, 0, 0, 0])
        self._send(DM_Motor.GetChannel(), 0x7FF, payload)
        if timeout and self._param_event.wait(timeout):
            if self.is_in_ranges(RID):
                return DM_Motor.get_param_as_uint32(RID)
            return DM_Motor.get_param_as_float(RID)
        return None

    def save_motor_param(self, DM_Motor: Motor):
        self.control_cmd(DM_Motor.GetCanId() + DM_Motor.GetMotorMode(), 0xFD, DM_Motor.GetChannel())
        time.sleep(0.01)
        self.read_write_save.set()
        can_id = DM_Motor.GetCanId()
        payload = bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, 0xAA, 0x01, 0, 0, 0, 0])
        self._send(DM_Motor.GetChannel(), 0x7FF, payload)
        time.sleep(0.1)

    def refresh_motor_status(self, motor: Motor):
        can_id = motor.GetCanId()
        payload = bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, 0xCC, 0x00])
        self._send(motor.GetChannel(), 0x7FF, payload)

    def control_cmd(self, id: int, cmd: int, ch: int = 0):
        return self._send(ch, id, bytes([0xFF] * 7 + [cmd]))

    def write_motor_param(self, DM_Motor: Motor, RID: int, data):
        self.read_write_save.set()
        self._param_event.clear()
        can_id = DM_Motor.GetCanId()
        data_bytes = bytes(data)
        if len(data_bytes) != 4:
            raise ValueError("data must contain exactly 4 bytes")
        payload = bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, 0x55, int(RID)]) + data_bytes
        return self._send(DM_Motor.GetChannel(), 0x7FF, payload)

    def set_zero_position(self, DM_Motor: Motor):
        self.control_cmd(DM_Motor.GetCanId() + DM_Motor.GetMotorMode(), 0xFE, DM_Motor.GetChannel())
        time.sleep(0.002)

    def control_mit(self, DM_Motor: Motor, kp: float, kd: float, q: float, dq: float, tau: float):
        if DM_Motor is None:
            raise ValueError("DM_Motor is None")
        key = (DM_Motor.GetChannel(), DM_Motor.GetCanId())
        if key not in self.motors:
            raise KeyError(f"Motor channel={key[0]} id={key[1]} is not registered")

        q_max, dq_max, tau_max = DM_Motor.get_limit_param()
        kp_uint = self._float_to_uint(kp, 0, 500, 12)
        kd_uint = self._float_to_uint(kd, 0, 5, 12)
        q_uint = self._float_to_uint(q, -q_max, q_max, 16)
        dq_uint = self._float_to_uint(dq, -dq_max, dq_max, 12)
        tau_uint = self._float_to_uint(tau, -tau_max, tau_max, 12)

        data = bytearray(8)
        data[0] = (q_uint >> 8) & 0xFF
        data[1] = q_uint & 0xFF
        data[2] = (dq_uint >> 4) & 0xFF
        data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
        data[4] = kp_uint & 0xFF
        data[5] = (kd_uint >> 4) & 0xFF
        data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
        data[7] = tau_uint & 0xFF
        return self._send(DM_Motor.GetChannel(), DM_Motor.GetCanId() + Control_Mode.MIT_MODE, data)

    def control_pos_vel(self, DM_Motor: Motor, pos: float, vel: float):
        if DM_Motor is None:
            raise ValueError("DM_Motor is None")
        data = struct.pack("<ff", pos, vel)
        return self._send(DM_Motor.GetChannel(), DM_Motor.GetCanId() + Control_Mode.POS_VEL_MODE, data)

    def control_vel(self, DM_Motor: Motor, vel: float):
        if DM_Motor is None:
            raise ValueError("DM_Motor is None")
        data = struct.pack("<f", vel)
        return self._send(DM_Motor.GetChannel(), DM_Motor.GetCanId() + Control_Mode.VEL_MODE, data)

    def receive_param(self, data: bytes, ch: int = 0):
        if len(data) < 8:
            return
        can_id = (data[1] << 8) | data[0]
        rid = data[3]
        motor = self.motors.get((int(ch), can_id))
        if motor is None:
            return

        if self.is_in_ranges(rid):
            value = int.from_bytes(data[4:8], byteorder="little", signed=False)
            motor.set_param(rid, value)
            if rid == DM_REG.CTRL_MODE:
                mode_map = {
                    1: Control_Mode.MIT_MODE,
                    2: Control_Mode.POS_VEL_MODE,
                    3: Control_Mode.VEL_MODE,
                    4: Control_Mode.POS_FORCE_MODE,
                }
                if value in mode_map:
                    motor.set_mode(mode_map[value])
        else:
            motor.set_param(rid, self.uint8_to_float(list(data[4:8])))
        self._param_event.set()

    def switchControlMode(self, DM_Motor: Motor, mode: Control_Mode_Code):
        mode = Control_Mode_Code(mode)
        ok = self.write_motor_param(DM_Motor, DM_REG.CTRL_MODE, bytes([mode, 0, 0, 0]))
        if ok:
            reverse = {
                Control_Mode_Code.MIT: Control_Mode.MIT_MODE,
                Control_Mode_Code.POS_VEL: Control_Mode.POS_VEL_MODE,
                Control_Mode_Code.VEL: Control_Mode.VEL_MODE,
                Control_Mode_Code.POS_FORCE: Control_Mode.POS_FORCE_MODE,
            }
            DM_Motor.set_mode(reverse[mode])
        return ok

    def change_motor_param(self, DM_Motor: Motor, RID, data):
        rid = int(RID)
        if self.is_in_ranges(rid):
            data_bytes = int(data).to_bytes(4, byteorder="little", signed=False)
        else:
            data_bytes = struct.pack("<f", float(data))
        return self.write_motor_param(DM_Motor, rid, data_bytes)

    def changeMotorLimit(self, DM_Motor: Motor, P_MAX, Q_MAX, T_MAX):
        DM_Motor.limit_param = [float(P_MAX), float(Q_MAX), float(T_MAX)]
        limit_param[DM_Motor.GetMotorType().value] = DM_Motor.limit_param

    def canframeCallback(self, device, frame: usb_rx_frame):
        with self._lock:
            can_id = int(frame.head.can_id)
            ch = int(frame.head.channel)
            dlc = int(frame.head.dlc)
            length = self._dlc_to_len(dlc)
            data = bytes(frame.payload[:length])
            if len(data) < 6:
                return

            if self.read_write_save.is_set():
                if len(data) >= 8 and data[2] in (0x33, 0x55, 0xAA):
                    if data[2] in (0x33, 0x55):
                        self.receive_param(data[:8], ch)
                    self.read_write_save.clear()
                    return

            motor = self.motors.get((ch, can_id))
            if motor is None:
                return

            err = (data[0] >> 4) & 0x0F
            q_uint = (data[1] << 8) | data[2]
            dq_uint = (data[3] << 4) | (data[4] >> 4)
            tau_uint = ((data[4] & 0x0F) << 8) | data[5]
            q_max, dq_max, tau_max = motor.get_limit_param()
            motor.receive_data(
                self._uint_to_float(q_uint, -q_max, q_max, 16),
                self._uint_to_float(dq_uint, -dq_max, dq_max, 12),
                self._uint_to_float(tau_uint, -tau_max, tau_max, 12),
                err,
            )
            motor.updateTimeInterval()

    @staticmethod
    def _dlc_to_len(dlc: int) -> int:
        return {9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64}.get(dlc, min(dlc, 8))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.disable_all()
        except Exception as exc:
            print(f"[Warn] disable_all failed during close: {exc}", file=sys.stderr)
        try:
            for ch in self._registered_channels() or {0}:
                self.device.enable_channel(ch, False)
        finally:
            if sys.platform == "win32":
                print("[Warn] skip SDK close on Windows to avoid dmcan shutdown hang", file=sys.stderr)
                return
            try:
                self.device.close()
            finally:
                try:
                    self.context.destroy()
                except OSError as exc:
                    print(f"[Warn] context destroy failed during close: {exc}", file=sys.stderr)


running = threading.Event()
running.set()


def signal_handler(signum, frame):
    running.clear()
    sys.stderr.write(f"\nInterrupt signal ({signum}) received.\n")
    sys.stderr.flush()


signal.signal(signal.SIGINT, signal_handler)



def ensure_pyserial():
    """Import pyserial; env_isaaclab may borrow the user's Python313 pure-Python package."""
    try:
        import serial  # type: ignore
        return serial
    except ImportError:
        user_site = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python313", "site-packages")
        if os.path.isdir(os.path.join(user_site, "serial")) and user_site not in sys.path:
            sys.path.insert(0, user_site)
        import serial  # type: ignore
        return serial


class DMImuSerialReader:
    """Minimal DM-IMU USB serial reader for frames: 55 AA sid reg 3*float32 crc crc 0A."""

    FRAME_LEN = 19
    HEADER = b"\x55\xAA"

    def __init__(self, port: str, baudrate: int = 921600, *, exit_setting_mode: bool = True):
        serial = ensure_pyserial()
        self.port = port
        self.baudrate = int(baudrate)
        self.ser = serial.Serial(port, self.baudrate, timeout=0, write_timeout=0)
        if exit_setting_mode:
            cmd = bytes([0xAA, 0x06, 0x00, 0x0D])
            for _ in range(10):
                self.ser.write(cmd)
                time.sleep(0.01)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._latest: Dict[int, Tuple[float, float, float]] = {}
        self._stamp: Dict[int, float] = {}
        self._counts: Dict[int, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while not self._stop.is_set():
                n = getattr(self.ser, "in_waiting", 0)
                if n:
                    self._buf.extend(self.ser.read(n))
                    self._parse_frames()
                else:
                    time.sleep(0.001)
        except Exception as exc:
            with self._lock:
                self._latest[-1] = (float("nan"), float("nan"), float("nan"))
                self._stamp[-1] = time.time()
                self._counts[-1] = self._counts.get(-1, 0) + 1
            print(f"[IMU] read thread stopped: {exc}", file=sys.stderr)

    def _parse_frames(self):
        while True:
            j = self._buf.find(self.HEADER)
            if j < 0:
                if len(self._buf) > 1:
                    del self._buf[:-1]
                return
            if j > 0:
                del self._buf[:j]
            if len(self._buf) < self.FRAME_LEN:
                return
            frame = bytes(self._buf[:self.FRAME_LEN])
            del self._buf[:self.FRAME_LEN]
            if frame[-1] != 0x0A:
                continue
            reg = int(frame[3])
            if reg not in (1, 2, 3, 4):
                continue
            vals = struct.unpack("<fff", frame[4:16])
            now = time.time()
            with self._lock:
                self._latest[reg] = (float(vals[0]), float(vals[1]), float(vals[2]))
                self._stamp[reg] = now
                self._counts[reg] = self._counts.get(reg, 0) + 1

    def get(self, reg: int) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self._latest.get(int(reg))

    def counts(self) -> Dict[int, int]:
        with self._lock:
            return dict(self._counts)

    def wait_ready(self, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if 2 in self._latest and 3 in self._latest:
                    return True
            time.sleep(0.01)
        return False

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass


def projected_gravity_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[float, float, float]:
    """Return gravity vector in body frame for ZYX yaw-pitch-roll convention."""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return (sp, -sr * cp, -cr * cp)


def parse_triplet(text_value: str, *, name: str) -> Tuple[float, float, float]:
    vals = [float(x.strip()) for x in text_value.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values")
    return (vals[0], vals[1], vals[2])


def clipped_scaled(values, lo: float, hi: float, scale: float = 1.0):
    return [max(lo, min(hi, float(v))) * scale for v in values]


class KeyboardCommandX:
    """Non-blocking keyboard control for crawl1 command_x and policy arming.

    Keys:
      Enter / Space: arm policy after safe-start
      PgUp: increase command_x
      PgDn: decrease command_x
      Home / 0 / X: set command_x to zero
      Q / Esc: request shutdown
    """

    def __init__(self, initial_x: float, min_x: float, max_x: float, step: float, auto_arm: bool = False):
        self.initial_x = float(initial_x)
        self.command_x = float(initial_x)
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.step = float(step)
        self.armed = bool(auto_arm)
        if self.min_x > self.max_x:
            raise ValueError("keyboard_min_x must be <= keyboard_max_x")
        self.command_x = max(self.min_x, min(self.max_x, self.command_x))

    def _apply_key(self, key: str, extended: bool = False) -> None:
        key_lower = key.lower() if len(key) == 1 else key
        if extended and key.upper() == "I":  # PgUp
            self.command_x = min(self.max_x, self.command_x + self.step)
            print(f"[KEYBOARD] PgUp command_x={self.command_x:.3f} m/s")
        elif extended and key.upper() == "Q":  # PgDn
            self.command_x = max(self.min_x, self.command_x - self.step)
            print(f"[KEYBOARD] PgDn command_x={self.command_x:.3f} m/s")
        elif extended and key.upper() == "G":  # Home
            self.command_x = 0.0
            print("[KEYBOARD] command_x zeroed to 0.000 m/s")
        elif key in ("\r", " "):
            if not self.armed:
                self.armed = True
                print("[KEYBOARD] policy ARMED")
        elif key_lower in ("0", "x"):
            self.command_x = 0.0
            print("[KEYBOARD] command_x zeroed to 0.000 m/s")
        elif key_lower == "w":  # fallback if PgUp is unavailable in a terminal
            self.command_x = min(self.max_x, self.command_x + self.step)
            print(f"[KEYBOARD] command_x={self.command_x:.3f} m/s")
        elif key_lower == "s":  # fallback if PgDn is unavailable in a terminal
            self.command_x = max(self.min_x, self.command_x - self.step)
            print(f"[KEYBOARD] command_x={self.command_x:.3f} m/s")
        elif key_lower in ("q", "\x1b"):
            print("[KEYBOARD] shutdown requested")
            running.clear()

    def poll(self) -> float:
        if sys.platform != "win32":
            return self.command_x
        try:
            import msvcrt
            while msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    self._apply_key(msvcrt.getwch(), extended=True)
                else:
                    self._apply_key(key, extended=False)
        except Exception as exc:
            print(f"[KEYBOARD] polling disabled after error: {exc}", file=sys.stderr)
        return self.command_x


def crawl_phase_terms(
    elapsed_s: float,
    frequency_hz: float,
    duty_factor: float,
    swing_height: float,
    command_x: float,
    command_deadband: float = COMMAND_DEADBAND_DEFAULT,
):
    moving = abs(float(command_x)) >= float(command_deadband)
    velocity_command = [float(command_x) if moving else 0.0, 0.0, 0.0]

    if not moving:
        crawl_phase = [0.0, 1.0]
        leg_phase_terms = [0.0, 1.0] * 4
        desired_contacts = [1.0, 1.0, 1.0, 1.0]
        gait_params = [0.0, 1.0, 0.0]
        return velocity_command, crawl_phase, leg_phase_terms, desired_contacts, gait_params, 0.0

    phase = (elapsed_s * frequency_hz) % 1.0
    phase_rad = 2.0 * math.pi * phase
    crawl_phase = [math.sin(phase_rad), math.cos(phase_rad)]

    # Order is FL, FR, RL, RR. Offsets match quad_leg_crawl1/mdp/gait_scheduler.py.
    offsets = (0.0, 0.5, 0.75, 0.25)
    swing_fraction = max(0.0, 1.0 - duty_factor)
    leg_phase_terms = []
    desired_contacts = []
    for offset in offsets:
        leg_phase = (phase - offset) % 1.0
        leg_phase_rad = 2.0 * math.pi * leg_phase
        leg_phase_terms.extend([math.sin(leg_phase_rad), math.cos(leg_phase_rad)])
        desired_contacts.append(0.0 if leg_phase < swing_fraction else 1.0)

    gait_params = [frequency_hz, duty_factor, swing_height]
    return velocity_command, crawl_phase, leg_phase_terms, desired_contacts, gait_params, phase


def fmt_vec(values, ndigits: int = 3) -> str:
    return "[" + ", ".join(f"{float(v):+.{ndigits}f}" for v in values) + "]"


if __name__ == "__main__":
    control = None
    imu = None
    try:
        # ========================= 参数说明 =========================
        # --policy
        #   要加载的 TorchScript 策略文件 policy.pt。默认使用 DEFAULT_POLICY。
        #
        # --imu_port
        #   IMU 串口号，例如 COM6。插拔 USB 后如果串口变了，就改这里。
        #
        # --imu_baudrate
        #   IMU 波特率。达妙 IMU 当前一般用 921600。
        #
        # --policy_rate_hz
        #   策略推理频率。默认从训练日志 env.yaml 读取；一般不用手动改。
        #
        # --output_scale
        #   策略动作额外缩放系数。真机首测建议 0.03 或 0.05，越小越保守。
        #
        # --action_scale_deg
        #   策略动作基础角度尺度。默认从 env.yaml 读取；一般不要手动改。
        #
        # --target_rate_limit_deg_s
        #   策略输出目标角的变化速度限制，单位 deg/s。限制 policy 上策略后的动作速度。
        #
        # --kp / --kd
        #   MIT 控制的刚度和阻尼。默认从训练 env.yaml 读取；真机不稳时可降低 kp。
        #
        # --warmup_s
        #   进入训练默认角并通过到位检测后，再等待多久才真正启动策略。
        #
        # --duration_s
        #   运行总时长，单位秒。0 表示一直运行，直到 Ctrl+C。
        #
        # --tau_limit
        #   力矩保护阈值，单位 Nm。超过后脚本会停机保护。
        #
        # --active_joints
        #   控制哪些关节：
        #   all = 四条腿 8 个关节全部控制；
        #   rr  = 只控制 RR 后右腿，用于单腿安全测试。
        #
        # --default_mode
        #   默认角映射方式：
        #   motor_zero   = 电机零位对应仿真零位，推荐当前真机使用；
        #   current_pose = 启动时当前姿态映射为训练默认姿态，调试用。
        #
        # --motor_zero_positions
        #   8 个电机零位偏移，顺序为：
        #   FL_thigh, FL_calf, FR_thigh, FR_calf, RL_thigh, RL_calf, RR_thigh, RR_calf
        #
        # --standup_s
        #   旧参数。现在主要作为 default_timeout_s 没写时的默认超时时间。
        #
        # --default_rate_limit_deg_s
        #   从当前姿态慢慢进入训练默认角的速度，单位 deg/s。越小越慢越安全。
        #
        # --default_reached_threshold_deg
        #   判断“已经到训练默认角”的角度误差阈值，单位 deg。
        #
        # --default_reached_hold_s
        #   进入默认角误差阈值后，需要连续保持多久才允许启动策略。
        #
        # --default_timeout_s
        #   进入训练默认角的最大等待时间。超时还没到位就退出保护。
        #
        # --dry_run
        #   只读取 IMU 和跑策略，不打开/控制电机，用于检查 IMU 和策略输出。
        #
        # --gyro_signs
        #   IMU 角速度 x/y/z 三轴符号修正，例如 1,1,1 或 -1,1,1。
        #
        # --gravity_signs
        #   projected_gravity 三轴符号修正，用于匹配训练观测坐标系。
        #
        # --roll_offset_deg / --pitch_offset_deg / --yaw_offset_deg
        #   IMU 欧拉角零偏补偿，单位 deg。IMU 安装有固定倾角时用这里修正。
        # ============================================================
        parser = argparse.ArgumentParser(description="Bennett quad_leg_crawl1 policy sim2real runner.")
        parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="Exported TorchScript policy.pt path.")
        parser.add_argument("--imu_port", type=str, default="COM6", help="DM-IMU USB serial port.")
        parser.add_argument("--imu_baudrate", type=int, default=921600, help="DM-IMU baudrate.")
        parser.add_argument("--policy_rate_hz", type=float, default=None, help="Policy inference rate. Defaults to env.yaml.")
        parser.add_argument("--output_scale", type=float, default=1, help="Extra safety scale on policy residual action.")
        parser.add_argument("--action_scale_deg", type=float, default=None, help="Action scale. Defaults to env.yaml joint_pos scale.")
        parser.add_argument("--target_rate_limit_deg_s", type=float, default=70, help="Residual target rate limit in deg/s. Default is effectively unbounded to match training action output.")
        parser.add_argument("--command_x", type=float, default=None, help="Forward velocity command in m/s. Defaults to env.yaml/fallback 0.10.")
        parser.add_argument("--keyboard", action="store_true", help="Enable live PgUp/PgDn keyboard control for command_x.")
        parser.add_argument("--keyboard_step_x", type=float, default=0.02, help="PgUp/PgDn command_x increment in m/s.")
        parser.add_argument("--keyboard_min_x", type=float, default=-0.16, help="Minimum keyboard command_x in m/s.")
        parser.add_argument("--keyboard_max_x", type=float, default=0.16, help="Maximum keyboard command_x in m/s.")
        parser.add_argument("--command_deadband", type=float, default=COMMAND_DEADBAND_DEFAULT, help="Abs(command_x) below this uses crawl1 stand-still observations.")
        parser.add_argument("--auto_arm", action="store_true", help="Start policy automatically after safe-start. By default --keyboard waits for Enter/Space.")
        parser.add_argument("--crawl_frequency_hz", type=float, default=None, help="Crawl scheduler frequency. Defaults to env.yaml/fallback 0.50.")
        parser.add_argument("--crawl_duty_factor", type=float, default=None, help="Crawl stance duty factor. Defaults to env.yaml/fallback 0.80.")
        parser.add_argument("--crawl_swing_height", type=float, default=None, help="Crawl swing height observation. Defaults to env.yaml/fallback 0.06.")
        parser.add_argument("--kp", type=float, default=None, help="MIT position gain. Defaults to env.yaml stiffness.")
        parser.add_argument("--kd", type=float, default=None, help="MIT velocity gain. Defaults to env.yaml damping.")
        parser.add_argument("--warmup_s", type=float, default=2.0, help="Hold current motor positions before policy starts.")
        parser.add_argument("--duration_s", type=float, default=0.0, help="Run duration. Use 0 for infinite.")
        parser.add_argument("--tau_limit", type=float, default=16.0, help="Runtime safety torque stop threshold in Nm.")
        parser.add_argument("--active_joints", choices=("all", "rr"), default="all", help="Command all joints or only RR for staged tests.")
        parser.add_argument(
            "--default_mode",
            choices=("motor_zero", "current_pose"),
            default="motor_zero",
            help="motor_zero: motor q=0 maps to sim q=0; current_pose: startup pose maps to trained default pose.",
        )
        parser.add_argument("--motor_zero_positions", type=str, default="0,0,0,0,0,0,0,0", help="8 motor zero positions in joint order, radians.")
        parser.add_argument("--standup_s", type=float, default=6.0, help="Deprecated alias for --default_timeout_s when --default_timeout_s is omitted.")
        parser.add_argument("--default_rate_limit_deg_s", type=float, default=20.0, help="Rate limit used while moving from current pose to trained default pose.")
        parser.add_argument("--default_reached_threshold_deg", type=float, default=2.0, help="Max active-joint error to trained default before policy is allowed.")
        parser.add_argument("--default_reached_hold_s", type=float, default=0.5, help="Required continuous in-threshold time before policy is allowed.")
        parser.add_argument("--default_timeout_s", type=float, default=None, help="Abort if trained default is not reached within this time. Defaults to --standup_s.")
        # parser.add_argument("--default_timeout_s", type=float, default=10, help="Abort if trained default is not reached within this time. Defaults to --standup_s.")
        parser.add_argument("--dry_run", action="store_true", help="Read IMU and run policy, but do not open or command motors.")
        parser.add_argument("--gyro_signs", type=str, default="1,1,1", help="Per-axis signs for gyro x,y,z after IMU read.")
        parser.add_argument("--gravity_signs", type=str, default="1,1,1", help="Per-axis signs for projected gravity x,y,z after RPY conversion.")
        parser.add_argument("--roll_offset_deg", type=float, default=0.0, help="Subtract this roll bias before gravity projection.")
        parser.add_argument("--pitch_offset_deg", type=float, default=0.0, help="Subtract this pitch bias before gravity projection.")
        parser.add_argument("--yaw_offset_deg", type=float, default=0.0, help="Subtract this yaw bias before gravity projection.")
        args = parser.parse_args()

        policy_path = resolve_policy_path(args.policy)
        if not policy_path.exists():
            raise FileNotFoundError(policy_path)

        train_align = load_train_alignment(policy_path)
        train_rate_hz = train_align.get("policy_rate_hz")
        if args.policy_rate_hz is None:
            args.policy_rate_hz = train_rate_hz if train_rate_hz is not None else 50.0
            policy_rate_source = "env.yaml" if train_rate_hz is not None else "fallback"
        else:
            policy_rate_source = "user"

        if args.action_scale_deg is None:
            action_scale_rad = train_align.get("action_scale_rad", 0.30)
            action_scale_source = "env.yaml" if "action_scale_rad" in train_align else "fallback"
        else:
            action_scale_rad = math.radians(args.action_scale_deg)
            action_scale_source = "user"

        if args.command_x is None:
            if args.keyboard:
                args.command_x = 0.0
                command_x_source = "keyboard_default"
            else:
                args.command_x = train_align.get("command_x", 0.10)
                command_x_source = "env.yaml" if "command_x" in train_align else "fallback"
        else:
            command_x_source = "user"
        if args.crawl_frequency_hz is None:
            args.crawl_frequency_hz = train_align.get("crawl_frequency_hz", 0.50)
            crawl_frequency_source = "env.yaml" if "crawl_frequency_hz" in train_align else "fallback"
        else:
            crawl_frequency_source = "user"
        if args.crawl_duty_factor is None:
            args.crawl_duty_factor = train_align.get("crawl_duty_factor", 0.80)
            crawl_duty_source = "env.yaml" if "crawl_duty_factor" in train_align else "fallback"
        else:
            crawl_duty_source = "user"
        if args.crawl_swing_height is None:
            args.crawl_swing_height = train_align.get("crawl_swing_height", 0.06)
            crawl_swing_source = "env.yaml" if "crawl_swing_height" in train_align else "fallback"
        else:
            crawl_swing_source = "user"

        target_rate_limit_rad_s = math.radians(args.target_rate_limit_deg_s)
        default_rate_limit_rad_s = math.radians(args.default_rate_limit_deg_s)
        default_reached_threshold_rad = math.radians(args.default_reached_threshold_deg)
        if args.default_timeout_s is None:
            args.default_timeout_s = args.standup_s
        mit_kp = args.kp if args.kp is not None else train_align.get("stiffness", 40.0)
        mit_kd = args.kd if args.kd is not None else train_align.get("damping", 1.5)
        kp_source = "user" if args.kp is not None else ("env.yaml" if "stiffness" in train_align else "fallback")
        kd_source = "user" if args.kd is not None else ("env.yaml" if "damping" in train_align else "fallback")

        gyro_signs = parse_triplet(args.gyro_signs, name="--gyro_signs")
        gravity_signs = parse_triplet(args.gravity_signs, name="--gravity_signs")

        joint_names = [
            "FL_thigh", "FL_calf", "FR_thigh", "FR_calf",
            "RL_thigh", "RL_calf", "RR_thigh", "RR_calf",
        ]
        train_default_values = [0.14, -0.28, -0.14, -0.28, 0.14, -0.28, -0.14, -0.28]
        # train_default_values = [0,0,0,0,0,0,0,0]
        train_default_by_name = dict(zip(joint_names, train_default_values))
        motor_zero_values = [float(x.strip()) for x in args.motor_zero_positions.split(",") if x.strip()]
        if len(motor_zero_values) != len(joint_names):
            raise ValueError("--motor_zero_positions must contain 8 comma-separated values in joint order")
        configured_motor_zero_by_name = dict(zip(joint_names, motor_zero_values))
        MOTOR_SPECS = [
            {"name": "FL_thigh", "channel": 0, "can_id": 0x01, "mst_id": 0x11, "sim_to_motor": +1.0},
            {"name": "FL_calf",  "channel": 0, "can_id": 0x02, "mst_id": 0x12, "sim_to_motor": +1.0},
            {"name": "FR_thigh", "channel": 1, "can_id": 0x03, "mst_id": 0x13, "sim_to_motor": +1.0},
            {"name": "FR_calf",  "channel": 1, "can_id": 0x04, "mst_id": 0x14, "sim_to_motor": -1.0},
            {"name": "RL_thigh", "channel": 0, "can_id": 0x05, "mst_id": 0x15, "sim_to_motor": +1.0},
            {"name": "RL_calf",  "channel": 0, "can_id": 0x06, "mst_id": 0x16, "sim_to_motor": +1.0},
            {"name": "RR_thigh", "channel": 1, "can_id": 0x07, "mst_id": 0x17, "sim_to_motor": +1.0},
            {"name": "RR_calf",  "channel": 1, "can_id": 0x08, "mst_id": 0x18, "sim_to_motor": -1.0},
        ]
        active_joint_names = tuple(joint_names if args.active_joints == "all" else ("RR_thigh", "RR_calf"))
        init_data = [
            DmActData(DM_Motor_Type.DM8006, Control_Mode.MIT_MODE, spec["can_id"], spec["mst_id"], spec["channel"])
            for spec in MOTOR_SPECS
        ]
        spec_by_name = {spec["name"]: spec for spec in MOTOR_SPECS}

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("quad_leg_xu\quad_leg_crawl1.py requires torch in the active Python environment.") from exc

        policy = torch.jit.load(str(policy_path), map_location="cpu")
        policy.eval()

        imu = DMImuSerialReader(args.imu_port, args.imu_baudrate, exit_setting_mode=True)
        if not imu.wait_ready(timeout_s=2.0):
            raise RuntimeError(f"IMU did not provide gyro/euler frames on {args.imu_port}")

        policy_dt = 1.0 / args.policy_rate_hz
        desired_duration = 1.0 / 1000.0
        next_policy_time = time.monotonic()
        start_time = time.monotonic()
        loop_count = 0
        last_action = torch.zeros(8, dtype=torch.float32)
        action = torch.zeros(8, dtype=torch.float32)
        desired_joint_offsets = torch.zeros(8, dtype=torch.float32)
        applied_joint_offsets = torch.zeros(8, dtype=torch.float32)

        def imu_obs_terms():
            gyro_raw = imu.get(2)
            rpy_raw = imu.get(3)
            if gyro_raw is None or rpy_raw is None:
                raise RuntimeError("missing IMU gyro/euler data")
            gyro = [float(gyro_raw[i]) * gyro_signs[i] for i in range(3)]
            roll = float(rpy_raw[0]) - args.roll_offset_deg
            pitch = float(rpy_raw[1]) - args.pitch_offset_deg
            yaw = float(rpy_raw[2]) - args.yaw_offset_deg
            grav = list(projected_gravity_from_rpy_deg(roll, pitch, yaw))
            grav = [grav[i] * gravity_signs[i] for i in range(3)]
            base_ang_vel_obs = clipped_scaled(gyro, -10.0, 10.0, 0.2)
            projected_gravity_obs = clipped_scaled(grav, -1.0, 1.0, 1.0)
            return gyro, (roll, pitch, yaw), grav, base_ang_vel_obs, projected_gravity_obs

        def make_policy_obs(base_ang_vel_obs, projected_gravity_obs, joint_pos_obs, joint_vel_obs, last_action_tensor, phase_elapsed_s: float):
            vel_cmd, crawl_phase, leg_phase, desired_contacts, gait_params, phase = crawl_phase_terms(
                phase_elapsed_s,
                args.crawl_frequency_hz,
                args.crawl_duty_factor,
                args.crawl_swing_height,
                args.command_x,
                args.command_deadband,
            )
            obs_values = (
                base_ang_vel_obs
                + projected_gravity_obs
                + vel_cmd
                + joint_pos_obs
                + joint_vel_obs
                + last_action_tensor.tolist()
                + crawl_phase
                + leg_phase
                + desired_contacts
                + gait_params
            )
            if len(obs_values) != 50:
                raise RuntimeError(f"crawl1 observation length mismatch: {len(obs_values)} != 50")
            return torch.tensor(obs_values, dtype=torch.float32).unsqueeze(0), phase, desired_contacts

        print(f"[POLICY-IN-DAMIAO-6-crawl1] loaded: {policy_path}")
        if train_align.get("env_yaml") is not None:
            print(f"[TRAIN-ALIGN] env_yaml: {train_align['env_yaml']}")
        else:
            print("[TRAIN-ALIGN] env_yaml: not found; using fallback deployment defaults", file=sys.stderr)
        print(
            f"[TRAIN-ALIGN] policy_rate={args.policy_rate_hz:.3f}Hz({policy_rate_source}) "
            f"action_scale={math.degrees(action_scale_rad):.2f}deg({action_scale_source}) "
            f"target_rate_limit={args.target_rate_limit_deg_s:.2f}deg/s(user/fallback) "
            f"target_step={args.target_rate_limit_deg_s / args.policy_rate_hz:.4f}deg/step"
        )
        print(
            f"[TRAIN-ALIGN] MIT kp={mit_kp:.3f}({kp_source}) kd={mit_kd:.3f}({kd_source}) "
            f"output_scale={args.output_scale:.3f} active_joints={args.active_joints} dry_run={args.dry_run}"
        )
        keyboard_command = None
        if args.keyboard:
            keyboard_command = KeyboardCommandX(
                initial_x=args.command_x,
                min_x=args.keyboard_min_x,
                max_x=args.keyboard_max_x,
                step=args.keyboard_step_x,
                auto_arm=args.auto_arm,
            )
            args.command_x = keyboard_command.command_x

        print(
            f"[CRAWL] command_x={args.command_x:.3f}({command_x_source}) "
            f"frequency={args.crawl_frequency_hz:.3f}Hz({crawl_frequency_source}) "
            f"duty={args.crawl_duty_factor:.3f}({crawl_duty_source}) "
            f"swing_height={args.crawl_swing_height:.3f}m({crawl_swing_source})"
        )
        if keyboard_command is not None:
            print(
                f"[KEYBOARD] enabled: PgUp/PgDn adjust command_x by {args.keyboard_step_x:.3f} m/s, "
                f"range=[{args.keyboard_min_x:.3f}, {args.keyboard_max_x:.3f}], Home/0/X zero, "
                f"Enter/Space arm, Q/Esc quit, auto_arm={args.auto_arm}."
            )
        print("[OBS] order = base_ang_vel(3), projected_gravity(3), velocity_command(3), joint_pos_rel(8), joint_vel_rel(8), last_action(8), crawl_phase(2), crawl_leg_phase(8), desired_contacts(4), gait_params(3)")
        print("[JOINT-ORDER] " + ", ".join(joint_names))
        print("[TRAIN-DEFAULT] " + ", ".join(f"{name}:{math.degrees(train_default_by_name[name]):+.1f}deg" for name in joint_names))
        print(
            "[ACTION] q_cmd = motor_zero + sim_to_motor * (train_default + policy_action * action_scale * output_scale); "
            f"default_mode={args.default_mode} default_rate_limit={args.default_rate_limit_deg_s:.2f}deg/s "
            f"default_threshold={args.default_reached_threshold_deg:.2f}deg default_timeout={args.default_timeout_s:.2f}s"
        )
        print("[IMU] counts=" + str(imu.counts()))

        if args.dry_run:
            print("[DRY-RUN] motors are not opened. Joint pos/vel terms are zeros.")
            while running.is_set():
                now = time.monotonic()
                elapsed_s = now - start_time
                if args.duration_s > 0.0 and elapsed_s >= args.duration_s:
                    break
                if now >= next_policy_time:
                    gyro, rpy, grav, base_ang_vel_obs, projected_gravity_obs = imu_obs_terms()
                    if keyboard_command is not None:
                        args.command_x = keyboard_command.poll()
                    obs, phase, desired_contacts = make_policy_obs(
                        base_ang_vel_obs,
                        projected_gravity_obs,
                        [0.0] * 8,
                        [0.0] * 8,
                        last_action,
                        elapsed_s,
                    )
                    if keyboard_command is not None and not keyboard_command.armed:
                        action = torch.zeros(8, dtype=torch.float32)
                    else:
                        with torch.no_grad():
                            action = policy(obs).squeeze(0).to(torch.float32).clamp(-1.0, 1.0)
                        if action.numel() != 8:
                            raise RuntimeError(f"policy action length mismatch: {action.numel()} != 8")
                    last_action = action.clone()
                    desired_joint_offsets = action * action_scale_rad * float(args.output_scale)
                    if loop_count % max(1, int(args.policy_rate_hz // 2)) == 0:
                        print(
                            f"armed={keyboard_command.armed if keyboard_command is not None else True} cmd_x={args.command_x:.3f} phase={phase:.3f} contacts={fmt_vec(desired_contacts, 0)} "
                            f"imu_rpy_deg={fmt_vec(rpy)} gyro={fmt_vec(gyro, 4)} proj_g={fmt_vec(grav)} "
                            f"action={fmt_vec(action.tolist())} residual_deg={fmt_vec([math.degrees(float(x)) for x in desired_joint_offsets])}"
                        )
                    next_policy_time += policy_dt
                    loop_count += 1
                time.sleep(0.001)
            raise SystemExit(0)

        control = Motor_Control(
            1000000,
            5000000,
            "D6977F56F86C64B77B316E7154FA6DF3",
            init_data,
            device_type=dmcan_device_type.USB2CANFD_DUAL,
        )
        motors = {spec["name"]: control.getMotor(spec["channel"], spec["can_id"]) for spec in MOTOR_SPECS}
        active_motors = {name: motors[name] for name in active_joint_names}

        feedback_start = {name: motors[name].last_time_ for name in joint_names}
        feedback_deadline = time.monotonic() + 1.0
        while time.monotonic() < feedback_deadline:
            for name in joint_names:
                motor = motors[name]
                control.refresh_motor_status(motor)
                control.control_mit(motor, 0.0, 1.5, 0.0, 0.0, 0.0)
            time.sleep(0.01)
            if all(motors[name].last_time_ > feedback_start[name] for name in joint_names):
                break
        if not all(motors[name].last_time_ > feedback_start[name] for name in joint_names):
            missing = [name for name in joint_names if motors[name].last_time_ <= feedback_start[name]]
            control.disable_all()
            raise RuntimeError(f"no fresh motor feedback during safe start: {missing}")

        startup_motor_pos = {name: motors[name].Get_Position() for name in joint_names}
        if args.default_mode == "motor_zero":
            motor_zero_by_name = dict(configured_motor_zero_by_name)
        else:
            # Make the current measured motor pose correspond to the trained default sim pose.
            motor_zero_by_name = {
                name: startup_motor_pos[name] - spec_by_name[name]["sim_to_motor"] * train_default_by_name[name]
                for name in joint_names
            }

        def motor_to_sim_pos(name: str, motor: Motor) -> float:
            spec = spec_by_name[name]
            return (motor.Get_Position() - motor_zero_by_name[name]) / spec["sim_to_motor"]

        def motor_to_policy_pos_rel(name: str, motor: Motor) -> float:
            return motor_to_sim_pos(name, motor) - train_default_by_name[name]

        def motor_to_policy_vel_rel(name: str, motor: Motor) -> float:
            spec = spec_by_name[name]
            return motor.Get_Velocity() / spec["sim_to_motor"]

        startup_sim_pos = torch.tensor([motor_to_sim_pos(name, motors[name]) for name in joint_names], dtype=torch.float32)
        train_default = torch.tensor([train_default_by_name[name] for name in joint_names], dtype=torch.float32)

        print(
            "[SAFE-START] startup_motor_pos_rad="
            + ", ".join(f"{name}:{startup_motor_pos[name]:+.4f}" for name in joint_names),
            file=sys.stderr,
        )
        print(
            "[SAFE-START] motor_zero_rad="
            + ", ".join(f"{name}:{motor_zero_by_name[name]:+.4f}" for name in joint_names),
            file=sys.stderr,
        )
        print(
            "[SAFE-START] startup_sim_deg="
            + ", ".join(f"{name}:{math.degrees(float(startup_sim_pos[i])):+.1f}" for i, name in enumerate(joint_names)),
            file=sys.stderr,
        )
        print(
            f"[SAFE-START] default_mode={args.default_mode}; first command equals current pose, "
            f"then rate-limit toward trained default at {args.default_rate_limit_deg_s:.2f}deg/s. "
            f"Policy starts only after active joints stay within {args.default_reached_threshold_deg:.2f}deg "
            f"for {args.default_reached_hold_s:.2f}s, then warmup {args.warmup_s:.2f}s. "
            f"default_timeout={args.default_timeout_s:.2f}s.",
            file=sys.stderr,
        )

        # Hardware timing starts only after feedback/default mapping is known. This prevents init time
        # from consuming standup_s and avoids first-frame jumps to the trained default pose.
        start_time = time.monotonic()
        next_policy_time = start_time
        loop_count = 0
        target_sim_pos = startup_sim_pos.clone()
        last_action = torch.zeros(8, dtype=torch.float32)
        action = torch.zeros(8, dtype=torch.float32)
        desired_joint_offsets = torch.zeros(8, dtype=torch.float32)
        applied_joint_offsets = torch.zeros(8, dtype=torch.float32)
        active_indices = torch.tensor([joint_names.index(name) for name in active_joint_names], dtype=torch.long)
        startup_phase = "ramp_default"
        default_reached_since = None
        warmup_start_time = None
        policy_phase_start_time = None
        last_phase = 0.0
        last_desired_contacts = [1.0, 1.0, 1.0, 1.0]
        last_control_time = start_time

        while running.is_set():
            current_time = time.perf_counter()
            now = time.monotonic()
            elapsed_s = now - start_time
            if args.duration_s > 0.0 and elapsed_s >= args.duration_s:
                break
            loop_count += 1

            real_offsets = torch.tensor([motor_to_policy_pos_rel(name, motors[name]) for name in joint_names], dtype=torch.float32)
            real_vels = torch.tensor([motor_to_policy_vel_rel(name, motors[name]) for name in joint_names], dtype=torch.float32)

            control_dt = max(0.0, min(now - last_control_time, 0.02))
            last_control_time = now

            if startup_phase == "ramp_default":
                action = torch.zeros(8, dtype=torch.float32)
                desired_joint_offsets = torch.zeros(8, dtype=torch.float32)
                applied_joint_offsets = torch.zeros(8, dtype=torch.float32)
                max_default_delta = default_rate_limit_rad_s * control_dt
                target_sim_pos += torch.clamp(train_default - target_sim_pos, -max_default_delta, max_default_delta)
                last_action = action.clone()
                next_policy_time = now + policy_dt

                target_error_rad = torch.max(torch.abs((target_sim_pos - train_default).index_select(0, active_indices))).item()
                real_error_rad = torch.max(torch.abs(real_offsets.index_select(0, active_indices))).item()
                if target_error_rad <= default_reached_threshold_rad and real_error_rad <= default_reached_threshold_rad:
                    if default_reached_since is None:
                        default_reached_since = now
                    elif now - default_reached_since >= args.default_reached_hold_s:
                        startup_phase = "warmup"
                        warmup_start_time = now
                        target_sim_pos = train_default.clone()
                        print(
                            f"[SAFE-START] trained default reached: real_err={math.degrees(real_error_rad):.2f}deg "
                            f"target_err={math.degrees(target_error_rad):.2f}deg; warmup {args.warmup_s:.2f}s before policy.",
                            file=sys.stderr,
                        )
                else:
                    default_reached_since = None

                if elapsed_s > args.default_timeout_s:
                    control.disable_all()
                    raise RuntimeError(
                        f"default pose not reached within {args.default_timeout_s:.2f}s: "
                        f"real_err={math.degrees(real_error_rad):.2f}deg "
                        f"target_err={math.degrees(target_error_rad):.2f}deg"
                    )
            elif startup_phase == "warmup":
                action = torch.zeros(8, dtype=torch.float32)
                desired_joint_offsets = torch.zeros(8, dtype=torch.float32)
                applied_joint_offsets = torch.zeros(8, dtype=torch.float32)
                target_sim_pos = train_default.clone()
                last_action = action.clone()
                next_policy_time = now + policy_dt
                if warmup_start_time is not None and now - warmup_start_time >= args.warmup_s:
                    if keyboard_command is not None and not keyboard_command.armed:
                        startup_phase = "wait_arm"
                        next_policy_time = now + policy_dt
                        print("[SAFE-START] default pose ready; waiting for Enter/Space to ARM policy.", file=sys.stderr)
                    else:
                        startup_phase = "policy"
                        policy_phase_start_time = now
                        next_policy_time = now
                        print("[SAFE-START] policy enabled after default-pose gate; crawl phase reset to 0.", file=sys.stderr)
            elif startup_phase == "wait_arm":
                action = torch.zeros(8, dtype=torch.float32)
                desired_joint_offsets = torch.zeros(8, dtype=torch.float32)
                applied_joint_offsets = torch.zeros(8, dtype=torch.float32)
                target_sim_pos = train_default.clone()
                last_action = action.clone()
                if keyboard_command is not None:
                    args.command_x = keyboard_command.poll()
                next_policy_time = now + policy_dt
                if keyboard_command is None or keyboard_command.armed:
                    startup_phase = "policy"
                    policy_phase_start_time = now
                    next_policy_time = now
                    print("[SAFE-START] policy ARMED by keyboard; crawl1 phase reset to 0.", file=sys.stderr)
            elif now >= next_policy_time:
                gyro, rpy, grav, base_ang_vel_obs, projected_gravity_obs = imu_obs_terms()
                if keyboard_command is not None:
                    args.command_x = keyboard_command.poll()
                joint_pos_obs = clipped_scaled(real_offsets.tolist(), -1.0, 1.0, 1.0)
                joint_vel_obs = clipped_scaled(real_vels.tolist(), -20.0, 20.0, 0.05)
                if policy_phase_start_time is None:
                    policy_phase_start_time = now
                obs, last_phase, last_desired_contacts = make_policy_obs(
                    base_ang_vel_obs,
                    projected_gravity_obs,
                    joint_pos_obs,
                    joint_vel_obs,
                    last_action,
                    now - policy_phase_start_time,
                )
                with torch.no_grad():
                    action = policy(obs).squeeze(0).to(torch.float32).clamp(-1.0, 1.0)
                if action.numel() != 8:
                    raise RuntimeError(f"policy action length mismatch: {action.numel()} != 8")
                desired_joint_offsets = action * action_scale_rad * float(args.output_scale)
                max_delta = target_rate_limit_rad_s * policy_dt
                applied_joint_offsets += torch.clamp(desired_joint_offsets - applied_joint_offsets, -max_delta, max_delta)
                target_sim_pos = train_default + applied_joint_offsets
                last_action = action.clone()
                next_policy_time += policy_dt
                if next_policy_time < now - policy_dt:
                    next_policy_time = now + policy_dt
            q_cmd = {}
            for i, name in enumerate(joint_names):
                spec = spec_by_name[name]
                q_cmd[name] = motor_zero_by_name[name] + spec["sim_to_motor"] * float(target_sim_pos[i])

            for name in active_joint_names:
                motor = active_motors[name]
                if motor.Get_err() not in (0, 1):
                    control.disable_all()
                    raise RuntimeError(f"{name} motor error: {motor.Get_err()}")
                if loop_count > 100 and abs(motor.Get_tau()) > args.tau_limit:
                    control.disable_all()
                    raise RuntimeError(f"{name} tau too high: {motor.Get_tau():.3f}")
                control.control_mit(motor, mit_kp, mit_kd, q_cmd[name], 0.0, 0.0)

            if loop_count % 200 == 0:
                tau_max = max(abs(motors[name].Get_tau()) for name in active_joint_names)
                print(
                    f"cmd_x={args.command_x:.3f} phase={last_phase:.3f} contacts={fmt_vec(last_desired_contacts, 0)} "
                    f"act={fmt_vec(action.tolist(), 2)} "
                    f"target_sim_deg={fmt_vec([math.degrees(float(x)) for x in target_sim_pos], 1)} "
                    f"rel_deg={fmt_vec([math.degrees(float(x)) for x in real_offsets], 1)} "
                    f"tau_max={tau_max:.3f}"
                )

            sleep_till = current_time + desired_duration
            sleep_now = time.perf_counter()
            if sleep_till > sleep_now:
                time.sleep(sleep_till - sleep_now)

        print("The program exited safely.")
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error: hardware interface exception: {e}", file=sys.stderr)
    finally:
        try:
            if control is not None:
                control.close()
        except Exception:
            pass
        try:
            if imu is not None:
                imu.close()
        except Exception:
            pass



