# 单腿RL policy sim2real
# 把 policy 集成进 damiao_3.py
# 优点：单进程，理论延迟低。
# 缺点：达妙 DLL + Torch + conda 环境混在一起，很容易出现 DLL、Python 版本、USB 库冲突。现在不建议。
# 改两处：
# damiao_3.py：每隔比如 50Hz 或 100Hz 把 m7/m8 真实反馈 UDP 发到 127.0.0.1:15002。
# single_leg_policy_udp_runner.py：绑定 15002 接收反馈，用真实反馈构造 observation。

# 重新训练后，搜索policy，接着改动

# 终端直接输入
# D:\Conda\envs\env_isaaclab\python.exe .\damiao_3.py --output_scale 1 --policy_rate_hz 50 

# 当前版本：8 个电机都会注册，但只有 RR_thigh/canid7 和 RR_calf/canid8 会实际发控制命令。
# Use this only when the Python environment can load both dmcan and torch.

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
    dll_dir = os.path.join(base_dir, "dlls")
    os.add_dll_directory(base_dir)
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


DEFAULT_POLICY = (
    # r"D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_15-43-00\exported\policy.pt"   #50Hz  
    # r"D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_18-45-06\exported\policy.pt"   #250Hz
    # r"D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_21-23-57\exported\policy.pt"   #100Hz
    # r"D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_21-49-38\exported\policy.pt"   #150Hz

    # r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\single_leg_rr_trace\2026-06-27_22-54-27_50hz\exported\policy.pt" # 50Hz 明显卡顿
    # r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\single_leg_rr_trace\2026-06-27_23-43-06_150hz\exported\policy.pt" # 150Hz  轻微卡顿
    # r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\single_leg_rr_trace\2026-06-28_11-00-04_200hz\exported\policy.pt" # 200Hz
    r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\single_leg_rr_trace\2026-06-28_11-27-13_250hz\exported\policy.pt" # 250Hz
    # r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\single_leg_rr_trace\2026-06-28_12-04-46_500hz\exported\policy.pt" # 500Hz

)


def build_reference(elapsed_s: float, amplitude_rad: float, max_speed_rad_s: float) -> tuple[float, float, float, float]:
    """Return phase sin/cos and thigh/calf reference offsets."""
    frequency_hz = max_speed_rad_s / max(2.0 * math.pi * amplitude_rad, 1.0e-6)
    phase = 2.0 * math.pi * frequency_hz * elapsed_s
    phase_sin = math.sin(phase)
    phase_cos = math.cos(phase)
    thigh_ref = amplitude_rad * phase_sin
    calf_ref = amplitude_rad * math.sin(phase + math.pi / 2.0)
    return phase_sin, phase_cos, thigh_ref, calf_ref


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


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Integrated Bennett single-leg policy sim2real on 8-joint motor map; RR active only.")
        parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="Exported TorchScript policy.pt path.")
        parser.add_argument("--policy_rate_hz", type=float, default=50.0, help="Policy inference rate.")
        parser.add_argument("--output_scale", type=float, default=1, help="Extra safety scale for policy output.")
        parser.add_argument("--amplitude_deg", type=float, default=20.0, help="Reference trajectory amplitude.")
        parser.add_argument("--speed_deg_s", type=float, default=35.0, help="Reference speed and trained action target rate.")
        parser.add_argument("--warmup_s", type=float, default=2.0, help="Hold current target before policy starts.")
        parser.add_argument("--duration_s", type=float, default=0.0, help="Run duration. Use 0 for infinite.")
        parser.add_argument("--tau_limit", type=float, default=3.2, help="Safety torque limit in Nm.")
        args = parser.parse_args()

        MOTOR_SPECS = [
            {"name": "FL_thigh", "channel": 0, "can_id": 0x01, "mst_id": 0x11, "default": 0.0, "sim_to_motor": -1.0},
            {"name": "FL_calf",  "channel": 0, "can_id": 0x02, "mst_id": 0x12, "default": 0.0, "sim_to_motor": -1.0},
            {"name": "FR_thigh", "channel": 1, "can_id": 0x03, "mst_id": 0x13, "default": 0.0, "sim_to_motor": +1.0},
            {"name": "FR_calf",  "channel": 1, "can_id": 0x04, "mst_id": 0x14, "default": 0.0, "sim_to_motor": -1.0},
            {"name": "RL_thigh", "channel": 0, "can_id": 0x05, "mst_id": 0x15, "default": 0.0, "sim_to_motor": -1.0},
            {"name": "RL_calf",  "channel": 0, "can_id": 0x06, "mst_id": 0x16, "default": 0.0, "sim_to_motor": -1.0},
            {"name": "RR_thigh", "channel": 1, "can_id": 0x07, "mst_id": 0x17, "default": 0.0, "sim_to_motor": +1.0},
            {"name": "RR_calf",  "channel": 1, "can_id": 0x08, "mst_id": 0x18, "default": 0.0, "sim_to_motor": -1.0},
        ]
        ACTIVE_JOINTS = ("RR_thigh", "RR_calf")
        # FL/FR/RL joints are registered but intentionally not commanded until those legs are trained.
        init_data = [
            DmActData(
                motorType=DM_Motor_Type.DM8006,
                mode=Control_Mode.MIT_MODE,
                can_id=spec["can_id"],
                mst_id=spec["mst_id"],
                channel=spec["channel"],
            )
            for spec in MOTOR_SPECS
        ]
        spec_by_name = {spec["name"]: spec for spec in MOTOR_SPECS}
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("damiao_3.py requires torch in the active Python environment.") from exc

        policy_path = Path(args.policy)
        if not policy_path.exists():
            raise FileNotFoundError(policy_path)
        policy = torch.jit.load(str(policy_path), map_location="cpu")
        policy.eval()

        action_scale_rad = math.radians(20.0)
        amplitude_rad = math.radians(args.amplitude_deg)
        max_speed_rad_s = math.radians(args.speed_deg_s)
        policy_dt = 1.0 / args.policy_rate_hz
        desired_duration = 1.0 / 1000.0 
        joint_limit = math.radians(20.0)
        next_policy_time = time.monotonic()
        start_time = time.monotonic()
        loop_count = 0
        last_action = torch.zeros(2, dtype=torch.float32)
        action = torch.zeros(2, dtype=torch.float32)
        desired_offset = torch.zeros(2, dtype=torch.float32)

        with Motor_Control(1000000, 5000000, "D6977F56F86C64B77B316E7154FA6DF3", init_data,
            device_type=dmcan_device_type.USB2CANFD_DUAL) as control:
            motors = {spec["name"]: control.getMotor(spec["channel"], spec["can_id"]) for spec in MOTOR_SPECS}
            active_motors = {name: motors[name] for name in ACTIVE_JOINTS}

            feedback_start = {name: active_motors[name].last_time_ for name in ACTIVE_JOINTS}
            feedback_deadline = time.monotonic() + 1.0
            while time.monotonic() < feedback_deadline:
                for motor in active_motors.values():
                    control.refresh_motor_status(motor)
                    control.control_mit(motor, 0.0, 1.5, 0.0, 0.0, 0.0)
                time.sleep(0.01)
                if all(active_motors[name].last_time_ > feedback_start[name] for name in ACTIVE_JOINTS):
                    break
            if not all(active_motors[name].last_time_ > feedback_start[name] for name in ACTIVE_JOINTS):
                control.disable_all()
                raise RuntimeError("no fresh RR_thigh/RR_calf feedback during safe start")

            real_thigh_offset = active_motors["RR_thigh"].Get_Position() - spec_by_name["RR_thigh"]["default"]
            real_calf_offset = -(active_motors["RR_calf"].Get_Position() - spec_by_name["RR_calf"]["default"])
            applied_offset = torch.tensor([real_thigh_offset, real_calf_offset], dtype=torch.float32)
            desired_offset = applied_offset.clone()

            print(f"[POLICY-IN-DAMIAO] loaded: {policy_path}")
            print(
                f"[POLICY-IN-DAMIAO] active=RR_thigh/RR_calf rate={args.policy_rate_hz:.1f}Hz "
                f"output_scale={args.output_scale:.3f} tau_limit={args.tau_limit:.2f}Nm"
            )
            print(
                f"[SAFE-START] initial_RR_offset_deg=({math.degrees(real_thigh_offset):+.2f},"
                f"{math.degrees(real_calf_offset):+.2f})",
                file=sys.stderr,
            )

            while running.is_set():
                current_time = time.perf_counter()
                now = time.monotonic()
                elapsed_s = now - start_time
                if args.duration_s > 0.0 and elapsed_s >= args.duration_s:
                    break
                loop_count += 1

                real_thigh_offset = active_motors["RR_thigh"].Get_Position() - spec_by_name["RR_thigh"]["default"]
                real_calf_offset = -(active_motors["RR_calf"].Get_Position() - spec_by_name["RR_calf"]["default"])
                real_thigh_vel = active_motors["RR_thigh"].Get_Velocity()
                real_calf_vel = -active_motors["RR_calf"].Get_Velocity()

                if now >= next_policy_time:
                    real_offset = torch.tensor([real_thigh_offset, real_calf_offset], dtype=torch.float32)
                    real_vel = torch.tensor([real_thigh_vel, real_calf_vel], dtype=torch.float32)
                    if elapsed_s < args.warmup_s:
                        action = torch.zeros(2, dtype=torch.float32)
                        desired_offset = applied_offset.clone()
                    else:
                        policy_elapsed_s = elapsed_s - args.warmup_s
                        phase_sin, phase_cos, thigh_ref, calf_ref = build_reference(
                            policy_elapsed_s, amplitude_rad, max_speed_rad_s
                        )
                        ref = torch.tensor([thigh_ref, calf_ref], dtype=torch.float32)
                        tracking_error = real_offset - ref
                        obs = torch.cat(
                            (
                                torch.tensor([phase_sin, phase_cos], dtype=torch.float32),
                                ref,
                                tracking_error,
                                real_offset,
                                real_vel,
                                last_action,
                            )
                        ).unsqueeze(0)
                        with torch.no_grad():
                            action = policy(obs).squeeze(0).to(torch.float32).clamp(-1.0, 1.0)
                        desired_offset = torch.clamp(action * action_scale_rad * float(args.output_scale), -joint_limit, joint_limit)
                        max_delta = max_speed_rad_s * policy_dt
                        applied_offset += torch.clamp(desired_offset - applied_offset, -max_delta, max_delta)
                    last_action = action.clone()
                    next_policy_time += policy_dt
                    if next_policy_time < now - policy_dt:
                        next_policy_time = now + policy_dt

                q_cmd = {
                    "RR_thigh": spec_by_name["RR_thigh"]["default"] + applied_offset[0].item(),
                    "RR_calf": spec_by_name["RR_calf"]["default"] - applied_offset[1].item(),
                }

                for name in ACTIVE_JOINTS:
                    motor = active_motors[name]
                    if motor.Get_err() not in (0, 1):
                        control.disable_all()
                        raise RuntimeError(f"{name} motor error: {motor.Get_err()}")
                    if loop_count > 100 and abs(motor.Get_tau()) > args.tau_limit:
                        control.disable_all()
                        raise RuntimeError(f"{name} tau too high: {motor.Get_tau():.3f}")
                    control.control_mit(motor, 40.0, 1.5, q_cmd[name], 0.0, 0.0)

                if loop_count % 200 == 0:
                    print(
                        f"target_deg=({math.degrees(float(applied_offset[0].item())):+.1f},"
                        f"{math.degrees(float(applied_offset[1].item())):+.1f}) "
                        f"desired_deg=({math.degrees(float(desired_offset[0].item())):+.1f},"
                        f"{math.degrees(float(desired_offset[1].item())):+.1f}) "
                        f"real_deg=({math.degrees(real_thigh_offset):+.1f},{math.degrees(real_calf_offset):+.1f}) "
                        f"action=({action[0].item():+.2f},{action[1].item():+.2f}) "
                        f"cmd7:{q_cmd['RR_thigh']:+.3f} cmd8:{q_cmd['RR_calf']:+.3f} "
                        f"m7 tau:{active_motors['RR_thigh'].Get_tau():+.3f} "
                        f"m8 tau:{active_motors['RR_calf'].Get_tau():+.3f}"
                    )

                sleep_till = current_time + desired_duration
                sleep_now = time.perf_counter()
                if sleep_till > sleep_now:
                    time.sleep(sleep_till - sleep_now)

        print("The program exited safely.")
    except Exception as e:
        print(f"Error: hardware interface exception: {e}", file=sys.stderr)
