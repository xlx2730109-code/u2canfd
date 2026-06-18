from __future__ import annotations

import os
import sys
import ctypes
import json
import socket

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

from dmcan import DmCanContext, dmcan_channel_can_info, dmcan_device_type, usb_rx_frame


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


if __name__ == "__main__":  #在main函数里定义id变量，并且在try块里进行电机控制的初始化和循环控制
    try:
        init_data1= []
        init_data2 = []
        canid1=0x01
        mstid1=0x11
        canid2=0x02
        mstid2=0x12
        canid3=0x03
        mstid3=0x13
        canid4=0x04
        mstid4=0x14
        canid5=0x05
        mstid5=0x15
        canid6=0x06
        mstid6=0x16
        canid7=0x07
        mstid7=0x17
        canid8=0x08
        mstid8=0x18
        canid9=0x09
        mstid9=0x19
        #定义电机信息列表：
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM8006,  # 你的电机型号
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid1,
                    mst_id=mstid1))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM8006,  # 或者具体类型，如 DM_Motor_Type.DM8006
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid2,
                    mst_id=mstid2))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM8006,  # 或者具体类型，如 DM_Motor_Type.DM8006
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid3,
                    mst_id=mstid3))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM8006,  # 或者具体类型，如 DM_Motor_Type.DM8006
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid4,
                    mst_id=mstid4))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM8006,  # 或者具体类型，如 DM_Motor_Type.DM8006
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid5,
                    mst_id=mstid5))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM8006,  # 或者具体类型，如 DM_Motor_Type.DM8006
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid6,
                    mst_id=mstid6))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM8006,  # 或者具体类型，如 DM_Motor_Type.DM8006
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid7,
                    mst_id=mstid7))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM8006,  # 或者具体类型，如 DM_Motor_Type.DM8006
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid8,
                    mst_id=mstid8))
        # init_data1.append(DmActData(
        #             motorType=DM_Motor_Type.DM8006,  # 或者具体类型，如 DM_Motor_Type.DM8006
        #             mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
        #             can_id=canid9,
        #             mst_id=mstid9))
       
        # with Motor_Control(1000000, 5000000, "EC5BDB0C47494471B94B9E637DF39DA1", init_data1,device_type=dmcan_device_type.USB2CANFD_DUAL) as control:
        #USB-CANFD 设备 SN
        device_sn = "D6977F56F86C64B77B316E7154FA6DF3"
        #CANFD 波特率 1000000 是仲裁段 1M，5000000 是数据段 5M。你、按文档 5M 就保持这样。初始化电机控制结构体
        with Motor_Control(1000000, 5000000, device_sn, init_data1,device_type=None) as control:
            # control.set_zero_position(control.getMotor(canid1)) # 设置电机零位
            # control.set_zero_position(control.getMotor(canid2))
            # control.set_zero_position(control.getMotor(canid3))
            # control.set_zero_position(control.getMotor(canid4))
            # control.set_zero_position(control.getMotor(canid5))
            # control.set_zero_position(control.getMotor(canid6))
            # control.set_zero_position(control.getMotor(canid7))
            # control.set_zero_position(control.getMotor(canid8))
            # control.set_zero_position(control.getMotor(canid9))
            loop_count = 0  #循环计数器
            fr_thigh_offset = 0.0   # FR_thigh 偏移量
            fr_calf_offset = 0.0   # FR_calf 偏移量
            fr_key_step = 3.141592653589793 / 180.0  # 1deg joint per key press
            fr_limit = 20.0 * 3.141592653589793 / 180.0 # 20deg joint limit
            udp_timeout_s = 0.5 # UDP 超时 0.5 秒
            udp_tau_limit = 2.0 # UDP tau limit 2.0 Nm
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_sock.bind(("127.0.0.1", 15001))
            udp_sock.setblocking(False)
            last_udp_time = time.monotonic()
            print("[UDP] listening on 127.0.0.1:15001; IsaacSim FR -> real RR canid7/canid8")
            while running.is_set():
                    #控制周期 现在是 10ms，即 100Hz(1/0.01s=100hz)。后面多电机测试稳定后再考虑更快。
                    desired_duration = 0.01  # 秒
                    current_time = time.perf_counter()
                    loop_count += 1

                    # control.control_mit(control.getMotor(0,canid3), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(1,canid4), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid1), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid2), 0.0, 0.0, 0.0, 0.0, 0.0)

                    #kp = 0.0； kd = 1.0； q = 0.0，因为 kp=0，位置目标基本不起作用 ；dq = 1.0，目标速度； tau = 0.0
                    # control.control_mit(control.getMotor(canid1), 0.0, 1.0, 0.0, 0.5, 0)
                    # control.control_mit(control.getMotor(canid2), 0.0, 1.0, 0.0, 0.7, 0)
                    # control.control_mit(control.getMotor(canid3), 0.0, 1.5, 0.0, 0.5, 0)
                    # control.control_mit(control.getMotor(canid4), 0.0, 1.5, 0.0, 0.7, 0)
                    # old: 只读 canid3/canid4 位置，用来记录默认姿态
                    # control.refresh_motor_status(control.getMotor(canid3))
                    # control.refresh_motor_status(control.getMotor(canid4))

                    # old: FR_thigh(canid3) 默认姿态附近 3 度安全慢速摆动测试
                    # old: 约定：FR_thigh -> canid3, FR_calf -> canid4, sign=-1, ratio=6:1
                    # old: m3_default = 1.637
                    # old: m4_default = 1.823
                    # old: m3_target = 1.323  # 1.637 - 6 * radians(3deg)
                    # old: m4_target = 1.823

                    # old: 按当前实测位置测试 canid3；canid4 只刷新状态，不做位置保持
                    # old: FR_thigh(canid3): 3deg joint = 6 * 0.05236 = 0.314rad motor
                    # old: m3_default = 1.455
                    # old: m3_target = 1.141  # 1.455 - 6 * radians(3deg)

                    # old: 测试 FR_calf(canid4)，canid3 只刷新状态
                    # old: FR_calf(canid4): 3deg joint = 6 * 0.05236 = 0.314rad motor
                    # old: m4_default = 1.628
                    # old: m4_target = 1.314  # 1.628 - 6 * radians(3deg)

                    # old: FR_thigh(canid3) + FR_calf(canid4) 同时 3 度慢速往返
                    # old: 3deg joint = 6 * 0.05236 = 0.314rad motor
                    # old: m3_target = 1.141  # 1.455 - 6 * radians(3deg)
                    # old: m4_target = 1.314  # 1.628 - 6 * radians(3deg)

                    # old: FR_thigh(canid3) + FR_calf(canid4) 同时 8 度慢速往返
                    # old: 8deg joint = 6 * 0.13963 = 0.838rad motor
                    # old: m3_target = 0.617  # 1.455 - 6 * radians(8deg)
                    # old: m4_target = 0.790  # 1.628 - 6 * radians(8deg)

                    # old: FR_thigh(canid3) + FR_calf(canid4) 同时 14 度慢速往返
                    # old: 14deg joint = 6 * 0.24435 = 1.466rad motor
                    # old: m3_target = -0.011  # 1.455 - 6 * radians(14deg)
                    # old: m4_target = 0.162  # 1.628 - 6 * radians(14deg)

                    # old: FR_thigh(canid3) + FR_calf(canid4) 同时 20 度慢速往返
                    # old: 20deg joint = 6 * 0.34907 = 2.094rad motor
                    # old: m3_target = -0.639  # 1.455 - 6 * radians(20deg)
                    # old: m4_target = -0.466  # 1.628 - 6 * radians(20deg)

                    # old: 只读 canid7/canid8，用来记录 RR-as-FR 默认姿态电机位置
                    # old: m7_default = 0.089
                    # old: m8_default = 1.433

                    # old: 键盘手动控制真实 RR 单腿临时冒充仿真 FR 腿
                    # old: PowerShell 方向键直接改 fr_thigh_offset/fr_calf_offset

                    # 新：接收 IsaacSim UDP，仿真 FR 腿目标同步到真实 RR 腿 canid7/canid8
                    # q_motor = q_motor_default + sign * ratio * q_joint_offset, sign=-1, ratio=6
                    m7_default = 0.089  # canid7 的默认位置，实测值，保持不变
                    m8_default = 1.433

                    motor7 = control.getMotor(canid7)
                    motor8 = control.getMotor(canid8)

                    while True:
                        try:
                            packet, _addr = udp_sock.recvfrom(4096)
                        except BlockingIOError:
                            break
                        try:
                            msg = json.loads(packet.decode("ascii"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        if msg.get("leg") != "FR":
                            continue
                        fr_thigh_offset = float(msg.get("thigh_offset_rad", fr_thigh_offset))
                        fr_calf_offset = float(msg.get("calf_offset_rad", fr_calf_offset))
                        last_udp_time = time.monotonic()

                    fr_thigh_offset = max(-fr_limit, min(fr_limit, fr_thigh_offset))
                    fr_calf_offset = max(-fr_limit, min(fr_limit, fr_calf_offset))
                    udp_age = time.monotonic() - last_udp_time

                    # old: q7_cmd = m7_default - 6.0 * fr_thigh_offset
                    q7_cmd = m7_default + 6.0 * fr_thigh_offset
                    q8_cmd = m8_default - 6.0 * fr_calf_offset

                    tau7 = motor7.Get_tau()
                    tau8 = motor8.Get_tau()
                    if motor7.Get_err() not in (0, 1) or motor8.Get_err() not in (0, 1):
                        control.disable_all()
                        raise RuntimeError(f"motor error: m7={motor7.Get_err()} m8={motor8.Get_err()}")
                    if loop_count > 100 and (abs(tau7) > udp_tau_limit or abs(tau8) > udp_tau_limit):
                        control.disable_all()
                        raise RuntimeError(f"tau too high: m7={tau7:.3f} m8={tau8:.3f}")

                    # UDP 超时就保持最后目标，不继续变化；收到新包后会自动更新
                    control.control_mit(motor7, 35.0, 1.5, q7_cmd, 0.0, 0.0)
                    control.control_mit(motor8, 35.0, 1.5, q8_cmd, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid5), 0.0, 1.0, 0.0, 0.5, 0)
                    # control.control_mit(control.getMotor(canid6), 0.0, 1.0, 0.0, 0.5, 0)
                    # control.control_mit(control.getMotor(canid7), 0.0, 1.5, 0.0, 0.7, 0)
                    # control.control_mit(control.getMotor(canid8), 0.0, 1.0, 0.0, 0.5, 0)

                    #状态打印  每 50 次打印一次。100Hz 下就是 0.5 秒一次。  一次是10ms
                    if loop_count % 50 == 0:
                        motor1 = control.getMotor(canid1)
                        motor2 = control.getMotor(canid2)
                        motor3 = control.getMotor(canid3)
                        motor4 = control.getMotor(canid4)
                        motor5 = control.getMotor(canid5)
                        motor6 = control.getMotor(canid6)
                        motor7 = control.getMotor(canid7)
                        motor8 = control.getMotor(canid8)


                        print(
                            # f"m1 pos:{motor1.Get_Position():.3f} vel:{motor1.Get_Velocity():.3f} err:{motor1.Get_err()} | "
                            # f"m2 pos:{motor2.Get_Position():.3f} vel:{motor2.Get_Velocity():.3f} err:{motor2.Get_err()} | "
                            f"thigh_deg:{fr_thigh_offset * 180.0 / 3.141592653589793:+.1f} "
                            f"calf_deg:{fr_calf_offset * 180.0 / 3.141592653589793:+.1f} "
                            f"udp_age:{udp_age:.2f}s "
                            f"cmd7:{q7_cmd:.3f} cmd8:{q8_cmd:.3f} "
                            f"m7 pos:{motor7.Get_Position():.3f} vel:{motor7.Get_Velocity():.3f} tau:{motor7.Get_tau():.3f} err:{motor7.Get_err()} | "
                            f"m8 pos:{motor8.Get_Position():.3f} vel:{motor8.Get_Velocity():.3f} tau:{motor8.Get_tau():.3f} err:{motor8.Get_err()} | "
                            # f"m5 pos:{motor5.Get_Position():.3f} vel:{motor5.Get_Velocity():.3f} err:{motor5.Get_err()} | "
                            # f"m6 pos:{motor6.Get_Position():.3f} vel:{motor6.Get_Velocity():.3f} err:{motor6.Get_err()} | "
                            # f"m7 pos:{motor7.Get_Position():.3f} vel:{motor7.Get_Velocity():.3f} err:{motor7.Get_err()} | "
                            # f"m8 pos:{motor8.Get_Position():.3f} vel:{motor8.Get_Velocity():.3f} err:{motor8.Get_err()} | "
                            # print(
                            # f"canid:{canid1} "
                            # f"pos:{motor1.Get_Position():.3f} "
                            # f"vel:{motor1.Get_Velocity():.3f} "
                            # f"tau:{motor1.Get_tau():.3f} "
                            # f"err:{motor1.Get_err()} "
                            # f"dt:{motor1.getTimeInterval():.4f}",
                            # flush=True,
                        )
                    # control.control_mit(control.getMotor(canid2), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid3), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid4), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid5), 0.0, 0.0, 0.0, 0.0, 0.0)  
                    # control.control_mit(control.getMotor(canid6), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid7), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid8), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid9), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # for id in range(1,10): 
                    #     pos = control.getMotor(id).Get_Position()
                    #     vel = control.getMotor(id).Get_Velocity()
                    #     tau = control.getMotor(id).Get_tau()
                    #     err = control.getMotor(id).Get_err()
                    #     interval = control.getMotor(id).getTimeInterval()
                    #     print(f"canid is: {id} pos: {pos:.3f} vel: {vel:.3f} effort: {tau:.3f} err: {err} time(s): {interval:.4f}", file=sys.stderr)
                    # print(
                    #     f"canid is: {1} pos: {control.getMotor(canid1).Get_Position():.6f} "
                    #     f"canid is: {2} pos: {control.getMotor(canid2).Get_Position():.6f} "
                    #     f"canid is: {3} pos: {control.getMotor(canid3).Get_Position():.6f} "
                    #     f"canid is: {4} pos: {control.getMotor(canid4).Get_Position():.6f} "
                    #     f"canid is: {5} pos: {control.getMotor(canid5).Get_Position():.6f} "
                    #     f"canid is: {6} pos: {control.getMotor(canid6).Get_Position():.6f} "
                    #     f"canid is: {7} pos: {control.getMotor(canid7).Get_Position():.6f} "
                    #     f"canid is: {8} pos: {control.getMotor(canid8).Get_Position():.6f} "
                    #     f"canid is: {9} pos: {control.getMotor(canid9).Get_Position():.6f}",
                    #     file=sys.stderr
                    # )     
                    sleep_till = current_time + desired_duration
                    now = time.perf_counter()
                    if sleep_till > now:
                        time.sleep(sleep_till - now)

        print("The program exited safely.") 
    except Exception as e:
        print(f"Error: hardware interface exception: {e}", file=sys.stderr)
