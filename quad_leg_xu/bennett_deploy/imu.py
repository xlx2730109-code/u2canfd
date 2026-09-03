from __future__ import annotations

import math
import os
import struct
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple


def ensure_pyserial():
    try:
        import serial  # type: ignore

        return serial
    except ImportError:
        user_site = os.path.join(
            os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python313", "site-packages"
        )
        if os.path.isdir(os.path.join(user_site, "serial")) and user_site not in sys.path:
            sys.path.insert(0, user_site)
        import serial  # type: ignore

        return serial


def dm_imu_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def dm_imu_crc16_appendix(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        index = ((crc >> 8) ^ value) & 0xFF
        table_value = index << 8
        for _ in range(8):
            table_value = (
                ((table_value << 1) ^ 0x1021) & 0xFFFF
                if table_value & 0x8000
                else (table_value << 1) & 0xFFFF
            )
        crc = ((crc << 1) ^ table_value) & 0xFFFF
    return crc


class FirstOrderLowPass:
    def __init__(self, cutoff_hz: float, sample_dt: float):
        self.alpha = (
            1.0
            if cutoff_hz <= 0.0
            else 1.0 - math.exp(-2.0 * math.pi * float(cutoff_hz) * float(sample_dt))
        )
        self._value: Optional[List[float]] = None

    def update(self, values: Iterable[float]) -> List[float]:
        sample = [float(value) for value in values]
        if self._value is None:
            self._value = sample
        else:
            self._value = [old + self.alpha * (new - old) for old, new in zip(self._value, sample)]
        return list(self._value)


class DMImuSerialReader:
    """DM-IMU USB reader for 55 AA sid reg 3*float32 crc crc 0A frames."""

    FRAME_LEN = 19
    HEADER = b"\x55\xAA"

    def __init__(self, port: str, baudrate: int = 921600, *, exit_setting_mode: bool = True):
        serial = ensure_pyserial()
        self.port = port
        self.baudrate = int(baudrate)
        self.ser = serial.Serial(port, self.baudrate, timeout=0, write_timeout=0)
        if exit_setting_mode:
            command = bytes([0xAA, 0x06, 0x00, 0x0D])
            for _ in range(10):
                self.ser.write(command)
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
        self._crc_errors = 0
        self._crc_modes = {"ccitt": 0, "appendix": 0}
        self._frame_errors = 0
        self._thread_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, name="dm-imu-reader", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        try:
            while not self._stop.is_set():
                waiting = getattr(self.ser, "in_waiting", 0)
                if waiting:
                    self._buf.extend(self.ser.read(waiting))
                    self._parse_frames()
                else:
                    time.sleep(0.001)
        except Exception as exc:
            with self._lock:
                self._thread_error = repr(exc)
            print(f"[IMU] read thread stopped: {exc}", file=sys.stderr)

    def _parse_frames(self) -> None:
        while True:
            start = self._buf.find(self.HEADER)
            if start < 0:
                if len(self._buf) > 1:
                    del self._buf[:-1]
                return
            if start > 0:
                del self._buf[:start]
            if len(self._buf) < self.FRAME_LEN:
                return
            frame = bytes(self._buf[: self.FRAME_LEN])
            if frame[-1] != 0x0A:
                del self._buf[0]
                with self._lock:
                    self._frame_errors += 1
                continue
            expected = int(frame[16]) | (int(frame[17]) << 8)
            ccitt = dm_imu_crc16(frame[:16])
            appendix = dm_imu_crc16_appendix(frame[:16])
            if expected not in (ccitt, appendix):
                del self._buf[0]
                with self._lock:
                    self._crc_errors += 1
                continue
            del self._buf[: self.FRAME_LEN]
            register = int(frame[3])
            if register not in (1, 2, 3, 4):
                continue
            values = struct.unpack("<fff", frame[4:16])
            if not all(math.isfinite(value) for value in values):
                with self._lock:
                    self._frame_errors += 1
                continue
            now = time.monotonic()
            with self._lock:
                self._crc_modes["ccitt" if expected == ccitt else "appendix"] += 1
                self._latest[register] = tuple(float(value) for value in values)
                self._stamp[register] = now
                self._counts[register] = self._counts.get(register, 0) + 1

    def get_with_age(self, register: int) -> Tuple[Optional[Tuple[float, float, float]], float]:
        with self._lock:
            if self._thread_error is not None:
                raise RuntimeError(f"IMU reader failed: {self._thread_error}")
            value = self._latest.get(int(register))
            stamp = self._stamp.get(int(register))
        return value, float("inf") if stamp is None else max(0.0, time.monotonic() - stamp)

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "counts": dict(self._counts),
                "crc_errors": self._crc_errors,
                "crc_modes": dict(self._crc_modes),
                "frame_errors": self._frame_errors,
                "thread_error": self._thread_error,
            }

    def wait_ready(self, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._thread_error is not None:
                    return False
                if 2 in self._latest and 3 in self._latest:
                    return True
            time.sleep(0.01)
        return False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.ser.close()
        except Exception:
            pass


def projected_gravity_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[float, float, float]:
    del yaw_deg
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return sp, -sr * cp, -cr * cp
