# read-only IMU probe: which COM emits 55AA 0x0A IMU frames with valid CRC (register 2/3)?
from __future__ import annotations
import os, struct, sys, time

sys.path.insert(0, r"C:\Users\xu\AppData\Roaming\Python\Python313\site-packages")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quad_leg_xu", "bennett_deploy"))
from imu import dm_imu_crc16, dm_imu_crc16_appendix  # noqa: E402

import serial  # noqa: E402

PORTS = ["COM6", "COM7"]
FRAME_LEN = 19
BAUD = 921600


def probe(port: str, dur_s: float = 1.5) -> None:
    try:
        ser = serial.Serial(port, BAUD, timeout=0, write_timeout=0)
    except Exception as exc:
        print(f"  [probe] {port}: open failed: {exc!r}")
        return
    ser.reset_input_buffer()
    buf = bytearray()
    counts = {}
    crc_bad = 0
    frame_bad = 0
    t0 = time.time()
    while time.time() - t0 < dur_s:
        waiting = getattr(ser, "in_waiting", 0)
        if not waiting:
            time.sleep(0.001)
            continue
        buf.extend(ser.read(waiting))
        while True:
            start = buf.find(b"\x55\xaa")
            if start < 0:
                if len(buf) > 1:
                    del buf[:-1]
                break
            if start > 0:
                del buf[:start]
            if len(buf) < FRAME_LEN:
                break
            frame = bytes(buf[:FRAME_LEN])
            if frame[-1] != 0x0A:
                del buf[0]
                frame_bad += 1
                continue
            expected = int(frame[16]) | (int(frame[17]) << 8)
            ccitt = dm_imu_crc16(frame[:16])
            appendix = dm_imu_crc16_appendix(frame[:16])
            if expected not in (ccitt, appendix):
                del buf[0]
                crc_bad += 1
                continue
            del buf[:FRAME_LEN]
            reg = int(frame[3])
            if reg in (2, 3, 4):
                counts[reg] = counts.get(reg, 0) + 1
        # decode newest reg=2 (gyro) + reg=3 (euler) if present for sanity
    ser.close()
    print(f"  [probe] {port}: frames/reg={counts} crc_bad={crc_bad} frame_bad={frame_bad}")


if __name__ == "__main__":
    for port in PORTS:
        probe(port)
    print("DONE")
