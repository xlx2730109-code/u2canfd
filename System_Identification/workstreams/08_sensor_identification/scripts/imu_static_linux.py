"""Linux：DM-IMU 静态采集与噪声分析（只读串口，不依赖 pyserial 与 numpy 之外的包）。

机器人静止（悬空且不摆动）时记录三个寄存器（1 加速度、2 角速度、3 欧拉角）共 --seconds 秒，输出：
  - 每个寄存器每轴的均值、标准差、帧率；
  - 角速度零偏（均值）与噪声密度、加速度范数与重力偏差、欧拉角均值与漂移（首末各 10% 的均值差）；
  - 角速度与加速度的 Allan 偏差（τ = 0.01 至 10 s），给出零偏不稳定性与随机游走的量级；
  - 原始样本 CSV（主机时刻 ns、寄存器、三个值）。
结果写到 --out 目录。IMU 帧格式、CRC 与 sim2real/bennett_deploy/imu.py 相同。

用法（Jetson）：
    python3 imu_static_linux.py --port /dev/ttyACM0 --seconds 120 --out imu_static_001
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import struct
import sys
import termios
import time
from pathlib import Path

import numpy as np

if sys.platform != "linux":
    raise SystemExit("仅 Linux（用 termios）")

FRAME_LEN = 19
HEADER = b"\x55\xAA"


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def crc16_appendix(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        index = ((crc >> 8) ^ value) & 0xFF
        table_value = index << 8
        for _ in range(8):
            table_value = (((table_value << 1) ^ 0x1021) & 0xFFFF if table_value & 0x8000 else (table_value << 1) & 0xFFFF)
        crc = ((crc << 1) ^ table_value) & 0xFFFF
    return crc


def open_raw(port: str):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = attrs[5] = termios.B921600
    attrs[6][termios.VMIN] = 1
    attrs[6][termios.VTIME] = 2
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIFLUSH)
    return fd


def in_waiting(fd: int) -> int:
    buf = bytearray(4)
    fcntl.ioctl(fd, termios.FIONREAD, buf)
    return int.from_bytes(buf, "little")


def allan(x: np.ndarray, fs: float, taus_s):
    """非重叠 Allan 偏差；x 为等间隔采样序列。"""
    out = []
    for tau in taus_s:
        m = int(round(tau * fs))
        if m < 1 or len(x) // m < 4:
            continue
        k = len(x) // m
        means = x[: k * m].reshape(k, m).mean(axis=1)
        out.append((tau, float(np.sqrt(0.5 * np.mean(np.diff(means) ** 2)))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exit_setting", action="store_true", help="若 0.5 s 内没有数据，发与部署相同的退出设置模式命令")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fd = open_raw(args.port)
    ns = time.perf_counter_ns
    buf = bytearray()
    rows = []
    crc_err = frame_err = 0
    t_end = ns() + int(args.seconds * 1e9)
    try:
        if args.exit_setting:
            time.sleep(0.5)
            if in_waiting(fd) == 0:
                for _ in range(10):
                    os.write(fd, bytes([0xAA, 0x06, 0x00, 0x0D]))
                    time.sleep(0.01)
                termios.tcflush(fd, termios.TCIFLUSH)
        while ns() < t_end:
            data = os.read(fd, 1)
            if not data:
                continue
            t = ns()
            extra = in_waiting(fd)
            if extra:
                data += os.read(fd, extra)
            buf.extend(data)
            while True:
                start = buf.find(HEADER)
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
                    frame_err += 1
                    continue
                expected = frame[16] | (frame[17] << 8)
                if expected not in (crc16_ccitt(frame[:16]), crc16_appendix(frame[:16])):
                    del buf[0]
                    crc_err += 1
                    continue
                del buf[:FRAME_LEN]
                reg = frame[3]
                if reg in (1, 2, 3):
                    rows.append((t, reg) + struct.unpack("<fff", frame[4:16]))
    finally:
        os.close(fd)

    with open(out / "raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_host_ns", "reg", "x", "y", "z"])
        w.writerows(rows)

    arr = np.array(rows, dtype=np.float64)
    summary = {"seconds": args.seconds, "crc_errors": crc_err, "frame_errors": frame_err, "registers": {}}
    labels = {1: "accel", 2: "gyro", 3: "euler_deg"}
    for reg in (1, 2, 3):
        a = arr[arr[:, 1] == reg]
        if len(a) < 100:
            summary["registers"][labels[reg]] = {"frames": int(len(a))}
            continue
        t = (a[:, 0] - a[0, 0]) / 1e9
        fs = (len(a) - 1) / t[-1]
        v = a[:, 2:5]
        k = max(1, len(v) // 10)
        info = {"frames": int(len(a)), "rate_hz": float(fs), "mean": v.mean(axis=0).tolist(), "std": v.std(axis=0).tolist(),
                "drift_first_last_10pct": (v[-k:].mean(axis=0) - v[:k].mean(axis=0)).tolist()}
        if reg == 1:
            norm = np.linalg.norm(v, axis=1)
            info["norm_mean"] = float(norm.mean())
            info["norm_std"] = float(norm.std())
        taus = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        info["allan_dev"] = {axis: allan(v[:, j], fs, taus) for j, axis in enumerate("xyz")}
        info["noise_density_per_sqrt_hz"] = (v.std(axis=0) / np.sqrt(fs / 2)).tolist()
        summary["registers"][labels[reg]] = info
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, info in summary["registers"].items():
        if "mean" in info:
            print(f"{name}: {info['rate_hz']:.1f} Hz，均值 {[round(x, 5) for x in info['mean']]}，标准差 {[round(x, 6) for x in info['std']]}")
    print(f"CRC 错 {crc_err}，帧错 {frame_err}，输出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
