"""Linux：DM-IMU 数据到达的新鲜度，对比两种读法（只读，不依赖 pyserial）。

  A = 部署现用读法：轮询 FIONREAD（等价于 pyserial 的 in_waiting）+ time.sleep(0.001)，有数据就一次读完
  B = 阻塞读：read(1) 阻塞到有数据，再把已缓存的一起读完（DeepSeek 在 PATCHES.md 提的改法）

IMU 帧：`55 AA` + 长度/ID + 寄存器 + 3×float32 + CRC16 + `0A`，共 19 字节；寄存器 1 加速度、2 角速度、3 欧拉角。
CRC 校验与 `sim2real/bennett_deploy/imu.py` 相同（两种 CRC 变体任一通过即可）。

不写串口，除非 --exit_setting 且 0.5 s 内没有任何数据（部署的读取器初始化时会发同样的"退出设置模式"命令）。
交替运行 A、B、A、B，每段 --seconds 秒；结果写 JSON：每个寄存器的帧率、帧间隔分布、在 50 Hz 与 1 kHz 采样拍上的时龄分布。

用法（Jetson）：
    python3 imu_freshness_linux.py --port /dev/ttyACM0 --seconds 30 --out imu_freshness.json
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics
import sys
import termios
import time

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


def open_raw(port: str, vtime_ds: int):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = attrs[5] = termios.B921600
    attrs[6][termios.VMIN] = 1
    attrs[6][termios.VTIME] = vtime_ds  # 十分之一秒；阻塞读的超时
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIFLUSH)
    return fd


def in_waiting(fd: int) -> int:
    buf = bytearray(4)
    fcntl.ioctl(fd, termios.FIONREAD, buf)
    return int.from_bytes(buf, "little")


class Parser:
    def __init__(self):
        self.buf = bytearray()
        self.stamps = {1: [], 2: [], 3: []}
        self.crc_errors = 0
        self.frame_errors = 0

    def feed(self, data: bytes, t_ns: int) -> None:
        self.buf.extend(data)
        while True:
            start = self.buf.find(HEADER)
            if start < 0:
                if len(self.buf) > 1:
                    del self.buf[:-1]
                return
            if start > 0:
                del self.buf[:start]
            if len(self.buf) < FRAME_LEN:
                return
            frame = bytes(self.buf[:FRAME_LEN])
            if frame[-1] != 0x0A:
                del self.buf[0]
                self.frame_errors += 1
                continue
            expected = frame[16] | (frame[17] << 8)
            if expected not in (crc16_ccitt(frame[:16]), crc16_appendix(frame[:16])):
                del self.buf[0]
                self.crc_errors += 1
                continue
            del self.buf[:FRAME_LEN]
            reg = frame[3]
            if reg in self.stamps:
                self.stamps[reg].append(t_ns)


def run(port: str, mode: str, seconds: float, exit_setting: bool) -> dict:
    fd = open_raw(port, 2 if mode == "B" else 0)
    ns = time.perf_counter_ns
    p = Parser()
    t_end = ns() + int(seconds * 1e9)
    try:
        if exit_setting:
            time.sleep(0.5)
            if in_waiting(fd) == 0:
                for _ in range(10):  # 与 imu.py 初始化相同的命令
                    os.write(fd, bytes([0xAA, 0x06, 0x00, 0x0D]))
                    time.sleep(0.01)
                termios.tcflush(fd, termios.TCIFLUSH)
        while ns() < t_end:
            if mode == "A":
                n = in_waiting(fd)
                if n:
                    data = os.read(fd, n)
                    p.feed(data, ns())
                else:
                    time.sleep(0.001)
            else:
                data = os.read(fd, 1)  # 阻塞，VTIME=0.2 s 超时返回空
                if not data:
                    continue
                t = ns()
                extra = in_waiting(fd)
                if extra:
                    data += os.read(fd, extra)
                p.feed(data, t)
    finally:
        os.close(fd)
    return summarize(p, seconds, mode)


def pct(x, q):
    s = sorted(x)
    return s[min(len(s) - 1, int(len(s) * q / 100))] if s else float("nan")


def summarize(p: Parser, seconds: float, mode: str) -> dict:
    out = {"mode": mode, "seconds": seconds, "crc_errors": p.crc_errors, "frame_errors": p.frame_errors, "registers": {}}
    t0 = min((v[0] for v in p.stamps.values() if v), default=0)
    t1 = max((v[-1] for v in p.stamps.values() if v), default=0)
    for reg, st in p.stamps.items():
        if len(st) < 10:
            out["registers"][reg] = {"frames": len(st)}
            continue
        gaps = [(b - a) / 1e6 for a, b in zip(st, st[1:])]
        info = {"frames": len(st), "rate_hz": len(st) / ((st[-1] - st[0]) / 1e9),
                "gap_ms": {"p50": statistics.median(gaps), "p99": pct(gaps, 99), "max": max(gaps)}}
        for name, step in (("age_at_50hz_ms", 20_000_000), ("age_at_1khz_ms", 1_000_000)):
            ages = []
            j = 0
            tick = st[0] + step
            while tick < st[-1]:
                while j + 1 < len(st) and st[j + 1] <= tick:
                    j += 1
                ages.append((tick - st[j]) / 1e6)
                tick += step
            info[name] = {"p50": pct(ages, 50), "p95": pct(ages, 95), "p99": pct(ages, 99), "max": max(ages)}
        out["registers"][reg] = info
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--order", default="ABAB")
    ap.add_argument("--exit_setting", action="store_true", help="若 0.5 s 内没有数据，发与部署相同的退出设置模式命令")
    ap.add_argument("--out", default="imu_freshness.json")
    args = ap.parse_args()
    results = []
    for i, mode in enumerate(args.order):
        print(f"[{i + 1}/{len(args.order)}] 读法 {mode}，{args.seconds:g} s ...", flush=True)
        results.append(run(args.port, mode, args.seconds, args.exit_setting and i == 0))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"host": os.uname().machine, "results": results}, f, ensure_ascii=False, indent=1)
    for r in results:
        reg = r["registers"].get(2) or {}
        if "age_at_50hz_ms" in reg:
            a = reg["age_at_50hz_ms"]
            print(f"  {r['mode']}: reg2 {reg['rate_hz']:.1f} Hz，50 Hz 拍时龄 p50 {a['p50']:.3f} p99 {a['p99']:.3f} max {a['max']:.3f} ms，CRC 错 {r['crc_errors']}")
        else:
            print(f"  {r['mode']}: reg2 数据不足 {reg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
