# -*- coding: utf-8 -*-
"""IMU 静态采集 —— 原始加速度/角速度/姿态 + 噪声 + 水平度检查。

安全声明: 仅串口读取 IMU, 与电机 CAN 总线完全无关, 不发任何电机帧。

用法:
    py -3.13 imu_static_capture.py --dur 6            # 当前姿态采 6s
    py -3.13 imu_static_capture.py --poses 原位,左倾,右倾,前倾,后倾 --dur 4
       # 每个姿态: 控制台提示 -> 操作者摆好按回车 -> 采 dur 秒

寄存器(按旧探测结论): reg1=加速度, reg2=角速度, reg3=欧拉角, reg4=待定(温度/四元数)
输出: 本工作包 data/raw/<日期>/<run_id>/ {manifest.yaml, imu.csv}
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, PARENT)
sys.path.insert(0, os.path.join(PARENT, "quad_leg_xu", "bennett_deploy"))

from imu import DMImuSerialReader  # noqa: E402

PORTS = ["COM6", "COM7"]
BAUD = 921600
REGS = (1, 2, 3, 4)


def capture(reader, dur_s, pose, out_rows, t_ns):
    print(f"    采集 {dur_s}s ...")
    t0 = t_ns()
    n = 0
    while t_ns() - t0 < dur_s * 1e9:
        for reg in REGS:
            v, age = reader.get_with_age(reg)
            if v is None:
                continue
            out_rows.append([t_ns(), pose, reg, v[0], v[1], v[2], age])
            n += 1
        time.sleep(0.001)
    return n


def summarize(rows, pose):
    print(f"  姿态[{pose}]:")
    for reg in REGS:
        vals = [r for r in rows if r[1] == pose and r[2] == reg]
        if len(vals) < 10:
            continue
        xs = [r[3] for r in vals]
        ys = [r[4] for r in vals]
        zs = [r[5] for r in vals]
        mx, my, mz = statistics.mean(xs), statistics.mean(ys), statistics.mean(zs)
        mag = (mx * mx + my * my + mz * mz) ** 0.5
        # 波动幅度(去均值后的极差), 反映噪声
        span = sum(statistics.pstdev(v) for v in (xs, ys, zs))
        print(f"    reg{reg}: n={len(vals):<5} 均值=({mx:+.4f},{my:+.4f},{mz:+.4f}) "
              f"|均值|={mag:.4f} 波动≈{span:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="串口号, 默认自动探测 COM6/COM7")
    ap.add_argument("--dur", type=float, default=6.0, help="每姿态采集秒数")
    ap.add_argument("--poses", default="当前姿态", help="逗号分隔的姿态名")
    ap.add_argument("--no-ask", action="store_true", help="不等待回车, 直接逐姿态采集")
    args = ap.parse_args()

    stamp = datetime.now()
    run_id = stamp.strftime("%Y%m%d_%H%M%S") + "_imu_static_r01"
    out_dir = os.path.join(WS, "08_sensor_identification", "data", "raw",
                           stamp.strftime("%Y-%m-%d"), run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] run_id: {run_id}")

    port = args.port
    reader = None
    for p in ([port] if port else PORTS):
        try:
            reader = DMImuSerialReader(p, BAUD, exit_setting_mode=True)
            if reader.wait_ready(2.0):
                port = p
                break
            reader.close()
            reader = None
        except Exception as exc:
            print(f"    {p}: {exc!r}")
            reader = None
    if reader is None:
        print("[!] 未找到 IMU, 检查串口线")
        return
    diag = reader.diagnostics()
    print(f"[+] IMU 就绪: {port} @ {BAUD}  diag={diag}")

    rows = []
    t_ns = time.perf_counter_ns
    poses = [p.strip() for p in args.poses.split(",") if p.strip()]
    for pose in poses:
        if not args.no_ask:
            input(f"[姿态] 请摆好「{pose}」后按回车开始采集 ...")
        capture(reader, args.dur, pose, rows, t_ns)
        summarize(rows, pose)

    with open(os.path.join(out_dir, "imu.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_host_mono_ns", "pose", "reg", "x", "y", "z", "age_s"])
        w.writerows(rows)

    manifest = f"""# IMU 静态采集 manifest
run_id: {run_id}
utc_start: {stamp.utcnow().isoformat()}Z
local_start: {stamp.isoformat()}
operator: xu
purpose: 08 工作包 - IMU 静态偏置/噪声/轴向/水平度
hardware:
  imu_port: {port}
  imu_baud: {BAUD}
  registers: reg1=加速度 reg2=角速度 reg3=欧拉角 reg4=待定(均待轴向验证确认)
safety:
  motor_related: 无(纯串口)
data_files:
  imu.csv: t_host_mono_ns, pose, reg, x, y, z, age_s
notes:
  - age_s 为主机最后一次收到该寄存器帧的时龄
  - reg 含义需通过多姿态重力方向验证后才能定论
"""
    with open(os.path.join(out_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
        f.write(manifest)
    print(f"\n[完成] {len(rows)} 行 -> {out_dir}")
    reader.close()


if __name__ == "__main__":
    main()
