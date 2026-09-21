# -*- coding: utf-8 -*-
"""1kHz 空命令流下的反馈新鲜度探测(真机控制同款流量模式)。

安全声明:
  - 发送的 MIT 帧为 kp=0, kd=0, tau_ff=0 —— 输出力矩恒为 0, 不会引起任何运动;
  - 不发送使能帧(0xFC); 若电机未使能则不应答, 自动跳过;
  - 目的: 测量在 1kHz 连续命令流(真机部署同款)下, 反馈帧到达主机的节奏。

输出: 本工作包 data/raw/<日期>/<run_id>/ {manifest.yaml, frames.csv, tx.csv}
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, PARENT)
os.chdir(PARENT)  # dmcan SDK 按相对路径 ./dlls/ 找 DLL

from damiao import Control_Mode, DM_Motor_Type, DmActData, Motor_Control  # noqa: E402
from dmcan import dmcan_device_type  # noqa: E402

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}
DUR_S = 3.0
LOOP_HZ = 1000

sys.setswitchinterval(0.0005)  # 让 RX 回调线程更快拿到 GIL, 减少量测失真


def main():
    stamp = datetime.now()
    run_id = stamp.strftime("%Y%m%d_%H%M%S") + "_feedback_age_1khz_r01"
    out_dir = os.path.join(WS, "02_timing_latency", "data", "raw", stamp.strftime("%Y-%m-%d"), run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] run_id: {run_id}")
    print("[+] 安全: MIT 帧 kp=0 kd=0 tau=0 (零力矩), 不使能, 不运动")

    frames = []
    counts = {}
    tx_log = []
    t_ns = time.perf_counter_ns

    control = None
    try:
        init = [DmActData(DM_Motor_Type.DM8006, Control_Mode.MIT_MODE, cid, 0x10 + cid, ch)
                for cid, ch in MOTORS.items()]
        control = Motor_Control(1_000_000, 5_000_000, SN, init,
                                device_type=dmcan_device_type.USB2CANFD_DUAL,
                                auto_enable=False)
        orig_cb = control.canframeCallback

        def ts_callback(device, frame):
            t = t_ns()
            try:
                cid = int(frame.head.can_id)
                n = {9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64}.get(
                    int(frame.head.dlc), min(int(frame.head.dlc), 8))
                frames.append((t, int(frame.head.channel), cid, n,
                               bytes(frame.payload[:n]).hex()))
                counts[cid] = counts.get(cid, 0) + 1
            except Exception:
                pass
            orig_cb(device, frame)

        control.getDevice().hook_recv_callback(ts_callback)

        motors = [(cid, control.getMotor(ch, cid)) for cid, ch in MOTORS.items()]

        # 先各发 3 条零力矩命令, 记录哪些电机在应答(未使能的不会有反馈)
        time.sleep(0.05)
        probe_before = dict(counts)
        for cid, m in motors:
            control.control_mit(m, 0.0, 0.0, 0.0, 0.0, 0.0)
            tx_log.append((t_ns(), "mit_zero", m.GetChannel(), cid))
        time.sleep(0.2)
        alive = [cid for cid, m in motors
                 if counts.get(0x10 + cid, 0) > probe_before.get(0x10 + cid, 0)]
        dead = [cid for cid, _ in motors if cid not in alive]
        print(f"[+] 应答电机: {alive}" + (f"   无应答(可能未使能): {dead}" if dead else ""))

        # 1kHz 空命令流, 轮询全部电机
        print(f"[+] {LOOP_HZ}Hz 命令流持续 {DUR_S}s ...")
        t_start = t_ns()
        n_tx = 0
        i = 0
        period = 1.0 / LOOP_HZ
        while True:
            now = t_ns()
            if now - t_start >= DUR_S * 1e9:
                break
            cid, m = motors[i % len(motors)]
            control.control_mit(m, 0.0, 0.0, 0.0, 0.0, 0.0)
            tx_log.append((now, "mit_zero", m.GetChannel(), cid))
            n_tx += 1
            i += 1
            nxt = t_start + i * period * 1e9
            if nxt > now:
                time.sleep((nxt - now) / 1e9)
        t_end = t_ns()

        with open(os.path.join(out_dir, "frames.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_host_mono_ns", "channel", "can_id", "dlc", "data_hex"])
            w.writerows(frames)
        with open(os.path.join(out_dir, "tx.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_host_mono_ns", "kind", "channel", "can_id"])
            w.writerows(tx_log)
        with open(os.path.join(out_dir, "windows.json"), "w", encoding="utf-8") as f:
            f.write(f'{{"t_start_ns": {t_start}, "t_end_ns": {t_end}, "n_tx": {n_tx}}}')

        manifest = f"""# 1kHz 空命令流反馈新鲜度 manifest
run_id: {run_id}
utc_start: {stamp.utcnow().isoformat()}Z
local_start: {stamp.isoformat()}
operator: xu
purpose: 02 工作包 - 真机同款 1kHz 命令流下的反馈到达节奏
safety:
  motion_commands: false
  frame_type: MIT kp=0 kd=0 tau_ff=0 (零力矩, 不使能)
  tx_rate: {LOOP_HZ}Hz 轮询 8 电机
hardware:
  can_device_sn: {SN}
software:
  python: {sys.version.split()[0]}
  switch_interval_s: 0.0005
notes:
  - 时间戳为主机回调时间, 含 USB/DLL 批量交付影响
"""
        with open(os.path.join(out_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
            f.write(manifest)

        # 现场速报
        import collections
        by = collections.defaultdict(list)
        for t, ch, cid, dl, hx in frames:
            if 0x11 <= cid <= 0x18 and t_start <= t <= t_end:
                by[cid].append(t)
        print("\n--- 反馈到达间隔 (ms) ---")
        for cid in sorted(by):
            ts = sorted(by[cid])
            d = sorted((b - a) / 1e6 for a, b in zip(ts, ts[1:]))
            if d:
                print(f"mst{cid:02X} n={len(ts):<5} p50={d[len(d)//2]:6.2f} "
                      f"p95={d[int(0.95*len(d))]:6.2f} max={d[-1]:7.2f} "
                      f"freq={1000/(sum(d)/len(d)):6.1f}Hz")
        print(f"\n[完成] 数据: {out_dir}")
    finally:
        if control is not None:
            try:
                control.close()
            except Exception as exc:
                print(f"[warn] close: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
