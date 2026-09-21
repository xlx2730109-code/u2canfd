# -*- coding: utf-8 -*-
"""G0 只读快照 —— 8 台电机寄存器 + 总线往返延迟 + 反馈新鲜度。

安全声明:
  - Motor_Control 以 auto_enable=False 构造, 全程不使能电机;
  - 仅发送两类帧: 寄存器读(0x33, 目标 0x7FF) 与 状态刷新(0xCC, 目标 0x7FF);
  - 不发送任何 MIT/位置/速度/力矩运动命令。

采集内容:
  1. 每台电机寄存器快照(sw_ver/Gr/Damp/Inertia/PMAX/VMAX/TMAX/CTRL_MODE/TIMEOUT/保护阈值等)
  2. 刷新请求 -> 反馈帧到达 的总线往返延迟 RTT(每台 60 次)
  3. 连续 3s 轮询刷新下的反馈到达间隔分布(抖动/丢帧)

输出(全部落在本工作包 data/raw/<日期>/<run_id>/):
  manifest.yaml / frames.csv(RX流) / commands.csv(TX流) / registers.csv / rtt.csv / status0.csv

用法(电机上电, USB-CANFD 已插):
    py -3.13 g0_snapshot.py
"""
from __future__ import annotations

import csv
import os
import platform
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(os.path.dirname(HERE))              # .../System Identification/workstreams
SYSID_ROOT = os.path.dirname(WS_ROOT)                          # .../System Identification
PARENT = os.path.dirname(SYSID_ROOT)                           # u2canfd (damiao.py 所在)
sys.path.insert(0, PARENT)
os.chdir(PARENT)  # dmcan SDK 按相对路径 ./dlls/ 找 DLL, 必须从父目录运行

from damiao import Control_Mode, DM_Motor_Type, DmActData, Motor_Control  # noqa: E402
from dmcan import dmcan_device_type  # noqa: E402

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {
    1: ("FL_thigh", 0), 2: ("FL_calf", 0),
    3: ("FR_thigh", 1), 4: ("FR_calf", 1),
    5: ("RL_thigh", 0), 6: ("RL_calf", 0),
    7: ("RR_thigh", 1), 8: ("RR_calf", 1),
}

UINT_REGS = [(7, "MST_ID"), (8, "ESC_ID"), (9, "TIMEOUT"), (10, "CTRL_MODE"),
             (13, "hw_ver"), (14, "sw_ver"), (15, "SN"), (16, "NPP"),
             (35, "can_br"), (36, "sub_ver")]
FLOAT_REGS = [(0, "UV_Value"), (1, "KT_Value"), (2, "OT_Value"), (3, "OC_Value"),
              (6, "MAX_SPD"), (11, "Damp"), (12, "Inertia"), (17, "Rs"), (18, "LS"),
              (19, "Flux"), (20, "Gr"), (21, "PMAX"), (22, "VMAX"), (23, "TMAX"),
              (24, "I_BW"), (25, "KP_ASR"), (26, "KI_ASR"), (27, "KP_APR"),
              (28, "KI_APR"), (29, "OV_Value"), (30, "GREF"), (31, "Deta"),
              (32, "V_BW"), (33, "IQ_c1"), (34, "VL_c1")]

RTT_TRIALS = 60
RTT_TIMEOUT_S = 0.10
JITTER_S = 3.0


def build_init_data():
    init = []
    for canid, (_name, ch) in MOTORS.items():
        init.append(DmActData(
            motorType=DM_Motor_Type.DM8006,
            mode=Control_Mode.MIT_MODE,
            can_id=canid,
            mst_id=0x10 + canid,
            channel=ch,
        ))
    return init


def main():
    stamp = datetime.now()
    date_str = stamp.strftime("%Y-%m-%d")
    run_id = stamp.strftime("%Y%m%d_%H%M%S") + "_powered_g0snapshot_r01"
    out_dir = os.path.join(WS_ROOT, "01_hardware_inventory", "data", "raw", date_str, run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] run_id: {run_id}")
    print(f"[+] 输出: {out_dir}")
    print("[+] 安全模式: auto_enable=False, 只发寄存器读(0x33)和状态刷新(0xCC)")

    frames = []            # RX 事件流: (t_ns, ch, can_id, dlc, hex)
    counts = {}            # (ch, can_id) -> 收帧数
    tx_log = []            # TX 事件流: (t_ns, kind, ch, can_id)
    t_ns = time.perf_counter_ns

    control = None
    try:
        control = Motor_Control(1_000_000, 5_000_000, SN, build_init_data(),
                                device_type=dmcan_device_type.USB2CANFD_DUAL,
                                auto_enable=False)
        print("[+] 设备已打开(未使能)")

        # -- 挂带时间戳的 RX 回调(替换式, 包住原回调) ------------------------
        orig_cb = control.canframeCallback

        def ts_callback(device, frame):
            t = t_ns()
            try:
                ch = int(frame.head.channel)
                cid = int(frame.head.can_id)
                dlc = int(frame.head.dlc)
                n = {9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64}.get(dlc, min(dlc, 8))
                data = bytes(frame.payload[:n])
                frames.append((t, ch, cid, dlc, data.hex()))
                key = (ch, cid)
                counts[key] = counts.get(key, 0) + 1
            except Exception:
                pass
            orig_cb(device, frame)

        control.getDevice().hook_recv_callback(ts_callback)

        # -- 1) 寄存器快照 ---------------------------------------------------
        print("\n[1/3] 读寄存器 ...")
        reg_rows = []
        for canid, (name, ch) in sorted(MOTORS.items()):
            m = control.getMotor(ch, canid)
            got = 0
            for rid, rname in UINT_REGS + FLOAT_REGS:
                tx_log.append((t_ns(), "reg_read", ch, canid))
                v = control.read_motor_param(m, rid, timeout=0.5)
                if v is not None:
                    got += 1
                reg_rows.append({"canid": canid, "name": name, "channel": ch,
                                 "rid": rid, "reg": rname,
                                 "value": "" if v is None else v})
            print(f"    canid{canid} {name:<10} {got}/{len(UINT_REGS) + len(FLOAT_REGS)} 项")

        # -- 2) 总线往返延迟 RTT ---------------------------------------------
        print(f"\n[2/3] RTT 探测(每台 {RTT_TRIALS} 次) ...")
        rtt_rows = []
        for canid, (name, ch) in sorted(MOTORS.items()):
            m = control.getMotor(ch, canid)
            mst = 0x10 + canid
            key = (ch, mst)
            lat, miss = [], 0
            for _ in range(RTT_TRIALS):
                before = counts.get(key, 0)
                t0 = t_ns()
                tx_log.append((t0, "refresh", ch, canid))
                control.refresh_motor_status(m)
                while counts.get(key, 0) == before:
                    if t_ns() - t0 > RTT_TIMEOUT_S * 1e9:
                        miss += 1
                        break
                else:
                    lat.append((t_ns() - t0) / 1e6)
            rtt_rows.append({"canid": canid, "name": name, "channel": ch,
                             "n_ok": len(lat), "n_miss": miss,
                             "p50_ms": sorted(lat)[len(lat) // 2] if lat else "",
                             "max_ms": max(lat) if lat else "",
                             "all_ms": ";".join(f"{x:.3f}" for x in lat)})
            if lat:
                print(f"    canid{canid} {name:<10} p50={sorted(lat)[len(lat)//2]:.2f} ms  "
                      f"max={max(lat):.2f} ms  miss={miss}")
            else:
                print(f"    canid{canid} {name:<10} 无反馈! miss={miss}")

        # -- 3) 反馈到达间隔(抖动/丢帧) ---------------------------------------
        print(f"\n[3/3] 连续轮询 {JITTER_S}s ...")
        t_start = t_ns()
        n_round = 0
        while t_ns() - t_start < JITTER_S * 1e9:
            for canid, (name, ch) in sorted(MOTORS.items()):
                m = control.getMotor(ch, canid)
                tx_log.append((t_ns(), "refresh", ch, canid))
                control.refresh_motor_status(m)
                time.sleep(0.001)
            n_round += 1
        t_end = t_ns()
        print(f"    轮询 {n_round} 圈")
        with open(os.path.join(out_dir, "windows.json"), "w", encoding="utf-8") as f:
            f.write(f'{{"jitter_start_ns": {t_start}, "jitter_end_ns": {t_end}, '
                    f'"rounds": {n_round}}}')

        # 各电机当前状态快照(零位参考, 供 03 工作包用)
        status0 = []
        for canid, (name, ch) in sorted(MOTORS.items()):
            m = control.getMotor(ch, canid)
            status0.append({"canid": canid, "name": name, "channel": ch,
                            "pos_rad": m.Get_Position(), "vel": m.Get_Velocity(),
                            "tau_nm": m.Get_tau(), "err": m.Get_err()})

        # -- 保存 ------------------------------------------------------------
        def save_csv(fn, rows):
            if not rows:
                return
            with open(os.path.join(out_dir, fn), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        save_csv("registers.csv", reg_rows)
        save_csv("rtt.csv", rtt_rows)
        save_csv("status0.csv", status0)
        with open(os.path.join(out_dir, "frames.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_host_mono_ns", "channel", "can_id", "dlc", "data_hex"])
            w.writerows(frames)
        with open(os.path.join(out_dir, "commands.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_host_mono_ns", "kind", "channel", "can_id"])
            w.writerows(tx_log)

        manifest = f"""# G0 只读快照 manifest
run_id: {run_id}
utc_start: {stamp.utcnow().isoformat()}Z
local_start: {stamp.isoformat()}
operator: xu
purpose: G0 硬件盘点 - 寄存器快照 + 总线RTT + 反馈新鲜度
safety:
  motor_enable: false
  motion_commands: false
  frames_sent: 寄存器读(0x33->0x7FF) 与 状态刷新(0xCC->0x7FF) 仅此两类
hardware:
  robot: Bennett 8关节四足
  motors: DM8006 (按现有代码假定, 型号以寄存器SN/ver为准)
  can_device_sn: {SN}
  canfd: 仲裁1Mbps 数据5Mbps, USB2CANFD_DUAL
  can_map: canid1-2=FL thigh/calf ch0, 3-4=FR ch1, 5-6=RL ch0, 7-8=RR ch1, mst_id=0x10+canid
software:
  python: {platform.python_version()}
  script: workstreams/01_hardware_inventory/scripts/g0_snapshot.py
  driver: damiao.py (auto_enable=False)
data_files:
  frames.csv: 全部RX帧(t_host_mono_ns, ch, can_id, dlc, hex)
  commands.csv: 全部TX事件
  registers.csv: 寄存器快照(空值=超时无响应)
  rtt.csv: 刷新->反馈 往返延迟
  status0.csv: 采集结束时各关节位置/速度/力矩/错误码
notes:
  - RTT 为主机发帧到收到反馈帧的主机侧往返, 含驱动与回调调度, 非纯总线时间
  - 到达间隔在 1kHz 轮询(每台约125Hz)下测得
"""
        with open(os.path.join(out_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
            f.write(manifest)

        print(f"\n[完成] 共 RX {len(frames)} 帧, TX {len(tx_log)} 条")
        print(f"[完成] 数据已保存: {out_dir}")
    finally:
        if control is not None:
            try:
                control.close()
            except Exception as exc:
                print(f"[warn] close: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
