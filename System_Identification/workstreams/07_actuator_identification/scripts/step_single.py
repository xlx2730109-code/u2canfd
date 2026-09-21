# -*- coding: utf-8 -*-
"""单关节位置阶跃实验 —— 07 工作包第一批运动数据(真机, 悬空).

安全声明:
  - 仅驱动一个关节(默认 FL_thigh), 其余电机不下令;
  - 训练同款增益 kp=28 kd=2, 默认幅值 0.15 rad(先小后大);
  - 硬保护: |tau|>8Nm / 位置越界 / 反馈超时 50ms -> 立即切零力矩阻尼并终止;
  - 电机未使能时命令无效果, 脚本会检测并拒绝继续。

流程: 增益爬升(0.6s, q=p0) -> 保持(0.8s) -> 台阶(2.0s) -> 回位(2.0s)
数据: data/raw/<日期>/<run_id>/
  command.csv  每个控制周期的指令状态(1kHz)
  feedback.csv 全部 RX 反馈帧(带主机时间戳, 含批交付影响)
  manifest.yaml
用法:
  py -3.13 step_single.py                # FL_thigh, 0.15 rad
  py -3.13 step_single.py --amp 0.30     # 确认正常后加大
  py -3.13 step_single.py --canid 2      # 换关节
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, PARENT)
os.chdir(PARENT)

from damiao import Control_Mode, Control_Mode_Code, DM_Motor_Type, DmActData, Motor_Control  # noqa: E402
from dmcan import dmcan_device_type  # noqa: E402

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: ("FL_thigh", 0), 2: ("FL_calf", 0), 3: ("FR_thigh", 1), 4: ("FR_calf", 1),
          5: ("RL_thigh", 0), 6: ("RL_calf", 0), 7: ("RR_thigh", 1), 8: ("RR_calf", 1)}
KP, KD = 28.0, 2.0
TAU_STOP = 8.0        # Nm
RAMP_S, HOLD_S, STEP_S, BACK_S = 0.6, 0.8, 2.0, 2.0
LOOP_HZ = 1000
BOUNDS = {"thigh": (-0.80, 0.80), "calf": (-0.90, 0.55)}

sys.setswitchinterval(0.0005)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canid", type=int, default=1, choices=sorted(MOTORS))
    ap.add_argument("--amp", type=float, default=0.15, help="台阶幅值 rad, 默认 0.15")
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--kd", type=float, default=KD)
    args = ap.parse_args()

    name, ch = MOTORS[args.canid]
    kind = "thigh" if "thigh" in name else "calf"
    lo, hi = BOUNDS[kind]

    stamp = datetime.now()
    run_id = stamp.strftime("%Y%m%d_%H%M%S") + f"_step_{name}_a{args.amp:+.2f}_r01"
    out_dir = os.path.join(WS, "07_actuator_identification", "data", "raw",
                           stamp.strftime("%Y-%m-%d"), run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] run_id: {run_id}")
    print(f"[+] 关节: {name} (canid{args.canid}, ch{ch})  幅值 {args.amp} rad  "
          f"kp={args.kp} kd={args.kd}")
    print("[!] 确认机器人悬空、急停在手。Ctrl+C 可随时中断。")

    frames = []          # RX 流
    cmds = []            # TX/周期流
    stop_reason = ""
    t_ns = time.perf_counter_ns

    control = None
    try:
        init = [DmActData(DM_Motor_Type.DM8006, Control_Mode.MIT_MODE, cid, 0x10 + cid, c)
                for cid, (_n, c) in MOTORS.items()]
        control = Motor_Control(1_000_000, 5_000_000, SN, init,
                                device_type=dmcan_device_type.USB2CANFD_DUAL,
                                auto_enable=False)   # 不额外使能; 电机保持上电时状态
        orig_cb = control.canframeCallback
        last_rx_by_id = {}          # 每个电机最近一帧反馈的主机时刻

        def ts_callback(device, frame):
            t = t_ns()
            try:
                cid = int(frame.head.can_id)
                frames.append((t, int(frame.head.channel), cid,
                               bytes(frame.payload[:min(int(frame.head.dlc), 8)]).hex()))
                last_rx_by_id[cid] = t
            except Exception:
                pass
            orig_cb(device, frame)

        control.getDevice().hook_recv_callback(ts_callback)
        m = control.getMotor(ch, args.canid)
        mst = 0x10 + args.canid

        # --- 使能本关节 (与 0_Return_to_zero.py 同套路: 切 MIT 模式 + 0xFC ×5) ---
        # 上电/上一脚本退出后电机默认失能; 只使能本关节, 其余电机不动
        if not control.switchControlMode(m, Control_Mode_Code.MIT):
            print("[!] CTRL_MODE 写入失败, 终止")
            return
        for _ in range(5):
            control.control_cmd(args.canid, 0xFC, ch)
            time.sleep(0.002)
        print(f"[+] 已使能 {name} (MIT + 0xFC ×5)")

        # --- 前置检查: 反馈新鲜 ------------------------------------------------
        time.sleep(0.1)
        n0 = sum(1 for f in frames if f[2] == mst)
        control.refresh_motor_status(m)
        time.sleep(0.15)
        n1 = sum(1 for f in frames if f[2] == mst)
        if n1 <= n0:
            print("[!] 该电机无反馈, 终止")
            return
        p0 = m.Get_Position()
        if abs(p0) > 3.0:
            print(f"[!] 位置异常 p0={p0}, 终止")
            return
        err = getattr(m, "Get_err", lambda: "?")()
        print(f"[+] 前置检查通过: p0={p0:+.4f} rad, err={err}")

        # 方向: 朝中性 0, 并限制在机械边界内
        toward = 1.0 if p0 <= 0 else -1.0
        target = min(max(p0 + toward * abs(args.amp), lo), hi)
        print(f"[+] 目标: {target:+.4f} rad (位移 {target - p0:+.4f})")

        period = 1.0 / LOOP_HZ
        t_start = t_ns()
        phase = "ramp"
        t_phase0 = t_start
        i_loop = 0
        q_cmd = p0
        # 注意: 反馈按 ~109ms 批量交付, 批间空档 ~110ms 属正常;
        # 超时阈值取 250ms, 只拦截真正的链路失联。
        RX_TIMEOUT_NS = 250e6
        moved = False

        while True:
            now = t_ns()
            el = (now - t_start) / 1e9
            # --- 阶段切换 ---
            if phase == "ramp" and el >= RAMP_S:
                phase, t_phase0 = "hold", now
            elif phase == "hold" and (now - t_phase0) / 1e9 >= HOLD_S:
                phase, t_phase0 = "step", now
                q_cmd = target
                print(f"[{el:6.2f}s] 台阶 -> {target:+.4f}")
            elif phase == "step" and (now - t_phase0) / 1e9 >= STEP_S:
                phase, t_phase0 = "back", now
                q_cmd = p0
                print(f"[{el:6.2f}s] 回位 -> {p0:+.4f}")
            elif phase == "back" and (now - t_phase0) / 1e9 >= BACK_S:
                stop_reason = "完成"
                break

            # --- 增益 ---
            if phase == "ramp":
                kp_now = args.kp * min(1.0, (now - t_start) / (RAMP_S * 1e9))
            else:
                kp_now = args.kp

            # --- 保护 ---
            q, dq, tau = m.Get_Position(), m.Get_Velocity(), m.Get_tau()
            age = now - last_rx_by_id.get(mst, now)
            bad = ""
            if abs(tau) > TAU_STOP:
                bad = f"tau={tau:.2f}"
            elif not (lo - 0.15 <= q <= hi + 0.15):
                bad = f"pos={q:.2f} 越界"
            elif age > RX_TIMEOUT_NS:
                bad = f"反馈超时 {age/1e6:.0f}ms"
            if bad:
                control.control_mit(m, 0.0, 2.0, q, 0.0, 0.0)
                stop_reason = f"保护触发: {bad}"
                print(f"\n[!] {stop_reason}, 已切零力矩阻尼")
                break

            # 未动检测: 台阶 0.5s 后位置仍无变化 -> 使能未生效, 别白跑
            if not moved and abs(q - p0) > max(0.02 * abs(args.amp), 0.005):
                moved = True
            if phase == "step" and not moved and (now - t_phase0) / 1e9 > 0.5:
                control.control_mit(m, 0.0, 2.0, q, 0.0, 0.0)
                stop_reason = "电机未动(使能未生效?)"
                print(f"\n[!] {stop_reason}, 已切零力矩阻尼")
                break

            # --- 下令 ---
            control.control_mit(m, kp_now, args.kd, q_cmd, 0.0, 0.0)
            cmds.append((now, el, phase, q_cmd, kp_now, args.kd, q, dq, tau))
            i_loop += 1
            nxt = t_start + i_loop * period * 1e9
            if nxt > now:
                time.sleep((nxt - now) / 1e9)

        # 收尾: 零力矩阻尼保持 0.3s, 让腿停在当前位置附近
        for _ in range(300):
            control.control_mit(m, 0.0, args.kd, m.Get_Position(), 0.0, 0.0)
            time.sleep(0.001)
        print(f"[+] 结束: {stop_reason}  最终位置 {m.Get_Position():+.4f} rad")

        # --- 保存 ---
        with open(os.path.join(out_dir, "command.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_host_mono_ns", "t_rel_s", "phase", "q_cmd_rad",
                        "kp", "kd", "q_real_rad", "dq_real", "tau_real_nm"])
            w.writerows(cmds)
        with open(os.path.join(out_dir, "feedback.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_host_mono_ns", "channel", "can_id", "data_hex"])
            w.writerows(frames)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            f.write(f'{{"run_id": "{run_id}", "joint": "{name}", "canid": {args.canid}, '
                    f'"channel": {ch}, "p0": {p0}, "target": {target}, '
                    f'"amp": {args.amp}, "kp": {args.kp}, "kd": {args.kd}, '
                    f'"stop_reason": "{stop_reason}", '
                    f'"p_final": {m.Get_Position()}}}')
        manifest = f"""# 单关节阶跃实验 manifest
run_id: {run_id}
utc_start: {stamp.utcnow().isoformat()}Z
local_start: {stamp.isoformat()}
operator: xu
purpose: 07 工作包 - 台阶响应(Kp/Kd 验证 + 摩擦 + 端到端延迟)
safety:
  amplitude_rad: {args.amp}
  kp: {args.kp}
  kd: {args.kd}
  tau_stop_nm: {TAU_STOP}
  pos_bounds: [{lo}, {hi}]
  feedback_timeout_ms: 250  (批交付周期~110ms, 阈值须大于它)
  result: {stop_reason}
hardware:
  joint: {name} canid{args.canid} ch{ch} mst 0x{mst:02X}
  can_device_sn: {SN}
notes:
  - 反馈时间戳含驱动批交付(~109ms 周期)影响, 起步延迟分析须说明
"""
        with open(os.path.join(out_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
            f.write(manifest)
        print(f"[+] 数据: {out_dir}")
    except KeyboardInterrupt:
        print("\n[!] 手动中断, 已停止下令")
    finally:
        if control is not None:
            try:
                control.close()
            except Exception:
                pass  # Windows 下 close() 偶发 access violation, 数据已落盘


if __name__ == "__main__":
    main()
