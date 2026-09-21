# -*- coding: utf-8 -*-
"""力矩前馈扫描 —— 把 KP 真实值和固定偏置分开量出来(真机, 悬空).

为什么不用台阶
--------------
台阶只能靠重力负载变化去推 Δq, 而悬空腿上重力力矩对付 16 mrad 的位移几乎不变,
Δq 的散布只有 1~2 mrad, 拟合出来的 KP 是噪声. 换一个思路:

    tau_out = KP*(q_des - q) + KD*(dq_des - dq) + tau_ff

在同一个位置(同一个重力负载)上, 主动扫 tau_ff:
    KP*Δq + tau_ff = tau_load  =>  Δq = (tau_load - tau_ff)/KP
tau_ff 扫 0.8 N·m, Δq 就跟着扫 0.8/KP ~ 28 mrad —— 而 Δq 是 16 位编码器,
1 LSB = 0.38 mrad, 这个量程有 70 多个 LSB, 信噪比极高.

于是对每个位置上的 5 个 tau_ff 点回归
    tau_rep = KP*Δq + c*tau_ff + b
斜率 KP 就是电机真正用的位置增益; c 顺便回答"反馈力矩里含不含前馈"(c≈1 含, c≈0 不含);
b 就是之前台阶数据里那个 -0.036 N·m 的固定偏置, 这次能单独量出来.

安全声明
--------
  - 只驱动一个关节(默认 FL_thigh), 其余电机不下令, 全程不下地;
  - **不调用 set_zero_position, 不发 0xFE, 不碰零位**;
  - q_des 始终在 p0 ±0.06 rad 内, 实际偏离 q_des 最多 (|tau_ff|+|负载|)/KP ~ 25 mrad = 1.4°;
  - 硬保护: |tau|>3Nm / 位置越界 / 反馈超时 250ms -> 立即切零力矩阻尼并终止;
  - 使能检查: 先走一个 0.03 rad 小台阶, 不动就判定使能未生效, 直接收工.

用法:
  py -3.13 tau_ff_sweep.py --dry-run     # 只打印动作序列, 不碰硬件
  py -3.13 tau_ff_sweep.py               # FL_thigh 实跑
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
TAU_STOP = 3.0            # Nm, 本实验力矩本来就小, 保护阈值收得比台阶实验更紧
LOOP_HZ = 1000
BOUNDS = {"thigh": (-0.80, 0.80), "calf": (-0.90, 0.55)}

RAMP_S = 0.8              # 增益爬升
CHK_S = 1.0               # 使能检查小台阶保持
MOVE_S = 0.4              # 位置组之间挪动
HOLD_S = 1.0              # 每个 tau_ff 点保持
REL_S = 0.5               # 收尾缓放

P_OFFS = (0.0, +0.06, -0.06)                    # 三个位置(相对 p0)
TFFS = (-0.4, -0.2, 0.0, +0.2, +0.4)            # 五档前馈力矩 N·m
CHK_AMP = 0.03                                  # 使能检查台阶 rad

sys.setswitchinterval(0.0005)


def build_sequence(p0):
    """返回 [(phase, q_from, q_to, tau_ff, dur)]; 位置组之间自动插入挪动段."""
    seq, prev = [], p0
    for po in P_OFFS:
        tgt = p0 + po
        if abs(tgt - prev) > 1e-9:
            seq.append((f"move_p{po:+.2f}", prev, tgt, 0.0, MOVE_S))
        for ff in TFFS:
            seq.append((f"p{po:+.2f}_ff{ff:+.1f}", tgt, tgt, ff, HOLD_S))
        prev = tgt
    if abs(prev - p0) > 1e-9:
        seq.append(("move_back", prev, p0, 0.0, MOVE_S))
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canid", type=int, default=1, choices=sorted(MOTORS))
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--kd", type=float, default=KD)
    ap.add_argument("--dry-run", action="store_true", help="只打印序列, 不连硬件")
    args = ap.parse_args()

    name, ch = MOTORS[args.canid]
    kind = "thigh" if "thigh" in name else "calf"
    lo, hi = BOUNDS[kind]

    if args.dry_run:
        print(f"[dry-run] {name}  canid{args.canid} ch{ch}  kp={args.kp} kd={args.kd}")
        print(f"[dry-run] 位置组 {P_OFFS} rad,  前馈档 {TFFS} N·m")
        seq = build_sequence(0.0)
        t = RAMP_S + CHK_S
        print(f"{'阶段':<20}{'q_from':>9}{'q_to':>9}{'tau_ff':>9}{'时长':>7}   累计(s)")
        for ph, a, b, ff, d in seq:
            print(f"{ph:<20}{a:>+9.4f}{b:>+9.4f}{ff:>+9.2f}{d:>7.1f}   {t:>7.1f}")
            t += d
        print(f"{'release':<20}{0.0:>+9.4f}{0.0:>+9.4f}{0.0:>+9.2f}{REL_S:>7.1f}   {t:>7.1f}")
        print(f"[dry-run] 总时长约 {t + REL_S:.1f} s;  最大力矩估计 "
              f"{max(abs(f) for f in TFFS) + 0.6:.1f} N·m < 保护 {TAU_STOP}")
        return

    stamp = datetime.now()
    run_id = stamp.strftime("%Y%m%d_%H%M%S") + f"_tauff_{name}_r01"
    out_dir = os.path.join(WS, "07_actuator_identification", "data", "raw",
                           stamp.strftime("%Y-%m-%d"), run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] run_id: {run_id}")
    print(f"[+] 关节 {name} (canid{args.canid} ch{ch})  kp={args.kp} kd={args.kd}")
    print("[!] 确认机器人悬空、急停在手。Ctrl+C 可随时中断。")

    frames, cmds = [], []
    stop_reason = ""
    t_ns = time.perf_counter_ns
    control = None
    p0 = None
    try:
        init = [DmActData(DM_Motor_Type.DM8006, Control_Mode.MIT_MODE, cid, 0x10 + cid, c)
                for cid, (_n, c) in MOTORS.items()]
        control = Motor_Control(1_000_000, 5_000_000, SN, init,
                                device_type=dmcan_device_type.USB2CANFD_DUAL,
                                auto_enable=False)
        orig_cb = control.canframeCallback
        last_rx_by_id = {}

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

        # --- 使能本关节 (只切 MIT 模式 + 0xFC, 绝不发 0xFE) ---
        if not control.switchControlMode(m, Control_Mode_Code.MIT):
            print("[!] CTRL_MODE 写入失败, 终止")
            return
        for _ in range(5):
            control.control_cmd(args.canid, 0xFC, ch)
            time.sleep(0.002)
        print(f"[+] 已使能 {name} (MIT + 0xFC x5; 未发 0xFE, 零位不动)")

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
        print(f"[+] 前置检查通过: p0={p0:+.4f} rad, err={getattr(m, 'Get_err', lambda: '?')()}")

        # 位置越界预检: 所有目标都必须留 0.1 rad 余量
        seq = build_sequence(p0)
        worst = min([b for _p, _a, b, _f, _d in seq] + [p0])
        best = max([b for _p, _a, b, _f, _d in seq] + [p0])
        if not (lo + 0.1 <= worst and best <= hi - 0.1):
            print(f"[!] 目标位置 {worst:+.3f}~{best:+.3f} 超出安全范围 [{lo}, {hi}], 终止")
            return

        period = 1.0 / LOOP_HZ
        t_start = t_ns()
        RX_TIMEOUT_NS = 250e6

        # ---------- 阶段 0: 增益爬升 ----------
        i_loop = 0
        while True:
            now = t_ns()
            el = (now - t_start) / 1e9
            if el >= RAMP_S:
                break
            kp_now = args.kp * min(1.0, el / RAMP_S)
            q, dq, tau = m.Get_Position(), m.Get_Velocity(), m.Get_tau()
            control.control_mit(m, kp_now, args.kd, p0, 0.0, 0.0)
            cmds.append((now, el, "ramp", p0, kp_now, args.kd, 0.0, 0.0, q, dq, tau))
            i_loop += 1
            nxt = t_start + i_loop * period * 1e9
            if nxt > now:
                time.sleep((nxt - now) / 1e9)

        # ---------- 阶段 1: 使能检查 (0.03 rad 小台阶) ----------
        t_chk = t_ns()
        q_chk = min(max(p0 + CHK_AMP, lo + 0.1), hi - 0.1)
        moved = False
        while True:
            now = t_ns()
            if (now - t_chk) / 1e9 >= CHK_S:
                break
            q = m.Get_Position()
            if abs(q - p0) > 0.008:
                moved = True
            if abs(m.Get_tau()) > TAU_STOP:
                control.control_mit(m, 0.0, args.kd, q, 0.0, 0.0)
                print("[!] 使能检查中力矩超限, 收工")
                stop_reason = "使能检查保护"
                break
            control.control_mit(m, args.kp, args.kd, q_chk, 0.0, 0.0)
            cmds.append((now, (now - t_start) / 1e9, "check", q_chk, args.kp, args.kd,
                         0.0, 0.0, q, m.Get_Velocity(), m.Get_tau()))
            time.sleep(period)
        # 回 p0
        t_bk = t_ns()
        while (t_ns() - t_bk) / 1e9 < MOVE_S:
            now = t_ns()
            s = min(1.0, (now - t_bk) / (MOVE_S * 1e9))
            qd = q_chk + (p0 - q_chk) * s
            q = m.Get_Position()
            control.control_mit(m, args.kp, args.kd, qd, 0.0, 0.0)
            cmds.append((now, (now - t_start) / 1e9, "check_back", qd, args.kp, args.kd,
                         0.0, 0.0, q, m.Get_Velocity(), m.Get_tau()))
            time.sleep(period)

        if not stop_reason and not moved:
            control.control_mit(m, 0.0, args.kd, m.Get_Position(), 0.0, 0.0)
            stop_reason = "电机未动(使能未生效?)"
            print(f"[!] {stop_reason} —— 小台阶 0.03 rad 后位置几乎没变, 已切零力矩阻尼")
        if stop_reason:
            pass                                  # 直接跳到收尾
        else:
            print("[+] 使能检查通过, 开始扫描")

            # ---------- 阶段 2: 主序列 ----------
            for bi, (ph, qa, qb, ff, dur) in enumerate(seq):
                tb = t_ns()
                print(f"[{ (tb - t_start)/1e9:6.2f}s] {bi+1:2d}/{len(seq)}  {ph}  "
                      f"q={qb:+.4f}  tau_ff={ff:+.2f}")
                while True:
                    now = t_ns()
                    s = min(1.0, (now - tb) / (dur * 1e9))
                    qd = qa + (qb - qa) * s
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
                        control.control_mit(m, 0.0, args.kd, q, 0.0, 0.0)
                        stop_reason = f"保护触发: {bad}"
                        print(f"\n[!] {stop_reason}, 已切零力矩阻尼")
                        break
                    control.control_mit(m, args.kp, args.kd, qd, 0.0, ff)
                    cmds.append((now, (now - t_start) / 1e9, ph, qd, args.kp, args.kd,
                                 0.0, ff, q, dq, tau))
                    if s >= 1.0:
                        break
                    time.sleep(period)
                if stop_reason:
                    break
            if not stop_reason:
                stop_reason = "完成"

        # ---------- 收尾: 零力矩阻尼缓放 ----------
        for _ in range(int(REL_S * 1000)):
            control.control_mit(m, 0.0, args.kd, m.Get_Position(), 0.0, 0.0)
            time.sleep(0.001)
        print(f"[+] 结束: {stop_reason}  最终位置 {m.Get_Position():+.4f} rad")

        with open(os.path.join(out_dir, "command.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_host_mono_ns", "t_rel_s", "phase", "q_cmd_rad", "kp", "kd",
                        "dq_cmd_rad_s", "tau_ff_nm", "q_real_rad", "dq_real", "tau_real_nm"])
            w.writerows(cmds)
        with open(os.path.join(out_dir, "feedback.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_host_mono_ns", "channel", "can_id", "data_hex"])
            w.writerows(frames)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            f.write(f'{{"run_id": "{run_id}", "joint": "{name}", "canid": {args.canid}, '
                    f'"channel": {ch}, "p0": {p0}, "kp": {args.kp}, "kd": {args.kd}, '
                    f'"p_offs": {list(P_OFFS)}, "tau_ffs": {list(TFFS)}, '
                    f'"stop_reason": "{stop_reason}", "p_final": {m.Get_Position()}}}')
        with open(os.path.join(out_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
            f.write(f"""# 力矩前馈扫描 manifest
run_id: {run_id}
utc_start: {stamp.utcnow().isoformat()}Z
local_start: {stamp.isoformat()}
operator: xu
purpose: 07 工作包 - 扫 tau_ff 分离 KP 真实值与固定偏置(台阶数据解不开的那两个数)
safety:
  kp: {args.kp}
  kd: {args.kd}
  tau_ff_range_nm: [{min(TFFS)}, {max(TFFS)}]
  pos_offsets_rad: {list(P_OFFS)}
  tau_stop_nm: {TAU_STOP}
  pos_bounds: [{lo}, {hi}]
  feedback_timeout_ms: 250
  zero_position_touched: false   # 全程未发 0xFE
  result: {stop_reason}
hardware:
  joint: {name} canid{args.canid} ch{ch} mst 0x{mst:02X}
  can_device_sn: {SN}
notes:
  - q_des 只在 p0 +-0.06 rad 内, 实际偏离由 tau_ff 决定, 上界 ~(|tau_ff|+负载)/KP ~ 25 mrad
  - 反馈时间戳含驱动批交付(~109ms)影响; 稳态窗口取每段末 0.4s, 不受影响
""")
        print(f"[+] 数据: {out_dir}")
    except KeyboardInterrupt:
        print("\n[!] 手动中断, 已停止下令")
    finally:
        # close() 在本机会卡死(已实测), 所以失能必须自己先发, 不能依赖它内部的 disable_all
        if control is not None:
            try:
                for _ in range(3):
                    control.control_cmd(args.canid, 0xFD, ch)
                    time.sleep(0.002)
                print("[+] 已显式发 0xFD 失能本关节")
            except Exception as exc:
                print(f"[!] 显式失能失败: {exc}")
            try:
                control.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
