# -*- coding: utf-8 -*-
"""匀速斜坡 —— 量 KD 真实值 + 摩擦阻尼(真机, 悬空).

为什么这样设计
--------------
MIT 律:  tau = KP*(q_des - q) + KD*(dq_des - dq)
静止时 dq≈0, KD 项整个消失(这就是台阶数据量不出 KD 的原因). 只有在动的时候
dq 才不为零, 但快速瞬态里电流环跟不上, 力矩会滞后 —— 除非**匀速**:
匀速时 Δq 收敛成常数、dq 也是常数, 所有量都不随时间变, 电流环滞后就失效了.

同一速度跑两遍, 只改 dq_des:
    关前馈: tau = KP*Δq1 - KD*(0 - v) = KP*Δq1 - KD*v
    开前馈: tau = KP*Δq2 - KD*(v - v) = KP*Δq2      <- 固件自己把 KD 项抵消掉
两遍的物理负载完全相同(同速度同位置), 所以 tau 相同:
    KD*v = KP*(Δq1 - Δq2)   =>   KD = KP*(Δq1 - Δq2)/v
Δq 是位置量, 16 位, 1 LSB = 0.38 mrad, 比力矩(12 位, 9.8 mN·m)精细得多.

摩擦阻尼顺带出:
  同一位置正反两向 tau_out(q) - tau_in(q) = 2*tau_c + 2*b_v*|v|
  => 对 |v| 拟合, 截距给库仑摩擦 tau_c, 斜率给粘性阻尼 b_v.

安全声明
--------
  - 只驱动一个关节(默认 FL_thigh), 其余电机不下令, 全程不下地;
  - **不调用 set_zero_position, 不发 0xFE, 不碰零位**;
  - 往返一次位移 v*(T_CR+T_ACC), 最快 0.15 rad/s 时 0.20 rad = 11.6°, 且必定回到 p0;
  - 硬保护: |tau|>3Nm / 位置越界 / 反馈超时 250ms -> 立即切零力矩阻尼并终止.

用法:
  py -3.13 ramp_velocity.py --dry-run     # 只打印动作序列, 不碰硬件
  py -3.13 ramp_velocity.py               # FL_thigh 实跑
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
TAU_STOP = 3.0
LOOP_HZ = 1000
BOUNDS = {"thigh": (-0.80, 0.80), "calf": (-0.90, 0.55)}

RAMP_S = 0.8
T_ACC = 0.35          # 加减速时间
T_CR = 0.9            # 匀速段
HOLD_S = 0.4          # 往返之间停一下
REL_S = 0.5
T_ONE = 2 * T_ACC + T_CR          # 单程时长
T_TRAV = 2 * T_ONE + HOLD_S       # 一个往返

SPEEDS = (0.05, 0.10, 0.15)       # rad/s
CHK_AMP = 0.03

sys.setswitchinterval(0.0005)


def leg_traj(s, qs, d, v):
    """单程轨迹. s∈[0,T_ONE], 起点 qs, 方向 d=±1, 速度 v. 返回 (q_des, dq_des, 段名)."""
    if s < T_ACC:                                   # 抛物加速: 速度连续, 末端正好到 v
        return (qs + d * v * s * s / (2 * T_ACC),
                d * v * s / T_ACC, "accel")
    if s < T_ACC + T_CR:                            # 匀速
        return (qs + d * v * (s - T_ACC / 2), d * v, "cruise")
    u = s - T_ACC - T_CR                            # 抛物减速
    return (qs + d * v * (T_CR + T_ACC / 2 + u - u * u / (2 * T_ACC)),
            d * v * (1 - u / T_ACC), "decel")


def traversal(t, p0, v, ff_on):
    """一个往返. t∈[0,T_TRAV]. 返回 (q_des, dq_des, 段名, 方向)."""
    travel = v * (T_CR + T_ACC)
    if t < T_ONE:
        q, dq, seg = leg_traj(t, p0, +1.0, v)
        d = +1
    elif t < 2 * T_ONE:
        q, dq, seg = leg_traj(t - T_ONE, p0 + travel, -1.0, v)
        d = -1
    else:
        return (p0, 0.0, "hold", 0)
    return (q, dq if ff_on else 0.0, seg, d)


def build_sequence(p0):
    """返回 [(phase, speed, ff_on, dur)]."""
    seq = []
    for v in SPEEDS:
        for ff_on in (False, True):
            tag = "ffon" if ff_on else "ff0"
            seq.append((f"v{v:.2f}_{tag}", v, ff_on, T_TRAV))
    return seq


def main():
    global T_ACC, T_CR, T_ONE, T_TRAV, SPEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--canid", type=int, default=1, choices=sorted(MOTORS))
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--kd", type=float, default=KD)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--speeds", type=str, default="",
                    help="逗号分隔, 覆盖默认速度表, 例如 0.03,0.08,0.13")
    ap.add_argument("--t-acc", type=float, default=T_ACC)
    ap.add_argument("--t-cr", type=float, default=T_CR,
                    help="匀速段时长; 加长可以让正反两趟的位置范围有重叠, 摩擦才解得出来")
    ap.add_argument("--tag", type=str, default="r01")
    args = ap.parse_args()

    # 允许改速度/时长: 摩擦配对法要求正反两趟的**实际位置范围**有重叠.
    # 重叠 = v*T_CR - (Δq_out-Δq_in) - 2*0.109*v, 速度越低重叠越小, 所以要靠加长 T_CR 补.
    T_ACC, T_CR = args.t_acc, args.t_cr
    T_ONE = 2 * T_ACC + T_CR
    T_TRAV = 2 * T_ONE + HOLD_S
    if args.speeds:
        SPEEDS = tuple(float(s) for s in args.speeds.split(",") if s.strip())

    name, ch = MOTORS[args.canid]
    kind = "thigh" if "thigh" in name else "calf"
    lo, hi = BOUNDS[kind]

    if args.dry_run:
        print(f"[dry-run] {name} canid{args.canid} ch{ch}  kp={args.kp} kd={args.kd}")
        print(f"[dry-run] 速度 {SPEEDS} rad/s, 每速两遍(dq_des=0 / dq_des=v)")
        print(f"[dry-run] 单程 {T_ONE:.2f}s (加速 {T_ACC} + 匀速 {T_CR} + 减速 {T_ACC}), "
              f"往返 {T_TRAV:.2f}s")
        t = RAMP_S + 1.0 + 0.4
        for ph, v, ff, d in build_sequence(0.0):
            print(f"  {ph:<14} v={v:.2f}  dq_des={'v' if ff else '0':<2}  "
                  f"位移±{v*(T_CR+T_ACC)*1000:5.1f} mrad ({v*(T_CR+T_ACC)*57.3:4.1f}°)  "
                  f"时长 {d:.1f}s  累计 {t:5.1f}s")
            t += d
        print(f"[dry-run] 总时长约 {t + REL_S:.1f} s")
        print(f"[dry-run] 最大力矩估计: 摩擦+重力 ~0.5 + KD*v {KD*max(SPEEDS):.2f} "
              f"+ 加速项 ~0.1 = {0.5 + KD*max(SPEEDS) + 0.1:.2f} N·m < 保护 {TAU_STOP}")
        return

    stamp = datetime.now()
    run_id = stamp.strftime("%Y%m%d_%H%M%S") + f"_ramp_{name}_{args.tag}"
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
        print(f"[+] 前置检查通过: p0={p0:+.4f} rad")

        reach = max(SPEEDS) * (T_CR + T_ACC)
        if not (lo + 0.1 <= p0 - reach and p0 + reach <= hi - 0.1):
            print(f"[!] p0±{reach:.3f} 超出安全范围 [{lo}, {hi}], 终止")
            return

        period = 1.0 / LOOP_HZ
        RX_TIMEOUT_NS = 250e6
        t_start = t_ns()
        i_loop = 0

        # ---------- 增益爬升 ----------
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

        # ---------- 使能检查 ----------
        t_chk = t_ns()
        q_chk = min(max(p0 + CHK_AMP, lo + 0.1), hi - 0.1)
        moved = False
        while (t_ns() - t_chk) / 1e9 < 1.0:
            now = t_ns()
            q = m.Get_Position()
            if abs(q - p0) > 0.008:
                moved = True
            if abs(m.Get_tau()) > TAU_STOP:
                stop_reason = "使能检查保护"
                break
            control.control_mit(m, args.kp, args.kd, q_chk, 0.0, 0.0)
            cmds.append((now, (now - t_start) / 1e9, "check", q_chk, args.kp, args.kd,
                         0.0, 0.0, q, m.Get_Velocity(), m.Get_tau()))
            time.sleep(period)
        t_bk = t_ns()
        while (t_ns() - t_bk) / 1e9 < 0.4:
            now = t_ns()
            s = min(1.0, (now - t_bk) / 0.4e9)
            qd = q_chk + (p0 - q_chk) * s
            q = m.Get_Position()
            control.control_mit(m, args.kp, args.kd, qd, 0.0, 0.0)
            cmds.append((now, (now - t_start) / 1e9, "check_back", qd, args.kp, args.kd,
                         0.0, 0.0, q, m.Get_Velocity(), m.Get_tau()))
            time.sleep(period)

        if not stop_reason and not moved:
            control.control_mit(m, 0.0, args.kd, m.Get_Position(), 0.0, 0.0)
            stop_reason = "电机未动(使能未生效?)"
            print(f"[!] {stop_reason}, 已切零力矩阻尼")
        if not stop_reason:
            print("[+] 使能检查通过, 开始斜坡")
            for ph, v, ff_on, dur in build_sequence(p0):
                tb = t_ns()
                print(f"[{(tb - t_start)/1e9:6.2f}s] {ph}  dq_des={'v' if ff_on else '0'}  "
                      f"位移 ±{v*(T_CR+T_ACC):.4f} rad")
                while True:
                    now = t_ns()
                    t = (now - tb) / 1e9
                    if t >= dur:
                        break
                    qd, dqd, seg, d = traversal(t, p0, v, ff_on)
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
                    control.control_mit(m, args.kp, args.kd, qd, dqd, 0.0)
                    cmds.append((now, (now - t_start) / 1e9,
                                 f"{ph}_{seg}_{'out' if d > 0 else ('in' if d < 0 else 'h')}",
                                 qd, args.kp, args.kd, dqd, 0.0, q, dq, tau))
                    time.sleep(period)
                if stop_reason:
                    break
            if not stop_reason:
                stop_reason = "完成"

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
                    f'"speeds": {list(SPEEDS)}, "t_acc": {T_ACC}, "t_cruise": {T_CR}, '
                    f'"stop_reason": "{stop_reason}", "p_final": {m.Get_Position()}}}')
        with open(os.path.join(out_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
            f.write(f"""# 匀速斜坡 manifest
run_id: {run_id}
utc_start: {stamp.utcnow().isoformat()}Z
local_start: {stamp.isoformat()}
operator: xu
purpose: 07 工作包 - 匀速段量 KD 真实值 + 库仑摩擦/粘性阻尼
design: 同一速度跑两遍, 只改 dq_des(0 vs v), 于是 KD*v = KP*(Δq_ff0 - Δq_ffv)
safety:
  kp: {args.kp}
  kd: {args.kd}
  speeds_rad_s: {list(SPEEDS)}
  travel_per_leg_rad: {[round(v*(T_CR+T_ACC), 4) for v in SPEEDS]}
  tau_stop_nm: {TAU_STOP}
  pos_bounds: [{lo}, {hi}]
  feedback_timeout_ms: 250
  zero_position_touched: false   # 全程未发 0xFE
  result: {stop_reason}
hardware:
  joint: {name} canid{args.canid} ch{ch} mst 0x{mst:02X}
  can_device_sn: {SN}
notes:
  - 每个往返必定回到 p0, 净位移为 0
  - 匀速窗口取 T_ACC+0.15 ~ T_ACC+T_CR-0.15, 躲开加减速
  - 反馈时间戳含驱动批交付(~109ms)影响
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
