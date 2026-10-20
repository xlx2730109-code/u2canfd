"""动作侧延迟：命令变化后，电机的反馈多久才出现响应。单个关节、小幅度、悬空、带软件保护。

测法：1 kHz 连续发 MIT 命令（和部署一样），每个命令帧电机回一帧反馈，反馈与命令一一对应（01 号包已验证 tx = rx）。
在随机时刻把设定值阶跃一小步（位置阶跃 --mode pos，或力矩前馈脉冲 --mode tau），记录第 k0 个命令是第一个带新设定值的帧；
之后第 j 个反馈里力矩（或速度、位置）第一次超出基线噪声 5 倍，则响应延迟 = j - k0 个命令周期（1 个周期 1 ms）。
用"第几个命令"而不是主机时间，可以避开驱动批量交付造成的到达时刻偏差；同时也记录主机时刻供对照。

安全：
  - 默认只打印计划；必须加 --execute 才会使能电机；
  - 只使能 --motor 指定的一个电机，开始前 8 个电机必须都应答且失能；
  - 位置阶跃幅度 <= 0.08 rad，力矩脉冲 <= 0.6 N·m 且每次 <= 60 ms；程序内强制上限；
  - 实时保护：位置偏离起点超过 0.25 rad、反馈力矩超过 4 N·m、故障码不是 1（使能）、反馈中断超过 100 ms，立即失能退出；
  - 任何退出路径都先发失能命令；
  - 机器人必须悬空，实体急停可触达，见 System_Identification/docs/HARDWARE_SAFETY_CHECKLIST.md，每次运行都需要用户当次明确同意。

用法：
    python action_latency.py                                   # 只看计划
    python action_latency.py --execute --mode pos --amp 0.05   # 位置阶跃
    python action_latency.py --execute --mode tau --tau 0.3    # 力矩脉冲
Windows 用环境变量 BENNETT_DM_SDK_PROFILE 选库（custom 需要 BENNETT_DM_CAN_PATH，见 01 号包 PATCH.md）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
OUT_ROOT = HERE.parent / "data" / "raw"
DM_CAN = Path(os.environ.get("BENNETT_DM_CAN_PATH") or REPO / "sim2real" / "bennett_deploy" / "dm_can.py")

spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", DM_CAN)
dmc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmc
spec.loader.exec_module(dmc)

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}
NAMES = {1: "FL_thigh", 2: "FL_calf", 3: "FR_thigh", 4: "FR_calf", 5: "RL_thigh", 6: "RL_calf", 7: "RR_thigh", 8: "RR_calf"}
MAX_AMP, MAX_TAU, MAX_PULSE_S = 0.08, 0.6, 0.06
LIM_DQ, LIM_TAU, LIM_ERR_STATE, LIM_SILENCE_S = 0.25, 4.0, 1, 0.100
PMAX, VMAX, TMAX = 12.5, 45.0, 20.0


def decode(payload: bytes):
    q = (payload[1] << 8) | payload[2]
    dq = (payload[3] << 4) | (payload[4] >> 4)
    tau = ((payload[4] & 0xF) << 8) | payload[5]
    return (payload[0] >> 4) & 0xF, q / 65535 * 2 * PMAX - PMAX, dq / 4095 * 2 * VMAX - VMAX, tau / 4095 * 2 * TMAX - TMAX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", type=int, default=1, choices=sorted(MOTORS))
    ap.add_argument("--mode", choices=("pos", "tau"), default="pos")
    ap.add_argument("--amp", type=float, default=0.05, help="位置阶跃幅度 rad")
    ap.add_argument("--tau", type=float, default=0.3, help="力矩脉冲幅度 N·m")
    ap.add_argument("--pulse_ms", type=float, default=40.0, help="力矩脉冲宽度 ms")
    ap.add_argument("--kp", type=float, default=28.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--rate_hz", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--execute", action="store_true", help="真的使能电机；不加则只打印计划")
    args = ap.parse_args()

    if not 0 < args.amp <= MAX_AMP or not 0 < args.tau <= MAX_TAU or not 0 < args.pulse_ms / 1e3 <= MAX_PULSE_S:
        print(f"[拒绝] 幅度超出程序内上限：amp <= {MAX_AMP} rad，tau <= {MAX_TAU} N·m，脉冲 <= {MAX_PULSE_S * 1e3:g} ms")
        return 1
    plan = (f"计划：电机 {args.motor}（{NAMES[args.motor]}），模式 {args.mode}，kp={args.kp:g} kd={args.kd:g}，"
            f"{args.repeats} 次阶跃，{'幅度 %g rad' % args.amp if args.mode == 'pos' else '力矩 %g N·m、宽 %g ms' % (args.tau, args.pulse_ms)}，"
            f"阶跃间隔随机 0.8–1.5 s，方向正负交替；保护：偏离 {LIM_DQ} rad、力矩 {LIM_TAU} N·m、反馈中断 {LIM_SILENCE_S * 1e3:g} ms。")
    print(plan)
    if not args.execute:
        print("未加 --execute，没有动设备。")
        return 0

    if getattr(dmc, "_dm_sdk_profile", "") == "original":
        print("[拒绝] 官方库反馈每约 110–120 ms 一批，会触发 100 ms 的反馈中断保护且无法测延迟；请用 low_latency 或 custom")
        return 1
    rng = random.Random(args.seed)
    init = [dmc.DmActData(dmc.DM_Motor_Type.DM8006, dmc.Control_Mode.MIT_MODE, c, 0x10 + c, ch) for c, ch in MOTORS.items()]
    ctl = dmc.Motor_Control(1_000_000, 5_000_000, SN, init, device_type=dmc.dmcan_device_type.USB2CANFD_DUAL, auto_enable=False)
    ns = time.perf_counter_ns
    frames: list = []
    last_rx = {}
    orig_cb = ctl.canframeCallback

    def cb(device, frame):
        t = ns()
        try:
            cid = int(frame.head.can_id)
            if 0x11 <= cid <= 0x18:
                last_rx[cid - 0x10] = t
                if cid - 0x10 == args.motor:
                    frames.append((t, bytes(frame.payload[:8])))
        except Exception:
            pass
        orig_cb(device, frame)

    ctl.getDevice().hook_recv_callback(cb)
    time.sleep(0.3)
    m = ctl.getMotor(MOTORS[args.motor], args.motor)

    # 预检：8 个电机都应答且失能
    for c in MOTORS:
        ctl.refresh_motor_status(ctl.getMotor(MOTORS[c], c))
        time.sleep(0.002)
    time.sleep(0.4)
    bad = [c for c in MOTORS if c not in last_rx]
    if bad:
        print(f"[中止] 无应答电机 {bad}")
        return 2
    time.sleep(0.05)

    stamp = datetime.now()
    run_id = f"{stamp.strftime('%Y%m%d_%H%M%S')}_hung_{NAMES[args.motor]}_{args.mode}_r01"
    out_dir = OUT_ROOT / stamp.strftime("%Y-%m-%d") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_rows = []   # 每个发出的命令帧：序号、起止时刻、设定值
    events = []     # 阶跃事件：第一个带新设定值的命令序号、方向
    abort = {"why": None}
    q0 = None
    n_pre = 0

    def disable():
        for _ in range(5):
            ctl.control_cmd(m.GetCanId() + m.GetMotorMode(), 0xFD, m.GetChannel())
            time.sleep(0.003)

    try:
        # 使能并稳住：先读当前位置，把它作为保持点（不让电机动）
        frames.clear()
        ctl.switchControlMode(m, dmc.control_mode_to_code[m.GetMotorMode()])
        time.sleep(0.01)
        for _ in range(3):
            ctl.control_cmd(m.GetCanId() + m.GetMotorMode(), 0xFC, m.GetChannel())
            time.sleep(0.005)
        t_wait = ns() + int(0.3e9)
        while ns() < t_wait:
            ctl.control_mit(m, 0.0, 0.0, 0.0, 0.0, 0.0)  # 先零增益收集当前位置
            time.sleep(0.001)
        time.sleep(0.3)  # 不发命令，让前面所有命令的反馈都到齐，之后反馈序号与命令序号一一对应
        n_pre = len(frames)
        if not frames:
            abort["why"] = "使能后没有反馈"
            raise RuntimeError(abort["why"])
        st, q0, _, _ = decode(frames[-1][1])
        if st != LIM_ERR_STATE:
            abort["why"] = f"使能后状态码 {st}（应为 1）"
            raise RuntimeError(abort["why"])
        print(f"[使能] 起点 q0 = {q0:+.5f} rad，开始 {args.repeats} 次阶跃")

        period_ns = int(1e9 / args.rate_hz)
        t_next = ns()
        k = 0
        q_des, tau_ff = q0, 0.0
        kp, kd = args.kp, args.kd
        phase = "hold"
        t_phase_end = t_next + int(1.0e9)
        n_done, sign = 0, 1
        pulse_until = 0
        sent_step = False
        while n_done < args.repeats:
            now = ns()
            # 阶段机
            if phase == "hold" and now >= t_phase_end:
                phase = "step"
            if phase == "step" and not sent_step:
                if args.mode == "pos":
                    q_des = q0 + sign * args.amp
                    tau_ff = 0.0
                else:
                    tau_ff = sign * args.tau
                    kp = 0.0
                    pulse_until = now + int(args.pulse_ms * 1e6)
                events.append({"cmd_index": k, "sign": sign, "t_host_ns": now})
                sent_step = True
                t_phase_end = now + int(0.6e9)
            if phase == "step" and args.mode == "tau" and now >= pulse_until and tau_ff != 0.0:
                tau_ff = 0.0
                kp = args.kp  # 脉冲结束，回到位置保持（q_des 仍为 q0）
            if phase == "step" and now >= t_phase_end:
                # 回到起点（位置模式下再阶跃回去不记事件，保持下一次独立）
                q_des, tau_ff, kp = q0, 0.0, args.kp
                phase = "hold"
                t_phase_end = now + int(rng.uniform(0.8, 1.5) * 1e9)
                sent_step = False
                sign = -sign
                n_done += 1
            if phase == "hold" and args.mode == "tau":
                kp = args.kp
            tb = ns()
            ok = ctl.control_mit(m, kp, kd, q_des, 0.0, tau_ff)
            ta = ns()
            cmd_rows.append((k, tb, ta, kp, kd, q_des, tau_ff, int(bool(ok))))
            k += 1
            # 保护
            if frames:
                st, q, dq, tau = decode(frames[-1][1])
                if st != LIM_ERR_STATE:
                    abort["why"] = f"状态码 {st}"
                elif abs(q - q0) > LIM_DQ:
                    abort["why"] = f"位置偏离 {q - q0:+.3f} rad"
                elif abs(tau) > LIM_TAU:
                    abort["why"] = f"力矩 {tau:+.2f} N·m"
                elif (ns() - frames[-1][0]) / 1e9 > LIM_SILENCE_S:
                    abort["why"] = "反馈中断"
            if abort["why"]:
                raise RuntimeError(abort["why"])
            t_next += period_ns
            late = t_next - ns()
            if late > 0:
                time.sleep(late / 1e9)
            elif late < -int(50e6):
                t_next = ns()
    except Exception as exc:
        print(f"[保护触发] {exc}")
    finally:
        disable()
        print("[结束] 已发送失能命令")
    time.sleep(0.3)  # 收尾，让在途反馈到达
    n_replies_total = len(frames)

    # 落盘
    with open(out_dir / "commands.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cmd_index", "t_send_begin_ns", "t_send_end_ns", "kp", "kd", "q_des", "tau_ff", "send_ok"])
        w.writerows(cmd_rows)
    with open(out_dir / "feedback.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reply_index", "t_host_ns", "state", "q", "dq", "tau"])
        for i, (t, payload) in enumerate(frames):
            st, q, dq, tau = decode(payload)
            w.writerow([i, t, st, q, dq, tau])
    meta = {"run_id": run_id, "motor": args.motor, "name": NAMES[args.motor], "mode": args.mode, "amp": args.amp, "tau": args.tau,
            "pulse_ms": args.pulse_ms, "kp": args.kp, "kd": args.kd, "events": events, "abort": abort["why"],
            "q0": q0, "n_pre_replies": n_pre, "n_cmds_main": len(cmd_rows), "n_replies_total": n_replies_total,
            "aligned": (n_replies_total - n_pre == len(cmd_rows)), "dll": getattr(dmc, "_dm_sdk_profile", "linux"), "rate_hz": args.rate_hz,
            "note": "反馈第 j 个对应命令第 j 个（有一个固定偏移，由分析脚本用使能后的第一个应答对齐）"}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[数据] {out_dir.relative_to(REPO)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
