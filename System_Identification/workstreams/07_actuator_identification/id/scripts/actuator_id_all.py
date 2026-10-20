"""八个关节同时的执行器辨识采集（悬空）：每个关节一组不同频率的多正弦位置参考，部署增益，记录全部命令与应答。

用途：单关节数据（actuator_id.py）得到每个电机的延迟、滞后、增益；本脚本给出"整机同时动作"的数据，用于
  - 检验单关节模型在八台同时出力、闭链腿相互耦合时是否仍成立（考试集）；
  - 在 MuJoCo/Isaac 整机模型里拟合电枢惯量、摩擦等与闭链耦合有关的量（PACE 式）。

安全设计（程序内强制）：
  - 默认只打印计划，必须加 --execute 才会使能电机；必须同时给 --timeout_ms 与 --timeout_unit_us，
    使能前给八台电机各写 TIMEOUT（运行期，不保存，读回核对），结束全部写回并读回；
  - 开始前八个电机必须都应答且状态码为 0；
  - 每个关节的实时保护：位置偏离起点超过 0.25 rad、|dq| 超过 3 rad/s、|力矩| 超过 4 N·m、故障码不是 1、反馈中断超过 100 ms，
    任何一个触发，八台全部立即失能退出；所有退出路径先发失能命令；
  - 位置幅值 <= 0.08 rad，前 1.5 s 保持、再用 1 s 把幅值从 0 升到目标；
  - 机器人必须悬空，实体急停可触达，见 System_Identification/docs/HARDWARE_SAFETY_CHECKLIST.md，每次运行都需要用户当次明确同意。

用法：
    python actuator_id_all.py --seed 1                       # 只看计划
    python actuator_id_all.py --seed 1 --execute --timeout_ms 300 --timeout_unit_us 50
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}
NAMES = {1: "FL_thigh", 2: "FL_calf", 3: "FR_thigh", 4: "FR_calf", 5: "RL_thigh", 6: "RL_calf", 7: "RR_thigh", 8: "RR_calf"}
PMAX, VMAX, TMAX = 12.5, 45.0, 20.0
MAX_AMP = 0.08
QLIM, DQLIM, TAULIM, SILENCE_S = 0.25, 3.0, 4.0, 0.100
RID_TIMEOUT = 9


def decode(payload: bytes):
    q = (payload[1] << 8) | payload[2]
    dq = (payload[3] << 4) | (payload[4] >> 4)
    tau = ((payload[4] & 0xF) << 8) | payload[5]
    return (payload[0] >> 4) & 0xF, q / 65535 * 2 * PMAX - PMAX, dq / 4095 * 2 * VMAX - VMAX, tau / 4095 * 2 * TMAX - TMAX


def reference(args, motor: int, t: float) -> float:
    """关节 motor 在时刻 t（相对保护起点）的位置参考偏移。"""
    rng = random.Random(args.seed * 100 + motor)
    freqs = [0.4, 0.9, 1.7, 3.1, 5.3, 7.7]
    phases = [rng.uniform(0, 2 * math.pi) for _ in freqs]
    ramp = min(1.0, max(0.0, (t - args.hold_s) / args.ramp_s))
    tail = min(1.0, max(0.0, (args.hold_s + args.ramp_s + args.duration_s - t) / args.ramp_s))
    y = sum(math.sin(2 * math.pi * f * (t - args.hold_s) + p) for f, p in zip(freqs, phases))
    return args.amp / len(freqs) * ramp * tail * y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate_hz", type=float, default=500.0)
    ap.add_argument("--kp", type=float, default=28.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--amp", type=float, default=0.06)
    ap.add_argument("--hold_s", type=float, default=1.5)
    ap.add_argument("--ramp_s", type=float, default=1.0)
    ap.add_argument("--duration_s", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout_ms", type=float, default=0.0)
    ap.add_argument("--timeout_unit_us", type=float, default=None)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if args.amp > MAX_AMP:
        print(f"[拒绝] 位置幅值 <= {MAX_AMP} rad")
        return 1
    if args.kp * args.amp > TAULIM * 0.8:
        print("[拒绝] kp*幅值超过力矩保护的 80%")
        return 1
    timeout_raw = 0
    if args.timeout_ms > 0:
        if not args.timeout_unit_us or args.timeout_unit_us <= 0 or args.timeout_ms < 50:
            print("[拒绝] --timeout_ms 需 >= 50 且必须同时给 --timeout_unit_us")
            return 1
        timeout_raw = round(args.timeout_ms * 1000 / args.timeout_unit_us)
        if not 1 <= timeout_raw < 2 ** 31:
            print("[拒绝] 换算后的寄存器值超出范围")
            return 1
    total_s = args.hold_s + args.ramp_s + args.duration_s + args.ramp_s + 1.0
    print(f"计划：八个关节同时，多正弦位置参考（每关节独立相位，0.4–7.7 Hz），幅值 {args.amp:g} rad，{args.rate_hz:g} Hz，保持增益 kp={args.kp:g} kd={args.kd:g}，"
          f"约 {total_s:.0f} s。" + (f" 看门狗 {args.timeout_ms:g} ms（寄存器值 {timeout_raw}，八台）。" if timeout_raw else " 看门狗：未设置。")
          + f" 保护：每关节偏离 {QLIM} rad、|dq| {DQLIM} rad/s、力矩 {TAULIM} N·m、反馈中断 {SILENCE_S * 1e3:g} ms，任一触发八台全部失能。")
    if not args.execute:
        print("未加 --execute，没有动设备。")
        return 0
    if not timeout_raw:
        print("[拒绝] 实测运行必须设置看门狗（--timeout_ms 与 --timeout_unit_us）")
        return 1

    spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", DM_CAN)
    dmc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dmc
    spec.loader.exec_module(dmc)
    profile = getattr(dmc, "_dm_sdk_profile", None) or getattr(dmc, "_dm_linux_profile", None) or os.environ.get("BENNETT_DM_SDK_PROFILE", "original")
    if profile == "original":
        print("[拒绝] 官方库反馈成批交付，会触发反馈中断保护；请用 low_latency 或 custom")
        return 1

    init = [dmc.DmActData(dmc.DM_Motor_Type.DM8006, dmc.Control_Mode.MIT_MODE, c, 0x10 + c, ch) for c, ch in MOTORS.items()]
    ctl = dmc.Motor_Control(1_000_000, 5_000_000, SN, init, device_type=dmc.dmcan_device_type.USB2CANFD_DUAL, auto_enable=False)
    ns = time.perf_counter_ns
    frames: list = []
    last_rx: dict = {}
    states: dict = {}
    latest: dict = {}
    orig_cb = ctl.canframeCallback

    def cb(device, frame):
        t = ns()
        try:
            cid = int(frame.head.can_id)
            if 0x11 <= cid <= 0x18:
                payload = bytes(frame.payload[:8])
                mid = cid - 0x10
                last_rx[mid] = t
                states[mid] = (payload[0] >> 4) & 0xF
                latest[mid] = (t, payload)
                frames.append((t, mid, payload))
        except Exception:
            pass
        orig_cb(device, frame)

    ctl.getDevice().hook_recv_callback(cb)
    time.sleep(0.3)
    motors = {c: ctl.getMotor(ch, c) for c, ch in MOTORS.items()}
    for c in MOTORS:
        ctl.refresh_motor_status(motors[c])
        time.sleep(0.002)
    time.sleep(0.4)
    print("[开始前状态码] " + ", ".join(f"{c}:{states.get(c, '无应答')}" for c in sorted(MOTORS)))
    if any(c not in last_rx for c in MOTORS) or any(states.get(c) != 0 for c in MOTORS):
        print("[中止] 要求八台都应答且状态码为 0；状态码 13 的电机需断电重启")
        return 2
    time.sleep(0.05)

    orig_to = {}
    armed = []

    def restore_timeouts():
        for c in armed:
            try:
                ctl.write_motor_param(motors[c], RID_TIMEOUT, int(orig_to[c]).to_bytes(4, "little"))
                time.sleep(0.1)
                back = ctl.read_motor_param(motors[c], RID_TIMEOUT, timeout=1.0)
                print(f"[看门狗] 电机 {c} 写回 TIMEOUT={orig_to[c]}，读回 {back}")
            except Exception as exc:
                print(f"[看门狗] 电机 {c} 写回失败：{exc}（运行期设置，断电后恢复）")

    for c in sorted(MOTORS):
        v = ctl.read_motor_param(motors[c], RID_TIMEOUT, timeout=1.0)
        if v is None:
            print(f"[中止] 电机 {c} 读不到原 TIMEOUT")
            restore_timeouts()
            return 2
        orig_to[c] = int(v)
        ctl.write_motor_param(motors[c], RID_TIMEOUT, int(timeout_raw).to_bytes(4, "little"))
        armed.append(c)
        time.sleep(0.1)
        back = ctl.read_motor_param(motors[c], RID_TIMEOUT, timeout=1.0)
        if back is None or int(back) != timeout_raw:
            print(f"[中止] 电机 {c} TIMEOUT 写入读回 {back}，应为 {timeout_raw}")
            restore_timeouts()
            return 2
    print(f"[看门狗] 八台 TIMEOUT -> {timeout_raw}（运行期）")

    stamp = datetime.now()
    run_id = f"{stamp.strftime('%Y%m%d_%H%M%S')}_idall_ms8_seed{args.seed}_r{int(args.rate_hz)}"
    out_dir = OUT_ROOT / stamp.strftime("%Y-%m-%d") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_rows: list = []
    abort = {"why": None}
    q0: dict = {}
    n_pre = 0

    def disable_all():
        for _ in range(5):
            for c in MOTORS:
                ctl.control_cmd(motors[c].GetCanId() + motors[c].GetMotorMode(), 0xFD, motors[c].GetChannel())
            time.sleep(0.003)

    try:
        frames.clear()
        for c in sorted(MOTORS):
            ctl.switchControlMode(motors[c], dmc.control_mode_to_code[motors[c].GetMotorMode()])
            time.sleep(0.005)
        for _ in range(3):
            for c in sorted(MOTORS):
                ctl.control_cmd(motors[c].GetCanId() + motors[c].GetMotorMode(), 0xFC, motors[c].GetChannel())
                time.sleep(0.002)
            time.sleep(0.005)
        t_wait = ns() + int(0.3e9)
        while ns() < t_wait:
            for c in sorted(MOTORS):
                ctl.control_mit(motors[c], 0.0, 0.0, 0.0, 0.0, 0.0)  # 零增益收集当前位置
            time.sleep(0.002)
        time.sleep(min(0.15, 0.4 * args.timeout_ms / 1e3))
        n_pre = len(frames)
        for c in MOTORS:
            if c not in latest:
                raise RuntimeError(f"电机 {c} 使能后没有反馈")
            st, q, _, _ = decode(latest[c][1])
            if st != 1:
                raise RuntimeError(f"电机 {c} 使能后状态码 {st}（应为 1）")
            q0[c] = q
        print("[使能] 起点 q0 = " + ", ".join(f"{c}:{q0[c]:+.4f}" for c in sorted(q0)))

        period_ns = int(1e9 / args.rate_hz)
        t_next = ns()
        t_loop0 = t_next
        k = 0
        n_total = round(total_s * args.rate_hz)
        while k < n_total:
            t = k / args.rate_hz
            for c in sorted(MOTORS):
                qd = q0[c] + reference(args, c, t)
                tb = ns()
                ok = ctl.control_mit(motors[c], args.kp, args.kd, qd, 0.0, 0.0)
                ta = ns()
                cmd_rows.append((k, c, tb, ta, args.kp, args.kd, qd, 0.0, int(bool(ok))))
            k += 1
            now = ns()
            for c in MOTORS:
                st, q, dq, tau = decode(latest[c][1])
                if st != 1:
                    abort["why"] = f"电机 {c} 状态码 {st}"
                elif abs(q - q0[c]) > QLIM:
                    abort["why"] = f"电机 {c} 位置偏离 {q - q0[c]:+.3f} rad"
                elif abs(dq) > DQLIM:
                    abort["why"] = f"电机 {c} |dq| {dq:+.2f} rad/s"
                elif abs(tau) > TAULIM:
                    abort["why"] = f"电机 {c} 力矩 {tau:+.2f} N·m"
                elif (now - max(latest[c][0], t_loop0)) / 1e9 > SILENCE_S:
                    abort["why"] = f"电机 {c} 反馈中断"
                if abort["why"]:
                    break
            if abort["why"]:
                raise RuntimeError(abort["why"])
            t_next += period_ns
            late = t_next - ns()
            if late > 0:
                time.sleep(late / 1e9)
            elif late < -int(50e6):
                t_next = ns()
    except Exception as exc:
        abort["why"] = abort["why"] or str(exc)
        print(f"[保护触发] {exc}")
    finally:
        disable_all()
        print("[结束] 已向八台发送失能命令")
    time.sleep(0.3)
    n_replies_total = len(frames)
    restore_timeouts()

    with open(out_dir / "commands.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cmd_index", "motor", "t_send_begin_ns", "t_send_end_ns", "kp", "kd", "q_des", "tau_ff", "send_ok"])
        w.writerows(cmd_rows)
    per_motor_replies = {c: 0 for c in MOTORS}
    with open(out_dir / "feedback.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reply_index", "motor", "t_host_ns", "state", "q", "dq", "tau"])
        for i, (t, mid, payload) in enumerate(frames):
            st, q, dq, tau = decode(payload)
            w.writerow([i, mid, t, st, q, dq, tau])
            if i >= n_pre and st == 1:
                per_motor_replies[mid] += 1
    n_cmd_motor = {c: sum(1 for r in cmd_rows if r[1] == c) for c in MOTORS}
    meta = {"run_id": run_id, "protocol": "ms8", "rate_hz": args.rate_hz, "kp": args.kp, "kd": args.kd, "amp": args.amp, "seed": args.seed,
            "q0": q0, "abort": abort["why"], "n_pre_replies_all": n_pre, "n_cmds_per_motor": n_cmd_motor,
            "state1_replies_per_motor_after_pre": per_motor_replies,
            "aligned": all(per_motor_replies[c] == n_cmd_motor[c] for c in MOTORS), "n_replies_total": n_replies_total,
            "dll": profile, "timeout_ms": args.timeout_ms, "timeout_raw": timeout_raw, "args": vars(args),
            "note": "分电机统计：使能后状态码 1 的应答数等于该电机命令数时对齐；预热期（n_pre_replies_all）的应答在 feedback.csv 的前部"}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[数据] {out_dir.relative_to(REPO)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
