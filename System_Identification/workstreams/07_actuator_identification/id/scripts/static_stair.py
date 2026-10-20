"""静态阶梯（单电机，伙伴失能，悬空）：在起点附近的一组角度上正反两个方向逼近，取稳态力矩，用于辨识零位偏差、重力力矩曲线与摩擦。

一次只使能一个电机，伙伴电机失能（自由运动），所以闭链不会产生内力（DeepSeek 的修正方案，也是所有单关节数据的做法）。
位姿（电机坐标，相对起点）：-0.3, -0.2, -0.1, 0, +0.1, +0.2, +0.3 rad；每个位姿先从下方逼近一次（先到 L-0.04 再到 L），
再从上方逼近一次（先到 L+0.04 再到 L），每次到位保持 2 s，取末 0.5 s 的位置与力矩均值；摩擦死区的方向性由正反平均消除，
(上行力矩 - 下行力矩) / 2 给出摩擦。保持增益 kp 28、kd 2。

安全设计（程序内强制）：默认只打印计划，必须加 --execute 与 --timeout_ms/--timeout_unit_us（使能前写看门狗，结束写回并读回）；
开始前八个电机都必须应答且状态码为 0；保护：|力矩| 超过 3.0 N·m、|位置 - 指令| 超过 0.10 rad（堵转）、|dq| 超过 1.5 rad/s、
故障码不是 1、反馈中断超过 100 ms，立即失能；机器人必须悬空、急停可触达，每次运行都需要用户当次明确同意。

用法：
    python static_stair.py --motor 1                                   # 只看计划
    python static_stair.py --motor 1 --execute --timeout_ms 300 --timeout_unit_us 50
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
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
LEGS = {"FL": (1, 2), "FR": (3, 4), "RL": (5, 6), "RR": (7, 8)}
PMAX, VMAX, TMAX = 12.5, 45.0, 20.0
LEVELS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
PRE = 0.04
TAULIM, ERRLIM, DQLIM, SILENCE_S = 3.0, 0.10, 1.5, 0.100
RID_TIMEOUT = 9


def decode(payload: bytes):
    q = (payload[1] << 8) | payload[2]
    dq = (payload[3] << 4) | (payload[4] >> 4)
    tau = ((payload[4] & 0xF) << 8) | payload[5]
    return (payload[0] >> 4) & 0xF, q / 65535 * 2 * PMAX - PMAX, dq / 4095 * 2 * VMAX - VMAX, tau / 4095 * 2 * TMAX - TMAX


def build_sequence(rate: float, ramp_s=1.0, hold_s=2.0):
    """返回 [(tag, offset, pose_idx)]：每个位姿两次逼近（from_below / from_above）。"""
    seq, prev, idx = [], 0.0, 0
    plan = []
    for L in LEVELS:
        plan.append((f"L{L:+.1f}_from_below", L - PRE, L))
        plan.append((f"L{L:+.1f}_from_above", L + PRE, L))
    n_r, n_h = round(ramp_s * rate), round(hold_s * rate)
    for k, (tag, pre, L) in enumerate(plan):
        for target, kind in ((pre, "pre"), (L, "ramp")):
            for i in range(n_r):
                w = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n_r)
                seq.append((f"{tag}_{kind}", prev + (target - prev) * w, k))
            prev = target
        for i in range(n_h):
            seq.append((f"{tag}_hold", L, k))
    return seq, plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", type=int, choices=sorted(MOTORS), required=True)
    ap.add_argument("--rate_hz", type=float, default=500.0)
    ap.add_argument("--kp", type=float, default=28.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--timeout_ms", type=float, default=0.0)
    ap.add_argument("--timeout_unit_us", type=float, default=None)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    timeout_raw = 0
    if args.timeout_ms > 0:
        if not args.timeout_unit_us or args.timeout_unit_us <= 0 or args.timeout_ms < 50:
            print("[拒绝] --timeout_ms 需 >= 50 且必须同时给 --timeout_unit_us")
            return 1
        timeout_raw = round(args.timeout_ms * 1000 / args.timeout_unit_us)
    seq, poses = build_sequence(args.rate_hz)
    ids = (args.motor,)
    print(f"计划：电机 {args.motor}（{NAMES[args.motor]}，伙伴失能），{len(LEVELS)} 个角度 x 正反两个方向，保持增益 kp={args.kp:g} kd={args.kd:g}，共 {len(seq) / args.rate_hz:.0f} s。"
          + (f" 看门狗 {args.timeout_ms:g} ms（寄存器值 {timeout_raw}）。" if timeout_raw else " 看门狗：未设置。")
          + f" 保护：力矩 {TAULIM} N·m、位置偏差 {ERRLIM} rad、|dq| {DQLIM} rad/s。")
    if not args.execute:
        print("未加 --execute，没有动设备。")
        return 0
    if not timeout_raw:
        print("[拒绝] 实测运行必须设置看门狗")
        return 1

    spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", DM_CAN)
    dmc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dmc
    spec.loader.exec_module(dmc)
    profile = getattr(dmc, "_dm_sdk_profile", None) or getattr(dmc, "_dm_linux_profile", None) or os.environ.get("BENNETT_DM_SDK_PROFILE", "original")
    if profile == "original":
        print("[拒绝] 官方库反馈成批交付；请用 low_latency 或 custom")
        return 1

    init = [dmc.DmActData(dmc.DM_Motor_Type.DM8006, dmc.Control_Mode.MIT_MODE, c, 0x10 + c, ch) for c, ch in MOTORS.items()]
    ctl = dmc.Motor_Control(1_000_000, 5_000_000, SN, init, device_type=dmc.dmcan_device_type.USB2CANFD_DUAL, auto_enable=False)
    ns = time.perf_counter_ns
    frames: list = []
    states: dict = {}
    latest: dict = {}
    last_rx: dict = {}
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
        print("[中止] 要求八台都应答且状态码为 0")
        return 2

    orig_to, armed = {}, []

    def restore():
        for c in armed:
            try:
                ctl.write_motor_param(motors[c], RID_TIMEOUT, int(orig_to[c]).to_bytes(4, "little"))
                time.sleep(0.1)
                back = ctl.read_motor_param(motors[c], RID_TIMEOUT, timeout=1.0)
                print(f"[看门狗] 电机 {c} 写回 TIMEOUT={orig_to[c]}，读回 {back}")
            except Exception as exc:
                print(f"[看门狗] 电机 {c} 写回失败：{exc}")

    for c in ids:
        v = ctl.read_motor_param(motors[c], RID_TIMEOUT, timeout=1.0)
        if v is None:
            print(f"[中止] 电机 {c} 读不到原 TIMEOUT")
            restore()
            return 2
        orig_to[c] = int(v)
        ctl.write_motor_param(motors[c], RID_TIMEOUT, int(timeout_raw).to_bytes(4, "little"))
        armed.append(c)
        time.sleep(0.1)
        back = ctl.read_motor_param(motors[c], RID_TIMEOUT, timeout=1.0)
        if back is None or int(back) != timeout_raw:
            print(f"[中止] 电机 {c} TIMEOUT 写入读回 {back}")
            restore()
            return 2
    print(f"[看门狗] 电机 {ids} TIMEOUT -> {timeout_raw}")

    stamp = datetime.now()
    run_id = f"{stamp.strftime('%Y%m%d_%H%M%S')}_stair_{NAMES[args.motor]}"
    out_dir = OUT_ROOT / stamp.strftime("%Y-%m-%d") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_rows, abort, q0 = [], {"why": None}, {}

    def disable():
        for _ in range(5):
            for c in ids:
                ctl.control_cmd(motors[c].GetCanId() + motors[c].GetMotorMode(), 0xFD, motors[c].GetChannel())
            time.sleep(0.003)

    n_pre = 0
    try:
        frames.clear()
        for c in ids:
            ctl.switchControlMode(motors[c], dmc.control_mode_to_code[motors[c].GetMotorMode()])
            time.sleep(0.005)
        for _ in range(3):
            for c in ids:
                ctl.control_cmd(motors[c].GetCanId() + motors[c].GetMotorMode(), 0xFC, motors[c].GetChannel())
                time.sleep(0.002)
            time.sleep(0.005)
        t_wait = ns() + int(0.3e9)
        while ns() < t_wait:
            for c in ids:
                ctl.control_mit(motors[c], 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.002)
        time.sleep(min(0.15, 0.4 * args.timeout_ms / 1e3))
        n_pre = len(frames)
        for c in ids:
            if c not in latest:
                raise RuntimeError(f"电机 {c} 使能后没有反馈")
            st, q, _, _ = decode(latest[c][1])
            if st != 1:
                raise RuntimeError(f"电机 {c} 使能后状态码 {st}")
            q0[c] = q
        print("[使能] 起点 q0 = " + ", ".join(f"{c}:{q0[c]:+.4f}" for c in ids))
        period_ns = int(1e9 / args.rate_hz)
        t_next = ns()
        t_loop0 = t_next
        pre = round(1.0 * args.rate_hz)           # 先在起点保持 1 s
        total = pre + len(seq)
        k = 0
        while k < total:
            if k < pre:
                tag, d1, pose = "pre", 0.0, -1
            else:
                tag, d1, pose = seq[k - pre]
            for c, dd in ((ids[0], d1),):
                qd = q0[c] + dd
                tb = ns()
                ok = ctl.control_mit(motors[c], args.kp, args.kd, qd, 0.0, 0.0)
                ta = ns()
                cmd_rows.append((k, c, tb, ta, args.kp, args.kd, qd, 0.0, tag, pose, int(bool(ok))))
            k += 1
            now = ns()
            for c, dd in ((ids[0], d1),):
                st, q, dq, tau = decode(latest[c][1])
                if st != 1:
                    abort["why"] = f"电机 {c} 状态码 {st}"
                elif abs(tau) > TAULIM:
                    abort["why"] = f"电机 {c} 力矩 {tau:+.2f} N·m（限 {TAULIM}）"
                elif abs(q - (q0[c] + dd)) > ERRLIM:
                    abort["why"] = f"电机 {c} 位置偏差 {q - (q0[c] + dd):+.3f} rad（限 {ERRLIM}）"
                elif abs(dq) > DQLIM:
                    abort["why"] = f"电机 {c} |dq| {dq:+.2f} rad/s"
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
        disable()
        print("[结束] 已发送失能命令")
    time.sleep(0.3)
    restore()

    with open(out_dir / "commands.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cmd_index", "motor", "t_send_begin_ns", "t_send_end_ns", "kp", "kd", "q_des", "tau_ff", "phase", "pose", "send_ok"])
        w.writerows(cmd_rows)
    with open(out_dir / "feedback.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reply_index", "motor", "t_host_ns", "state", "q", "dq", "tau"])
        for i, (t, mid, payload) in enumerate(frames):
            st, q, dq, tau = decode(payload)
            w.writerow([i, mid, t, st, q, dq, tau])
    meta = {"run_id": run_id, "protocol": "static_stair", "motor": args.motor, "name": NAMES[args.motor], "motors": list(ids), "levels": list(LEVELS), "rate_hz": args.rate_hz,
            "kp": args.kp, "kd": args.kd, "q0": {str(c): v for c, v in q0.items()}, "plan": poses, "abort": abort["why"], "n_pre_replies_all": n_pre,
            "dll": profile, "timeout_ms": args.timeout_ms, "timeout_raw": timeout_raw}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[数据] {out_dir.relative_to(REPO)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
