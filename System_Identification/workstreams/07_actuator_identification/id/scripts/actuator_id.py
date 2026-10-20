"""执行器辨识采集：单个悬空关节，按协议发令，记录每个命令与每个应答，供 GPU 拟合仿真执行器参数。

协议（--protocol）：
  tau_steps   开环力矩阶跃：脉冲期间 kp=kd=0，只给前馈力矩 tau_ff = 保持力矩偏置 + 幅值；多幅值、多宽度、正负极性；
              测力矩建立（上升、下降）的形状、是否与幅值有关（限幅、变化率限制）、速度响应。
  chirp       位置对数扫频 q_des = q0 + a sin(...)，部署增益（缺省 kp 28、kd 2、速度项 0）；测闭环带宽、等效惯量、阻尼。
  ramp        慢速三角位置参考（三档速度）；测库仑与粘性摩擦、重力随角度变化（静摩擦由换向处的力矩跳变得到）。
  multisine   随机相位多正弦；考试集，只用来检验已拟合的模型，不参与拟合。

安全设计（程序内强制）：
  - 默认只打印计划，必须加 --execute 才会使能电机；execute 必须同时给 --timeout_ms 与 --timeout_unit_us，
    使能前把电机看门狗写成非零（运行期，不保存），结束写回并读回；没有看门狗不开始；
  - 只使能 --motor 指定的一个电机，开始前 8 个电机必须都应答；被测电机状态码必须为 0（失能）；
  - 实时保护：位置偏离起点超过 QLIM（保持阶段 0.25 rad，自由阶段 0.12 rad）、|dq| 超过 3 rad/s、|力矩| 超过 4 N·m、
    故障码不是 1、反馈中断超过 100 ms，立即失能退出；所有退出路径先发失能命令；
  - 自由阶段（kp=kd=0）每次不超过 250 ms，且前馈中含保持力矩偏置（关节自身重力），所以关节不会因重力下落；
  - 幅值上限：力矩阶跃 <= 0.8 N·m，位置幅值 <= 0.12 rad，chirp 频率 <= 10 Hz。
  - 机器人必须悬空，实体急停可触达，见 System_Identification/docs/HARDWARE_SAFETY_CHECKLIST.md，每次运行都需要用户当次明确同意。

用法：
    python actuator_id.py --protocol tau_steps --motor 4                       # 只看计划
    python actuator_id.py --protocol chirp --motor 4 --execute --timeout_ms 300 --timeout_unit_us 50
Windows 用环境变量 BENNETT_DM_SDK_PROFILE 选库；Jetson 用 low_latency（0.5 ms 副本）。
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
MAX_TAU_STEP, MAX_POS_AMP, MAX_CHIRP_HZ, MAX_FREE_S = 0.8, 0.12, 10.0, 0.25
QLIM_HOLD, QLIM_FREE, DQLIM, TAULIM, SILENCE_S = 0.25, 0.12, 3.0, 4.0, 0.100
STATE_ENABLED = 1
RID_TIMEOUT = 9


def decode(payload: bytes):
    q = (payload[1] << 8) | payload[2]
    dq = (payload[3] << 4) | (payload[4] >> 4)
    tau = ((payload[4] & 0xF) << 8) | payload[5]
    return (payload[0] >> 4) & 0xF, q / 65535 * 2 * PMAX - PMAX, dq / 4095 * 2 * VMAX - VMAX, tau / 4095 * 2 * TMAX - TMAX


class Cmd:
    """一个周期的命令：kp、kd、位置、前馈力矩，以及阶段标签与事件编号（用于分析）。"""
    __slots__ = ("kp", "kd", "q", "tau", "phase", "event", "free")

    def __init__(self, kp, kd, q, tau, phase, event=-1, free=False):
        self.kp, self.kd, self.q, self.tau, self.phase, self.event, self.free = kp, kd, q, tau, phase, event, free


def build_tau_steps(args, rate, q0, bias):
    """返回按周期展开的命令列表。bias 是保持阶段测得的平均反馈力矩（重力等），自由阶段用它做前馈。"""
    dt = 1.0 / rate
    n = lambda s: max(1, round(s * rate))
    seq: list = []
    ev = 0
    cases = list(args.cases)
    order = []
    for rep in range(args.repeats):
        for a, w in cases:
            for sgn in (1, -1):
                order.append((sgn * a, w))
    random.Random(args.seed).shuffle(order)
    hold = lambda s, tag: [Cmd(args.kp, args.kd, q0, 0.0, tag) for _ in range(n(s))]
    seq += hold(1.0, "hold0")
    for amp, w in order:
        seq += hold(0.6, "hold")
        seq += [Cmd(0.0, 0.0, q0, bias, "free_pre", ev, True) for _ in range(n(0.02))]
        seq += [Cmd(0.0, 0.0, q0, bias + amp, "pulse", ev, True) for _ in range(n(w / 1000))]
        seq += [Cmd(0.0, 0.0, q0, bias, "free_post", ev, True) for _ in range(n(0.06))]
        ev += 1
    seq += hold(0.5, "hold_end")
    return seq, {"cases": order, "bias": bias}


def build_chirp(args, rate, q0, bias):
    n_total = round(args.duration_s * rate)
    f0, f1 = args.f0, args.f1
    k = math.log(f1 / f0) / args.duration_s
    seq = [Cmd(args.kp, args.kd, q0, 0.0, "hold0") for _ in range(round(1.0 * rate))]
    phase = 0.0
    for i in range(n_total):
        t = i / rate
        f = f0 * math.exp(k * t)
        phase += 2 * math.pi * f / rate
        env = min(1.0, t / 1.0, (args.duration_s - t) / 1.0)
        seq.append(Cmd(args.kp, args.kd, q0 + args.amp * max(0.0, env) * math.sin(phase), 0.0, "chirp"))
    seq += [Cmd(args.kp, args.kd, q0, 0.0, "hold_end") for _ in range(round(1.0 * rate))]
    return seq, {"f0": f0, "f1": f1, "amp": args.amp, "duration_s": args.duration_s}


def build_ramp(args, rate, q0, bias):
    seq = [Cmd(args.kp, args.kd, q0, 0.0, "hold0") for _ in range(round(1.0 * rate))]
    for speed in args.speeds:                       # rad/s
        # 三角波：q0 -> q0+A -> q0-A -> q0，每档两个周期
        for rep in range(2):
            pts = [0.0, args.amp, -args.amp, 0.0]
            for a, b in zip(pts[:-1], pts[1:]):
                steps = max(1, round(abs(b - a) / speed * rate))
                for i in range(steps):
                    seq.append(Cmd(args.kp, args.kd, q0 + a + (b - a) * (i + 1) / steps, 0.0, f"ramp_{speed:g}"))
            seq += [Cmd(args.kp, args.kd, q0, 0.0, "settle") for _ in range(round(0.5 * rate))]
    seq += [Cmd(args.kp, args.kd, q0, 0.0, "hold_end") for _ in range(round(1.0 * rate))]
    return seq, {"speeds": args.speeds, "amp": args.amp}


def build_multisine(args, rate, q0, bias):
    rng = random.Random(args.seed + 100)
    freqs = [0.4, 0.9, 1.7, 3.1, 5.3, 7.7]
    phases = [rng.uniform(0, 2 * math.pi) for _ in freqs]
    peak = sum(1.0 for _ in freqs)
    scale = args.amp / peak
    seq = [Cmd(args.kp, args.kd, q0, 0.0, "hold0") for _ in range(round(1.0 * rate))]
    for i in range(round(args.duration_s * rate)):
        t = i / rate
        env = min(1.0, t / 1.0, (args.duration_s - t) / 1.0)
        y = sum(math.sin(2 * math.pi * f * t + p) for f, p in zip(freqs, phases))
        seq.append(Cmd(args.kp, args.kd, q0 + scale * max(0.0, env) * y, 0.0, "multisine"))
    seq += [Cmd(args.kp, args.kd, q0, 0.0, "hold_end") for _ in range(round(1.0 * rate))]
    return seq, {"freqs": freqs, "phases": phases, "amp": args.amp, "duration_s": args.duration_s}


def build_pd_grid(args, rate, q0, bias):
    """PD 扫描：每个 (kp, kd) 组合依次做 q0 -> +a -> q0 -> -a -> q0 的位置阶跃，每段 0.5 s，重复两遍；组合按阻尼比从大到小排序（最稳的先做）。"""
    inertia = 0.045
    combos = list(args.pd_grid)
    combos.sort(key=lambda kk: -(kk[1] / (2 * math.sqrt(kk[0] * inertia))))
    n = lambda sec: max(1, round(sec * rate))
    seq = [Cmd(args.kp, args.kd, q0, 0.0, "hold0") for _ in range(n(1.0))]
    for i, (kp, kd) in enumerate(combos):
        seq += [Cmd(kp, kd, q0, 0.0, "pd_settle", i) for _ in range(n(0.5))]
        for rep in range(2):
            for off in (args.amp, 0.0, -args.amp, 0.0):
                seq += [Cmd(kp, kd, q0 + off, 0.0, "pd_up" if off > 0 else ("pd_dn" if off < 0 else "pd_back"), i) for _ in range(n(0.5))]
    seq += [Cmd(args.kp, args.kd, q0, 0.0, "hold_end") for _ in range(n(0.5))]
    return seq, {"combos": combos, "amp": args.amp}


BUILDERS = {"tau_steps": build_tau_steps, "chirp": build_chirp, "ramp": build_ramp, "multisine": build_multisine, "pd_grid": build_pd_grid}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=sorted(BUILDERS), required=True)
    ap.add_argument("--motor", type=int, default=4, choices=sorted(MOTORS))
    ap.add_argument("--rate_hz", type=float, default=500.0)
    ap.add_argument("--kp", type=float, default=28.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--amp", type=float, default=0.06, help="chirp / ramp / multisine 的位置幅值 rad")
    ap.add_argument("--duration_s", type=float, default=25.0)
    ap.add_argument("--f0", type=float, default=0.2)
    ap.add_argument("--f1", type=float, default=8.0)
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.05, 0.1, 0.2], help="ramp 的速度 rad/s")
    ap.add_argument("--cases", type=lambda t: [(float(a), float(w)) for a, w in (x.split(":") for x in t.split(","))],
                    default="0.2:20,0.2:60,0.2:150,0.4:20,0.4:60,0.4:100,0.6:20,0.6:40,0.6:60",
                    help="tau_steps 的 幅值N·m:宽度ms 列表；位移约与 幅值*宽度² 成正比，0.6 N·m 在 150 ms 时 FR_calf 位移已达 0.12 rad，所以大幅值配短宽度")
    ap.add_argument("--pd_grid", type=lambda t: [(float(a), float(b)) for a, b in (x.split(":") for x in t.split(","))],
                    default=",".join(f"{kp}:{kd}" for kp in (14, 20, 28, 40, 56) for kd in (0.5, 1, 1.5, 2, 3, 4)),
                    help="pd_grid 的 kp:kd 列表")
    ap.add_argument("--repeats", type=int, default=3, help="tau_steps 每个(幅值,宽度)的重复次数（正负各一次）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout_ms", type=float, default=0.0)
    ap.add_argument("--timeout_unit_us", type=float, default=None)
    ap.add_argument("--execute", action="store_true", help="真的使能电机；不加则只打印计划")
    args = ap.parse_args()

    if args.protocol == "tau_steps" and (max(a for a, _ in args.cases) > MAX_TAU_STEP or max(w for _, w in args.cases) / 1000 > MAX_FREE_S - 0.08
                                         or any(a * w > 60.0 for a, w in args.cases)):
        print(f"[拒绝] 力矩幅值 <= {MAX_TAU_STEP} N·m，脉冲宽度 <= {(MAX_FREE_S - 0.08) * 1e3:g} ms，幅值*宽度(N·m·ms) <= 60")
        return 1
    if args.amp > MAX_POS_AMP or args.f1 > MAX_CHIRP_HZ or args.f0 <= 0 or args.f1 <= args.f0:
        print(f"[拒绝] 位置幅值 <= {MAX_POS_AMP} rad，chirp 频率范围需 0 < f0 < f1 <= {MAX_CHIRP_HZ} Hz")
        return 1
    if args.protocol == "pd_grid" and (max(kp for kp, _ in args.pd_grid) * args.amp > TAULIM * 0.65 or min(kd for _, kd in args.pd_grid) < 0.3):
        print("[拒绝] pd_grid：kp*幅值 超过力矩保护的 65%，或 kd < 0.3（kd 过小可能振荡）")
        return 1
    if args.kp * args.amp > TAULIM * 0.8 and args.protocol not in ("tau_steps", "pd_grid"):
        print(f"[拒绝] kp*幅值 = {args.kp * args.amp:.2f} N·m 超过力矩保护的 80%")
        return 1
    timeout_raw = 0
    if args.timeout_ms > 0:
        if not args.timeout_unit_us or args.timeout_unit_us <= 0:
            print("[拒绝] 使用 --timeout_ms 必须同时给 --timeout_unit_us")
            return 1
        if args.timeout_ms < 50:
            print("[拒绝] --timeout_ms 小于 50 ms：主机偶发停顿可达 10–30 ms，会误触发看门狗")
            return 1
        timeout_raw = round(args.timeout_ms * 1000 / args.timeout_unit_us)
        if not 1 <= timeout_raw < 2 ** 31:
            print("[拒绝] 换算后的寄存器值超出范围")
            return 1

    # 计划：用 q0=0、bias=0 展开一次，只为得到时长与事件数
    plan_seq, plan_info = BUILDERS[args.protocol](args, args.rate_hz, 0.0, 0.0)
    print(f"计划：电机 {args.motor}（{NAMES[args.motor]}），协议 {args.protocol}，{args.rate_hz:g} Hz，保持增益 kp={args.kp:g} kd={args.kd:g}，"
          f"共 {len(plan_seq)} 个周期（{len(plan_seq) / args.rate_hz:.0f} s）。"
          + (f"事件数 {len(plan_info['cases'])}。" if "cases" in plan_info else "")
          + (f"PD 组合 {len(plan_info['combos'])} 个。" if "combos" in plan_info else "")
          + (f" 看门狗 {args.timeout_ms:g} ms（寄存器值 {timeout_raw}）。" if timeout_raw else " 看门狗：未设置。")
          + f" 保护：位置偏离 {QLIM_HOLD}/{QLIM_FREE} rad、|dq| {DQLIM} rad/s、力矩 {TAULIM} N·m、反馈中断 {SILENCE_S * 1e3:g} ms。")
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
    orig_cb = ctl.canframeCallback

    def cb(device, frame):
        t = ns()
        try:
            cid = int(frame.head.can_id)
            if 0x11 <= cid <= 0x18:
                payload = bytes(frame.payload[:8])
                last_rx[cid - 0x10] = t
                states[cid - 0x10] = (payload[0] >> 4) & 0xF
                if cid - 0x10 == args.motor:
                    frames.append((t, payload))
        except Exception:
            pass
        orig_cb(device, frame)

    ctl.getDevice().hook_recv_callback(cb)
    time.sleep(0.3)
    m = ctl.getMotor(MOTORS[args.motor], args.motor)

    for c in MOTORS:
        ctl.refresh_motor_status(ctl.getMotor(MOTORS[c], c))
        time.sleep(0.002)
    time.sleep(0.4)
    bad = [c for c in MOTORS if c not in last_rx]
    print("[开始前状态码] " + ", ".join(f"{c}:{states.get(c, '无应答')}" for c in sorted(MOTORS)))
    if bad:
        print(f"[中止] 无应答电机 {bad}")
        return 2
    if states.get(args.motor) != 0:
        print(f"[中止] 被测电机状态码 {states.get(args.motor)}，必须为 0（失能）；若为 13 需断电重启")
        return 2
    time.sleep(0.05)

    timeout_orig = ctl.read_motor_param(m, RID_TIMEOUT, timeout=1.0)
    if timeout_orig is None:
        print("[中止] 读不到原 TIMEOUT 值")
        return 2
    ctl.write_motor_param(m, RID_TIMEOUT, int(timeout_raw).to_bytes(4, "little"))
    back = ctl.read_motor_param(m, RID_TIMEOUT, timeout=1.0)
    if back is None or int(back) != timeout_raw:
        print(f"[中止] TIMEOUT 写入后读回 {back}，应为 {timeout_raw}")
        ctl.write_motor_param(m, RID_TIMEOUT, int(timeout_orig).to_bytes(4, "little"))
        return 2
    print(f"[看门狗] TIMEOUT {timeout_orig} -> {timeout_raw}（运行期）")

    stamp = datetime.now()
    run_id = f"{stamp.strftime('%Y%m%d_%H%M%S')}_id_{NAMES[args.motor]}_{args.protocol}_r{int(args.rate_hz)}"
    out_dir = OUT_ROOT / stamp.strftime("%Y-%m-%d") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_rows: list = []
    abort = {"why": None}
    info: dict = {}
    q0 = None
    bias = 0.0
    n_pre = 0
    t_loop0 = 0

    def disable():
        for _ in range(5):
            ctl.control_cmd(m.GetCanId() + m.GetMotorMode(), 0xFD, m.GetChannel())
            time.sleep(0.003)

    try:
        frames.clear()
        ctl.switchControlMode(m, dmc.control_mode_to_code[m.GetMotorMode()])
        time.sleep(0.01)
        for _ in range(3):
            ctl.control_cmd(m.GetCanId() + m.GetMotorMode(), 0xFC, m.GetChannel())
            time.sleep(0.005)
        t_wait = ns() + int(0.3e9)
        while ns() < t_wait:
            ctl.control_mit(m, 0.0, 0.0, 0.0, 0.0, 0.0)  # 零增益收集当前位置
            time.sleep(0.001)
        time.sleep(min(0.15, 0.4 * args.timeout_ms / 1e3))  # 静默期，让在途应答到齐；比看门狗短
        n_pre = len(frames)
        if not frames:
            raise RuntimeError("使能后没有反馈")
        st, q0, _, _ = decode(frames[-1][1])
        if st != STATE_ENABLED:
            raise RuntimeError(f"使能后状态码 {st}（应为 1）")
        print(f"[使能] 起点 q0 = {q0:+.5f} rad")

        # 保持 1.5 s 估计保持力矩偏置（重力等），再展开协议
        period_ns = int(1e9 / args.rate_hz)
        t_next = ns()
        t_loop0 = t_next
        hold_frames = round(1.5 * args.rate_hz)
        seq: list = []
        k = 0
        built = False
        total = None
        while True:
            if k == hold_frames and not built:
                tail = [fr for fr in frames[n_pre + hold_frames // 2: n_pre + hold_frames]]
                bias = sum(decode(p)[3] for _, p in tail) / max(1, len(tail))
                info["bias_nm"] = bias
                print(f"[保持力矩偏置] {bias:+.4f} N·m（{len(tail)} 个应答的均值）")
                seq, extra = BUILDERS[args.protocol](args, args.rate_hz, q0, bias)
                info.update(extra)
                built = True
                total = hold_frames + len(seq)
            if built and k >= total:
                break
            if k < hold_frames:
                c = Cmd(args.kp, args.kd, q0, 0.0, "pre_hold")
            else:
                c = seq[k - hold_frames]
            tb = ns()
            ok = ctl.control_mit(m, c.kp, c.kd, c.q, 0.0, c.tau)
            ta = ns()
            cmd_rows.append((k, tb, ta, c.kp, c.kd, c.q, c.tau, c.phase, c.event, int(bool(ok))))
            k += 1
            if frames:
                st, q, dq, tau = decode(frames[-1][1])
                qlim = QLIM_FREE if c.free else QLIM_HOLD
                if st != STATE_ENABLED:
                    abort["why"] = f"状态码 {st}"
                elif abs(q - q0) > qlim:
                    abort["why"] = f"位置偏离 {q - q0:+.3f} rad（限 {qlim}）"
                elif abs(dq) > DQLIM:
                    abort["why"] = f"|dq| {dq:+.2f} rad/s"
                elif abs(tau) > TAULIM:
                    abort["why"] = f"力矩 {tau:+.2f} N·m"
                elif (ns() - max(frames[-1][0], t_loop0)) / 1e9 > SILENCE_S:
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
        abort["why"] = abort["why"] or str(exc)
        print(f"[保护触发] {exc}")
    finally:
        disable()
        print("[结束] 已发送失能命令")
    time.sleep(0.3)
    n_replies_total = len(frames)
    try:
        ctl.write_motor_param(m, RID_TIMEOUT, int(timeout_orig).to_bytes(4, "little"))
        back = ctl.read_motor_param(m, RID_TIMEOUT, timeout=1.0)
        print(f"[看门狗] 已写回 TIMEOUT={timeout_orig}，读回 {back}")
    except Exception as exc:
        print(f"[看门狗] 写回失败：{exc}（运行期设置，电机断电后恢复）")

    with open(out_dir / "commands.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cmd_index", "t_send_begin_ns", "t_send_end_ns", "kp", "kd", "q_des", "tau_ff", "phase", "event", "send_ok"])
        w.writerows(cmd_rows)
    with open(out_dir / "feedback.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reply_index", "t_host_ns", "state", "q", "dq", "tau"])
        for i, (t, payload) in enumerate(frames):
            st, q, dq, tau = decode(payload)
            w.writerow([i, t, st, q, dq, tau])
    n_state1 = sum(1 for fr in frames[n_pre:] if decode(fr[1])[0] == STATE_ENABLED)
    meta = {"run_id": run_id, "motor": args.motor, "name": NAMES[args.motor], "protocol": args.protocol, "rate_hz": args.rate_hz,
            "kp": args.kp, "kd": args.kd, "q0": q0, "bias_nm": bias, "abort": abort["why"], "n_pre_replies": n_pre,
            "n_cmds": len(cmd_rows), "n_replies_total": n_replies_total, "aligned": n_state1 == len(cmd_rows),
            "dll": profile, "timeout_ms": args.timeout_ms, "timeout_raw": timeout_raw, "args": {k_: v for k_, v in vars(args).items()},
            "info": info, "note": "反馈第 n_pre+j 个对应命令第 j 个（状态码 1 的应答数等于命令数时 aligned 为真）"}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[数据] {out_dir.relative_to(REPO)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
