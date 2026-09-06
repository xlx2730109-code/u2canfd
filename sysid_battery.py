# -*- coding: utf-8 -*-
"""DM-J8006 系统辨识电池 —— 一次若干阶段,每阶段产 CSV + 图.

全部在 SIM 单位(偏移量)下测量:sim_offset = (Get_Position - default) / sim_to_motor,
与训练端直接可比。SIM_TO_MOTOR(±1,仅符号)照抄 3_quad_leg_track.py 的 MOTOR_SPECS。

启动先按 3_quad_leg_track.py 安全启动方式唤醒全部 8 电机(refresh + 纯阻尼 control_mit(0,1.5,0,0,0)),
只有唤醒成功的电机才会被驱动;没回传的会显式标出(这就是"排除故障"的一部分)。

用法(电机已上电、悬空、急停就绪):
    cd E:\\HuanCun\\Desktop\\u2canfd
    python sysid_battery.py alive                       # 唤醒+确认 8 电机全部回传 (最安全,先跑)
    python sysid_battery.py map                         # 8 电机各 ±0.05 小阶跃 -> 映射比+方向
    python sysid_battery.py kp  --motor FL_thigh        # 阶梯 MIT kp -> 静态刚度标定曲线
    python sysid_battery.py lin --motor FL_thigh        # 多幅值 -> 线性度
    python sysid_battery.py damp --motor FL_thigh       # 小摆荡 -> kd + 库仑摩擦
    python sysid_battery.py lat --motor FL_thigh        # 阶跃 -> 环路延迟(互相关)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damiao import Control_Mode, DM_Motor_Type, DmActData, Motor_Control  # noqa: E402
from dmcan import dmcan_device_type  # noqa: E402

SN = "D6977F56F86C64B77B316E7154FA6DF3"
# 与 3_quad_leg_track.py MOTOR_SPECS 一致:default=0, sim_to_motor=±1(仅符号)
MOTOR_SPECS = [
    {"name": "FL_thigh", "channel": 0, "can_id": 0x01, "mst_id": 0x11, "default": 0.0, "sim_to_motor": -1.0},
    {"name": "FL_calf", "channel": 0, "can_id": 0x02, "mst_id": 0x12, "default": 0.0, "sim_to_motor": +1.0},
    {"name": "FR_thigh", "channel": 1, "can_id": 0x03, "mst_id": 0x13, "default": 0.0, "sim_to_motor": +1.0},
    {"name": "FR_calf", "channel": 1, "can_id": 0x04, "mst_id": 0x14, "default": 0.0, "sim_to_motor": -1.0},
    {"name": "RL_thigh", "channel": 0, "can_id": 0x05, "mst_id": 0x15, "default": 0.0, "sim_to_motor": -1.0},
    {"name": "RL_calf", "channel": 0, "can_id": 0x06, "mst_id": 0x16, "default": 0.0, "sim_to_motor": +1.0},
    {"name": "RR_thigh", "channel": 1, "can_id": 0x07, "mst_id": 0x17, "default": 0.0, "sim_to_motor": +1.0},
    {"name": "RR_calf", "channel": 1, "can_id": 0x08, "mst_id": 0x18, "default": 0.0, "sim_to_motor": -1.0},
]
by_name = {s["name"]: s for s in MOTOR_SPECS}
LIMITS = {"thigh": (-0.80, 0.80), "calf": (-0.90, 0.55)}


DEFAULT_KP = 28.0  # 单关节实验默认 MIT kp,可用 --kp 覆盖


def is_thigh(name):
    return name.endswith("_thigh")


def clamp_sim(name, off):
    lo, hi = LIMITS["thigh" if is_thigh(name) else "calf"]
    return float(min(max(off, lo), hi))


def q_cmd(name, sim_off):
    sp = by_name[name]
    return sp["default"] + sp["sim_to_motor"] * sim_off


def sim_offset(name, motor):
    sp = by_name[name]
    return (motor.Get_Position() - sp["default"]) / sp["sim_to_motor"]


def sim_vel(name, motor):
    sp = by_name[name]
    return motor.Get_Velocity() / sp["sim_to_motor"]


def connect():
    init = [DmActData(motorType=DM_Motor_Type.DM8006, mode=Control_Mode.MIT_MODE,
                      can_id=s["can_id"], mst_id=s["mst_id"], channel=s["channel"])
            for s in MOTOR_SPECS]
    return Motor_Control(1_000_000, 5_000_000, SN, init,
                         device_type=dmcan_device_type.USB2CANFD_DUAL)


def wake_and_check(control):
    """按 3_quad_leg_track.py 安全启动唤醒全部电机:refresh + 纯阻尼保持,确认每台 last_time_ 更新."""
    motors = {s["name"]: control.getMotor(s["channel"], s["can_id"]) for s in MOTOR_SPECS}
    fb_start = {s["name"]: motors[s["name"]].last_time_ for s in MOTOR_SPECS}
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        for s in MOTOR_SPECS:
            m = motors[s["name"]]
            control.refresh_motor_status(m)
            control.control_mit(m, 0.0, 1.5, 0.0, 0.0, 0.0)
        time.sleep(0.01)
        if all(motors[s["name"]].last_time_ > fb_start[s["name"]] for s in MOTOR_SPECS):
            break
    ok = {s["name"]: (motors[s["name"]].last_time_ > fb_start[s["name"]]) for s in MOTOR_SPECS}
    return motors, ok


def save_csv(path, recs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)


def out_paths(outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    st = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(outdir, f"{tag}_{st}.csv"), os.path.join(outdir, f"{tag}_{st}.png")


def plt_xy(path, xs, ys, xlab, ylab, title, fit_slope=None, ms=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "-o", ms=ms)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.grid(True); ax.set_title(title)
    if fit_slope is not None and np.ptp(xs) > 0:
        m_, b_ = np.polyfit(xs, ys, 1)
        xs2 = np.linspace(xs.min(), xs.max(), 20)
        ax.plot(xs2, m_ * xs2 + b_, "r--", label=f"fit slope={m_:.3f}")
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def hold_stage(control, m, name, target, kp, kd, dur_s, recs, t0, hz=1000.0):
    """把电机指令到 target(sim 偏移),保持 dur_s 秒并持续记录。

    用单调时钟忙等做节拍,避免 Windows time.sleep 分辨率(~15.6ms)把 1kHz 撑成 ~64Hz。
    """
    q = q_cmd(name, clamp_sim(name, target))
    dt = 1.0 / hz
    next_t = time.perf_counter()
    end_t = next_t + dur_s
    n = 0
    while next_t < end_t:
        now = time.perf_counter()
        recs.append({"t": now - t0,
                     "target_sim": target, "q_cmd": q,
                     "sim_off": sim_offset(name, m), "sim_vel": sim_vel(name, m),
                     "tau": m.Get_tau(), "err": m.Get_err()})
        control.control_mit(m, kp, kd, q, 0.0, 0.0)
        next_t += dt
        while time.perf_counter() < next_t:
            pass  # 忙等到下一个采样时刻
        n += 1
    return n


def phase_alive(control, outdir):
    motors, ok = wake_and_check(control)
    rows = []
    for s in MOTOR_SPECS:
        m = motors[s["name"]]
        rows.append({"name": s["name"], "sim_to_motor": s["sim_to_motor"], "ok": ok[s["name"]],
                     "pos_rad": m.Get_Position(), "sim_off": sim_offset(s["name"], m),
                     "vel": m.Get_Velocity(), "tau": m.Get_tau(), "err": m.Get_err(),
                     "age": m.getTimeInterval()})
    csv_path, png_path = out_paths(outdir, "alive")
    save_csv(csv_path, rows)
    print(f"{'name':<10}{'s2m':>4}  ok   pos(rad)    sim_off    tau(Nm)  err   age_ms")
    for r in rows:
        print(f"{r['name']:<10}{r['sim_to_motor']:>+4.0f}  "
              f"{'YES' if r['ok'] else 'NO '}  {r['pos_rad']:>+9.4f}  {r['sim_off']:>+9.4f}  "
              f"{r['tau']:>+7.3f}   {r['err']}  {r['age'] * 1000:>6.1f}")
    bad = [r["name"] for r in rows if not r["ok"]]
    print(f"\n[ALIVE] 全部回传? {'是' if not bad else '否'}   未回传: {bad if bad else '无'}")
    make_alive_plot(png_path, rows)
    print(f"已保存 {png_path}")


def make_alive_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["name"] for r in rows]
    ok = [1 if r["ok"] else 0 for r in rows]
    sim = [r["sim_off"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(names, ok, color=["tab:green" if o else "tab:red" for o in ok])
    ax[0].set_ylim(-0.1, 1.2); ax[0].set_title("Alive (fresh feedback)")
    ax[0].tick_params(axis="x", rotation=45)
    ax[1].bar(names, sim, color="tab:blue")
    ax[1].set_title("sim_offset at hold"); ax[1].tick_params(axis="x", rotation=45)
    ax[1].grid(True, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def phase_map(control, outdir):
    """8 电机各 +0.05(sim)小阶跃,量 Δsim_off/Δcmd —— 应≈+1(映射比 1:1,方向对齐 sim)。"""
    motors, ok = wake_and_check(control)
    step = 0.05
    rows = []
    print(f"{'name':<10}{'cmd':>6}{'actual':>9}{'ratio':>8}  sim_to_motor")
    for s in MOTOR_SPECS:
        name = s["name"]; m = motors[name]
        if not ok[name]:
            print(f"{name:<10}  SKIP (未回传)")
            continue
        p0 = sim_offset(name, m)
        recs = []; t0 = time.perf_counter()
        hold_stage(control, m, name, p0, DEFAULT_KP, 2, 0.4, recs, t0)
        hold_stage(control, m, name, p0 + step, DEFAULT_KP, 2, 0.5, recs, t0)
        sim = np.array([r["sim_off"] for r in recs])
        n = len(recs); seg = n // 2
        a = sim[:seg].mean(); b = sim[-int(0.6 * seg):].mean()
        actual = b - a
        rows.append({"name": name, "cmd": step, "actual": actual, "ratio": actual / step,
                     "sim_to_motor": s["sim_to_motor"]})
        print(f"{name:<10}{step:>+6.2f}{actual:>+9.4f}{actual / step:>+8.3f}  {s['sim_to_motor']:+.0f}")
        hold_stage(control, m, name, p0, DEFAULT_KP, 2, 0.3, recs, t0)  # 回到起点
    csv_path, png_path = out_paths(outdir, "map")
    save_csv(csv_path, rows)
    make_map_plot(png_path, rows)
    print(f"已保存 {png_path}")


def make_map_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["name"] for r in rows]
    actual = [r["actual"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(names, actual, color="tab:purple"); ax.axhline(0.05, color="r", ls="--", label="+0.05 cmd")
    ax.set_title("map: Δsim_offset for Δcmd=+0.05 (all ≈ +0.05 → ratio 1:1)")
    ax.tick_params(axis="x", rotation=45); ax.grid(True, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def phase_kp(control, outdir, motor="FL_thigh"):
    motors, ok = wake_and_check(control)
    m = motors[motor]
    if not ok[motor]:
        print(f"[KP] {motor} 未回传,退出"); return
    kps = [20, 28, 40, 60, 100]
    rows = []
    p0 = sim_offset(motor, m)
    print(f"{'kp':>4}{'err_st':>11}{'tau_st':>11}{'kp_static':>11}")
    for kp in kps:
        target = p0 + 0.05
        recs = []; t0 = time.perf_counter()
        hold_stage(control, m, motor, p0, kp, 2, 0.4, recs, t0)
        hold_stage(control, m, motor, target, kp, 2, 1.2, recs, t0)
        tau = np.array([r["tau"] for r in recs]); sim = np.array([r["sim_off"] for r in recs])
        n = len(recs); st = int(0.85 * n)
        err_st = target - sim[st:].mean()
        tau_st = tau[st:].mean() * by_name[motor]["sim_to_motor"]  # 电机扭矩 -> sim 扭矩(符号随 s2m)
        kp_static = tau_st / err_st if abs(err_st) > 1e-5 else float("nan")
        rows.append({"kp": kp, "err_st": err_st, "tau_st": tau_st, "kp_static": kp_static})
        print(f"{kp:>4}{err_st:>+11.5f}{tau_st:>+11.4f}{kp_static:>+11.2f}")
        hold_stage(control, m, motor, p0, kp, 2, 0.3, recs, t0)
    csv_path, png_path = out_paths(outdir, "kp")
    save_csv(csv_path, rows)
    plt_xy(png_path, [r["kp"] for r in rows], [r["kp_static"] for r in rows],
           "MIT kp", "kp_static Nm/rad", "Static stiffness vs MIT kp", fit_slope=None)
    print(f"已保存 {png_path}")


def phase_lin(control, outdir, motor="FL_thigh"):
    motors, ok = wake_and_check(control)
    m = motors[motor]
    if not ok[motor]:
        print(f"[LIN] {motor} 未回传,退出"); return
    amps = [0.05, 0.15, 0.30, 0.45, 0.30, 0.15, 0.05]
    p0 = sim_offset(motor, m)
    recs = []; t0 = time.perf_counter()
    hold_stage(control, m, motor, p0, DEFAULT_KP, 2, 0.4, recs, t0)
    stages = []
    for a in amps:
        hold_stage(control, m, motor, p0 + a, DEFAULT_KP, 2, 0.5, recs, t0)
        stages.append(p0 + a)
    sim = np.array([r["sim_off"] for r in recs])
    n = len(recs); seg = n // (1 + len(amps))
    base = sim[int(0.4 * seg):seg].mean()  # stage0 = 阶跃前保持段
    rows = []
    for i, a in enumerate(amps):
        lo = (i + 1) * seg + int(0.3 * seg); hi = (i + 1) * seg + int(0.8 * seg)
        act = sim[lo:hi].mean() - base
        rows.append({"cmd": a, "actual": act, "ratio": act / a})
    csv_path, png_path = out_paths(outdir, "lin")
    save_csv(csv_path, rows)
    plt_xy(png_path, [r["cmd"] for r in rows], [r["actual"] for r in rows],
           "cmd (sim rad)", "actual (sim rad)", f"Linearity {motor}", fit_slope=1.0)
    hold_stage(control, m, motor, p0, DEFAULT_KP, 2, 0.3, recs, t0)
    print(f"[LIN] 已保存 {png_path}")


def phase_lat(control, outdir, motor="FL_thigh"):
    motors, ok = wake_and_check(control)
    m = motors[motor]
    if not ok[motor]:
        print(f"[LAT] {motor} 未回传,退出"); return
    p0 = sim_offset(motor, m)
    recs = []; t0 = time.perf_counter()
    hold_stage(control, m, motor, p0, DEFAULT_KP, 2, 0.4, recs, t0)
    hold_stage(control, m, motor, p0 + 0.30, DEFAULT_KP, 2, 2.0, recs, t0)
    t = np.array([r["t"] for r in recs])
    target = np.array([r["target_sim"] for r in recs])
    sim = np.array([r["sim_off"] for r in recs])
    lag = cross_corr_lag(t, target, sim)
    print(f"[LAT] 环路延迟 ≈ {lag * 1000:.0f} ms (互相关峰值)")
    make_lat_plot(*out_paths(outdir, "lat"), t, target, sim, lag)
    print(f"已保存 {outdir}")


def cross_corr_lag(t, target, sim):
    """返回 target 信号超前 sim 的时延(秒),峰值互相关。"""
    tx_d = np.gradient(target); sx_d = np.gradient(sim)
    tx_d = tx_d - tx_d.mean(); sx_d = sx_d - sx_d.mean()
    if np.linalg.norm(tx_d) < 1e-9 or np.linalg.norm(sx_d) < 1e-9:
        return 0.0
    corr = np.correlate(sx_d, tx_d, mode="full")
    lag_samples = np.argmax(corr) - (len(t) - 1)
    return lag_samples / 1000.0


def make_lat_plot(csv_path, png_path, t, target, sim, lag_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, np.degrees(target), "--", label="target (sim deg)")
    ax.plot(t - lag_s, np.degrees(sim), label=f"actual shifted {lag_s * 1000:.0f}ms")
    ax.set_xlabel("t (s)"); ax.set_ylabel("deg"); ax.grid(True); ax.legend()
    ax.set_title(f"Loop latency ≈ {lag_s * 1000:.0f} ms")
    fig.tight_layout(); fig.savefig(png_path, dpi=110); plt.close(fig)


def phase_damp(control, outdir, motor="FL_thigh"):
    motors, ok = wake_and_check(control)
    m = motors[motor]
    if not ok[motor]:
        print(f"[DAMP] {motor} 未回传,退出"); return
    p0 = sim_offset(motor, m)
    freq, amp = 0.5, 0.15  # 低频、较大幅值:惯性扭矩 ~ I*w^2*A 可忽略,err/dq 正交可分离
    recs = []; t0 = time.perf_counter()
    dt = 1.0 / 1000.0
    next_t = time.perf_counter()
    dur = 5.0
    while next_t - t0 < dur:
        t = next_t - t0
        target = p0 + amp * np.sin(2 * np.pi * freq * t)
        q = q_cmd(motor, clamp_sim(motor, target))
        recs.append({"t": t, "target_sim": target, "sim_off": sim_offset(motor, m),
                     "sim_vel": sim_vel(motor, m), "tau": m.Get_tau(), "err": m.Get_err()})
        control.control_mit(m, DEFAULT_KP, 2, q, 0.0, 0.0)
        next_t += dt
        while time.perf_counter() < next_t:
            pass  # 忙等保证 1kHz 采样
    err = np.array([r["target_sim"] - r["sim_off"] for r in recs])
    dq = np.array([r["sim_vel"] for r in recs])
    tau = np.array([r["tau"] for r in recs]) * by_name[motor]["sim_to_motor"]  # -> sim 扭矩
    X = np.column_stack([err, dq, np.sign(dq), np.ones_like(dq)])
    coef, *_ = np.linalg.lstsq(X, tau, rcond=None)
    kp, kd, fc, c0 = coef
    print(f"[DAMP] fit tau_sim = {kp:+.2f}*err + {kd:+.2f}*dq + {fc:+.2f}*sign(dq) + {c0:+.2f}")
    csv_path, png_path = out_paths(outdir, "damp")
    save_csv(csv_path, recs)
    make_damp_plot(png_path, recs, coef)
    print(f"已保存 {png_path}")


def make_damp_plot(path, recs, coef):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dq = np.array([r["sim_vel"] for r in recs]); tau = np.array([r["tau"] for r in recs])
    sim = np.array([r["sim_off"] for r in recs])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(dq, tau, ".", ms=2)
    ax[0].set_xlabel("sim_vel (rad/s)"); ax[0].set_ylabel("tau (Nm)")
    ax[0].set_title("tau vs velocity (friction / damping)"); ax[0].grid(True)
    ax[1].plot(np.arange(len(sim)) * 0.001, np.degrees(sim))
    ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("sim deg"); ax[1].set_title("position under sine")
    ax[1].grid(True)
    ax[1].text(0.02, 0.95, f"kp={coef[0]:+.2f} kd={coef[1]:+.2f} fc={coef[2]:+.2f}",
               transform=ax[1].transAxes, va="top")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def phase_encnoise(control, outdir, motor="FR_thigh"):
    """静止保持,量化编码器噪声底(std)与漂移(peak-peak):也侧面反映回差/松紧."""
    motors, ok = wake_and_check(control)
    m = motors[motor]
    if not ok[motor]:
        print(f"[ENCNOISE] {motor} 未回传,退出"); return
    p0 = sim_offset(motor, m)
    pos, vel = [], []
    t0 = time.perf_counter()
    dt = 5.0 / 1000.0
    next_t = time.perf_counter()
    while next_t - t0 < 3.0:
        control.control_mit(m, DEFAULT_KP, 2, q_cmd(motor, p0), 0.0, 0.0)
        pos.append(m.Get_Position()); vel.append(m.Get_Velocity())
        next_t += dt
        while time.perf_counter() < next_t:
            pass
    pos = np.array(pos); vel = np.array(vel)
    print(f"[ENCNOISE] {motor}: std={pos.std()*1000:.3f} mrad  peakpk={(pos.max()-pos.min())*1000:.3f} mrad  "
          f"vel_rms={np.sqrt((vel**2).mean()):.5f} rad/s  n={len(pos)}")
    csv_path, png_path = out_paths(outdir, "encnoise")
    save_csv(csv_path, [{"t": i * dt, "pos": p, "vel": v} for i, (p, v) in enumerate(zip(pos, vel))])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(np.arange(len(pos)) * dt, np.degrees(pos));
    ax[0].set_ylabel("deg"); ax[0].grid(True); ax[0].set_title(f"Encoder rest noise {motor}")
    ax[1].plot(np.arange(len(vel)) * dt, vel)
    ax[1].set_ylabel("vel rad/s"); ax[1].grid(True)
    fig.tight_layout(); fig.savefig(png_path, dpi=110); plt.close(fig)
    print(f"已保存 {png_path}")


def safe_release(control):
    """把各电机切到轻柔阻尼保持(kp=0),再让进程硬退出,避免 close() 卡死/访问违例."""
    try:
        for s in MOTOR_SPECS:
            m = control.getMotor(s["channel"], s["can_id"])
            control.control_mit(m, 0.0, 2.0, m.Get_Position(), 0.0, 0.0)
    except Exception:
        pass
    time.sleep(0.1)


PHASES = {"alive": phase_alive, "map": phase_map, "kp": phase_kp, "lin": phase_lin,
          "damp": phase_damp, "lat": phase_lat, "encnoise": phase_encnoise}
MOTOR_PHASES = {"kp", "lin", "damp", "lat", "encnoise"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(PHASES))
    ap.add_argument("--motor", default="FL_thigh", help="单关节实验默认电机")
    ap.add_argument("--kp", type=float, default=28.0, help="单关节实验 MIT kp(默认28;映射用高kp如300可压重力误差)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sysid_out"))
    args = ap.parse_args()
    global DEFAULT_KP
    DEFAULT_KP = args.kp

    control = None
    try:
        control = connect()
        print("**********Motor_Control init success**********")
        if args.phase in MOTOR_PHASES:
            PHASES[args.phase](control, args.outdir, motor=args.motor)
        else:
            PHASES[args.phase](control, args.outdir)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if control is not None:
            safe_release(control)  # 全程不调 close(),最后硬退出,reclaim 由 OS 负责
        print("done")
        os._exit(0)


if __name__ == "__main__":
    main()
