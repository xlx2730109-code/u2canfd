# -*- coding: utf-8 -*-
"""从已有台阶数据里反解电机 MIT 增益的"真实值".

原理
----
MIT 模式下力矩由电机自己算:
    tau = KP * (q_des - q) + KD * (dq_des - dq) + tau_ff
台阶脚本发的是 control_mit(m, kp, kd, q_cmd, 0, 0) => dq_des = 0, tau_ff = 0,
所以:  tau = KP * (q_cmd - q) - KD * dq

tau / q / dq 来自同一帧反馈(三者在电机内部同一采样时刻自洽), q_cmd 按时间插值.
于是可以问三个层次的问题:
  1) 稳态(dq~0): tau = KP * Δq          -> 直接量出 KP
  2) 全段 2 参数最小二乘                  -> KP KD 一起估
  3) 单参数力矩比例 gamma: tau = gamma*(kp*Δq + kd*(-dq))
       gamma=1 说明固件如实执行了指令增益; gamma≠1 说明输出力矩被整体缩放

注意
----
  * 反馈流里混有非 MIT 帧(使能/寄存器应答), byte0 高半字节不是 1, 必须剔除;
  * 批量交付下同一主机时间戳有多帧, 但每帧内部自洽, 不影响逐帧回归;
  * 只在 q_des 恒定的阶段做回归, 躲开时间戳 0~109ms 相位模糊.

只读 data/raw 下的 csv, 不改任何原始文件, 也不碰硬件.
用法:
  py -3.13 identify_kp_kd.py --run <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

try:                                    # Windows 控制台是 GBK, 别让生僻符号把脚本打断
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

# DM8006 量程(damiao.py 的限幅表 [P_MAX, V_MAX, T_MAX])
P_MAX, V_MAX, T_MAX = 12.5, 45.0, 20.0
DQ_MOVE = 0.05          # rad/s, 超过算"在动"
Q_PLAUSIBLE = 1.0       # rad, 关节工作范围附近


def decode(hexstr):
    """把 8 字节反馈帧解成 (err, q, dq, tau), 公式与 damiao.py 一致."""
    b = bytes.fromhex(hexstr)
    err = (b[0] >> 4) & 0x0F
    q_uint = (b[1] << 8) | b[2]
    dq_uint = (b[3] << 4) | (b[4] >> 4)
    tau_uint = ((b[4] & 0x0F) << 8) | b[5]
    q = q_uint / 65535.0 * 2 * P_MAX - P_MAX
    dq = dq_uint / 4095.0 * 2 * V_MAX - V_MAX
    tau = tau_uint / 4095.0 * 2 * T_MAX - T_MAX
    return err, q, dq, tau


def load_feedback(run_dir, can_id):
    """读反馈并剔除非 MIT 帧(寄存器应答等)."""
    good, bad = [], 0
    with open(os.path.join(run_dir, "feedback.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["can_id"]) != can_id:
                continue
            b0 = bytes.fromhex(r["data_hex"])[0]
            err, q, dq, tau = decode(r["data_hex"])
            if (b0 >> 4) & 0x0F not in (0, 1) or abs(q) > Q_PLAUSIBLE or abs(dq) > 10.0:
                bad += 1                      # 非 MIT 布局或物理上不可能 -> 丢弃
                continue
            good.append((int(r["t_host_mono_ns"]), err, q, dq, tau))
    good.sort(key=lambda v: v[0])
    return np.array(good, dtype=np.float64), bad


def load_command(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "command.csv"), encoding="utf-8")))
    g = lambda k, f=float: np.array([f(r[k]) for r in rows])
    return (g("t_host_mono_ns", int), g("q_cmd_rad"), g("kp"), g("kd"),
            g("q_real_rad"), [r["phase"] for r in rows])


def fit(A, y):
    """最小二乘 + 标准误 + R2 + 条件数."""
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(1, len(y) - A.shape[1])
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(A.T @ A)), 0, None))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss if ss > 0 else float("nan")
    return coef, se, r2, float(np.linalg.cond(A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--canid", type=int, default=1, help="被测电机 canid, 默认 1(FL_thigh)")
    args = ap.parse_args()

    run = args.run.rstrip("\\/")
    fb, nbad = load_feedback(run, 0x10 + args.canid)
    tc, q_cmd, kp_c, kd_c, snap_q, phase = load_command(run)
    t0 = tc[0]
    trel = (tc - t0) / 1e9

    print(f"[+] run: {os.path.basename(run)}")
    print(f"[+] 反馈帧: 采用 {len(fb)}  剔除 {nbad}(非 MIT 布局/物理不合理)")

    # 解码自检: 与驱动运行时快照比
    idx = np.clip(np.searchsorted(fb[:, 0], tc) - 1, 0, len(fb) - 1)
    err_q = np.abs(fb[idx, 2] - snap_q)
    print(f"[+] 解码自检: |Δq| 中位 {np.median(err_q)*1e3:.4f} mrad, 最大 {err_q.max()*1e3:.4f} mrad")

    t_fb = (fb[:, 0] - t0) / 1e9
    q_des = np.interp(t_fb, trel, q_cmd)
    kp_at = np.interp(t_fb, trel, kp_c)
    kd_at = np.interp(t_fb, trel, kd_c)
    q_m, dq_m, tau_m = fb[:, 2], fb[:, 3], fb[:, 4]
    eq, edq = q_des - q_m, -dq_m

    ph = np.array(phase)
    marks = {}
    for nm in ("ramp", "hold", "step", "back"):
        w = np.where(ph == nm)[0]
        if len(w):
            marks[nm] = (trel[w[0]], trel[w[-1]])
    print("[+] 阶段: " + "  ".join(f"{k} {v[0]:.2f}~{v[1]:.2f}s" for k, v in marks.items()))

    # 只在 q_des 恒定、且离开切换点 0.15s 的窗口里回归
    t_step = marks["step"][0]
    t_end = marks["back"][1]
    win = (t_fb > t_step + 0.15) & (t_fb < t_end)
    mv = win & (np.abs(dq_m) > DQ_MOVE)
    # 准静止必须取"每段真正稳下来的末段", 否则会把 q_des 已跳、力矩还没跟上的瞬态点算进去
    settled = np.zeros_like(win)
    for nm in ("step", "back"):
        a_, b_ = marks[nm][1] - 0.6, marks[nm][1]
        settled |= (t_fb > a_) & (t_fb < b_)
    st = win & settled

    print(f"\n[+] 窗口: 阶跃后 0.15s ~ 结束   总 {win.sum()} 帧, 其中运动 {mv.sum()} / 稳态 {st.sum()}")
    out = {"run": os.path.basename(run), "canid": args.canid,
           "kp_cmd": float(kp_c.max()), "kd_cmd": float(kd_c.max()),
           "n_used": int(len(fb)), "n_dropped": int(nbad)}

    # --- 1) 稳态段: tau = KP * Δq ---
    if st.sum() > 20:
        c, se, r2, _ = fit(eq[st].reshape(-1, 1), tau_m[st])
        print(f"\n[1] 稳态段(每段末 0.6s) tau = KP·Δq")
        print(f"    KP = {c[0]:.2f} ± {se[0]:.2f}   R2={r2:.4f}   "
              f"(下发 {kp_at[st].mean():.1f}  => 实际/下发 = {c[0]/kp_at[st].mean():.3f})")
        for nm in ("step", "back"):
            a_, b_ = marks[nm][1] - 0.6, marks[nm][1]
            m = win & (t_fb > a_) & (t_fb < b_)
            if m.sum() > 5:
                print(f"      [{nm}] Δq={np.median(eq[m])*1e3:+7.2f} mrad  "
                      f"tau={np.median(tau_m[m]):+.4f} N·m  =>  tau/Δq = {np.median(tau_m[m]/eq[m]):.2f}")
        out["static_kp"] = {"kp": float(c[0]), "se": float(se[0]), "r2": float(r2),
                            "kp_cmd": float(kp_at[st].mean())}

    # --- 2) 运动段: 两参数 ---
    if mv.sum() > 20:
        c, se, r2, cond = fit(np.column_stack([eq[mv], edq[mv]]), tau_m[mv])
        print(f"\n[2] 运动段(|dq|>{DQ_MOVE}) tau = KP·Δq + KD·Δdq")
        print(f"    KP = {c[0]:.2f} ± {se[0]:.2f}   KD = {c[1]:.2f} ± {se[1]:.2f}   "
              f"R2={r2:.4f}  cond={cond:.1f}")
        print(f"    相对下发: KP {c[0]/kp_at[mv].mean():.3f}×    KD {c[1]/kd_at[mv].mean():.3f}×")

        # --- 3) 单参数力矩比例 gamma ---
        pred = kp_at[mv] * eq[mv] + kd_at[mv] * edq[mv]
        g, gse, gr2, _ = fit(pred.reshape(-1, 1), tau_m[mv])
        print(f"\n[3] 力矩比例 (把两个增益绑在一起) tau = γ·(kp·Δq + kd·Δdq)")
        print(f"    γ = {g[0]:.4f} ± {gse[0]:.4f}   R2={gr2:.4f}   "
              f"即电机实际输出只有指令的 {g[0]*100:.1f}%")
        out["moving_2p"] = {"kp": float(c[0]), "kp_se": float(se[0]),
                            "kd": float(c[1]), "kd_se": float(se[1]),
                            "r2": float(r2), "cond": cond,
                            "kp_cmd": float(kp_at[mv].mean()), "kd_cmd": float(kd_at[mv].mean())}
        out["gamma"] = {"gamma": float(g[0]), "se": float(gse[0]), "r2": float(gr2)}

    proc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "processed", os.path.basename(run))
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(proc, "kp_kd_ident.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 汇总: {os.path.join(proc, 'kp_kd_ident.json')}")


if __name__ == "__main__":
    main()
