# -*- coding: utf-8 -*-
"""KP/KD 辨识的诊断图 —— 说明"单次台阶为什么不够".

(a) 全程时间序列: q_cmd 阶梯 / q 实际 / tau, 标出两个真正稳住的平台
(b) tau–Δq 平面: 运动段散点 vs 稳态散点, 对照 kp=28 的理论线
    —— 两个稳态点的比值分别是 25.0 和 33.0, 但过这两点的直线斜率恰好 ~28:
       说明中间夹着一个固定偏置, 而两点只有两个方程, 解不开 KP / KD / 偏置

只读数据, 不碰硬件.
用法:
  py -3.13 plot_kp_kd_diag.py --run <run_dir> --out fig3_kp_kd_diag
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from identify_kp_kd import DQ_MOVE, load_command, load_feedback  # noqa: E402

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

C_CMD, C_REAL, C_TAU = "#1A4E8A", "#D9822B", "#B03A2E"
C_MOVE, C_SET = "#B8B8B8", "#1A4E8A"


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--canid", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = args.run.rstrip("\\/")
    fb, _ = load_feedback(run, 0x10 + args.canid)
    tc, q_cmd, kp_c, kd_c, _, phase = load_command(run)
    t0 = tc[0]
    trel = (tc - t0) / 1e9
    t_fb = (fb[:, 0] - t0) / 1e9
    q_des = np.interp(t_fb, trel, q_cmd)
    kp_at = np.interp(t_fb, trel, kp_c)
    q_m, dq_m, tau_m = fb[:, 2], fb[:, 3], fb[:, 4]
    eq = q_des - q_m

    ph = np.array(phase)
    marks = {}
    for nm in ("ramp", "hold", "step", "back"):
        w = np.where(ph == nm)[0]
        if len(w):
            marks[nm] = (trel[w[0]], trel[w[-1]])
    t_step = marks["step"][0]
    win = (t_fb > t_step + 0.15) & (t_fb < marks["back"][1])
    mv = win & (np.abs(dq_m) > DQ_MOVE)
    setm = np.zeros_like(win)
    for nm in ("step", "back"):
        setm |= (t_fb > marks[nm][1] - 0.6) & (t_fb < marks[nm][1])
    st = setm

    KP_CMD = float(kp_c.max())

    fig = plt.figure(figsize=(9.8, 3.9), dpi=2000)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.26,
                          left=0.062, right=0.975, top=0.88, bottom=0.155)

    # ---------------- (a) 时间序列 ----------------
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(t_fb, q_des, color=C_CMD, lw=1.0, ls="--", label="q_cmd（指令角）")
    axA.plot(t_fb, q_m, color=C_REAL, lw=0.9, label="q（实际角）")
    axA.set_xlabel("时间 (s)", fontsize=9.5)
    axA.set_ylabel("关节角 (rad)", fontsize=9.5)
    axA.set_xlim(0, t_fb.max())
    axA.legend(frameon=False, fontsize=8.4, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    for nm, txt in (("step", "稳态点 A"), ("back", "稳态点 B")):
        a_, b_ = marks[nm][1] - 0.6, marks[nm][1]
        axA.axvspan(a_, b_, color="#1A4E8A", alpha=0.09, zorder=0)
        axA.text((a_ + b_) / 2, axA.get_ylim()[0] * 0.98, txt, fontsize=8,
                 color="#1A4E8A", ha="center", va="bottom")
    style_ax(axA)

    axT = axA.twinx()
    axT.plot(t_fb, tau_m, color=C_TAU, lw=0.8, alpha=0.85)
    axT.set_ylabel("力矩 (N·m)", fontsize=9.5, color=C_TAU)
    axT.tick_params(axis="y", colors=C_TAU, width=0.8, labelsize=9)
    axT.spines["top"].set_visible(False)
    axT.spines["right"].set_linewidth(0.8)
    axA.text(-0.052, 1.05, "(a)", fontsize=11, fontweight="bold", transform=axA.transAxes)
    axA.set_title("台阶全程：红=力矩，灰带=真正稳住的平台", fontsize=9.3, pad=6)

    # ---------------- (b) tau–Δq 平面: 聚焦稳态区 ----------------
    axB = fig.add_subplot(gs[0, 1])
    axB.axhline(0, color="#E0E0E0", lw=0.7, zorder=0)
    axB.axvline(0, color="#E0E0E0", lw=0.7, zorder=0)
    axB.scatter(eq[st] * 1e3, tau_m[st], s=6, color=C_SET, alpha=0.42,
                linewidths=0, label="稳态段实测（可信）", zorder=3)

    xs = np.array([-14.0, 14.0])
    axB.plot(xs, KP_CMD * xs * 1e-3, color="#444444", lw=1.0, ls=":",
             label=f"指令 kp={KP_CMD:.0f} 的理论线（过原点）", zorder=4)
    pts = []
    for nm, lab in (("step", "A"), ("back", "B")):
        a_, b_ = marks[nm][1] - 0.6, marks[nm][1]
        m = (t_fb > a_) & (t_fb < b_)
        pts.append((float(np.median(eq[m])) * 1e3, float(np.median(tau_m[m])), lab))
    (x1, y1, l1), (x2, y2, l2) = pts
    k_fit = (y1 - y2) / ((x1 - x2) * 1e-3)
    b_fit = y1 - k_fit * x1 * 1e-3
    axB.plot(xs, k_fit * xs * 1e-3 + b_fit, color=C_TAU, lw=1.1, ls="-",
             label=f"过两稳态点的直线（斜率 {k_fit:.1f}）", zorder=4)

    for (x_, y_, lab), (dx, dy, ha, va) in zip(
            pts, ((-1.8, 0.055, "right", "center"), (-1.6, 0.0, "right", "center"))):
        axB.plot([x_], [y_], marker="*", ms=15, color=C_SET, zorder=5,
                 markeredgecolor="white", markeredgewidth=0.7)
        axB.annotate(f"{lab}  Δq={x_:+.1f} mrad\n     τ/Δq={y_/(x_*1e-3):.1f}",
                     xy=(x_, y_), xytext=(x_ + dx, y_ + dy),
                     fontsize=8.3, color=C_SET, ha=ha, va=va,
                     bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#DDDDDD", lw=0.6),
                     arrowprops=dict(arrowstyle="-", lw=0.6, color=C_SET), zorder=6)

    axB.set_xlim(-15.5, 15.5)
    axB.set_ylim(-0.40, 0.46)
    axB.set_xlabel("位置误差 Δq (mrad)", fontsize=9.5)
    axB.set_ylabel("电机输出力矩 τ (N·m)", fontsize=9.5)
    axB.legend(frameon=False, fontsize=8.0, loc="lower right")
    axB.text(0.03, 0.97, f"截距 = {b_fit:+.3f} N·m → 存在固定偏置",
             transform=axB.transAxes, va="top", fontsize=8.4, color="#333333")
    style_ax(axB)
    axB.text(-0.075, 1.05, "(b)", fontsize=11, fontweight="bold", transform=axB.transAxes)
    axB.set_title("两个稳态点比值不同 → 中间夹着固定偏置", fontsize=9.3, pad=6)

    fig_dir = os.path.join(WS, "07_actuator_identification", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, args.out + ".png")
    fig.savefig(out, dpi=2000)
    plt.close(fig)
    print(f"[+] 图: {out}")


if __name__ == "__main__":
    main()
