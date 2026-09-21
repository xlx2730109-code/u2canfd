# -*- coding: utf-8 -*-
"""07 工作包结果图: KP 真实值 / KD 真实值 / 库仑摩擦+粘性阻尼.

(a) KP: 力矩前馈扫描的 15 个稳态点, 扣掉前馈后 tau 对 Δq 是一条直线, 斜率就是真 KP
        —— 直接回答"固件到底拿多大的 KP 在算"
(b) KD: 同一速度跑两遍(一遍 dq_des=0, 一遍 dq_des=v), Δq 的差值正比于 dq_des,
        斜率 = KD/KP. 两批独立实验(不同速度组)各给一条线, 重合就说明不是巧合
(c) 摩擦: 同一位置正反两向比力矩, 重力项严格消掉, 差值/2 就是摩擦力矩

只读 data/processed 的 json, 不碰硬件.
用法:
  py -3.13 plot_kpkd_fric.py --kp-run <tauff_run> --kd-run <ramp_run> [--kd-run <ramp_run2>]
                             --out fig4_kpkd_fric
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

C_FIT = "#1A4E8A"
C_R1 = "#D9822B"
C_R2 = "#2E8B57"
C_CMD = "#B03A2E"
GRP_COLORS = ("#1A4E8A", "#D9822B", "#2E8B57")


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=9)


def speed_label(segs):
    """从段名里取出这组实验用了哪几个速度, 用来当图例(两批实验就靠这个区分)."""
    vs = sorted({float(k.split("|")[0].split("_")[0][1:]) for k in segs})
    return " / ".join(f"{v:g}" for v in vs)


def load_json(run, name):
    p = os.path.join(WS, "07_actuator_identification", "data", "processed",
                     os.path.basename(run.rstrip("\\/")), name)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-run", required=True, help="tau_ff_sweep 的 run 目录")
    ap.add_argument("--kd-run", action="append", required=True, help="ramp 的 run 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    outdir = os.path.join(WS, "07_actuator_identification", "results", "figures")
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.145, wspace=0.27)

    # ---------------- (a) KP ----------------
    axA = axes[0]
    K = load_json(args.kp_run, "kp_offset_ident.json")
    fit_all = K["fits"]["全部合并"]
    kp_fit, c_ff, b_fit = fit_all["kp"], fit_all["c_ff"], fit_all["b"]
    pts = K["points"]
    grps = list(dict.fromkeys(p["phase"].split("_")[0] for p in pts))
    for gi, g in enumerate(grps):
        sub = [p for p in pts if p["phase"].startswith(g)]
        axA.plot([p["dq_eq"] * 1e3 for p in sub],
                 [p["tau"] - c_ff * p["ff"] for p in sub], "o", ms=5.2,
                 color=GRP_COLORS[gi % 3], label=f"位置偏移 {g}", zorder=3,
                 markeredgecolor="white", markeredgewidth=0.5)
    xs = np.array([p["dq_eq"] * 1e3 for p in pts])
    lo, hi = xs.min() - 1.5, xs.max() + 1.5
    xx = np.array([lo, hi])
    axA.plot(xx, kp_fit * xx * 1e-3 + b_fit, "-", color=C_FIT, lw=1.6,
             label=f"拟合  KP = {kp_fit:.2f}", zorder=4)
    kp_cmd = 28.0
    axA.plot(xx, kp_cmd * xx * 1e-3, "--", color=C_CMD, lw=1.3,
             label=f"下发值  KP = {kp_cmd:.1f}", zorder=4)
    axA.set_xlabel("位置误差  Δq = q_des − q  (mrad)", fontsize=9.5)
    axA.set_ylabel("反馈力矩 − 前馈  (N·m)", fontsize=9.5)
    axA.set_title("(a) KP 真实值：前馈扫描", fontsize=10.5, pad=7)
    axA.text(0.04, 0.965,
             f"KP = {kp_fit:.2f} ± {fit_all['kp_se']:.2f}\n"
             f"实际/下发 = {kp_fit/kp_cmd:.3f}\n"
             f"R² = {fit_all['r2']:.5f}   n = {fit_all['n']}",
             transform=axA.transAxes, va="top", fontsize=8.6, color="#222222",
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#DDDDDD", lw=0.6))
    axA.legend(frameon=False, fontsize=8.0, loc="lower right")
    style_ax(axA)

    # ---------------- (b) KD ----------------
    # 坐标一律用 mrad/s 与 mrad, 斜率因此是无量纲的 KD/KP —— 早先 x 用 rad/s、画线又用
    # mrad/s, 线比数据陡了 1000 倍, 图上看就是"数据全趴在 0 上, 两条线飞到天上去".
    axB = axes[1]
    kp_used = 28.410
    runs = []
    for r in args.kd_run:
        J = load_json(r, "kd_friction_ident.json")
        segs = J["segments"]
        dx, dy = [], []
        for key, s in segs.items():
            vt, d = key.split("|")
            if not vt.endswith("ff0"):
                continue
            other = segs.get(f"{vt[:-3]}ffon|{d}")
            if other is None:
                continue
            e1 = s["q_cmd"] - s["q"]
            e2 = other["q_cmd"] - other["q"]
            if abs(e1 - e2) / 0.381e-3 < 10.0:   # <10 个编码器 LSB 的点不可信
                continue
            dx.append(other["dqd_cmd"] * 1e3)    # mrad/s
            dy.append((e1 - e2) * 1e3)           # mrad
        xa, ya = np.array(dx), np.array(dy)
        sl = np.polyfit(xa, ya, 1)               # 斜率无量纲 = KD/KP
        runs.append({"nm": speed_label(segs), "x": xa, "y": ya,
                     "kd": kp_used * sl[0], "sl": sl})
    allx = np.concatenate([r["x"] for r in runs])
    ally = np.concatenate([r["y"] for r in runs])
    sl_all = np.polyfit(allx, ally, 1)
    kd_est = kp_used * sl_all[0]
    resid = ally - np.polyval(sl_all, allx)
    r2 = 1 - float(resid @ resid) / float(((ally - ally.mean()) ** 2).sum())

    padx = 0.10 * (allx.max() - allx.min())
    pady = 0.22 * (ally.max() - ally.min())
    xl = np.array([allx.min() - padx, allx.max() + padx])
    for ri, r in enumerate(runs):
        col = (C_R1, C_R2)[ri % 2]
        axB.plot(r["x"], r["y"], "o" if ri == 0 else "s", ms=6.0, color=col,
                 zorder=3, markeredgecolor="white", markeredgewidth=0.6,
                 label=f"实验 {'AB'[ri]}：{r['nm']} rad/s  →  KD = {r['kd']:.3f}")
        axB.plot(xl, np.polyval(r["sl"], xl), "--", color=col, lw=1.1,
                 alpha=0.85, zorder=4)
    axB.plot(xl, np.polyval(sl_all, xl), "-", color=C_FIT, lw=2.0, zorder=5,
             label=f"合并拟合  →  KD = {kd_est:.3f}")
    axB.plot(xl, (2.0 / kp_used) * xl, ":", color=C_CMD, lw=1.4, zorder=4,
             label="下发值  KD = 2.0")
    axB.axhline(0, color="#CCCCCC", lw=0.8, zorder=1)
    axB.axvline(0, color="#CCCCCC", lw=0.8, zorder=1)
    axB.set_xlim(xl[0], xl[1])
    axB.set_ylim(ally.min() - pady, ally.max() + pady)
    axB.set_xlabel("指令速度  dq_des  (mrad/s)", fontsize=9.5)
    axB.set_ylabel("Δq(dq_des=0) − Δq(dq_des=v)   (mrad)", fontsize=9.5)
    axB.set_title("(b) KD 真实值：同速两遍差分", fontsize=10.5, pad=7)
    axB.text(0.04, 0.96,
             f"两批独立实验各自拟合，互相重合到 "
             f"{abs(runs[0]['kd']-runs[1]['kd'])/kd_est*100:.1f}%，\n"
             f"都落在下发值 2.0 上方 1~3%\n"
             f"合并 R² = {r2:.4f}   n = {len(allx)}",
             transform=axB.transAxes, va="top", fontsize=8.6, color="#222222",
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#DDDDDD", lw=0.6))
    axB.legend(frameon=False, fontsize=8.0, loc="lower right")
    style_ax(axB)

    # ---------------- (c) 摩擦 ----------------
    axC = axes[2]
    Jf = None
    axC_pts, axC_fr = [], []
    for ri, r in enumerate(args.kd_run):
        J = load_json(r, "kd_friction_ident.json")
        fp = J.get("friction_pairwise")
        if not fp:
            continue
        xx = np.array(fp["v_cmd"]) * 1e3
        yy = np.array(fp["half_dtau"])
        axC_pts.append(yy)
        mk = "o" if ri == 0 else "s"
        col = (C_R1, C_R2)[ri % 2]
        axC.plot(xx, yy, mk, ms=6.5, color=col, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.6,
                 label=f"实验 {'AB'[ri]}：{speed_label(J['segments'])} rad/s")
        if J.get("friction_fit"):
            Jf = J["friction_fit"]
            axC_fr = Jf["tau_fric"]
    if Jf is not None:
        # y 轴贴着"所有画出来的点"放 —— 0~0.30 会把三个点压成一条线, 8.7% 的起伏看不出来;
        # 但也别裁掉实验 A 那个 0.233 的离群点, 那正是它作为对照要展示的东西
        ylo = min(min(v) for v in axC_pts + [axC_fr]) - 0.010
        yhi = max(max(v) for v in axC_pts + [axC_fr]) + 0.012
        vmax = max(Jf["v_cmd"])
        axC.plot([0.0, vmax * 1e3 * 1.10],
                 [Jf["tau_c"], Jf["b_v"] * vmax + Jf["tau_c"]],
                 "-", color=C_FIT, lw=1.6, label="实验 B 线性拟合", zorder=4)
        axC.axhline(Jf["tau_c"], color=C_FIT, lw=0.9, ls=":", zorder=2)
        axC.text(vmax * 1e3 * 1.20, Jf["tau_c"] + 0.001,
                 f"τ_c = {Jf['tau_c']:.3f}", color=C_FIT,
                 fontsize=8.4, va="bottom", ha="right")
    else:
        ylo, yhi = 0.0, 0.30
    axC.set_xlim(0, None)
    axC.set_ylim(ylo, yhi)
    axC.set_xlabel("速度  |v|  (mrad/s)", fontsize=9.5)
    axC.set_ylabel("摩擦力矩  [τ_out(q) − τ_in(q)] / 2   (N·m)", fontsize=9.5)
    axC.set_title("(c) 摩擦：同位置正反两向差分", fontsize=10.5, pad=7)
    if Jf is not None:
        sp = (max(Jf["tau_fric"]) - min(Jf["tau_fric"])) / np.mean(Jf["tau_fric"])
        axC.text(0.04, 0.96,
                 f"实验 B 三点：{min(Jf['tau_fric']):.3f} ~ {max(Jf['tau_fric']):.3f} N·m\n"
                 f"（起伏 {sp*100:.0f}%，每点 400~700 个匹配样本）\n"
                 f"→ 0.03~0.13 rad/s 内基本是常数：库仑摩擦",
                 transform=axC.transAxes, va="top", fontsize=8.4, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#DDDDDD", lw=0.6))
        axC.text(0.04, 0.05, f"b_v = {Jf['b_v']:+.3f}：自由度只有 1，别当数用",
                 transform=axC.transAxes, va="bottom", fontsize=7.6, color="#777777")
    axC.legend(frameon=False, fontsize=8.0, loc="lower right")
    style_ax(axC)

    name = args.out + (f"_{args.tag}" if args.tag else "") + ".png"
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=2000)
    print(f"[+] 图: {path}")
    print(f"    (a) KP = {kp_fit:.3f} ± {fit_all['kp_se']:.3f}  (下发 28.0)")
    print(f"    (b) KD = {kd_est:.3f}  R2 = {r2:.5f}  点 {len(allx)} 个")
    if Jf is not None:
        print(f"    (c) tau_c = {Jf['tau_c']:.4f}  b_v = {Jf['b_v']:+.4f}  "
              f"R2 = {Jf['r2']:.4f}")


if __name__ == "__main__":
    main()
