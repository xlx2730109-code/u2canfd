# -*- coding: utf-8 -*-
"""IMU 调平前后对比图 —— 一张图讲完"调平前 1.13° → 调平后 0.72°"。

原图 fig2a(调平前) / fig2b(调平后) 保持不动, 本图是给答辩用的合并版。

(a) 气泡水平仪对比: 调平前/后两个均值星标画在同一 roll-pitch 平面, 灰箭头表示调整方向,
    灰色虚线圈 0.5°/1.0° 作参考 —— 调平前的星在 1.0° 圈外, 调平后进到圈内
(b) 柱状对比: roll / pitch / 合倾角 三个量的调平前 vs 调平后, 柱顶标数值

配色: 原始(调平前)=蓝, 后期(调平后)=红.

用法:
  py -3.13 plot_imu_compare.py --before <run_dir> --after <run_dir> --out fig2c_imu_level_compare
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

from plot_imu_static import load_run, style_ax  # noqa: E402

C_BEFORE = "#1A4E8A"     # 调平前(原始) —— 蓝
C_AFTER = "#B03A2E"      # 调平后(后期) —— 红
C_ARROW = "#555555"      # 调整方向箭头(中性灰, 不跟蓝红抢)


def metrics(run_dir):
    """读一次采集, 返回 roll/pitch/合倾角序列和均值."""
    reg = load_run(run_dir)
    a = reg[1]
    ax_, ay_, az_ = a[:, 1], a[:, 2], a[:, 3]
    g = np.sqrt(ax_ ** 2 + ay_ ** 2 + az_ ** 2)
    roll = np.degrees(np.arctan2(ay_, az_))
    pitch = np.degrees(np.arctan2(-ax_, np.sqrt(ay_ ** 2 + az_ ** 2)))
    tilt = np.degrees(np.arccos(np.clip(az_ / g, -1, 1)))
    return {"roll": roll, "pitch": pitch, "tilt": tilt,
            "r_m": float(roll.mean()), "p_m": float(pitch.mean()),
            "t_m": float(tilt.mean()), "t_s": float(tilt.std()),
            "g_m": float(g.mean()), "n": int(len(a)),
            "run": os.path.basename(run_dir.rstrip("\\/"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="调平前的 run 目录")
    ap.add_argument("--after", required=True, help="调平后的 run 目录")
    ap.add_argument("--out", required=True, help="输出图名(不含扩展名)")
    args = ap.parse_args()

    mb, ma = metrics(args.before), metrics(args.after)
    drop = 100.0 * (1.0 - ma["t_m"] / mb["t_m"])
    print(f"[+] 调平前 {mb['run']}: roll {mb['r_m']:+.3f}  pitch {mb['p_m']:+.3f}  "
          f"合倾角 {mb['t_m']:.3f} ± {mb['t_s']:.3f}  (n={mb['n']})")
    print(f"[+] 调平后 {ma['run']}: roll {ma['r_m']:+.3f}  pitch {ma['p_m']:+.3f}  "
          f"合倾角 {ma['t_m']:.3f} ± {ma['t_s']:.3f}  (n={ma['n']})")
    print(f"[+] 合倾角下降 {drop:.1f}%   |g| {mb['g_m']:.4f} → {ma['g_m']:.4f} m/s^2")

    fig = plt.figure(figsize=(9.8, 4.3), dpi=2000)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.0], wspace=0.24,
                          left=0.062, right=0.985, top=0.885, bottom=0.155)

    # ---------------- (a) 气泡水平仪: 两个星标同框 ----------------
    axA = fig.add_subplot(gs[0, 0])
    lim = max(1.35, 1.42 * max(abs(mb["r_m"]), abs(mb["p_m"]),
                               abs(ma["r_m"]), abs(ma["p_m"])))
    for rad, lab in ((0.5, "0.5°"), (1.0, "1.0°")):
        axA.add_patch(plt.Circle((0, 0), rad, fill=False, ec="#C4C4C4",
                                 lw=0.8, ls="--", zorder=1))
        axA.text(rad * 0.707, -rad * 0.707, lab, fontsize=7.5, color="#A8A8A8",
                 ha="left", va="top", zorder=1)
    axA.axhline(0, color="#E4E4E4", lw=0.7, zorder=0)
    axA.axvline(0, color="#E4E4E4", lw=0.7, zorder=0)
    axA.plot([0], [0], marker="+", ms=11, mew=1.4, color="#333333", zorder=3)

    for m, col in ((mb, C_BEFORE), (ma, C_AFTER)):
        st = max(1, m["n"] // 700)      # 抽稀, 否则点太密
        axA.scatter(m["roll"][::st], m["pitch"][::st], s=4, color=col,
                    alpha=0.22, linewidths=0, zorder=2)

    # 调整方向箭头(起点终点各留出星标的空档)
    axA.annotate("", xy=(ma["r_m"], ma["p_m"]), xytext=(mb["r_m"], mb["p_m"]),
                 arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.3,
                                 shrinkA=10, shrinkB=12,
                                 connectionstyle="arc3,rad=0.16"), zorder=4)
    axA.plot([mb["r_m"]], [mb["p_m"]], marker="*", ms=16, color=C_BEFORE,
             zorder=5, markeredgecolor="white", markeredgewidth=0.7)
    axA.plot([ma["r_m"]], [ma["p_m"]], marker="*", ms=19, color=C_AFTER,
             zorder=6, markeredgecolor="white", markeredgewidth=0.7)

    for m, col, dy, va in ((mb, C_BEFORE, -0.26, "top"), (ma, C_AFTER, 0.26, "bottom")):
        axA.annotate(f"调平前\n{m['t_m']:.2f}°" if col == C_BEFORE else f"调平后\n{m['t_m']:.2f}°",
                     xy=(m["r_m"], m["p_m"]),
                     xytext=(m["r_m"], m["p_m"] + dy), fontsize=9,
                     color=col, ha="center", va=va, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#DDDDDD", lw=0.6),
                     arrowprops=dict(arrowstyle="-", lw=0.6, color=col), zorder=7)

    axA.set_xlim(-lim, lim)
    axA.set_ylim(-lim, lim)
    axA.set_aspect("equal")
    axA.set_xlabel("roll (deg)", fontsize=9.5)
    axA.set_ylabel("pitch (deg)", fontsize=9.5)
    style_ax(axA)
    axA.set_title("气泡水平仪：越靠中心越平", fontsize=9.5, pad=7)
    axA.text(-0.055, 1.055, "(a)", fontsize=11, fontweight="bold",
             transform=axA.transAxes)

    # ---------------- (b) 三个量的柱状对比 ----------------
    axB = fig.add_subplot(gs[0, 1])
    cats = ["roll", "pitch", "合倾角"]
    bv = [mb["r_m"], mb["p_m"], mb["t_m"]]
    av = [ma["r_m"], ma["p_m"], ma["t_m"]]
    x = np.arange(len(cats))
    w = 0.34
    b1 = axB.bar(x - w / 2, bv, w, color=C_BEFORE, label="调平前", zorder=3)
    b2 = axB.bar(x + w / 2, av, w, color=C_AFTER, label="调平后", zorder=3)
    axB.axhline(0, color="#666666", lw=0.8, zorder=2)
    for rects in (b1, b2):
        for r in rects:
            h = r.get_height()
            axB.text(r.get_x() + r.get_width() / 2, h + (0.055 if h >= 0 else -0.055),
                     f"{h:.2f}°", ha="center", va="bottom" if h >= 0 else "top",
                     fontsize=8.6, zorder=4)
    axB.set_xticks(x)
    axB.set_xticklabels(cats, fontsize=9.5)
    axB.set_ylabel("倾角 (deg)", fontsize=9.5)
    axB.set_ylim(-1.38, 1.52)
    axB.legend(frameon=False, fontsize=9, loc="upper left", ncol=1)
    axB.annotate(f"合倾角 {mb['t_m']:.2f}° → {ma['t_m']:.2f}°（↓{drop:.0f}%）",
                 xy=(0.97, 0.055), xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=9.2, color=C_AFTER, fontweight="bold")
    style_ax(axB)
    axB.set_title("各轴倾角：调平前 vs 调平后", fontsize=9.5, pad=7)
    axB.text(-0.075, 1.055, "(b)", fontsize=11, fontweight="bold",
             transform=axB.transAxes)

    # ---------------- 输出 ----------------
    fig_dir = os.path.join(WS, "08_sensor_identification", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, args.out + ".png")
    fig.savefig(out, dpi=2000)
    plt.close(fig)

    proc = os.path.join(WS, "08_sensor_identification", "data", "processed",
                        "level_compare_before_vs_after")
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(proc, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "before": {k: mb[k] for k in ("run", "n", "r_m", "p_m", "t_m", "t_s", "g_m")},
            "after": {k: ma[k] for k in ("run", "n", "r_m", "p_m", "t_m", "t_s", "g_m")},
            "tilt_drop_percent": round(drop, 2),
        }, f, ensure_ascii=False, indent=2)
    print(f"[+] 图: {out}\n[+] 汇总: {os.path.join(proc, 'compare_summary.json')}")


if __name__ == "__main__":
    main()
