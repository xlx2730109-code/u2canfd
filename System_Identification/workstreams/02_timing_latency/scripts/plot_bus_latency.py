# -*- coding: utf-8 -*-
"""图① 总线延迟与反馈节奏 —— 学术出版风格。

输入: data/raw/<日期>/<run_id>/ (frames.csv + tx.csv + windows.json)
输出:
  data/processed/<run_id>/  {bus_latency_summary.csv, quality_report.json, source_runs.txt}
  results/figures/fig1_bus_latency.png   (300 dpi)
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, PARENT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 学术风: Times New Roman (西文/数字) + 宋体 (中文), matplotlib>=3.6 逐字形回退
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

C_TX = "#9E9E9E"      # 命令: 灰
C_RX = "#1A4E8A"      # 反馈: 深蓝
C_MED = "#B03A2E"     # 中位线: 暗红

NAMES = {0x11: "FL thigh", 0x12: "FL calf", 0x13: "FR thigh", 0x14: "FR calf",
         0x15: "RL thigh", 0x16: "RL calf", 0x17: "RR thigh", 0x18: "RR calf"}


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()

    if args.run:
        run_dir = args.run
    else:
        cands = sorted(glob.glob(os.path.join(WS, "02_timing_latency", "data", "raw", "*", "*feedback_age*")))
        if not cands:
            print("[!] 没找到 1kHz 探测数据")
            return
        run_dir = cands[-1]
    print(f"[+] 数据: {run_dir}")

    w = json.load(open(os.path.join(run_dir, "windows.json")))
    t0w, t1w = w["t_start_ns"], w["t_end_ns"]

    by = collections.defaultdict(list)
    with open(os.path.join(run_dir, "frames.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t, cid = int(row["t_host_mono_ns"]), int(row["can_id"])
            if t0w <= t <= t1w and cid in NAMES:
                by[cid].append(t)
    tx = []
    with open(os.path.join(run_dir, "tx.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = int(row["t_host_mono_ns"])
            if t0w <= t <= t1w:
                tx.append(t)

    # 统计 + processed
    stats_rows = []
    for cid in sorted(by):
        ts = np.array(sorted(by[cid]), dtype=np.int64)
        d = np.diff(ts) / 1e6
        inter = d[d >= 1.0]
        stats_rows.append({
            "mst_id": f"0x{cid:02X}", "joint": NAMES[cid], "n_frames": int(len(ts)),
            "freq_hz": round(1000.0 / d.mean(), 1),
            "batch_gap_p50_ms": round(float(np.median(inter)), 2),
            "batch_gap_p95_ms": round(float(np.percentile(inter, 95)), 2),
            "batch_gap_max_ms": round(float(inter.max()), 2),
        })
    proc_dir = os.path.join(WS, "02_timing_latency", "data", "processed", os.path.basename(run_dir))
    os.makedirs(proc_dir, exist_ok=True)
    with open(os.path.join(proc_dir, "bus_latency_summary.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        wr.writeheader()
        wr.writerows(stats_rows)
    with open(os.path.join(proc_dir, "source_runs.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.basename(run_dir) + "\n")
    with open(os.path.join(proc_dir, "quality_report.json"), "w", encoding="utf-8") as f:
        json.dump({"frames_total": int(sum(r["n_frames"] for r in stats_rows)),
                   "all_motors_streaming": all(r["n_frames"] > 300 for r in stats_rows),
                   "median_batch_gap_ms": float(np.median([r["batch_gap_p50_ms"] for r in stats_rows])),
                   "note": "批间隔=同电机相邻两批到达的间隔, 近似反馈最大时龄"}, f, ensure_ascii=False, indent=2)

    # ---- 图 ----------------------------------------------------------------
    fig = plt.figure(figsize=(9.6, 3.9), dpi=2000)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.26,
                          left=0.065, right=0.988, top=0.90, bottom=0.16)

    # (a) 事件栅格: 上=命令(1kHz 连续), 下=反馈(成批)
    axA = fig.add_subplot(gs[0, 0])
    t_ref = tx[0]
    WIN = 0.45e9
    tx_s = sorted((t - t_ref) / 1e9 for t in tx if (t - t_ref) < WIN)
    rx = sorted(by[0x11])
    rx_s = [(t - t_ref) / 1e9 for t in rx if (t - t_ref) < WIN]
    axA.vlines(tx_s, 1.15, 1.62, color=C_TX, lw=0.5, alpha=0.85)
    axA.vlines(rx_s, 0.18, 0.66, color=C_RX, lw=2.6)
    # 批间隔双向箭头
    rg = np.diff(rx_s) * 1000
    big = rg[rg > 5]
    for i in range(min(2, len(big))):
        idx = np.where(rg > 5)[0][i]
        x0, x1 = rx_s[idx], rx_s[idx + 1]
        y = 0.86
        axA.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="<->", lw=0.9, color="k"))
        axA.text((x0 + x1) / 2, y + 0.06, f"~{big[i]:.0f} ms",
                 ha="center", fontsize=8.5)
    axA.text(0.012, 1.70, "命令流（1 kHz 连续发送）", fontsize=9, color="#555")
    axA.text(0.012, 0.72, "反馈到达", fontsize=9, color=C_RX)
    axA.set_ylim(0, 1.95)
    axA.set_xlim(-0.005, 0.46)
    axA.set_yticks([])
    axA.set_xlabel("时间 (s)", fontsize=9.5)
    style_ax(axA)
    axA.text(-0.02, 1.06, "(a)", fontsize=11, fontweight="bold", transform=axA.transAxes)

    # (b) 箱线图: 8 关节批间隔分布
    axB = fig.add_subplot(gs[0, 1])
    data, pts, xs = [], [], []
    for i, cid in enumerate(sorted(by)):
        ts = np.array(sorted(by[cid]), dtype=np.int64)
        d = np.diff(ts) / 1e6
        inter = d[d >= 1.0]
        data.append(inter)
        pts.append(inter)
        xs.append(i + 1)
    bp = axB.boxplot(data, positions=xs, widths=0.5, showfliers=False, patch_artist=True,
                     medianprops=dict(color="k", lw=1.0),
                     boxprops=dict(facecolor="#D6E4F0", edgecolor="k", lw=0.7),
                     whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    rng = np.random.default_rng(0)
    for x, inter in zip(xs, pts):
        axB.scatter(np.full(len(inter), x) + rng.uniform(-0.13, 0.13, len(inter)),
                    inter, s=3.5, color=C_RX, alpha=0.45, zorder=3, linewidths=0)
    med_all = np.median([r["batch_gap_p50_ms"] for r in stats_rows])
    axB.axhline(med_all, color=C_MED, ls="--", lw=0.9)
    axB.text(8.42, med_all + 2.5, f"中位 {med_all:.0f} ms", color=C_MED,
             fontsize=8.5, ha="right")
    axB.set_xticks(xs)
    axB.set_xticklabels([r["joint"].replace(" ", "\n") for r in stats_rows], fontsize=8)
    axB.set_ylabel("反馈批间隔 (ms)", fontsize=9.5)
    axB.set_ylim(95, 118)
    style_ax(axB)
    axB.text(-0.13, 1.04, "(b)", fontsize=11, fontweight="bold", transform=axB.transAxes)

    fig_dir = os.path.join(WS, "02_timing_latency", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, "fig1_bus_latency.png")
    fig.savefig(out_png, dpi=2000)
    plt.close(fig)
    print(f"[+] 图: {out_png}")
    for r in stats_rows:
        print(f"    {r['joint']:<9} n={r['n_frames']:<4} {r['freq_hz']}Hz  "
              f"批间隔 p50={r['batch_gap_p50_ms']}ms max={r['batch_gap_max_ms']}ms")


if __name__ == "__main__":
    main()
