# -*- coding: utf-8 -*-
"""图①b 延迟"现象 + 原因"双联图 —— 复刻旧图 lat_20260905_034522.png 思路的定稿版.

(a) 现象: 台阶响应起步滞后.
    数据: 本会话实测 (07 工作包 2026-09-17 台阶, FL_thigh 0.15 rad kp=28 kd=2),
    取 data/raw 下最新一次 step_FL_thigh 运行.
    延迟定义: 起步时刻差 = q_real 幅值达 10% 的时刻 - q_cmd 达 10% 的时刻.
    (不用整体均方误差平移: 响应两段式上升, SSE 对齐被波形带偏, 旧图 93ms 即源于此;
     起步时刻是稳健估计)
(b) 原因: 反馈按批到达.
    数据: 本工作包 1kHz 零力矩探测 frames.csv (自己实测, 2026-09-16).
    同一电机的反馈帧到达时刻栅格 —— 批内 <1ms, 批间隔 ~109ms;
    批间隔中位数现算出来, 直接写进 (a) 的结论框, 不写死.

输出: results/figures/fig1b_latency_shift.png (300 dpi)
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, PARENT)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

SRC_CSV = None  # (a) 数据改为本会话实测, 见 load_step()
C_TGT = "#1A4E8A"     # 目标: 深蓝
C_ACT = "#D9822B"     # 实际: 橙
C_GHOST = "#EBCBA6"   # 原始实际: 浅橙
C_RX = "#1A4E8A"
C_TX = "#9E9E9E"


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=9)


def onset(t, y, frac=0.1):
    """首次越过总变化 10% 的时刻."""
    lo, hi = y[0], y[-1]
    i = int(np.argmax(np.abs(y - lo) > frac * abs(hi - lo)))
    return t[i]


def load_step():
    """本会话实测台阶: 07 工作包最新一次 FL_thigh 台阶运行的 command.csv + feedback.csv."""
    cands = sorted(glob.glob(os.path.join(
        WS, "07_actuator_identification", "data", "raw", "*", "*step_FL_thigh*")))
    if not cands:
        raise FileNotFoundError("没有找到 step_FL_thigh 台阶数据, 先跑 "
                                "07_actuator_identification/scripts/step_single.py")
    run = cands[-1]
    rows = list(csv.DictReader(open(os.path.join(run, "command.csv"), encoding="utf-8")))
    t = np.array([float(r["t_rel_s"]) for r in rows])
    t0_cmd = int(rows[0]["t_host_mono_ns"])
    qc = np.degrees(np.array([float(r["q_cmd_rad"]) for r in rows]))
    qr = np.degrees(np.array([float(r["q_real_rad"]) for r in rows]))
    # 同一份 feedback.csv: 0x11 反馈帧到达时刻(与 command.csv 同一单调时钟), 聚成批
    fb = np.array([int(r["t_host_mono_ns"]) for r in
                   csv.DictReader(open(os.path.join(run, "feedback.csv"), encoding="utf-8"))
                   if int(r["can_id"]) == 0x11], dtype=np.int64)
    fb_rel = (fb - t0_cmd) / 1e9 + t[0]
    br = np.where(np.diff(fb_rel) > 0.002)[0]
    batch_t = np.concatenate(([fb_rel[0]], fb_rel[br + 1]))
    print(f"[+] 台阶数据: {os.path.basename(run)} (07 工作包实测, {len(batch_t)} 个反馈批)")
    return t, qc, qr, batch_t


def load_probe():
    """1kHz 探测: 返回 (tx 时刻列表, FL_thigh RX 时刻列表, 批间隔中位 ms)."""
    cands = sorted(glob.glob(os.path.join(
        WS, "02_timing_latency", "data", "raw", "*", "*feedback_age*")))
    run = cands[-1]
    by = collections.defaultdict(list)
    tx = []
    with open(os.path.join(run, "frames.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["can_id"])
            if cid == 0x11:
                by[cid].append(int(row["t_host_mono_ns"]))
    with open(os.path.join(run, "tx.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tx.append(int(row["t_host_mono_ns"]))
    rx = np.array(sorted(by[0x11]), dtype=np.int64)
    gaps = np.diff(rx) / 1e6
    batch_med = float(np.median(gaps[gaps >= 1.0]))
    return np.array(sorted(tx), dtype=np.int64), rx, batch_med


def main():
    t, qc, qr, batch_t = load_step()
    i_step = int(np.argmax(np.abs(qc - qc[0]) > 0.05))
    t_step = t[i_step]
    d_onset = onset(t, qr) - onset(t, qc)
    d_ms = d_onset * 1000
    tx, rx, batch_med = load_probe()
    print(f"[+] 台阶: 起步滞后 Δ = {d_ms:.1f} ms;  探测: 批间隔中位 {batch_med:.1f} ms")

    fig = plt.figure(figsize=(9.6, 3.9), dpi=2000)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.24,
                          left=0.062, right=0.988, top=0.90, bottom=0.155)

    # ---------------- (a) 台阶平移对齐 ----------------
    axA = fig.add_subplot(gs[0, 0])
    t0, t1 = t_step - 0.20, t_step + 0.80
    m = (t >= t0) & (t <= t1)
    axA.plot(t[m], qc[m], "--", color=C_TGT, lw=1.5, label="目标角度")
    axA.plot(t[m], qr[m], color=C_GHOST, lw=1.3, label="实际角度（原始）")
    axA.plot(t[m] - d_onset, qr[m], color=C_ACT, lw=1.7,
             label=f"实际角度（提前 {d_ms:.0f} ms 平移后）")
    axA.axvline(t_step, color="#999999", lw=0.7, ls=":")

    ymin = qr[m].min() - 5.5
    y_lvl = max(qc[m].max(), qr[m].max())
    axA.set_ylim(ymin, y_lvl + 2.8)
    axA.set_xlim(t0, t1)

    y_dim = y_lvl + 1.7
    axA.text(t_step + d_onset / 2, y_dim + 0.5, f"Δ ≈ {d_ms:.0f} ms",
             ha="center", fontsize=9.5)
    for x_end in (t_step, t_step + d_onset):
        axA.plot([x_end, x_end], [y_lvl - 0.1, y_dim], ls=":", lw=0.7,
                 color="#999999", zorder=1)

    # 反馈批到达刻度(顶部) + 一个台阶周期箭头: 上升沿每 ~109 ms 跳一阶
    post = batch_t[(batch_t >= t_step) & (batch_t <= t1)]
    y_top = y_lvl + 0.95
    axA.plot(post, np.full(len(post), y_top), linestyle="None", marker="|",
             ms=7, mew=1.3, color=C_RX, zorder=3)
    if len(post):
        axA.text(post[0] - 0.008, y_top, "反馈批到达", fontsize=8, color=C_RX,
                 ha="right", va="center")
    if len(post) >= 2:
        y_arr = y_lvl * 0.52
        axA.annotate("", xy=(post[1], y_arr), xytext=(post[0], y_arr),
                     arrowprops=dict(arrowstyle="<->", lw=0.9, color="k"))
        axA.text((post[0] + post[1]) / 2, y_arr + 0.45, f"~{batch_med:.0f} ms",
                 ha="center", fontsize=8.5)

    axA.text(0.985, 0.42,
             f"起步 Δ 随批相位变 (0~{batch_med:.0f} ms)\n"
             f"上升沿每 {batch_med:.0f} ms 跳一阶 = 反馈批交付",
             transform=axA.transAxes, ha="right", va="bottom", fontsize=8.2,
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#BBBBBB", lw=0.7))

    axA.set_xlabel("时间 (s)", fontsize=9.5)
    axA.set_ylabel("关节角度 (deg)", fontsize=9.5)
    axA.legend(frameon=False, fontsize=8.2, loc="lower left")
    style_ax(axA)
    axA.text(-0.02, 1.06, "(a)", fontsize=11, fontweight="bold",
             transform=axA.transAxes)

    # ---------------- (b) 反馈批到达栅格 ----------------
    axB = fig.add_subplot(gs[0, 1])
    # 基准取启动空档(>10ms)之后: 开头第一轮 8 条命令挤在 ~0.7ms 内,
    # 随后 200ms 空档, 之后的 1kHz 流才是稳态; 否则出现孤线 + 假长批间隔
    dtx = np.diff(tx)
    holes = np.where(dtx > 10e6)[0]
    i0 = int(holes[0]) + 1 if len(holes) else 0
    t_ref = int(tx[i0])
    WIN = 0.45e9
    rx_s = [(x - t_ref) / 1e9 for x in rx if 0 <= x - t_ref < WIN]
    tx_s = sorted((x - t_ref) / 1e9 for x in tx if 0 <= x - t_ref < WIN)
    axB.vlines(tx_s, 1.15, 1.62, color=C_TX, lw=0.5, alpha=0.85)
    axB.vlines(rx_s, 0.18, 0.66, color=C_RX, lw=2.6)
    rg = np.diff(rx_s) * 1000
    big_idx = np.where(rg > 5)[0]
    for i in range(min(2, len(big_idx))):
        x0, x1 = rx_s[big_idx[i]], rx_s[big_idx[i] + 1]
        axB.annotate("", xy=(x1, 0.86), xytext=(x0, 0.86),
                     arrowprops=dict(arrowstyle="<->", lw=0.9, color="k"))
        axB.text((x0 + x1) / 2, 0.92, f"~{rg[big_idx[i]]:.0f} ms",
                 ha="center", fontsize=8.5)
    axB.text(0.012, 1.70, "命令流（1 kHz 连续发送）", fontsize=9, color="#555")
    axB.text(0.012, 0.72, "反馈到达", fontsize=9, color=C_RX)
    axB.set_ylim(0, 1.95)
    axB.set_xlim(-0.005, 0.46)
    axB.set_yticks([])
    axB.set_xlabel("时间 (s)", fontsize=9.5)
    style_ax(axB)
    axB.text(-0.02, 1.06, "(b)", fontsize=11, fontweight="bold",
             transform=axB.transAxes)

    fig_dir = os.path.join(WS, "02_timing_latency", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "fig1b_latency_shift.png")
    fig.savefig(out, dpi=2000)
    plt.close(fig)
    print(f"[+] 图: {out}")


if __name__ == "__main__":
    main()
