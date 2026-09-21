# -*- coding: utf-8 -*-
"""IMU 静态水平度图 —— 学术风, 每次采集出一张.

(a) 倾角时间序列: 由加速度算 roll/pitch (deg), 画均值线, 看"平不平 + 稳不稳"
(b) 气泡水平仪: roll-pitch 平面散点, 0.5°/1.0° 同心圆, 均值星标 = 机身倾角

轴向约定(与 IMU 自带欧拉角 reg3 对过, 差 <0.05°):
  roll  = atan2(ay, az)
  pitch = atan2(-ax, sqrt(ay^2+az^2))
  |g|   = sqrt(ax^2+ay^2+az^2)
用法:
  py -3.13 plot_imu_static.py --run <run_dir> --label 调平前 --out fig2a_imu_level_before
  py -3.13 plot_imu_static.py --run <run_dir> --label 调平后 --out fig2b_imu_level_after
"""
from __future__ import annotations

import argparse
import csv
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

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

C_ROLL = "#1A4E8A"
C_PITCH = "#D9822B"
C_MEAN = "#B03A2E"


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=9)


def load_run(run_dir):
    t, reg = [], {}
    with open(os.path.join(run_dir, "imu.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            reg.setdefault(int(r["reg"]), []).append(
                (int(r["t_host_mono_ns"]), float(r["x"]), float(r["y"]), float(r["z"])))
    for k in reg:
        reg[k].sort(key=lambda v: v[0])
        reg[k] = np.array(reg[k], dtype=np.float64)
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", default="", help="图内标注, 如 调平前/调平后")
    ap.add_argument("--out", required=True, help="输出图名(不含扩展名)")
    args = ap.parse_args()

    reg = load_run(args.run)
    a, g, e = reg[1], reg[2], reg.get(3)

    t0 = a[0, 0]
    ts = (a[:, 0] - t0) / 1e9
    ax_, ay_, az_ = a[:, 1], a[:, 2], a[:, 3]
    gmag = np.sqrt(ax_ ** 2 + ay_ ** 2 + az_ ** 2)
    roll = np.degrees(np.arctan2(ay_, az_))
    pitch = np.degrees(np.arctan2(-ax_, np.sqrt(ay_ ** 2 + az_ ** 2)))
    tilt = np.degrees(np.arccos(np.clip(az_ / gmag, -1, 1)))

    r_m, p_m, t_m = roll.mean(), pitch.mean(), tilt.mean()
    r_s, p_s, t_s = roll.std(), pitch.std(), tilt.std()
    g_m = gmag.mean()

    # 陀螺零偏/噪声
    gx, gy, gz = g[:, 1], g[:, 2], g[:, 3]
    gyro_bias = np.array([gx.mean(), gy.mean(), gz.mean()])
    gyro_noise = np.array([gx.std(), gy.std(), gz.std()])

    print(f"[+] {args.label}  {os.path.basename(args.run)}")
    print(f"    n={len(a)}  |g|={g_m:.4f} m/s^2")
    print(f"    roll ={r_m:+.3f}° ± {r_s:.4f}   pitch={p_m:+.3f}° ± {p_s:.4f}")
    print(f"    合倾角={t_m:.3f}° ± {t_s:.4f}")
    print(f"    陀螺零偏={np.round(gyro_bias, 5)} °/s  噪声σ={np.round(gyro_noise, 5)} °/s")
    if e is not None:
        print(f"    自报欧拉角均值=(roll {e[:,1].mean():+.3f}, pitch {e[:,2].mean():+.3f}, "
              f"yaw {e[:,3].mean():+.2f})")

    # --------- 图: (a) 倾角时间序列  (b) 气泡水平仪 ---------
    fig = plt.figure(figsize=(9.6, 3.9), dpi=2000)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22,
                          left=0.062, right=0.985, top=0.885, bottom=0.16)

    axA = fig.add_subplot(gs[0, 0])
    axA.plot(ts, roll, color=C_ROLL, lw=0.9, label="roll（绕机身 x 轴）")
    axA.plot(ts, pitch, color=C_PITCH, lw=0.9, label="pitch（绕机身 y 轴）")
    axA.axhline(r_m, color=C_ROLL, ls="--", lw=0.7, alpha=0.6)
    axA.axhline(p_m, color=C_PITCH, ls="--", lw=0.7, alpha=0.6)
    axA.axhline(0.0, color="#888888", lw=0.8, ls="-")
    lim = max(1.0, 1.35 * max(abs(roll).max(), abs(pitch).max()))
    axA.set_ylim(-lim, lim)
    axA.set_xlim(0, ts[-1])
    axA.set_xlabel("时间 (s)", fontsize=9.5)
    axA.set_ylabel("倾角 (deg)", fontsize=9.5)
    axA.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)
    axA.text(0.015, 0.94,
             f"均值  roll {r_m:+.2f}° / pitch {p_m:+.2f}°\n"
             f"波动  σ = {t_s:.3f}°   |g| = {g_m:.4f} m/s²",
             transform=axA.transAxes, va="top", fontsize=8.4, color="#333333")
    style_ax(axA)
    axA.text(-0.045, 1.06, "(a)", fontsize=11, fontweight="bold", transform=axA.transAxes)

    # (b) 气泡水平仪
    axB = fig.add_subplot(gs[0, 1])
    lim_b = max(1.0, 1.15 * max(np.percentile(np.abs(roll), 99.5),
                                np.percentile(np.abs(pitch), 99.5)))
    for rad, lab in ((0.5, "0.5°"), (1.0, "1.0°")):
        if rad <= lim_b:
            axB.add_patch(plt.Circle((0, 0), rad, fill=False, ec="#BBBBBB",
                                     lw=0.7, ls="--", zorder=1))
            axB.text(rad * 0.707, -rad * 0.707, lab, fontsize=7.5, color="#999999",
                     ha="left", va="top", zorder=1)
    step = max(1, len(roll) // 1500)     # 抽稀, 否则点数过多成一坨
    axB.scatter(roll[::step], pitch[::step], s=4, color=C_ROLL, alpha=0.30,
                linewidths=0, zorder=2)
    axB.plot([0], [0], marker="+", ms=11, mew=1.4, color="#333333", zorder=4)
    axB.plot([r_m], [p_m], marker="*", ms=17, color=C_MEAN, zorder=5,
             markeredgecolor="white", markeredgewidth=0.7)
    # 标签始终朝画面内侧摆, 免得星标贴边时文字被切掉
    sx = -1.0 if r_m > 0 else 1.0
    sy = -1.0 if p_m > 0 else 1.0
    axB.annotate(f"均值 ({r_m:+.2f}, {p_m:+.2f})°", xy=(r_m, p_m),
                 xytext=(r_m + sx * 0.34 * lim_b, p_m + sy * 0.36 * lim_b),
                 fontsize=8.2, color=C_MEAN,
                 ha="right" if sx < 0 else "left",
                 va="top" if sy < 0 else "bottom",
                 bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#DDDDDD", lw=0.6),
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=C_MEAN), zorder=6)
    axB.axhline(0, color="#DDDDDD", lw=0.6, zorder=0)
    axB.axvline(0, color="#DDDDDD", lw=0.6, zorder=0)
    axB.set_xlim(-lim_b, lim_b)
    axB.set_ylim(-lim_b, lim_b)
    axB.set_aspect("equal")
    axB.set_xlabel("roll (deg)", fontsize=9.5)
    axB.set_ylabel("pitch (deg)", fontsize=9.5)
    axB.text(0.03, 0.03, f"{args.label}：{t_m:.2f}°", transform=axB.transAxes,
             fontsize=9.5, color=C_MEAN, va="bottom")
    style_ax(axB)
    axB.text(-0.06, 1.06, "(b)", fontsize=11, fontweight="bold", transform=axB.transAxes)

    fig_dir = os.path.join(WS, "08_sensor_identification", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, args.out + ".png")
    fig.savefig(out, dpi=2000)
    plt.close(fig)

    # --------- processed 汇总 ---------
    proc = os.path.join(WS, "08_sensor_identification", "data", "processed",
                        os.path.basename(args.run.rstrip("\\/")))
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(proc, "level_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run": os.path.basename(args.run.rstrip("\\/")), "label": args.label,
            "n_samples": int(len(a)), "duration_s": float(ts[-1]),
            "g_magnitude_ms2": float(g_m),
            "roll_mean_deg": float(r_m), "pitch_mean_deg": float(p_m),
            "tilt_mean_deg": float(t_m), "tilt_std_deg": float(t_s),
            "roll_std_deg": float(r_s), "pitch_std_deg": float(p_s),
            "gyro_bias_dps": [float(v) for v in gyro_bias],
            "gyro_noise_std_dps": [float(v) for v in gyro_noise],
            "euler_mean_deg": ([float(e[:, 1].mean()), float(e[:, 2].mean()), float(e[:, 3].mean())]
                               if e is not None else None),
            "axis_convention": "roll=atan2(ay,az), pitch=atan2(-ax,sqrt(ay^2+az^2))",
        }, f, ensure_ascii=False, indent=2)
    print(f"[+] 图: {out}\n[+] 汇总: {proc}")


if __name__ == "__main__":
    main()
