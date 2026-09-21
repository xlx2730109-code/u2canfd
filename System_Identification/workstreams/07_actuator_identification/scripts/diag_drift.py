# -*- coding: utf-8 -*-
"""诊断: 匀速窗内 Δq 到底稳没稳.

把每个匀速窗切两半分别算 Δq, 两半差多少一目了然. 实测窗内 Δq 会漂 1~7 mrad,
所以**算 Δq 必须用 command.csv 同一行的配对数据**(q_cmd_rad 与 q_real_rad 同拍记录).
早先这里 q_cmd 取命令行、q 取 CAN 帧, 两套样本权重不同, 漂移会被算成假的差值 ——
identify_kd_friction 之前 KD 散到 ±1.1 就是这个原因.

只读 data/raw.
用法:  py -3.13 diag_drift.py --run <run_dir> [--run <run_dir> ...]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from identify_kp_kd import load_feedback  # noqa: E402

GUARD = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--canid", type=int, default=1)
    args = ap.parse_args()

    for run in args.run:
        run = run.rstrip("\\/")
        rows = list(csv.DictReader(open(os.path.join(run, "command.csv"), encoding="utf-8")))
        t = np.array([int(r["t_host_mono_ns"]) for r in rows])
        ph = np.array([r["phase"] for r in rows])
        qc = np.array([float(r["q_cmd_rad"]) for r in rows])
        qr = np.array([float(r["q_real_rad"]) for r in rows])   # 与 qc 同拍, 配对!
        dqv = np.array([float(r["dq_real"]) for r in rows])
        t0 = t[0]
        t_cmd = (t - t0) / 1e9

        print(f"\n=== {os.path.basename(run)} ===")
        print(f"{'段':<22}{'Δq前半':>10}{'Δq后半':>10}{'漂移':>9}"
              f"{'dq前半':>9}{'dq后半':>9}")
        for name in dict.fromkeys(ph):
            parts = name.split("_")
            if len(parts) != 4 or parts[2] != "cruise":
                continue
            m = ph == name
            tt = t_cmd[m]
            a, b = tt[0] + GUARD, tt[-1] - GUARD
            if b - a < 0.3:
                continue
            mid = (a + b) / 2
            res = []
            for lo_, hi_ in ((a, mid), (mid, b)):
                cs = (t_cmd > lo_) & (t_cmd < hi_)
                if cs.sum() < 5:
                    res.append((float("nan"), float("nan")))
                    continue
                res.append((float(np.mean(qc[cs] - qr[cs])), float(np.mean(dqv[cs]))))
            d1, d2 = res
            print(f"{name:<22}{d1[0]*1e3:>+10.2f}{d2[0]*1e3:>+10.2f}"
                  f"{(d2[0]-d1[0])*1e3:>+9.2f}{d1[1]:>+9.4f}{d2[1]:>+9.4f}")


if __name__ == "__main__":
    main()
