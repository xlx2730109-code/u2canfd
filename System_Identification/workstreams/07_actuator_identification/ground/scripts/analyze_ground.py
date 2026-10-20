"""落地日志分析：读取部署程序 `--log_csv` 生成的 deploy.csv（每关节每策略步一行），输出各阶段的跟踪、力矩、IMU、反馈时龄统计。

只读，不连设备。用法：
    python analyze_ground.py <deploy.csv> [--out <json>]
阶段（startup_phase）：ramp_default、warmup、wait_arm、policy；主要看 policy，warmup 是默认姿态静止保持段。
指令接近零（|command| < 0.02）的策略步视为站立段，单独统计各关节平均力矩，用来和仿真站立力矩对比（07-T5 后续）。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NAMES = ["FL_thigh", "FL_calf", "FR_thigh", "FR_calf", "RL_thigh", "RL_calf", "RR_thigh", "RR_calf"]


def load(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def col(rows, key, dtype=float):
    return np.array([dtype(r[key]) for r in rows])


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load(Path(args.csv))
    if not rows:
        print("空日志")
        return 1
    phases = defaultdict(list)
    for r in rows:
        phases[r["startup_phase"]].append(r)
    res = {"file": args.csv, "rows": len(rows), "phases": {k: len(v) // 8 for k, v in phases.items()}}
    pol = phases.get("policy", [])
    if not pol:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        print("没有 policy 阶段的数据")
        return 1
    t = col(pol, "elapsed_s")
    steps = col(pol, "policy_step", int)
    first = np.unique(steps)
    t_step = np.array([t[steps == s][0] for s in first])
    dt = np.diff(t_step)
    res["policy_duration_s"] = float(t_step[-1] - t_step[0])
    res["policy_step_dt_ms"] = {"mean": float(dt.mean() * 1e3), "p99": pct(dt, 99) * 1e3, "max": float(dt.max() * 1e3)}
    res["feedback_age_ms"] = {"p50": pct(col(pol, "feedback_age_s"), 50) * 1e3, "p99": pct(col(pol, "feedback_age_s"), 99) * 1e3}
    res["imu_age_ms_max"] = float(max(col(pol, "imu_gyro_age_s").max(), col(pol, "imu_rpy_age_s").max()) * 1e3)
    res["motor_err_nonzero_rows"] = int((col(pol, "motor_err", int) != 1).sum())
    gyro = np.stack([col(pol, f"base_ang_vel_{a}") for a in "xyz"], axis=1)
    grav = np.stack([col(pol, f"projected_gravity_{a}") for a in "xyz"], axis=1)
    res["imu"] = {"gyro_obs_std": gyro.std(axis=0).tolist(), "gyro_obs_absmax": np.abs(gyro).max(axis=0).tolist(),
                  "gravity_obs_std": grav.std(axis=0).tolist(), "gravity_obs_range": [grav.min(axis=0).tolist(), grav.max(axis=0).tolist()]}
    cmd = np.stack([col(pol, k) for k in ("command_x", "command_y", "command_yaw")], axis=1)
    res["command_x_values"] = sorted({round(float(v), 3) for v in cmd[:, 0]})[:12]
    joints = {}
    names = [r["joint_name"] for r in pol]
    idx = {n: np.array([i for i, x in enumerate(names) if x == n]) for n in NAMES}
    qreal, qcmd, tau, dq, qrel = (col(pol, k) for k in ("q_real_motor_rad", "q_cmd_motor_rad", "tau_nm", "dq_real_motor_rad_s", "rel_rad"))
    stand = np.linalg.norm(cmd, axis=1) < 0.02
    for n in NAMES:
        i = idx[n]
        e = qreal[i] - qcmd[i]
        j = {"track_err_rms_mrad": float(np.sqrt(np.mean(e ** 2)) * 1e3), "track_err_max_mrad": float(np.abs(e).max() * 1e3),
             "tau_rms": float(np.sqrt(np.mean(tau[i] ** 2))), "tau_absmax": float(np.abs(tau[i]).max()),
             "tau_gt8_pct": float((np.abs(tau[i]) > 8).mean() * 100), "tau_gt12_pct": float((np.abs(tau[i]) > 12).mean() * 100),
             "dq_absmax": float(np.abs(dq[i]).max()), "rel_pos_range_rad": [float(qrel[i].min()), float(qrel[i].max())]}
        if stand[i].any():
            j["stand_tau_mean"] = float(tau[i][stand[i]].mean())
            j["stand_steps"] = int(stand[i].sum())
        joints[n] = j
    # 默认姿态静止保持段（warmup）：落地后机器人由 PD 撑住，用来核对力矩标尺与整机重量（需要 --warmup_s 足够长）
    war = phases.get("warmup", [])
    if war:
        wn = [r["joint_name"] for r in war]
        wt = col(war, "tau_nm"); wq = col(war, "q_real_motor_rad"); wc = col(war, "q_cmd_motor_rad")
        res["warmup"] = {"steps": len(war) // 8, "joints": {}}
        for n in NAMES:
            k = np.array([i for i, x in enumerate(wn) if x == n])
            h = k[len(k) // 2:]   # 取后一半，避开刚到位的瞬态
            res["warmup"]["joints"][n] = {"tau_mean": float(wt[h].mean()), "tau_std": float(wt[h].std()), "q_real_mean": float(wq[h].mean()), "q_cmd_mean": float(wc[h].mean())}
        res["warmup"]["tau_sum_abs"] = float(sum(abs(v["tau_mean"]) for v in res["warmup"]["joints"].values()))
    res["joints"] = joints
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
