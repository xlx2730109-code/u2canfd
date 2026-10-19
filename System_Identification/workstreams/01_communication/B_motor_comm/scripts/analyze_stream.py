"""分析 zero_torque_stream.py 的一次运行：反馈到达节奏、设备时间戳、发令节拍、观测时龄。

用法：
    python analyze_stream.py <run 目录> [<run 目录> ...]
输出：对应的 data/processed/<日期>/<run_id>/summary.json（raw 目录只追加，不写派生结果），并打印对比表。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BATCH_GAP_MS = 5.0  # 相邻两帧（所有电机）间隔超过它，视为两批之间的空档


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def stats(x_ms):
    x = np.asarray(x_ms, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "p50": pct(x, 50), "p95": pct(x, 95),
            "p99": pct(x, 99), "max": float(x.max()), "min": float(x.min())}


def analyze(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    fr = pd.read_csv(run / "frames.csv")
    tx = pd.read_csv(run / "tx.csv")
    t0, t1 = meta["t_start_ns"], meta["t_end_ns"]
    # 只用发令窗口内的反馈（去掉预检的 8 帧）
    fb = fr[(fr.can_id >= 0x11) & (fr.can_id <= 0x18) & (fr.t_host_mono_ns >= t0)].copy()
    fb["motor"] = fb.can_id - 0x10
    fb = fb.sort_values("t_host_mono_ns")
    out = {"run": run.name, "tx_frames": meta["tx_frames"], "tx_fail": meta["tx_fail"], "rx_in_window": int(len(fb))}

    dur_s = (t1 - t0) / 1e9
    out["rx_rate_hz_per_motor"] = {int(m): round(len(g) / dur_s, 2) for m, g in fb.groupby("motor")}
    out["frame_flags"] = {"canfd": sorted(fb.canfd.unique().tolist()), "brs": sorted(fb.brs.unique().tolist()),
                          "dir": sorted(fb.dir.unique().tolist()), "esi": sorted(fb.esi.unique().tolist())}

    # 1) 主机到达间隔（所有电机合并）与批
    th = fb.t_host_mono_ns.to_numpy()
    gaps_ms = np.diff(th) / 1e6
    out["host_arrival_gap_ms_all_motors"] = stats(gaps_ms)
    big = np.where(gaps_ms > BATCH_GAP_MS)[0]
    batch_starts = np.r_[0, big + 1]
    batch_sizes = np.diff(np.r_[batch_starts, len(th)])
    batch_period_ms = np.diff(th[batch_starts]) / 1e6
    out["batches"] = {"count": int(len(batch_starts)),
                      "period_ms": stats(batch_period_ms) if len(batch_period_ms) else None,
                      "frames_per_batch": stats(batch_sizes)}
    per_motor_gap = []
    for m, g in fb.groupby("motor"):
        per_motor_gap.append(np.diff(g.t_host_mono_ns.to_numpy()) / 1e6)
    out["host_arrival_gap_ms_per_motor"] = stats(np.concatenate(per_motor_gap))

    # 2) 设备时间戳：只检查单位与跳变，不再用它推交付滞后和 8 电机时间差
    #    原因（DeepSeek 复核后核实）：帧头时间戳约每秒向前跳 3.3 s（跨度是主机的 4.1 倍），
    #    任何滚动基线都会被重置，交付滞后没有依据；时间戳分辨率约 1 ms，按 1 ms 分箱算 8 电机差是分箱伪像。
    dev = fb.dev_timestamp.to_numpy(dtype=np.int64)
    host = th.astype(np.int64)
    one_dev = fb[fb.motor == 1].dev_timestamp.to_numpy(dtype=np.int64)
    out["dev_gap_ms_motor1_median"] = float(np.median(np.diff(one_dev)) / 1e6)  # 约 1.0 表示单位为 ns
    out["dev_timestamp_span_over_host_span"] = float((dev.max() - dev.min()) / max(1, host.max() - host.min()))
    out["dev_timestamp_forward_jumps_over_1s"] = int((np.diff(one_dev) > 1_000_000_000).sum())
    out["dev_timestamp_backsteps"] = int((np.diff(dev) < 0).sum())
    out["delivery_lag_ms"] = None
    out["dev_skew_within_1ms_bin_ms"] = None
    out["invalid_metrics_note"] = ("delivery_lag_ms 与 dev_skew_within_1ms_bin_ms 依赖有周期性前跳的设备时间戳，已作废；"
                                   "时龄只用主机侧口径")

    # 4) 发令节拍与发送调用耗时
    cyc = tx.drop_duplicates("cycle")
    dt_ms = np.diff(cyc.t_cycle_ns.to_numpy()) / 1e6
    out["cycle_period_ms"] = stats(dt_ms)
    out["cycles_over_2ms"] = int((dt_ms > 2.0).sum())
    out["cycles_over_5ms"] = int((dt_ms > 5.0).sum())
    send_us = (tx.t_tx_end_ns - tx.t_tx_begin_ns).to_numpy() / 1e3
    out["send_call_us"] = stats(send_us)

    # 5) 控制器看到的反馈时龄：每个发令周期开始时，各电机最近一次反馈已到达多久（主机时间）
    ages = []
    cyc_t = cyc.t_cycle_ns.to_numpy()[::10]  # 每 10 ms 取一个周期，足够密
    for m, g in fb.groupby("motor"):
        ta = g.t_host_mono_ns.to_numpy()
        idx = np.searchsorted(ta, cyc_t, side="right") - 1
        ok = idx >= 0
        ages.append((cyc_t[ok] - ta[idx[ok]]) / 1e6)
    out["feedback_age_ms_at_cycle_start"] = stats(np.concatenate(ages))
    return out


def main() -> int:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 1
    res = []
    for r in runs:
        s = analyze(r)
        parts = list(r.resolve().parts)
        i = len(parts) - 1 - parts[::-1].index("raw")  # .../data/raw/<日期>/<run_id> -> .../data/processed/<日期>/<run_id>
        proc = Path(*parts[:i], "processed", *parts[i + 1:])
        proc.mkdir(parents=True, exist_ok=True)
        (proc / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        res.append(s)
    keys = [
        ("每电机反馈频率 Hz(均)", lambda s: np.mean(list(s["rx_rate_hz_per_motor"].values()))),
        ("到达批数", lambda s: s["batches"]["count"]),
        ("批间隔 p50 ms", lambda s: s["batches"]["period_ms"]["p50"] if s["batches"]["period_ms"] else float("nan")),
        ("批间隔 max ms", lambda s: s["batches"]["period_ms"]["max"] if s["batches"]["period_ms"] else float("nan")),
        ("每批帧数 p50", lambda s: s["batches"]["frames_per_batch"]["p50"]),
        ("设备时间戳单位检查(ms)", lambda s: s["dev_gap_ms_motor1_median"]),
        ("设备时间戳前跳次数(电机1)", lambda s: s["dev_timestamp_forward_jumps_over_1s"]),
        ("反馈时龄 mean ms", lambda s: s["feedback_age_ms_at_cycle_start"]["mean"]),
        ("反馈时龄 p99 ms", lambda s: s["feedback_age_ms_at_cycle_start"]["p99"]),
        ("反馈时龄 max ms", lambda s: s["feedback_age_ms_at_cycle_start"]["max"]),
        ("发令周期 mean ms", lambda s: s["cycle_period_ms"]["mean"]),
        ("发令周期 p99 ms", lambda s: s["cycle_period_ms"]["p99"]),
        ("发令周期 max ms", lambda s: s["cycle_period_ms"]["max"]),
        ("周期>2ms 次数", lambda s: s["cycles_over_2ms"]),
        ("发送调用 p50 us", lambda s: s["send_call_us"]["p50"]),
        ("发送调用 max us", lambda s: s["send_call_us"]["max"]),
        ("发送失败", lambda s: s["tx_fail"]),
    ]
    w = max(len(k) for k, _ in keys)
    print(" " * (w + 2) + "  ".join(f"{s['run'].split('_powered_')[-1][:28]:>28}" for s in res))
    for k, f in keys:
        print(f"{k:<{w}}  " + "  ".join(f"{f(s):>28.4g}" for s in res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
