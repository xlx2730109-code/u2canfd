"""对比逐帧发送（A）与队列发送（B）：每周期 8 帧的发送跨度、单次调用耗时、尖峰个数、周期、tx/rx、反馈时龄。

用法：
    python compare_send_ab.py <run 目录> [<run 目录> ...]     # 目录名含 _q_ 的按队列发送（B）处理
跨度口径：A = 第一次发送开始到最后一次发送结束；B = 第一次入队开始到提交（flush）结束。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def pct(x, q):
    return float(np.percentile(x, q))


def one(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    tx = pd.read_csv(run / "tx.csv")
    queue = (run / "flush.csv").exists()
    g = tx.groupby("cycle")
    first = g.t_tx_begin_ns.min()
    last = g.t_tx_end_ns.max()
    if queue:
        fl = pd.read_csv(run / "flush.csv").set_index("cycle")
        last = fl.t_flush_end_ns.reindex(first.index)
        flush_us = (fl.t_flush_end_ns - fl.t_flush_begin_ns) / 1e3
    span_ms = (last - first) / 1e6
    call_us = (tx.t_tx_end_ns - tx.t_tx_begin_ns) / 1e3
    cyc = tx.drop_duplicates("cycle")
    dt = np.diff(cyc.t_cycle_ns.to_numpy()) / 1e6
    fr = pd.read_csv(run / "frames.csv", usecols=["t_host_mono_ns", "can_id"])
    fb = fr[(fr.can_id >= 0x11) & (fr.can_id <= 0x18) & (fr.t_host_mono_ns >= meta["t_start_ns"])]
    ages = []
    cyc_t = cyc.t_cycle_ns.to_numpy()[::10]
    for _, gg in fb.groupby("can_id"):
        ta = np.sort(gg.t_host_mono_ns.to_numpy())
        idx = np.searchsorted(ta, cyc_t, side="right") - 1
        ok = idx >= 0
        ages.append((cyc_t[ok] - ta[idx[ok]]) / 1e6)
    ages = np.concatenate(ages)
    out = {
        "run": run.name,
        "mode": "B queue" if queue else "A per-frame",
        "tx": meta["tx_frames"], "rx_in_window": int(len(fb)), "tx_fail": meta["tx_fail"],
        "span_ms": {"p50": pct(span_ms, 50), "p99": pct(span_ms, 99), "max": float(span_ms.max()),
                    "cycles_over_1ms": int((span_ms > 1).sum()), "cycles_over_3ms": int((span_ms > 3).sum())},
        "call_us": {"p50": pct(call_us, 50), "p99": pct(call_us, 99), "max": float(call_us.max()),
                    "calls_over_3ms": int((call_us > 3000).sum())},
        "cycle_ms": {"mean": float(dt.mean()), "p99": pct(dt, 99), "max": float(dt.max()),
                     "over_2ms": int((dt > 2).sum()), "over_5ms": int((dt > 5).sum())},
        "age_ms": {"mean": float(ages.mean()), "p99": pct(ages, 99), "max": float(ages.max())},
    }
    if queue:
        out["flush_us"] = {"p50": pct(flush_us, 50), "p99": pct(flush_us, 99), "max": float(flush_us.max())}
    return out


def main() -> int:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 1
    res = [one(r) for r in runs]
    parts = list(runs[0].resolve().parts)
    i = len(parts) - 1 - parts[::-1].index("raw")
    proc = Path(*parts[:i], "processed", "send_ab_summary.json")
    proc.parent.mkdir(parents=True, exist_ok=True)
    proc.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [
        ("模式", lambda r: r["mode"]),
        ("发送帧 / 窗口内接收", lambda r: f"{r['tx']}/{r['rx_in_window']}"),
        ("跨度 p50 / ms", lambda r: f"{r['span_ms']['p50']:.3f}"),
        ("跨度 p99 / ms", lambda r: f"{r['span_ms']['p99']:.3f}"),
        ("跨度 最大 / ms", lambda r: f"{r['span_ms']['max']:.2f}"),
        ("跨度>1ms 周期数", lambda r: r["span_ms"]["cycles_over_1ms"]),
        ("跨度>3ms 周期数", lambda r: r["span_ms"]["cycles_over_3ms"]),
        ("单次调用 p99 / µs", lambda r: f"{r['call_us']['p99']:.0f}"),
        ("单次调用>3ms 次数", lambda r: r["call_us"]["calls_over_3ms"]),
        ("发令周期 p99 / ms", lambda r: f"{r['cycle_ms']['p99']:.3f}"),
        ("发令周期 最大 / ms", lambda r: f"{r['cycle_ms']['max']:.2f}"),
        ("周期>2ms 次数", lambda r: r["cycle_ms"]["over_2ms"]),
        ("周期>5ms 次数", lambda r: r["cycle_ms"]["over_5ms"]),
        ("时龄 均值 / ms", lambda r: f"{r['age_ms']['mean']:.2f}"),
        ("时龄 p99 / ms", lambda r: f"{r['age_ms']['p99']:.2f}"),
        ("提交(flush) p50/p99 / µs", lambda r: "—" if "flush_us" not in r else f"{r['flush_us']['p50']:.0f}/{r['flush_us']['p99']:.0f}"),
    ]
    print("%-24s" % "" + "".join("%18s" % r["run"].split("_r")[-1] for r in res))
    for name, f in rows:
        print("%-24s" % name + "".join("%18s" % f(r) for r in res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
