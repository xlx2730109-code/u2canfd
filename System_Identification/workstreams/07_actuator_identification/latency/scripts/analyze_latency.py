"""分析 action_latency.py 的一次运行：命令变化后，力矩、速度、位置反馈第一次出现响应用了多少个命令周期（1 ms 一个）。

反馈与命令一一对应时（meta.aligned 为真），响应延迟 = 第一个超出阈值的反馈序号 - 阶跃命令序号，与驱动的批量交付无关；
同时给出主机时刻差作对照（含交付延迟）。阈值 = 基线均值 ± max(5 倍标准差, 1.5 倍量化步长)，连续 3 个反馈超过才算。

用法：
    python analyze_latency.py <run 目录>
结果写到对应的 data/processed/<日期>/<run_id>/latency_summary.json 并打印。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STEP = {"tau": 40 / 4095, "dq": 90 / 4095, "q": 25 / 65535}  # 反馈编码的量化步长（DM8006 量程）


def first_response(sig: np.ndarray, start: int, sign: int, base_lo: int, base_hi: int, step: float) -> int | None:
    base = sig[base_lo:base_hi]
    thr = max(5 * base.std(), 1.5 * step)
    dev = (sig[start:start + 400] - base.mean()) * sign
    over = dev > thr
    for i in range(len(over) - 2):
        if over[i] and over[i + 1] and over[i + 2]:
            return i  # 相对 start 的偏移
    return None


def analyze(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    cmd = pd.read_csv(run / "commands.csv")
    fb = pd.read_csv(run / "feedback.csv")
    n_pre = int(meta["n_pre_replies"])
    out = {"run": run.name, "aligned": bool(meta.get("aligned")), "events": len(meta["events"]), "mode": meta["mode"], "abort": meta.get("abort")}
    sig = {"tau": fb.tau.to_numpy(), "dq": fb.dq.to_numpy(), "q": fb.q.to_numpy()}
    res = {k: {"frames": [], "host_ms": []} for k in sig}
    for ev in meta["events"]:
        k0 = int(ev["cmd_index"])
        j0 = n_pre + k0
        if j0 - 160 < 0 or j0 + 400 > len(fb):
            continue
        sign = int(ev["sign"])
        t_cmd = int(cmd.t_send_begin_ns.iloc[k0])
        for name, s in sig.items():
            off = first_response(s, j0, sign, j0 - 150, j0 - 5, STEP[name])
            if off is None:
                continue
            res[name]["frames"].append(off)
            res[name]["host_ms"].append((int(fb.t_host_ns.iloc[j0 + off]) - t_cmd) / 1e6)
    for name, r in res.items():
        f, h = np.array(r["frames"]), np.array(r["host_ms"])
        out[name] = {
            "n_responded": int(len(f)),
            "latency_frames": None if len(f) == 0 else {"p10": float(np.percentile(f, 10)), "median": float(np.median(f)),
                                                         "p90": float(np.percentile(f, 90)), "min": int(f.min()), "max": int(f.max())},
            "host_ms": None if len(h) == 0 else {"median": float(np.median(h)), "p10": float(np.percentile(h, 10)), "p90": float(np.percentile(h, 90))},
        }
    if not out["aligned"]:
        out["warning"] = "反馈与命令未对齐（有丢帧或多帧），延迟帧数不可信，只看主机时刻差"
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    run = Path(sys.argv[1])
    out = analyze(run)
    parts = list(run.resolve().parts)
    if "raw" in parts:
        i = len(parts) - 1 - parts[::-1].index("raw")
        proc = Path(*parts[:i], "processed", *parts[i + 1:])
        proc.mkdir(parents=True, exist_ok=True)
        (proc / "latency_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
