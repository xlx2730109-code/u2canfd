"""汇总 `zero_torque_stream.py --lite` 的长稳运行：每电机收帧数、时龄分布、周期、发送耗时、进程内存与 CPU 的变化。

用法：
    python summarize_lite.py <run 目录> [<run 目录> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 1
    for r in runs:
        lite = json.loads((r / "lite.json").read_text(encoding="utf-8"))
        meta = json.loads((r / "meta.json").read_text(encoding="utf-8"))
        res = meta.get("resources_every_5s", [])
        rx = lite["rx_per_motor"]
        tx_per_motor = meta["cycles"]
        miss = {k: tx_per_motor - v for k, v in rx.items()}
        print(f"== {r.name}")
        print(f"  时长 {meta['seconds']} s，周期 {meta['cycles']}，发送失败 {meta['tx_fail']}，进程 CPU/墙钟 {meta.get('cpu_over_wall')}")
        print(f"  每电机缺帧（发送周期数 - 收到）: {miss}")
        a, c, s = lite["age_ms"], lite["cycle_ms"], lite["send_call"]
        print(f"  反馈时龄 ms：均值 {a['mean']:.2f}  p50 {a['p50']:.2f}  p99 {a['p99']:.2f}  最大 {a['max']:.2f}")
        print(f"  发令周期 ms：均值 {c['mean']:.3f}  p99 {c['p99']:.3f}  最大 {c['max']:.1f}  >2ms {c['over_2ms']}  >5ms {c['over_5ms']}")
        print(f"  单次发送：最大 {s['max_us']:.0f} µs  >1ms {s['over1ms']}  >3ms {s['over3ms']}  共 {s['n']}")
        if res:
            first, last = res[0], res[-1]
            q = len(res) // 4 or 1
            print(f"  进程内存 MB：开始 {first[1]}  结束 {last[1]}  增长 {last[1] - first[1]:+.1f}  最大 {max(x[1] for x in res)}")
            print(f"  CPU %（psutil，单核满载为 100）：前段均值 {sum(x[2] for x in res[:q]) / q:.0f}  后段均值 {sum(x[2] for x in res[-q:]) / q:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
