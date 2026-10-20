"""结构对比与共享参数检验（调用 fit_actuator_v2 的 GPU 仿真与 CMA-ES）。

1. 结构对比：同一数据上比较 全模型（延迟+一阶滞后，滞后同时作用于施加力矩与反馈力矩）、
   仅反馈滞后（施加力矩无滞后）、仅延迟（无滞后）、无延迟、仅变化率限制，看哪种结构的代价最低，
   用来判断滞后是真实的力矩建立过程，还是只出现在上报的力矩里。
2. 共享参数：把"执行器本身"的参数（延迟、滞后、kp/kd 比例、力矩比例）固定为各关节拟合的中位数，
   只重拟合关节自己的量（惯量、摩擦、重力），看代价是否明显变差；差不多说明电机是同一个模型。
3. 考试：把每个关节拟合出的模型拿去预测"八关节同时动"的数据（多正弦），与理想仿真对比。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_actuator_v2 as fa  # noqa: E402

NAMES8 = {1: "FL_thigh", 2: "FL_calf", 3: "FR_thigh", 4: "FR_calf", 5: "RL_thigh", 6: "RL_calf", 7: "RR_thigh", 8: "RR_calf"}


def idx(name):
    return fa.NAMES.index(name)


def run_fit(ds, gens, lam, restarts, mode=0, fixed=None, seed0=0):
    """fixed: {参数名: 值}。通过临时收紧上下界实现。"""
    lo, hi = fa.LO.copy(), fa.HI.copy()
    try:
        for k, v in (fixed or {}).items():
            fa.LO[idx(k)] = v
            fa.HI[idx(k)] = v
        best = None
        for r in range(restarts):
            x, f, _ = fa.fit(ds, gens, lam, seed=seed0 + r, verbose=False, mode=mode)
            if best is None or f < best[1]:
                best = (x, f)
        phys = fa.from_unit(best[0][None, :])[0]
        return phys, best[1]
    finally:
        fa.LO[:], fa.HI[:] = lo, hi


def split_runs(root, motor):
    def ok(d):
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return m.get("abort") is None and m.get("aligned")
    runs = sorted(d for d in root.iterdir() if f"_id_{NAMES8[motor]}_" in d.name and ok(d))
    return [d for d in runs if "multisine" not in d.name], [d for d in runs if "multisine" in d.name]


def convert_ms8(root, tmp: Path):
    """把八关节同时的数据拆成每个电机一个单关节格式的目录。"""
    out = {m: [] for m in NAMES8}
    for d in sorted(root.iterdir()):
        if "_idall_" not in d.name:
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("abort") is not None or not meta.get("aligned"):
            continue
        cmd = fa.read_csv(d / "commands.csv")
        fb = fa.read_csv(d / "feedback.csv")
        n_pre = int(meta["n_pre_replies_all"])
        for m in NAMES8:
            c_mask = cmd["motor"] == m
            f_mask = (fb["motor"] == m) & (np.arange(len(fb["motor"])) >= n_pre) & (fb["state"] == 1)
            sub = tmp / f"{d.name}_m{m}"
            sub.mkdir(parents=True, exist_ok=True)
            n = int(c_mask.sum())
            np.savetxt(sub / "commands.csv", np.c_[np.arange(n), cmd["kp"][c_mask], cmd["kd"][c_mask], cmd["q_des"][c_mask], cmd["tau_ff"][c_mask]],
                       delimiter=",", header="cmd_index,kp,kd,q_des,tau_ff", comments="")
            nf = int(f_mask.sum())
            np.savetxt(sub / "feedback.csv", np.c_[np.arange(nf), fb["t_host_ns"][f_mask], fb["state"][f_mask], fb["q"][f_mask], fb["dq"][f_mask], fb["tau"][f_mask]],
                       delimiter=",", header="reply_index,t_host_ns,state,q,dq,tau", comments="")
            (sub / "meta.json").write_text(json.dumps({"n_pre_replies": 0, "rate_hz": meta["rate_hz"], "abort": None, "aligned": True}), encoding="utf-8")
            out[m].append(sub)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--lam", type=int, default=2048)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--motors", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    args = ap.parse_args()
    root = Path(args.data)
    out = {}
    ms8 = convert_ms8(root, Path(tempfile.mkdtemp()))
    # ---- 1. 结构对比 ----
    structs = {
        "full": dict(mode=0, fixed={}),
        "report_lag_only": dict(mode=1, fixed={}),
        "delay_only": dict(mode=0, fixed={"tau_c_ms": 0.05, "rate_lim": 3000.0}),
        "no_delay": dict(mode=0, fixed={"delay_ms": 0.0}),
        "rate_only": dict(mode=0, fixed={"tau_c_ms": 0.05}),
    }
    for m in args.motors:
        fit_runs, exam_runs = split_runs(root, m)
        ds = fa.Dataset(fit_runs, args.device)
        de = fa.Dataset(exam_runs, args.device)
        d8 = fa.Dataset(ms8[m], args.device)
        res = {}
        for name, cfg in structs.items():
            phys, f = run_fit(ds, args.gens, args.lam, args.restarts, mode=cfg["mode"], fixed=cfg["fixed"])
            ce = fa.total_cost(de.evaluate(phys[None, :], mode=cfg["mode"]))[0]
            c8 = fa.total_cost(d8.evaluate(phys[None, :], mode=cfg["mode"]))[0]
            ch = ds.evaluate(phys[None, :], mode=cfg["mode"])[0]  # [runs,3]
            res[name] = {"fit": float(f), "exam_single": float(ce), "exam_ms8": float(c8), "params": dict(zip(fa.NAMES, map(float, phys))),
                         "fit_channel_mse": [float(ch[:, 0].mean()), float(ch[:, 1].mean()), float(ch[:, 2].mean())]}
        res["ideal_sim"] = {"fit": float(fa.total_cost(ds.evaluate(fa.BASE[None, :]))[0]),
                            "exam_single": float(fa.total_cost(de.evaluate(fa.BASE[None, :]))[0]),
                            "exam_ms8": float(fa.total_cost(d8.evaluate(fa.BASE[None, :]))[0])}
        out[NAMES8[m]] = res
        print(NAMES8[m], {k: (round(v["fit"], 1), round(v["exam_single"], 2), round(v["exam_ms8"], 2)) for k, v in res.items()}, flush=True)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "study_structures.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
