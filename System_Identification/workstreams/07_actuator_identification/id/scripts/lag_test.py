"""真实力矩滞后检验：只用开环力矩阶跃（kp=kd=0）的数据，只用位置与速度误差（不看上报的力矩）拟合三种结构，
比较位置、速度的拟合误差：若\"施加力矩有一阶滞后\"明显优于\"施加力矩无滞后（仅上报滞后）\"和\"只有延迟\"，则滞后是真实的力矩建立过程。"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_actuator as fa
import study_models as sm

root = Path(sys.argv[1]); out = Path(sys.argv[2])
fa.W_T = 0.0  # 不看力矩读数，只看位置、速度
structs = {"lag": dict(mode=0, fixed={"kp_s": 1.0, "kd_s": 1.0, "ff_s": 1.0, "tau_gain": 1.0}),
           "report_lag_only": dict(mode=1, fixed={"kp_s": 1.0, "kd_s": 1.0, "ff_s": 1.0, "tau_gain": 1.0}),
           "delay_only": dict(mode=0, fixed={"tau_c_ms": 0.05, "rate_lim": 3000.0, "kp_s": 1.0, "kd_s": 1.0, "ff_s": 1.0, "tau_gain": 1.0})}
res = {}
for m, name in sm.NAMES8.items():
    runs = [d for d in sorted(root.iterdir()) if f"_id_{name}_tau_steps_" in d.name and json.loads((d / "meta.json").read_text(encoding="utf-8")).get("abort") is None]
    ds = fa.Dataset(runs, "cuda:0")
    r = {}
    for sname, cfg in structs.items():
        phys, f = sm.run_fit(ds, 300, 2048, 4, mode=cfg["mode"], fixed=cfg["fixed"])
        c = ds.evaluate(phys[None, :], mode=cfg["mode"])[0].mean(axis=0)
        r[sname] = {"q_rmse_mrad": float(np.sqrt(c[0]) * 1e3), "dq_rmse": float(np.sqrt(c[1])), "tau_c_ms": float(phys[1]), "delay_ms": float(phys[0]), "inertia": float(phys[3])}
    res[name] = r
    print(name, {k: (round(v["q_rmse_mrad"], 2), round(v["dq_rmse"], 4), round(v["tau_c_ms"], 1), round(v["delay_ms"], 1)) for k, v in r.items()}, flush=True)
    (out / "lag_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
