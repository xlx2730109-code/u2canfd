"""速度滤波假设检验：kd 项用的速度是否经过一阶滤波（寄存器 V_BW=40 的猜测）。
比较两种结构（都固定 kp、kd 比例为 1、无一阶力矩滞后）：A 仅命令延迟；B 命令延迟加速度滤波时间常数 vf。
若 B 明显优于 A 且 vf 接近 1/(2π·40Hz)=4 ms，则支持\"等效延迟来自电机内部速度估计滤波\"。"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_vfilter as fa
import study_models as sm
sm.fa = fa
root = Path(sys.argv[1]); out = Path(sys.argv[2])
res = {}
nom = {"tau_c_ms": 0.05, "rate_lim": 3000.0, "kp_s": 1.0, "kd_s": 1.0, "ff_s": 1.0, "tau_gain": 1.0, "q_bias": 0.0}
for m, name in sm.NAMES8.items():
    fit_runs, exam_runs = sm.split_runs(root, m)
    ds, de = fa.Dataset(fit_runs, "cuda:0"), fa.Dataset(exam_runs, "cuda:0")
    r = {}
    for label, fixed in (("A_仅延迟", dict(nom, vf_ms=0.05)), ("B_延迟加速度滤波", dict(nom))):
        phys, f = sm.run_fit(ds, 300, 2048, 4, fixed=fixed)
        r[label] = {"fit": float(f), "exam": float(fa.total_cost(de.evaluate(phys[None, :]))[0]), "delay_ms": float(phys[0]), "vf_ms": float(phys[13])}
    res[name] = r
    print(name, {k: (round(v["fit"], 1), round(v["exam"], 2), round(v["delay_ms"], 1), round(v["vf_ms"], 1)) for k, v in r.items()}, flush=True)
    (out / "vf_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
