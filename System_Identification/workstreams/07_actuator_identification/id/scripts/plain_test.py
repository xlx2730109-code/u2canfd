"""延迟与滞后到底有多重要：比较\"朴素执行器\"（标称 kp/kd、无延迟无滞后，只拟合惯量、摩擦、重力）与\"延迟+滞后+增益比例\"全模型。
在拟合集、考试集（多正弦）、八关节同时（ms8）上比较位置、速度、力矩误差；用来回答训练里要不要加延迟、滞后、增益缩放。"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_actuator as fa
import study_models as sm

root = Path(sys.argv[1]); res = Path(sys.argv[2])
ms8 = sm.convert_ms8(root, Path(tempfile.mkdtemp()))
plain = {"delay_ms": 0.0, "tau_c_ms": 0.05, "rate_lim": 3000.0, "kp_s": 1.0, "kd_s": 1.0, "ff_s": 1.0, "tau_gain": 1.0, "q_bias": 0.0}
only_delay_nom = {"tau_c_ms": 0.05, "rate_lim": 3000.0, "kp_s": 1.0, "kd_s": 1.0, "ff_s": 1.0, "tau_gain": 1.0, "q_bias": 0.0}   # 只允许延迟
out = {}
def rm(c): return [float(np.sqrt(c[0]) * 1e3), float(np.sqrt(c[1])), float(np.sqrt(c[2]))]
for m, name in sm.NAMES8.items():
    fit_runs, exam_runs = sm.split_runs(root, m)
    ds, de, d8 = fa.Dataset(fit_runs, "cuda:0"), fa.Dataset(exam_runs, "cuda:0"), fa.Dataset(ms8[m], "cuda:0")
    full = json.loads((res / f"fit_{name}.json").read_text(encoding="utf-8"))["params"]
    full_p = np.array([full[k] for k in fa.NAMES])
    r = {}
    for label, fixed in (("plain", plain), ("nominal_gain+delay", only_delay_nom)):
        phys, f = sm.run_fit(ds, 300, 2048, 4, fixed=fixed)
        r[label] = {"exam_multisine": rm(de.evaluate(phys[None, :])[0].mean(axis=0)), "exam_ms8": rm(d8.evaluate(phys[None, :])[0].mean(axis=0)),
                    "params": dict(zip(fa.NAMES, map(float, phys)))}
    r["full"] = {"exam_multisine": rm(de.evaluate(full_p[None, :])[0].mean(axis=0)), "exam_ms8": rm(d8.evaluate(full_p[None, :])[0].mean(axis=0))}
    out[name] = r
    f3 = lambda k, e: " ".join(f"{v:.3g}" for v in r[k][e])
    print(f"{name:9s} ms8[q mrad, dq, tau]  plain {f3('plain','exam_ms8')} | 名义增益+延迟 {f3('nominal_gain+delay','exam_ms8')} | 全模型 {f3('full','exam_ms8')}   (delay={r['nominal_gain+delay']['params']['delay_ms']:.1f} ms)", flush=True)
    (res / "plain_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
