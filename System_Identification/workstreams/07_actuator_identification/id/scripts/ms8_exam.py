"""八关节同时动作的考试：用单关节拟合出的参数预测八关节同时的多正弦数据，与理想仿真对比（只评估，不拟合）。"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_actuator as fa
import study_models as sm

root = Path(sys.argv[1]); res = Path(sys.argv[2])
ms8 = sm.convert_ms8(root, Path(tempfile.mkdtemp()))
rows = []
tot = {"fit": [0, 0, 0], "ideal": [0, 0, 0]}
for m, name in sm.NAMES8.items():
    j = json.loads((res / f"fit_{name}.json").read_text(encoding="utf-8"))
    phys = np.array([j["params"][k] for k in fa.NAMES])
    d = fa.Dataset(ms8[m], "cuda:0")
    cf = d.evaluate(phys[None, :])[0].mean(axis=0)          # [q, dq, tau] 均方误差，两次运行平均
    ci = d.evaluate(fa.BASE[None, :])[0].mean(axis=0)
    rows.append((name, np.sqrt(cf), np.sqrt(ci)))
    print(f"{name:9s} RMSE  拟合模型 q {np.sqrt(cf[0])*1e3:6.2f} mrad  dq {np.sqrt(cf[1]):.3f} rad/s  tau {np.sqrt(cf[2]):.3f} N·m | 理想仿真 q {np.sqrt(ci[0])*1e3:6.2f} mrad  dq {np.sqrt(ci[1]):.3f}  tau {np.sqrt(ci[2]):.3f}")
out = {n: {"fit_rmse": [float(x) for x in f], "ideal_rmse": [float(x) for x in i]} for n, f, i in rows}
(res / "exam_ms8.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
