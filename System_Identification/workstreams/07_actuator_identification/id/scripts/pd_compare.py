"""PD 扫描数据的模型检验：把 pd_grid 的每个 (kp, kd) 组合当作独立短运行，用已拟合的模型直接预测（不重新拟合），
比较三种模型在各组合上的位置、速度误差：
  A 全模型（延迟+一阶滞后，kp/kd 比例来自拟合，07-K13）
  B 仅延迟模型（无滞后，kp/kd 比例来自结构对比 study_structures.json 的 delay_only）
  C 理想仿真（标称 kp、kd，无延迟无滞后）
用来回答：电机的 PD 增益到底是不是标称值（B 的比例接近 1 且误差最小 → 是，差异来自约 8 ms 的等效延迟）。
用法：python pd_compare.py <data/raw/日期> <results 目录> <电机号>"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_actuator as fa

root = Path(sys.argv[1]); res = Path(sys.argv[2]); motor = int(sys.argv[3])
names = {1: "FL_thigh", 2: "FL_calf", 3: "FR_thigh", 4: "FR_calf", 5: "RL_thigh", 6: "RL_calf", 7: "RR_thigh", 8: "RR_calf"}
name = names[motor]
runs = [d for d in sorted(root.iterdir()) if f"_id_{name}_pd_grid_" in d.name and json.loads((d / "meta.json").read_text(encoding="utf-8")).get("abort") is None]
run = runs[-1]
meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
cmd = fa.read_csv(run / "commands.csv"); fb = fa.read_csv(run / "feedback.csv")
n_pre = int(meta["n_pre_replies"]); keep = np.where((np.arange(len(fb["state"])) >= n_pre) & (fb["state"] == 1))[0]
q, dq, tau = fb["q"][keep], fb["dq"][keep], fb["tau"][keep]
N = min(len(q), len(cmd["kp"]))
ev = cmd["event"][:N].astype(int)
tmp = Path(tempfile.mkdtemp()); combos = []
for e in sorted(set(ev[ev >= 0])):
    idx = np.where(ev == e)[0]
    sub = tmp / f"c{e:02d}"; sub.mkdir()
    np.savetxt(sub / "commands.csv", np.c_[np.arange(len(idx)), cmd["kp"][idx], cmd["kd"][idx], cmd["q_des"][idx], cmd["tau_ff"][idx]], delimiter=",", header="cmd_index,kp,kd,q_des,tau_ff", comments="")
    np.savetxt(sub / "feedback.csv", np.c_[np.arange(len(idx)), np.arange(len(idx)), np.ones(len(idx)), q[idx], dq[idx], tau[idx]], delimiter=",", header="reply_index,t_host_ns,state,q,dq,tau", comments="")
    (sub / "meta.json").write_text(json.dumps({"n_pre_replies": 0, "rate_hz": meta["rate_hz"], "abort": None, "aligned": True}), encoding="utf-8")
    combos.append((float(cmd["kp"][idx[0]]), float(cmd["kd"][idx[0]]), sub))
# 每个组合只评估阶跃部分：跳过前 0.5 s 的 settle（Dataset 的 skip 为前 1 s，这里改成 0）
ds = fa.Dataset([c[2] for c in combos], "cuda:0")
ds.skip = fa.wp.array(np.zeros(len(combos), dtype=np.int32), dtype=fa.wp.int32, device="cuda:0")
full = json.loads((res / f"fit_{name}.json").read_text(encoding="utf-8"))["params"]
st = json.loads((res / "study_structures.json").read_text(encoding="utf-8")).get(name, {})
models = {"A_全模型": np.array([full[k] for k in fa.NAMES])}
if "delay_only" in st:
    models["B_仅延迟"] = np.array([st["delay_only"]["params"][k] for k in fa.NAMES])
models["C_理想仿真"] = fa.BASE.copy()
# 惯量、摩擦等沿用全模型/各自拟合；理想仿真只换成标称 kp、kd 与零延迟
for k, p in models.items():
    if k == "C_理想仿真":
        p[fa.NAMES.index("inertia")] = full["inertia"]; p[fa.NAMES.index("coul")] = full["coul"]; p[fa.NAMES.index("g0")] = full["g0"]; p[fa.NAMES.index("g1")] = full["g1"]
out = {}
print(f"{name}  PD 扫描 {len(combos)} 个组合；位置误差 RMSE（mrad）")
print(" kp   kd | " + " ".join(f"{m:>10s}" for m in models) + " | 速度误差 RMSE（rad/s）")
rows = []
cs = {m: ds.evaluate(p[None, :])[0] for m, p in models.items()}  # [combos,3]
for i, (kp, kd, _) in enumerate(combos):
    rows.append((kp, kd, [np.sqrt(cs[m][i, 0]) * 1e3 for m in models], [np.sqrt(cs[m][i, 1]) for m in models]))
for kp, kd, eq, edq in sorted(rows):
    print(f"{kp:4.0f} {kd:4.1f} | " + " ".join(f"{v:10.2f}" for v in eq) + " | " + " ".join(f"{v:.3f}" for v in edq))
tot = {m: (float(np.sqrt(np.mean([r[2][j] ** 2 for r in rows]))), float(np.sqrt(np.mean([r[3][j] ** 2 for r in rows])))) for j, m in enumerate(models)}
print("总计 位置mrad/速度rad/s:", {m: (round(a, 2), round(b, 3)) for m, (a, b) in tot.items()})
(res / f"pd_compare_{name}.json").write_text(json.dumps({"models": {m: dict(zip(fa.NAMES, map(float, p))) for m, p in models.items()}, "total": tot,
                                                         "rows": [{"kp": r[0], "kd": r[1], "q_rmse_mrad": [float(x) for x in r[2]], "dq_rmse": [float(x) for x in r[3]]} for r in rows]}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
