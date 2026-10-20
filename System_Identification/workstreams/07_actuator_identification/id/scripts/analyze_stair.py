"""静态阶梯分析：由 static_stair.py 的数据得到各关节的零位偏差、重力力矩曲线的对照、摩擦。

模型：电机坐标 q_m，仿真坐标 q_sim = s * (q_m - z)（s 为符号，z 为零位偏差），
实测力矩（上行下行平均）= s * tau_model(q_sim) + c0，tau_model 取自 gravity_curves.py 的 MuJoCo 重力曲线（伙伴自由）；
拟合 z 与 c0（力矩读数偏置）；摩擦 = （从下方逼近的力矩 - 从上方逼近的力矩）/ 2 的均值。
用法：python analyze_stair.py <data/raw/日期> <results 目录> [--selftest]"""
import json, sys
from pathlib import Path
import numpy as np

S2M = {1: 1, 2: 1, 3: 1, 4: -1, 5: 1, 6: 1, 7: 1, 8: -1}
NAMES = {1: "FL_thigh", 2: "FL_calf", 3: "FR_thigh", 4: "FR_calf", 5: "RL_thigh", 6: "RL_calf", 7: "RR_thigh", 8: "RR_calf"}


def read_csv(p):
    a = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {k: np.asarray(a[k]) for k in a.dtype.names}


def segments(run: Path):
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    mot = meta["motor"]; n_pre = int(meta["n_pre_replies_all"]); rate = meta["rate_hz"]
    c = read_csv(run / "commands.csv"); f = read_csv(run / "feedback.csv")
    keep = np.where((f["motor"] == mot) & (np.arange(len(f["motor"])) >= n_pre) & (f["state"] == 1))[0]
    q, tau = f["q"][keep], f["tau"][keep]
    ph = c["phase"][c["motor"] == mot]
    n = min(len(q), len(ph))
    rows = []
    for tag in sorted(set(p for p in ph[:n] if str(p).endswith("_hold"))):
        idx = np.where(ph[:n] == tag)[0]
        last = idx[-int(0.5 * rate):]
        L = float(str(tag).split("_")[0][1:]); kind = "below" if "from_below" in tag else "above"
        rows.append({"level": L, "kind": kind, "q": float(q[last].mean()), "tau": float(tau[last].mean()), "tau_std": float(tau[last].std())})
    return meta, rows


def fit_zero(rows, s, grid_q, curve, qmax=0.5, c0_fixed=None):
    lv = sorted(set(r["level"] for r in rows))
    q_m = np.array([np.mean([r["q"] for r in rows if r["level"] == L]) for L in lv])
    tau = np.array([np.mean([r["tau"] for r in rows if r["level"] == L]) for L in lv])
    best = None
    for z in np.arange(-0.4, 0.4, 0.001):
        qs = s * (q_m - z)
        ok = np.abs(qs) <= qmax
        if ok.sum() < 4:
            continue
        pred = s * np.interp(qs[ok], grid_q, curve)
        c0 = float(np.mean(tau[ok] - pred)) if c0_fixed is None else c0_fixed
        res = tau[ok] - pred - c0
        rms = float(np.sqrt(np.mean(res ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(z), c0, int(ok.sum()))
    return best, q_m, tau


def main():
    root = Path(sys.argv[1]); res = Path(sys.argv[2])
    gc = json.loads((res / "gravity_curves.json").read_text(encoding="utf-8"))
    grid = np.array(gc["grid_q_sim"])
    out = {}
    runs = [(segments(r), r) for r in sorted(root.glob("*_stair_*"))]
    runs = [x for x in runs if not x[0][0].get("abort")]
    # 第一阶段：小腿（重力曲线有曲率，z 与力矩偏置 c0 可分开），取 c0 的均值作为全体共用的力矩偏置；第二阶段：大腿固定 c0（曲线近似线性，z 与 c0 不可分）
    c0s = []
    for (meta, rows), _ in runs:
        if "calf" in NAMES[meta["motor"]]:
            best, _, _ = fit_zero(rows, S2M[meta["motor"]], grid, np.array(gc["curves"][NAMES[meta["motor"]]]))
            c0s.append(best[2])
    c0_common = float(np.mean(c0s)) if c0s else -0.035
    print(f"共用力矩偏置 c0 = {c0_common:+.3f} N·m（来自小腿 {len(c0s)} 个关节，标准差 {np.std(c0s) if c0s else 0:.3f}）")
    for (meta, rows), run in runs:
        mot = meta["motor"]; name = NAMES[mot]
        calf = "calf" in name
        best, q_m, tau = fit_zero(rows, S2M[mot], grid, np.array(gc["curves"][name]), c0_fixed=None if calf else c0_common)
        fr = []
        for L in sorted(set(r["level"] for r in rows)):
            b = [r["tau"] for r in rows if r["level"] == L and r["kind"] == "below"]; a_ = [r["tau"] for r in rows if r["level"] == L and r["kind"] == "above"]
            if b and a_: fr.append((b[0] - a_[0]) / 2)
        out[name] = {"zero_offset_rad": best[1], "torque_offset_nm": best[2], "fit_rms_nm": best[0], "n_levels": best[3], "friction_half_diff_nm": float(np.mean(np.abs(fr))) if fr else None,
                     "hang_q": float(meta["q0"][str(mot)]), "c0_fixed": not calf, "levels": rows}
        print(f"{name:9s} 零位偏差 z={best[1]:+.3f} rad ({np.degrees(best[1]):+.1f}°)  力矩偏置 {best[2]:+.3f}{'(固定)' if not calf else ''}  拟合 RMS {best[0]:.3f} N·m  摩擦(半差) {out[name]['friction_half_diff_nm']:.3f}")
    (res / "stair_fit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def selftest():
    """用模型曲线加已知零位偏差、力矩偏置、摩擦、噪声合成数据，检验能否找回。"""
    res = Path(sys.argv[1]); gc = json.loads((res / "gravity_curves.json").read_text(encoding="utf-8")); grid = np.array(gc["grid_q_sim"])
    rng = np.random.default_rng(0)
    for name, mot, z_true in (("FL_thigh", 1, 0.04), ("FR_calf", 4, -0.06), ("RL_calf", 6, 0.09)):
        s = S2M[mot]; hang = 0.1
        rows = []
        for L in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
            for kind, sg in (("below", +1), ("above", -1)):
                qm = hang + L
                tau = s * np.interp(s * (qm - z_true), grid, gc["curves"][name]) - 0.035 + sg * 0.25 * s + rng.normal(0, 0.01)
                rows.append({"level": L, "kind": kind, "q": qm + rng.normal(0, 0.003), "tau": float(tau), "tau_std": 0.01})
        best, _, _ = fit_zero(rows, s, grid, np.array(gc["curves"][name]), c0_fixed=None if "calf" in name else -0.035)
        print(f"selftest {name}: 真值 z={z_true:+.3f} 找回 {best[1]:+.3f}  c0 真值 -0.035 找回 {best[2]:+.3f}  rms {best[0]:.3f}")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
