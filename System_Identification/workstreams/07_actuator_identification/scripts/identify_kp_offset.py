# -*- coding: utf-8 -*-
"""从力矩前馈扫描数据里解出 KP 真实值和固定偏置.

回归式(每个 tau_ff 平台取稳态中位数):
    tau_rep = KP * Δq + c * tau_ff + b
  KP : 电机真正用的位置增益(要的就是这个)
  c  : 反馈力矩里含不含前馈(c≈1 含, c≈0 不含) —— 顺便验出来
  b  : 固定偏置(台阶数据里那个 -0.036 N·m 的真身)

三个位置各拟合一次 + 全部合并拟合一次, 用来检查 b 是不是真的"固定".

只读 data/raw, 不碰硬件.
用法:  py -3.13 identify_kp_offset.py --run <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from identify_kp_kd import decode, load_feedback, fit  # noqa: E402

SETTLE_FRAC = 0.5      # 每个平台只取后 50% 当稳态


def load_cmd(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "command.csv"), encoding="utf-8")))
    g = lambda k, f=float: np.array([f(r[k]) for r in rows])
    return {"t": g("t_host_mono_ns", int), "phase": [r["phase"] for r in rows],
            "q_cmd": g("q_cmd_rad"), "ff": g("tau_ff_nm"),
            "kp": g("kp"), "kd": g("kd"), "q": g("q_real_rad"),
            "dq": g("dq_real"), "tau": g("tau_real_nm")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--canid", type=int, default=1)
    args = ap.parse_args()

    run = args.run.rstrip("\\/")
    fb, nbad = load_feedback(run, 0x10 + args.canid)
    C = load_cmd(run)
    t0 = C["t"][0]
    t_cmd = (C["t"] - t0) / 1e9
    t_fb = (fb[:, 0] - t0) / 1e9

    print(f"[+] run: {os.path.basename(run)}")
    print(f"[+] 反馈帧 {len(fb)} 采用 / {nbad} 剔除(非 MIT 布局)")
    errs = np.unique(fb[:, 1].astype(int))
    print(f"[+] 反馈里出现的 err 码: {errs.tolist()}")

    # 按 phase 分段, 只留 hold 段(p开头), 每段取后 50%
    ph = np.array(C["phase"])
    pts = []
    for name in dict.fromkeys(ph):
        if not name.startswith("p"):
            continue
        m = ph == name
        tt = t_cmd[m]
        a, b = tt[0], tt[-1]
        ca, cb = a + (b - a) * SETTLE_FRAC, b
        fs = (t_fb > ca) & (t_fb < cb)
        cs = (t_cmd > ca) & (t_cmd < cb)
        if fs.sum() < 5 or cs.sum() < 5:
            continue
        q_des = float(np.median(C["q_cmd"][cs]))
        q_m = float(np.median(fb[fs, 2]))
        dq_m = float(np.median(fb[fs, 3]))
        tau_m = float(np.median(fb[fs, 4]))
        ff = float(np.median(C["ff"][cs]))
        pts.append({"phase": name, "q_des": q_des, "q": q_m, "dq": dq_m,
                    "tau": tau_m, "ff": ff, "n_fb": int(fs.sum()),
                    "dq_eq": q_des - q_m, "err": int(np.median(fb[fs, 1]))})

    print(f"\n{'平台':<18}{'q_des':>9}{'q':>9}{'Δq(mrad)':>10}{'dq':>9}"
          f"{'tau_rep':>9}{'tau_ff':>9}{'err':>5}")
    print("-" * 80)
    for p in pts:
        print(f"{p['phase']:<18}{p['q_des']:>+9.4f}{p['q']:>+9.4f}"
              f"{p['dq_eq']*1e3:>+10.2f}{p['dq']:>+9.4f}{p['tau']:>+9.4f}"
              f"{p['ff']:>+9.2f}{p['err']:>5}")

    eq = np.array([p["dq_eq"] for p in pts])
    tau = np.array([p["tau"] for p in pts])
    ff = np.array([p["ff"] for p in pts])
    grp = np.array([p["phase"].split("_")[0] for p in pts])

    out = {"run": os.path.basename(run), "canid": args.canid, "points": pts,
           "fits": {}}

    def show(tag, mask):
        if mask.sum() < 3:
            return
        A = np.column_stack([eq[mask], ff[mask], np.ones(mask.sum())])
        # tau_rep = KP*Δq + c*tau_ff + b  (b 由第三列给出)
        coef, se, r2, cond = fit(A, tau[mask])
        print(f"\n[{tag}]  tau_rep = KP·Δq + c·tau_ff + b     n={int(mask.sum())}")
        kp_cmd = float(C["kp"].max())
        print(f"    KP = {coef[0]:8.3f} ± {se[0]:.3f}"
              f"     (下发 {kp_cmd:.1f} => 实际/下发 = {coef[0]/kp_cmd:.4f})")
        print(f"    c  = {coef[1]:8.3f} ± {se[1]:.3f}"
              f"     ({'反馈力矩含前馈' if coef[1] > 0.5 else '反馈力矩不含前馈'})")
        print(f"    b  = {coef[2]:+8.4f} ± {se[2]:.4f} N·m   R2={r2:.5f}  cond={cond:.1f}")
        # 对照: 不含 tau_ff 的单变量回归(把前馈当噪声)
        c2, s2, r22, _ = fit(eq[mask].reshape(-1, 1), tau[mask])
        print(f"    [对照] 不建模 tau_ff: KP={c2[0]:.3f}±{s2[0]:.3f} b={0.0:+.4f} "
              f"R2={r22:.5f}  <- 台阶数据的做法, 偏置被塞进截距")
        out["fits"][tag] = {"kp": float(coef[0]), "kp_se": float(se[0]),
                            "c_ff": float(coef[1]), "c_ff_se": float(se[1]),
                            "b": float(coef[2]), "b_se": float(se[2]),
                            "r2": float(r2), "cond": float(cond), "n": int(mask.sum())}

    show("全部合并", np.ones(len(pts), bool))
    for g_ in dict.fromkeys(grp):
        show(f"位置 {g_}", grp == g_)

    proc = os.path.join(WS, "07_actuator_identification", "data", "processed",
                        os.path.basename(run))
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(proc, "kp_offset_ident.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 汇总: {os.path.join(proc, 'kp_offset_ident.json')}")


if __name__ == "__main__":
    main()
