# -*- coding: utf-8 -*-
"""从匀速斜坡数据里解 KD 真实值和摩擦阻尼.

KD 怎么出来的
-------------
MIT 律:  tau = KP*(q_des - q) + KD*(dq_des - dq)
同一速度 v 跑两遍, 只改 dq_des:
    dq_des=0 : tau = KP*Δq1 + KD*(0 - dq) = KP*Δq1 - KD*dq
    dq_des=dq: tau = KP*Δq2 + KD*(dq - dq) = KP*Δq2      <- 固件自己把 KD 项抵消
两遍的物理负载完全一样(同速度同位置) => tau 相同:
    KD*dq = KP*(Δq1 - Δq2)   =>   KD = KP*(Δq_ff0 - Δq_ffon)/dq
dq 是实测速度(带正负号), 所以正反两向用同一个公式.

摩擦怎么出来的
-------------
用 dq_des=0 的两遍(正反方向各一次), 在同一位置 q 上比较力矩:
    tau_out(q) = tau_grav(q) + tau_c + b_v*v
    tau_in(q)  = tau_grav(q) - tau_c - b_v*v
    => [tau_out(q) - tau_in(q)]/2 = tau_c + b_v*v
对 |v| 拟合: 截距 = 库仑摩擦 tau_c, 斜率 = 粘性阻尼 b_v.

只读 data/raw, 不碰硬件.
用法:  py -3.13 identify_kd_friction.py --run <run_dir> --kp 28.41
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

from identify_kp_kd import load_feedback, fit  # noqa: E402

GUARD = 0.15       # 匀速窗口两端各躲开 0.15s(加减速余波)


def load_cmd(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "command.csv"), encoding="utf-8")))
    g = lambda k, f=float: np.array([f(r[k]) for r in rows])
    return {"t": g("t_host_mono_ns", int), "phase": [r["phase"] for r in rows],
            "q_cmd": g("q_cmd_rad"), "dq_cmd": g("dq_cmd_rad_s"), "ff": g("tau_ff_nm"),
            "kp": g("kp"), "kd": g("kd"), "q": g("q_real_rad"),
            "dq": g("dq_real"), "tau": g("tau_real_nm")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--canid", type=int, default=1)
    ap.add_argument("--kp", type=float, default=28.41, help="experiment A 量出的 KP")
    args = ap.parse_args()

    run = args.run.rstrip("\\/")
    fb, nbad = load_feedback(run, 0x10 + args.canid)
    C = load_cmd(run)
    t0 = C["t"][0]
    t_cmd = (C["t"] - t0) / 1e9
    t_fb = (fb[:, 0] - t0) / 1e9
    print(f"[+] run: {os.path.basename(run)}")
    print(f"[+] 反馈帧 {len(fb)} 采用 / {nbad} 剔除   "
          f"err 码 {np.unique(fb[:,1].astype(int)).tolist()}")
    print(f"[+] 用 KP = {args.kp:.3f} (来自力矩前馈扫描)")

    # --- 逐段取匀速窗口 ---
    # 一律用 command.csv **同一行**的配对数据(q_cmd_rad 和 q_real_rad 是同一拍记的).
    # 早先版本 q_cmd 取命令行、q 取 CAN 帧, 两套样本权重不一样(457/443 vs 412/465);
    # 匀速窗内 Δq 其实还在缓慢漂(实测 1~7 mrad/窗), 权重一错配, 漂移就被算成假的 ΔΔq,
    # KD 从 ±0.1 散到 ±1.1 就是这么来的. 配对后权重错配和驱动批交付时延一起消失.
    ph = np.array(C["phase"])
    dq_paired = C["q_cmd"] - C["q"]
    segs = {}
    for name in dict.fromkeys(ph):
        parts = name.split("_")            # v0.05_ff0_cruise_out
        if len(parts) != 4 or parts[2] != "cruise":
            continue
        key = (f"{parts[0]}_{parts[1]}", parts[3])      # (v0.05_ff0, out/in)
        m = ph == name
        tt = t_cmd[m]
        a, b = tt[0] + GUARD, tt[-1] - GUARD
        if b - a < 0.3:
            continue
        cs = (t_cmd > a) & (t_cmd < b)
        if cs.sum() < 50:
            continue
        e = dq_paired[cs]
        half = len(e) // 2
        te = t_cmd[cs]
        qcs = C["q"][cs]
        # 独立校核: 用编码器在窗口两端的位移差算速度, 跟电机自报的 dq 对一下
        span = float(np.mean(te[-30:]) - np.mean(te[:30]))
        v_enc = ((float(np.median(qcs[-30:])) - float(np.median(qcs[:30]))) / span
                 if span > 0.2 else float("nan"))
        segs[key] = {
            "v_tag": f"{parts[0]}_{parts[1]}", "dir": parts[3],
            "q_cmd": float(np.mean(C["q_cmd"][cs])),
            "dqd_cmd": float(np.mean(C["dq_cmd"][cs])),
            "q": float(np.mean(qcs)),
            "dq": float(np.mean(C["dq"][cs])),
            "dq_enc": float(v_enc),
            "tau": float(np.mean(C["tau"][cs])),
            "dq_1": float(np.mean(e[:half])), "dq_2": float(np.mean(e[half:])),
            "q_arr": qcs.copy(), "tau_arr": C["tau"][cs].copy(),
            "n": int(cs.sum()),
        }

    print(f"\n{'段':<14}{'方向':>5}{'Δq(mrad)':>11}{'窗内漂移':>10}{'dq自报':>10}"
          f"{'dq编码器算':>12}{'dq指令':>10}{'tau(N·m)':>10}{'拍':>6}")
    print("-" * 88)
    for k in sorted(segs):
        s = segs[k]
        print(f"{s['v_tag']:<14}{s['dir']:>5}{(s['q_cmd']-s['q'])*1e3:>+11.2f}"
              f"{(s['dq_2']-s['dq_1'])*1e3:>+10.2f}"
              f"{s['dq']:>+10.4f}{s['dq_enc']:>+12.4f}{s['dqd_cmd']:>+10.4f}"
              f"{s['tau']:>+10.4f}{s['n']:>6}")

    out = {"run": os.path.basename(run), "canid": args.canid, "kp_used": args.kp,
           "segments": {f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items()
                                           if not kk.endswith("_arr")}
                        for k, v in segs.items()}}

    # --- KD: 同一速度、同一方向, ff0 vs ffon ---
    # 推导: tau_ff0 = KP*Δq1 + KD*(0 - dq);  tau_ffon = KP*Δq2 + KD*(dq_des - dq)
    # 同一速度同一位置物理负载相同 => 两式相等 => KP*(Δq1-Δq2) = KD*dq_des
    # 注意分母是 **dq_des**, 不是实测 dq —— 两边的 -KD*dq 项自己抵消掉了
    print(f"\n[1] KD = KP*(Δq_ff0 - Δq_ffon)/dq_des")
    print(f"    {'速度':>7}{'方向':>5}{'Δq_ff0':>10}{'Δq_ffon':>10}"
          f"{'ΔΔq':>9}{'ΔΔq前半':>10}{'ΔΔq后半':>10}{'dq_des':>9}{'KD':>9}{'ΔΔq/LSB':>10}")
    kds, kd_v, kd_lsb = [], [], []
    for v_tag in [t for t in sorted({k[0] for k in segs}) if t.endswith("ff0")]:
        for d in ("out", "in"):
            a, b = segs.get((v_tag, d)), segs.get((v_tag[:-3] + "ffon", d))
            if a is None or b is None:
                continue
            e1 = a["q_cmd"] - a["q"]        # Δq at dq_des=0
            e2 = b["q_cmd"] - b["q"]        # Δq at dq_des=dq
            v_cmd = b["dqd_cmd"]            # 分母是 ffon 段的 dq_des(不是 ff0 段的 0!)
            kd = args.kp * (e1 - e2) / v_cmd if abs(v_cmd) > 1e-6 else float("nan")
            lsb = abs(e1 - e2) / 0.381e-3   # 16 位编码器 1 LSB = 0.381 mrad
            # 窗内稳定性: 两个半窗各自算 ΔΔq, 差得多说明这拍没稳态
            h1 = b["dq_1"] - a["dq_1"]
            h2 = b["dq_2"] - a["dq_2"]
            kds.append(kd); kd_v.append(abs(v_cmd)); kd_lsb.append(lsb)
            print(f"{v_tag.split('_')[0]:>7}{d:>5}{e1*1e3:>+10.2f}{e2*1e3:>+10.2f}"
                  f"{(e1-e2)*1e3:>+9.2f}{h1*1e3:>+10.2f}{h2*1e3:>+10.2f}"
                  f"{v_cmd:>+9.3f}{kd:>9.3f}{lsb:>10.1f}")
    if kds:
        kds_a, kd_v_a = np.array(kds), np.array(kd_v)
        # 可信判据用编码器分辨力卡, 不用速度卡: ΔΔq 少于 10 个 LSB(3.8 mrad)就没意义.
        # 速度是间接指标, LSB 才是直接决定 KD 精度的量.
        keep = np.array(kd_lsb) >= 10.0
        print(f"    全部 {len(kds)} 段: KD = {kds_a.mean():.3f} ± {kds_a.std(ddof=1):.3f}")
        if keep.sum() >= 2:
            print(f"    只用 ΔΔq>=10 LSB 的 {int(keep.sum())} 段: "
                  f"KD = {kds_a[keep].mean():.3f} ± {kds_a[keep].std(ddof=1):.3f}  "
                  f"<- 可信")
            print(f"      这几段: " + ", ".join(f"{v:.3f}" for v in kds_a[keep]))
        print(f"    (下发 kd={float(np.max(C['kd'])):.1f})")
        out["kd"] = {"all": [float(v) for v in kds_a],
                     "all_v": [float(v) for v in kd_v_a],
                     "mean_trusted": float(kds_a[keep].mean()) if keep.sum() else None,
                     "std_trusted": float(kds_a[keep].std(ddof=1)) if keep.sum() > 1 else None,
                     "kd_cmd": float(np.max(C["kd"]))}
        if keep.sum() >= 2:
            print(f"    => KD = {kds_a[keep].mean():.3f} ± "
                  f"{kds_a[keep].std(ddof=1):.3f}  (只取可信段, 下发 kd="
                  f"{float(np.max(C['kd'])):.1f})")

    # --- 二维回归交叉验证: tau = KP*Δq + KD*(dq - dq_des) + b ---
    keys = sorted(segs)
    eq = np.array([segs[k]["q_cmd"] - segs[k]["q"] for k in keys])
    edq = np.array([segs[k]["dqd_cmd"] - segs[k]["dq"] for k in keys])   # 正号
    tau = np.array([segs[k]["tau"] for k in keys])
    A = np.column_stack([eq, edq, np.ones(len(keys))])
    coef, se, r2, cond = fit(A, tau)
    print(f"\n[2] 交叉验证  tau = KP·Δq + KD·(dq_des-dq) + b   (n={len(keys)})")
    print(f"    注意: Δq 与 (dq_des-dq) 高度共线(KP 大时 Δq 主要由 KD*dq 撑起来), "
          f"这个回归只能当方向性检查, 以 [1] 的配对差分为准")
    print(f"    KP = {coef[0]:7.3f} ± {se[0]:.3f}   "
          f"KD = {coef[1]:6.3f} ± {se[1]:.3f}   b = {coef[2]:+.4f} ± {se[2]:.4f}")
    print(f"    R2 = {r2:.5f}   cond = {cond:.1f}")
    out["regress_2d"] = {"kp": float(coef[0]), "kp_se": float(se[0]),
                         "kd": float(coef[1]), "kd_se": float(se[1]),
                         "b": float(coef[2]), "b_se": float(se[2]),
                         "r2": float(r2), "cond": float(cond)}

    # --- 摩擦: 正反两遍在同一位置比 (直观读数) ---
    print(f"\n[3] 摩擦阻尼 (配对法): [tau_out(q) - tau_in(q)]/2 对 |v|   (只用 dq_des=0 的段)")
    print(f"    注意: 出/回两趟的实际位置范围差着 (Δq_out-Δq_in) 加上驱动批交付的"
          f"~109ms 时延,\n"
          f"    低速时两趟能对上的位置很少, 所以低速点会缺 —— 定量的以 [4] 全局拟合为准")
    vs, dts, vc = [], [], []
    for v_tag in [t for t in sorted({k[0] for k in segs}) if t.endswith("ff0")]:
        o, i = segs.get((v_tag, "out")), segs.get((v_tag, "in"))
        if o is None or i is None:
            continue
        # 把 in 的 tau 插值到 out 的位置上, 保证同一 q (重力项相消)
        qo, to = o["q_arr"], o["tau_arr"]
        qi, ti = np.sort(i["q_arr"]), i["tau_arr"][np.argsort(i["q_arr"])]
        lo, hi = max(qo.min(), qi.min()), min(qo.max(), qi.max())
        m = (qo >= lo) & (qo <= hi)
        if m.sum() < 10:
            print(f"    v={v_tag.split('_')[0]:>6}  |dq|={abs(o['dq']):.4f}  "
                  f"位置重叠不足 ({int(m.sum())} 点), 跳过")
            continue
        d = to[m] - np.interp(qo[m], qi, ti)
        v_abs = abs(o["dq"])
        vs.append(v_abs); dts.append(float(np.mean(d)) / 2.0)
        # ff0 段的 dq_des 按定义就是 0, 速度只能从段名 "v0.05_ff0" 里取
        v_c = abs(float(v_tag.split("_")[0][1:]))
        vc.append(v_c)
        print(f"    v_cmd={v_c:.3f}  |dq|自报={v_abs:.4f}  "
              f"匹配 {m.sum():4d} 点   Δτ/2 = {np.mean(d)/2:+.4f} N·m")
    if vs:
        out["friction_pairwise"] = {"v_reported": [float(v) for v in vs],
                                    "v_cmd": [float(v) for v in vc],
                                    "half_dtau": [float(d) for d in dts]}
        # 用**指令速度**拟合: 自报 dq 有个约 -0.016 rad/s 的固定偏置(实测),
        # 指令速度是精确已知量, 拿它当自变量更靠谱
        if len(vc) >= 3:
            xv, yv = np.array(vc), np.array(dts)
            Av = np.column_stack([xv, np.ones_like(xv)])
            cf, sf, rf, _ = fit(Av, yv)
            dof = len(xv) - 2
            print(f"    线性拟合 ({len(xv)} 点, {dof} 自由度): "
                  f"tau_fric = {cf[0]:+.4f}·|v| {cf[1]:+.4f}   R2 = {rf:.4f}")
            print(f"    => 库仑摩擦 tau_c = {cf[1]:+.4f} ± {sf[1]:.4f} N·m")
            print(f"    => 粘性阻尼 b_v   = {cf[0]:+.4f} ± {sf[0]:.4f} N·m·s/rad"
                  f"   (自由度只有 {dof}, 这个误差棒本身不可信, 看散布)")
            print(f"    三点散布: {min(dts):.4f} ~ {max(dts):.4f} N·m "
                  f"(极差 {max(dts)-min(dts):.4f}, 占均值 "
                  f"{(max(dts)-min(dts))/np.mean(dts)*100:.1f}%)")
            out["friction_fit"] = {"v_cmd": [float(v) for v in xv],
                                   "tau_fric": [float(v) for v in yv],
                                   "tau_c": float(cf[1]), "tau_c_se": float(sf[1]),
                                   "b_v": float(cf[0]), "b_v_se": float(sf[0]),
                                   "r2": float(rf), "dof": dof}

    # --- 摩擦: 全局拟合 (把所有遍历帧都用上, 不受位置重叠限制) ---
    #   tau = tau_g(q) + tau_c*sign(dq) + b_v*dq
    #   tau_g(q) 就是"不动机器人时该位置要扛的重力/弹性负载", 用三次多项式吃掉;
    #   剩下 sign(dq) 的系数就是库仑摩擦, dq 的系数就是粘性阻尼.
    #   好处: 加减速段(0 -> v 全速度范围)全部变成 b_v 的量程, 不再只有 2 个速度点.
    print(f"\n[4] 摩擦阻尼 (全局): tau = c0+c1·q+c2·q²+c3·q³ + tau_c·sign(dq) + b_v·dq")
    Gq, Gdq, Gt, Gg = [], [], [], []
    for name in dict.fromkeys(ph):
        parts = name.split("_")
        if len(parts) != 4 or parts[2] not in ("accel", "cruise", "decel"):
            continue
        m = ph == name
        tt = t_cmd[m]
        a, b = tt[0] + 0.02, tt[-1] - 0.02
        cs = (t_cmd > a) & (t_cmd < b)
        if cs.sum() < 5:
            continue
        Gq.append(C["q"][cs]); Gdq.append(C["dq"][cs]); Gt.append(C["tau"][cs])
        Gg.append(np.full(int(cs.sum()), f"{parts[0]}_{parts[1]}_{parts[3]}"))
    if Gq:
        qg = np.concatenate(Gq); dqg = np.concatenate(Gdq)
        tg = np.concatenate(Gt); gg = np.concatenate(Gg)
        # 必须中心化: q 的绝对值 ~1.5 rad, 直接拿 q^3 当列条件数会炸
        q0g = float(np.mean(qg))
        qc = qg - q0g
        Ag = np.column_stack([np.ones_like(qc), qc, qc ** 2, np.sign(dqg), dqg])
        cg, sg, r2g, condg = fit(Ag, tg)
        k_tc, k_bv = 3, 4
        print(f"    用了 {len(qg)} 帧, q 中心 {q0g:+.4f} rad, "
              f"|dq| 覆盖 0 ~ {np.abs(dqg).max():.3f} rad/s")
        print(f"    tau_c (库仑, sign(dq) 系数) = {cg[k_tc]:+.4f} ± {sg[k_tc]:.4f} N·m")
        print(f"    b_v   (粘性, dq 系数)       = {cg[k_bv]:+.4f} ± {sg[k_bv]:.4f} N·m·s/rad")
        print(f"    重力项 (中心化二次) c0,c1,c2 = {cg[0]:+.4f}, {cg[1]:+.3f}, {cg[2]:+.3f}"
              f"   R2 = {r2g:.4f}  cond = {condg:.1f}")
        # 分组核对: 每组平均残差 = tau_c*<sign(dq)> + b_v*<dq>
        print(f"\n    {'组':<16}{'<dq>':>9}{'帧':>7}{'实测tau':>11}"
              f"{'重力项':>10}{'残差':>10}{'τ_c+b_v·v 预测':>16}")
        by_v = []
        for g_ in dict.fromkeys(gg):
            m = gg == g_
            grav = float(np.mean(Ag[m, :3] @ cg[:3]))
            res = float(np.mean(tg[m] - Ag[m, :3] @ cg[:3]))
            v_bar = float(np.mean(dqg[m]))
            pred = cg[k_tc] * float(np.mean(np.sign(dqg[m]))) + cg[k_bv] * v_bar
            by_v.append({"group": g_, "dq_mean": v_bar, "tau_mean": float(np.mean(tg[m])),
                         "grav": grav, "resid": res, "pred": pred, "n": int(m.sum())})
            print(f"    {g_:<16}{v_bar:>+9.4f}{int(m.sum()):>7}{np.mean(tg[m]):>+11.4f}"
                  f"{grav:>+10.4f}{res:>+10.4f}{pred:>+16.4f}")
        out["friction"] = {"method": "global", "q_center": q0g,
                           "tau_c": float(cg[k_tc]), "tau_c_se": float(sg[k_tc]),
                           "b_v": float(cg[k_bv]), "b_v_se": float(sg[k_bv]),
                           "grav_poly": [float(v) for v in cg[:3]],
                           "r2": float(r2g), "cond": float(condg),
                           "n_frames": int(len(qg)), "by_group": by_v}

    proc = os.path.join(WS, "07_actuator_identification", "data", "processed",
                        os.path.basename(run))
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(proc, "kd_friction_ident.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[+] 汇总: {os.path.join(proc, 'kd_friction_ident.json')}")


if __name__ == "__main__":
    main()
