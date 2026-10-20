"""PD 扫描分析：每个 (kp, kd) 的位置阶跃响应指标（上升时间、超调、调节时间、稳态误差、静止时的速度/力矩噪声、振铃频率），
并给出与\"理想仿真\"（标称 kp、kd 的二阶系统）的对照。用法：python analyze_pd.py <run 目录> [惯量 kg·m²]"""
import json, sys
from pathlib import Path
import numpy as np

def read(p):
    a = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {k: np.asarray(a[k]) for k in a.dtype.names}

def analyze(run: Path, inertia: float = 0.045):
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    c = read(run / "commands.csv"); f = read(run / "feedback.csv")
    n = int(meta["n_pre_replies"]); keep = np.where((np.arange(len(f["state"])) >= n) & (f["state"] == 1))[0]
    q = f["q"][keep]; dq = f["dq"][keep]; tau = f["tau"][keep]; N = min(len(q), len(c["kp"]))
    q, dq, tau = q[:N], dq[:N], tau[:N]
    rate = meta["rate_hz"]; dt = 1 / rate; q0 = meta["q0"]
    kp, kd, qd, ev, ph = c["kp"][:N], c["kd"][:N], c["q_des"][:N], c["event"][:N].astype(int), c["phase"][:N]
    amp = meta["info"]["amp"]
    rows = []
    for e in sorted(set(ev[ev >= 0])):
        idx = np.where(ev == e)[0]
        segs = []
        for tag in ("pd_up", "pd_dn"):
            ii = idx[ph[idx] == tag]
            if len(ii) == 0: continue
            # 分成连续段
            br = np.where(np.diff(ii) > 1)[0] + 1
            for s in np.split(ii, br):
                if len(s) < 100: continue
                sign = 1 if tag == "pd_up" else -1
                y = (q[s] - q[s[0] - 1]) * sign / amp   # 相对上一段末位置的归一化响应（目标 = 1）
                # 以命令目标为终点：target = qd - q_prev
                tgt = (qd[s[0]] - q[s[0] - 1]) * sign / amp
                y = y / tgt if abs(tgt) > 0.5 else y
                t = np.arange(len(y)) * dt
                def tcross(fr):
                    j = np.where(y >= fr)[0]; return t[j[0]] if len(j) else np.nan
                tr = tcross(0.9) - tcross(0.1)
                os_ = max(0.0, y.max() - 1.0)
                outside = np.where(np.abs(y - 1) > 0.05)[0]
                ts = (t[outside[-1]] if len(outside) else 0.0)
                last = slice(-int(0.2 * rate), None)
                sse = 1 - y[last].mean()
                noise_dq = dq[s][last].std(); noise_tau = tau[s][last].std()
                pk_tau = np.abs(tau[s]).max()
                # 振铃频率：超调后的过零
                yy = y - 1.0; zc = np.where(np.diff(np.sign(yy[: int(0.4 * rate)])) != 0)[0]
                ring = (len(zc) - 1) / (2 * (t[zc[-1]] - t[zc[0]])) if len(zc) >= 3 else np.nan
                segs.append((tr, os_, ts, sse, noise_dq, noise_tau, pk_tau, ring))
        if not segs: continue
        m = np.nanmean(np.array(segs), axis=0)
        rows.append({"kp": float(kp[idx[0]]), "kd": float(kd[idx[0]]), "rise_10_90_ms": m[0] * 1e3, "overshoot": m[1], "settle5_ms": m[2] * 1e3, "sse": m[3],
                     "dq_noise": m[4], "tau_noise": m[5], "peak_tau": m[6], "ring_hz": m[7], "n_steps": len(segs)})
    # 理想二阶（无延迟）的阻尼比与固有频率
    for r in rows:
        wn = (r["kp"] / inertia) ** 0.5; z = r["kd"] / (2 * (r["kp"] * inertia) ** 0.5)
        r["zeta_ideal"] = z; r["fn_ideal_hz"] = wn / (2 * np.pi)
    return rows, meta

if __name__ == "__main__":
    rows, meta = analyze(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 0.045)
    print(f"{meta['name']}  abort={meta['abort']}  aligned={meta['aligned']}")
    print(" kp   kd | 上升ms 超调  调节5%ms 稳态误差 | 静止dq噪声 静止力矩噪声 峰值力矩 | 振铃Hz | 理想ζ 理想fn")
    for r in sorted(rows, key=lambda r: (r["kp"], r["kd"])):
        print(f"{r['kp']:4.0f} {r['kd']:4.1f} | {r['rise_10_90_ms']:6.0f} {r['overshoot']*100:5.0f}% {r['settle5_ms']:7.0f} {r['sse']*100:7.1f}% | {r['dq_noise']:8.3f} {r['tau_noise']:9.3f} {r['peak_tau']:8.2f} | {r['ring_hz']:6.1f} | {r['zeta_ideal']:.2f} {r['fn_ideal_hz']:.1f}")
