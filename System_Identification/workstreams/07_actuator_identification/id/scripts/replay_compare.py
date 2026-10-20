"""真机命令重放（Isaac Lab，`bennett_rl/scripts/tools/replay_real_vs_sim.py` 的输出 npz）与真机数据的对比。

指标：
  1. 速度、力矩的均方根误差（与零位无关）；
  2. 位置：先用 1.5 s 滑动平均去掉慢漂移（仿真腿会向模型的平衡点漂，真机停在摩擦死区内，且零位偏差未知），再比较剩余部分；
  3. 多正弦各频率（0.4–7.7 Hz）的闭环增益与相位：|q / q_cmd|，真机与仿真并列。
用法：python replay_compare.py <npz 1> [<npz 2> ...]   （文件名里的任务名作标签）
"""
import sys
from pathlib import Path
import numpy as np

FREQS = (0.4, 0.9, 1.7, 3.1, 5.3, 7.7)


def highpass(x, win):
    k = np.ones(win) / win
    pad = np.pad(x, ((win // 2, win - win // 2 - 1), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], axis=1)
    return x - sm


def sinfit(y, t, f):
    X = np.c_[np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones(len(t))]
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return complex(c[0], -c[1])


def analyze(path):
    d = np.load(path)
    dt = float(d["dt"]); skip = int(1.0 / dt)
    sq, sdq, stau = d["sim_q"], d["sim_dq"], d["sim_tau"]
    mq, mdq, mtau = d["meas_q"], d["meas_dq"], d["meas_tau"]
    cmd = d["cmd_q"]
    out = {}
    out["dq_rmse"] = float(np.sqrt(np.mean((sdq[skip:] - mdq[skip:]) ** 2)))
    out["tau_rmse"] = float(np.sqrt(np.mean((stau[skip:] - mtau[skip:]) ** 2)))
    win = int(1.5 / dt)
    hs, hm = highpass(sq - sq[0], win), highpass(mq - mq[0], win)
    out["q_hp_rmse_mrad"] = float(np.sqrt(np.mean((hs[skip:] - hm[skip:]) ** 2)) * 1e3)
    out["q_hp_rms_meas_mrad"] = float(np.sqrt(np.mean(hm[skip:] ** 2)) * 1e3)
    t = np.arange(len(sq)) * dt
    sl = slice(int(2.5 / dt), len(sq) - int(1.5 / dt))
    fr = {}
    for f in FREQS:
        gs, gm, ps, pm = [], [], [], []
        for j in range(sq.shape[1]):
            zc = sinfit(cmd[sl, j], t[sl], f)
            zs = sinfit(sq[sl, j], t[sl], f); zm = sinfit(mq[sl, j], t[sl], f)
            gs.append(abs(zs / zc)); gm.append(abs(zm / zc)); ps.append(-np.degrees(np.angle(zs / zc))); pm.append(-np.degrees(np.angle(zm / zc)))
        fr[f] = (np.mean(gs), np.mean(gm), np.mean(ps), np.mean(pm))
    out["freq"] = fr
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = analyze(p)
        print(f"== {Path(p).stem}  速度 RMSE {r['dq_rmse']:.3f} rad/s  力矩 RMSE {r['tau_rmse']:.3f} N·m  位置(去慢漂移) RMSE {r['q_hp_rmse_mrad']:.2f} mrad（真机该分量 RMS {r['q_hp_rms_meas_mrad']:.2f}）")
        print("   频率 Hz | 增益 仿真/真机 | 相位滞后° 仿真/真机")
        for f, (gs, gm, ps, pm) in r["freq"].items():
            print(f"   {f:4.1f}   | {gs:.2f} / {gm:.2f}   | {ps:+5.0f} / {pm:+5.0f}")
