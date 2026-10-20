"""执行器参数拟合（GPU）：用 actuator_id.py 采到的单关节数据，拟合仿真执行器模型的参数。

模型（单自由度，每周期 S 个子步展开）：
    命令延迟 D（ZOH）、MIT 力矩 tau_cmd = kp_s*kp*(q_des + q_bias - q) - kd_s*kd*dq + ff_s*tau_ff，限幅，
    一阶滞后（时间常数 tau_c）与变化率限制 R，
    刚体 I*ddq = tau_m - g0 - g1*(q-q0) - b*dq - c*tanh(dq/0.02)，
    反馈力矩 = tau_gain * tau_m。
13 个参数，用 CMA-ES 在 GPU 上并行评估（Warp 内核，一个线程仿真一个候选在一次运行上的全部周期）。

拟合集与考试集分开：协议 tau_steps、chirp、ramp 拟合，multisine 只用来检验。

用法（GPU 环境）：
    python fit_actuator.py --motor 4 --data <data/raw/日期> --out <目录>
    python fit_actuator.py --selftest        # 合成数据自检：已知参数能否被找回
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

wp.config.quiet = True


def read_csv(path: Path) -> dict:
    """读 CSV 为 {列名: 数组}（数值列为 float，其余为字符串），不依赖 pandas。"""
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {k: np.asarray(arr[k]) for k in arr.dtype.names}

NAMES = ["delay_ms", "tau_c_ms", "rate_lim", "inertia", "visc", "coul", "g0", "g1", "kp_s", "kd_s", "ff_s", "q_bias", "tau_gain"]
LO = np.array([0.0, 0.05, 5.0, 5e-4, 0.0, 0.0, -3.0, -10.0, 0.6, 0.3, 0.9, -0.03, 0.9], dtype=np.float64)
HI = np.array([12.0, 60.0, 3000.0, 0.3, 1.5, 1.0, 3.0, 10.0, 1.4, 2.5, 1.1, 0.03, 1.1], dtype=np.float64)
LOG = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
BASE = np.array([0.0, 0.05, 3000.0, 0.01, 0.01, 0.01, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])  # "理想仿真"：无延迟无滞后，其余取标称
P = len(NAMES)
S_SUB = 4
W_Q, W_DQ, W_T = 1 / 0.005 ** 2, 1 / 0.1 ** 2, 1 / 0.05 ** 2


def from_unit(u: np.ndarray) -> np.ndarray:
    """[0,1]^P -> 物理参数。"""
    u = np.clip(u, 0.0, 1.0)
    out = np.empty_like(u)
    for j in range(P):
        if LOG[j]:
            out[..., j] = np.exp(np.log(LO[j]) + u[..., j] * (np.log(HI[j]) - np.log(LO[j])))
        else:
            out[..., j] = LO[j] + u[..., j] * (HI[j] - LO[j])
    return out


def to_unit(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    for j in range(P):
        if LOG[j]:
            out[..., j] = (np.log(np.clip(x[..., j], LO[j], HI[j])) - np.log(LO[j])) / (np.log(HI[j]) - np.log(LO[j]))
        else:
            out[..., j] = (np.clip(x[..., j], LO[j], HI[j]) - LO[j]) / (HI[j] - LO[j])
    return out


@wp.kernel
def sim_kernel(params: wp.array2d(dtype=wp.float32),
               kp_a: wp.array(dtype=wp.float32), kd_a: wp.array(dtype=wp.float32), qd_a: wp.array(dtype=wp.float32), ff_a: wp.array(dtype=wp.float32),
               qm_a: wp.array(dtype=wp.float32), dqm_a: wp.array(dtype=wp.float32), tm_a: wp.array(dtype=wp.float32),
               run_start: wp.array(dtype=wp.int32), run_len: wp.array(dtype=wp.int32), run_q0: wp.array(dtype=wp.float32),
               run_t0: wp.array(dtype=wp.float32), run_skip: wp.array(dtype=wp.int32),
               n_runs: int, period: float, sub: int, tmax: float,
               out: wp.array2d(dtype=wp.float32), pred_q: wp.array(dtype=wp.float32), pred_dq: wp.array(dtype=wp.float32), pred_t: wp.array(dtype=wp.float32),
               save_pred: int):
    tid = wp.tid()
    i = tid / n_runs
    r = tid - i * n_runs
    delay = params[i, 0] * 0.001
    tc = params[i, 1] * 0.001
    rate_lim = params[i, 2]
    inertia = params[i, 3]
    visc = params[i, 4]
    coul = params[i, 5]
    g0 = params[i, 6]
    g1 = params[i, 7]
    kp_s = params[i, 8]
    kd_s = params[i, 9]
    ff_s = params[i, 10]
    q_bias = params[i, 11]
    t_gain = params[i, 12]
    dt = period / float(sub)
    dsteps = int(delay / dt + 0.5)
    start = run_start[r]
    n = run_len[r]
    q0 = run_q0[r]
    q = q0
    dq = float(0.0)
    tm = run_t0[r]
    alpha = 1.0 - wp.exp(-dt / tc)
    e_q = float(0.0)
    e_dq = float(0.0)
    e_t = float(0.0)
    cnt = float(0.0)
    skip = run_skip[r]
    for j in range(n):
        for s in range(sub):
            u = j * sub + s - dsteps
            c = 0
            if u > 0:
                c = u / sub
            if c > n - 1:
                c = n - 1
            kp = kp_a[start + c]
            kd = kd_a[start + c]
            qd = qd_a[start + c]
            ff = ff_a[start + c]
            tau_cmd = kp_s * kp * (qd + q_bias - q) - kd_s * kd * dq + ff_s * ff
            tau_cmd = wp.clamp(tau_cmd, -tmax, tmax)
            d = (tau_cmd - tm) * alpha
            lim = rate_lim * dt
            d = wp.clamp(d, -lim, lim)
            tm = tm + d
            acc = (tm - g0 - g1 * (q - q0) - visc * dq - coul * wp.tanh(dq / 0.02)) / inertia
            dq = dq + acc * dt
            q = q + dq * dt
        if save_pred != 0:
            pred_q[i * 0 + start + j] = q
            pred_dq[start + j] = dq
            pred_t[start + j] = t_gain * tm
        if j >= skip:
            eq = q - qm_a[start + j]
            edq = dq - dqm_a[start + j]
            et = t_gain * tm - tm_a[start + j]
            e_q += eq * eq
            e_dq += edq * edq
            e_t += et * et
            cnt += 1.0
    cnt = wp.max(cnt, 1.0)
    out[tid, 0] = e_q / cnt
    out[tid, 1] = e_dq / cnt
    out[tid, 2] = e_t / cnt


class Dataset:
    def __init__(self, runs: list[Path], device: str):
        kp, kd, qd, ff, qm, dqm, tm = [], [], [], [], [], [], []
        starts, lens, q0s, t0s, skips, names = [], [], [], [], [], []
        pos = 0
        self.period = None
        for d in runs:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            cmd = read_csv(d / "commands.csv")
            fb = read_csv(d / "feedback.csv")
            n_pre = int(meta["n_pre_replies"])
            keep = np.where((np.arange(len(fb["state"])) >= n_pre) & (fb["state"] == 1))[0]
            post = {k: v[keep] for k, v in fb.items() if k in ("q", "dq", "tau")}
            n = min(len(cmd["kp"]), len(post["q"]))
            period = 1.0 / float(meta["rate_hz"])
            if self.period is None:
                self.period = period
            assert abs(self.period - period) < 1e-9, "同一批数据的发令频率必须相同"
            kp.append(cmd["kp"][:n]); kd.append(cmd["kd"][:n]); qd.append(cmd["q_des"][:n]); ff.append(cmd["tau_ff"][:n])
            qm.append(post["q"][:n]); dqm.append(post["dq"][:n]); tm.append(post["tau"][:n])
            starts.append(pos); lens.append(n); q0s.append(float(post["q"][0])); t0s.append(float(post["tau"][0]))
            skips.append(int(1.0 / period))  # 前 1 s 保持阶段不计入误差
            names.append(d.name)
            pos += n
        f32 = lambda a: wp.array(np.concatenate(a).astype(np.float32), dtype=wp.float32, device=device)
        self.kp, self.kd, self.qd, self.ff = f32(kp), f32(kd), f32(qd), f32(ff)
        self.qm, self.dqm, self.tm = f32(qm), f32(dqm), f32(tm)
        self.start = wp.array(np.array(starts, dtype=np.int32), dtype=wp.int32, device=device)
        self.len = wp.array(np.array(lens, dtype=np.int32), dtype=wp.int32, device=device)
        self.q0 = wp.array(np.array(q0s, dtype=np.float32), dtype=wp.float32, device=device)
        self.t0 = wp.array(np.array(t0s, dtype=np.float32), dtype=wp.float32, device=device)
        self.skip = wp.array(np.array(skips, dtype=np.int32), dtype=wp.int32, device=device)
        self.n_runs = len(runs)
        self.names = names
        self.total = pos
        self.device = device
        self.var = [float(np.var(np.concatenate(qm))), float(np.var(np.concatenate(dqm))), float(np.var(np.concatenate(tm)))]

    def evaluate(self, phys: np.ndarray, tmax: float = 12.0, save_pred: bool = False):
        """phys: [N,P] 物理参数；返回 cost[N,runs,3]（均方误差）及可选的预测。"""
        n = phys.shape[0]
        params = wp.array(phys.astype(np.float32), dtype=wp.float32, device=self.device)
        out = wp.zeros((n * self.n_runs, 3), dtype=wp.float32, device=self.device)
        size = self.total if save_pred else 1
        pq = wp.zeros(size, dtype=wp.float32, device=self.device)
        pdq = wp.zeros(size, dtype=wp.float32, device=self.device)
        pt = wp.zeros(size, dtype=wp.float32, device=self.device)
        wp.launch(sim_kernel, dim=n * self.n_runs, inputs=[params, self.kp, self.kd, self.qd, self.ff, self.qm, self.dqm, self.tm,
                                                           self.start, self.len, self.q0, self.t0, self.skip, self.n_runs, self.period, S_SUB, tmax,
                                                           out, pq, pdq, pt, 1 if save_pred else 0], device=self.device)
        c = out.numpy().reshape(n, self.n_runs, 3)
        if save_pred:
            return c, pq.numpy(), pdq.numpy(), pt.numpy()
        return c


def total_cost(c: np.ndarray, run_w: np.ndarray | None = None) -> np.ndarray:
    """cost[N,R,3] -> [N]：各通道按噪声尺度加权的均方误差之和。"""
    per = c[..., 0] * W_Q + c[..., 1] * W_DQ + c[..., 2] * W_T
    if run_w is not None:
        per = per * run_w[None, :]
    per = np.where(np.isfinite(per), per, 1e12)
    return per.sum(axis=1)


class CMAES:
    def __init__(self, x0, sigma, lam, seed=0):
        self.n = len(x0)
        self.mean = np.array(x0, dtype=np.float64)
        self.sigma = sigma
        self.lam = lam
        self.rng = np.random.default_rng(seed)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mu = mu
        self.mueff = 1 / np.sum(self.w ** 2)
        n = self.n
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.C = np.eye(n)
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.gen = 0

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        self.y = z @ (self.B * self.D).T
        return self.mean + self.sigma * self.y

    def tell(self, x, f):
        idx = np.argsort(f)[: self.mu]
        y_w = (self.y[idx] * self.w[:, None]).sum(axis=0)
        self.mean = self.mean + self.sigma * y_w
        invsqrtC_y = self.B @ ((self.B.T @ y_w) / self.D)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_y
        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + (math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w if hsig else 0)
        yk = self.y[idx]
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (0 if hsig else self.cc * (2 - self.cc)) * self.C)
                  + self.cmu * (yk.T * self.w) @ yk)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-3, 3.0))
        self.C = (self.C + self.C.T) / 2
        D2, B = np.linalg.eigh(self.C)
        D2 = np.clip(D2, 1e-12, None)
        self.D, self.B = np.sqrt(D2), B
        self.gen += 1


def fit(ds: Dataset, gens: int, lam: int, seed: int, x0_unit=None, verbose=True, run_w=None):
    best_x, best_f = None, np.inf
    rng = np.random.default_rng(seed)
    # CMA 在无约束空间 z 里搜索，u = sigmoid(z) 映射到 [0,1]，避免边界截断破坏步长自适应
    z0 = rng.normal(0.0, 0.8, P) if x0_unit is None else np.log(np.clip(x0_unit, 1e-3, 1 - 1e-3) / (1 - np.clip(x0_unit, 1e-3, 1 - 1e-3)))
    es = CMAES(z0, 1.0, lam, seed)
    t0 = time.time()
    hist = []
    for g in range(gens):
        zs = es.ask()
        us = 1.0 / (1.0 + np.exp(-zs))
        c = ds.evaluate(from_unit(us))
        f = total_cost(c, run_w)
        es.tell(zs, f)
        k = int(np.argmin(f))
        if f[k] < best_f:
            best_f, best_x = float(f[k]), us[k].copy()
        hist.append(best_f)
        if verbose and (g % 20 == 0 or g == gens - 1):
            print(f"  gen {g:4d}  best {best_f:.4g}  sigma {es.sigma:.3f}  {time.time() - t0:.0f}s", flush=True)
    return best_x, best_f, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", type=int, default=4)
    ap.add_argument("--data", type=str, default=None, help="data/raw/<日期> 目录")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--lam", type=int, default=2048)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exam", default="multisine")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest(args)
    names = {1: "FL_thigh", 2: "FL_calf", 3: "FR_thigh", 4: "FR_calf", 5: "RL_thigh", 6: "RL_calf", 7: "RR_thigh", 8: "RR_calf"}
    root = Path(args.data)
    def ok(d):
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return m.get("abort") is None and m.get("aligned")
    runs = sorted(d for d in root.iterdir() if f"_id_{names[args.motor]}_" in d.name and ok(d))
    fit_runs = [d for d in runs if args.exam not in d.name]
    exam_runs = [d for d in runs if args.exam in d.name]
    print(f"拟合集 {[d.name for d in fit_runs]}\n考试集 {[d.name for d in exam_runs]}")
    ds = Dataset(fit_runs, args.device)
    best = None
    for r in range(args.restarts):
        print(f"重启 {r}")
        x, f, _ = fit(ds, args.gens, args.lam, seed=r)
        if best is None or f < best[1]:
            best = (x, f)
    phys = from_unit(best[0][None, :])[0]
    # 近似最优解的范围：在最优点附近随机采样，保留代价不超过最优 1.1 倍的候选，给出各参数的 5%–95% 区间（可辨识性）
    rng = np.random.default_rng(123)
    zb = np.log(np.clip(best[0], 1e-3, 1 - 1e-3) / (1 - np.clip(best[0], 1e-3, 1 - 1e-3)))
    keep = []
    for _ in range(40):
        zs = zb + rng.normal(0, 0.35, (8192, P))
        ph = from_unit(1.0 / (1.0 + np.exp(-zs)))
        f = total_cost(ds.evaluate(ph))
        keep.append(ph[f <= best[1] * 1.1])
    keep = np.concatenate(keep) if keep else np.zeros((0, P))
    ranges = {n: [float(np.percentile(keep[:, j], 5)), float(np.percentile(keep[:, j], 95))] for j, n in enumerate(NAMES)} if len(keep) > 20 else {}
    out = {"near_optimal_count": int(len(keep)), "param_ranges_5_95": ranges,
           "motor": args.motor, "name": names[args.motor], "fit_cost": best[1], "params": dict(zip(NAMES, map(float, phys)))}
    base_c = total_cost(ds.evaluate(BASE[None, :]))[0]
    out["fit_cost_ideal_sim"] = float(base_c)
    cr = ds.evaluate(phys[None, :])[0]
    cb = ds.evaluate(BASE[None, :])[0]
    out["per_run"] = {n: {"fit": [float(x) for x in cr[i]], "ideal_sim": [float(x) for x in cb[i]]} for i, n in enumerate(ds.names)}
    out["rmse_note"] = "per_run 为 [q 均方误差 rad², dq rad²/s², 力矩 N²m²]；ideal_sim 为无延迟无滞后的标称仿真"
    if exam_runs:
        de = Dataset(exam_runs, args.device)
        out["exam_cost"] = float(total_cost(de.evaluate(phys[None, :]))[0])
        out["exam_cost_ideal_sim"] = float(total_cost(de.evaluate(BASE[None, :]))[0])
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"fit_{names[args.motor]}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


def selftest(args) -> int:
    """合成数据：用已知参数生成一次 chirp 与一次力矩阶跃，再拟合，看参数能否找回。"""
    import tempfile
    rng = np.random.default_rng(0)
    true = np.array([2.5, 11.0, 400.0, 0.02, 0.05, 0.03, 0.2, 1.0, 0.95, 1.2, 1.0, 0.004, 1.0])
    tmp = Path(tempfile.mkdtemp())
    rate = 500.0
    T = 1 / rate
    runs = []
    for name, kind in (("a_chirp", "chirp"), ("b_steps", "steps"), ("c_ramp", "ramp")):
        n = 12000
        kp = np.full(n, 28.0); kd = np.full(n, 2.0); qd = np.zeros(n); ff = np.zeros(n)
        t = np.arange(n) * T
        if kind == "chirp":
            f = 0.2 * np.exp(math.log(8 / 0.2) / 24 * t)
            qd = 0.06 * np.sin(np.cumsum(2 * np.pi * f * T))
        elif kind == "steps":
            kp[:] = 0; kd[:] = 0
            for k0 in range(500, n - 500, 600):
                ff[k0:k0 + int(rng.choice([10, 30, 75]))] = rng.choice([-0.6, -0.3, 0.3, 0.6]) + 0.2
            ff = np.where(ff == 0, 0.2, ff)
            kp[:500] = 28; kd[:500] = 2; ff[:500] = 0
        else:
            qd = 0.06 * np.sign(np.sin(2 * np.pi * t / 8)) * 0  # 占位
            qd = 0.06 * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * t / 6))
        d = tmp / f"20260101_000000_id_FR_calf_{name}_r500"
        d.mkdir()
        # 先用真参数仿真得到"测量"
        np.savetxt(d / "commands.csv", np.c_[np.arange(n), kp, kd, qd, ff], delimiter=",", header="cmd_index,kp,kd,q_des,tau_ff", comments="")
        np.savetxt(d / "feedback.csv", np.c_[np.arange(n), np.arange(n), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)], delimiter=",",
                   header="reply_index,t_host_ns,state,q,dq,tau", comments="")
        (d / "meta.json").write_text(json.dumps({"n_pre_replies": 0, "rate_hz": rate}), encoding="utf-8")
        runs.append(d)
    ds0 = Dataset(runs, args.device)
    c, pq, pdq, pt = ds0.evaluate(true[None, :], save_pred=True)
    # 把预测当作"测量"，加噪声与量化，写回文件
    pos = 0
    for d in runs:
        n = 12000
        qq = np.round((pq[pos:pos + n] + rng.normal(0, 3e-4, n)) / 3.8e-4) * 3.8e-4
        dd = np.round((pdq[pos:pos + n] + rng.normal(0, 0.02, n)) / 0.022) * 0.022
        tt = np.round((pt[pos:pos + n] + rng.normal(0, 0.01, n)) / 0.0098) * 0.0098
        np.savetxt(d / "feedback.csv", np.c_[np.arange(n), np.arange(n), np.ones(n), qq, dd, tt], delimiter=",",
                   header="reply_index,t_host_ns,state,q,dq,tau", comments="")
        pos += n
    ds = Dataset(runs, args.device)
    t0 = time.time()
    x, f, _ = fit(ds, args.gens, args.lam, seed=0)
    phys = from_unit(x[None, :])[0]
    print(f"用时 {time.time() - t0:.0f} s，拟合代价 {f:.4g}，真值代价 {total_cost(ds.evaluate(true[None, :]))[0]:.4g}")
    print("参数    真值      拟合")
    for nme, a, b in zip(NAMES, true, phys):
        print(f"{nme:9s} {a:9.4g} {b:9.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
