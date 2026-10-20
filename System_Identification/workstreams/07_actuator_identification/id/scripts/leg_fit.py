"""整机模型拟合（GPU，MuJoCo Warp，PACE 式）：在真实的 bennett_3 闭链模型里拟合执行器与关节参数。

与 fit_actuator.py 的区别：fit_actuator.py 把每个关节当作单自由度，闭链腿的耦合（被动关节、伙伴电机）没有进模型，
小腿的结果因此不可靠；本脚本直接在整机 MJCF 里仿真（底座固定，机器人悬空），八个关节同时处在模型里，
闭链约束由 MuJoCo 处理，所以小腿与大腿用同一套模型。

被拟合的量（参数向量，CMA-ES）：
  全局：delay_ms（命令到力矩的等效纯延迟）、kp_s、kd_s（增益比例）、tau_off（力矩读数偏置）；
  每个主动关节（8 个）：fric（库仑摩擦，MuJoCo frictionloss，约束求解）、visc（粘性阻尼）、arm（附加电枢惯量，
  在折算转子惯量 4.3e-3 之上）。
执行器：tau = kp_s*kp*(q_des - q) - kd_s*kd*dq + tau_ff，ZOH 命令延迟（分数周期按线性混合），限幅 ±12 N·m；
被动关节保持 MJCF 缺省。拟合数据：八关节同时多正弦 seed 1，加各关节的 tau_steps 与 chirp（伙伴关节的姿态取八关节数据的起点）；
考试数据：八关节 seed 2、各关节的 multisine 与 ramp。

零位偏差（07-T11）：--zero_offsets z1,...,z8（电机坐标，与 analyze_stair.py 同约定 q_sim = s*(q_m - z)）把全部命令、反馈、
起点统一平移后再拟合，检验静态阶梯估计的零位偏差能否降低拟合代价；--out_name 另存结果，不覆盖 leg_fit.json。

用法（GPU 环境）：
    python leg_fit.py --xml <bennett_3.xml> --data <id/data/raw/日期> --out <id/results> [--gens 150 --cands 512]
        [--zero_offsets -0.20,0,-0.07,0,-0.24,0,-0.03,0 --out_name leg_fit_zero.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_actuator as fa  # noqa: E402  (CMAES、read_csv)

wp.config.quiet = True
NAMES = ["FL_thigh", "FL_calf", "FR_thigh", "FR_calf", "RL_thigh", "RL_calf", "RR_thigh", "RR_calf"]
S2M = np.array([1, 1, 1, -1, 1, 1, 1, -1], dtype=np.float64)   # 仿真关节角 = S2M * 电机角
MOT_ID = {n: i + 1 for i, n in enumerate(NAMES)}
ROTOR = 0.0043                     # 折算转子惯量 kg·m²（Inertia 寄存器 1.2e-4 × 6²）
TMAX = 12.0
W_Q, W_DQ, W_T = 1 / 0.005 ** 2, 1 / 0.1 ** 2, 1 / 0.05 ** 2

ZERO_M = np.zeros(8)               # 电机坐标零位偏差 z，由 --zero_offsets 设置
PNAMES = ["delay_ms", "kp_s", "kd_s", "tau_off"] + [f"fric_{n}" for n in NAMES] + [f"visc_{n}" for n in NAMES] + [f"arm_{n}" for n in NAMES]
LO = np.array([0.0, 0.7, 0.5, -0.08] + [0.0] * 8 + [0.0] * 8 + [0.0] * 8)
HI = np.array([16.0, 1.3, 1.5, 0.02] + [0.8] * 8 + [0.3] * 8 + [0.06] * 8)
NP = len(PNAMES)
BASE = np.array([0.0, 1.0, 1.0, 0.0] + [0.05] * 8 + [0.001] * 8 + [0.0] * 8)   # 训练缺省：延迟 0、标称增益、frictionloss 0.05、damping 0.001


def phys(z):
    return LO + (HI - LO) / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------- 模型与数据
def load_model(xml: Path):
    src = xml.read_text(encoding="utf-8")
    meshdir = (xml.parent / "meshes").as_posix() + "/"
    src = src.replace('meshdir="meshes/"', f'meshdir="{meshdir}"')
    src = src.replace('<joint name="floating_base_joint" type="free"/>', "")
    src = src.replace('<flag energy="disable" contact="enable"/>', '<flag energy="disable" contact="disable"/>')
    m = mujoco.MjModel.from_xml_string(src)
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_gainprm[:, 1:] = 0
    m.actuator_biasprm[:, :] = 0
    m.actuator_ctrllimited[:] = 0
    m.actuator_forcelimited[:] = 0
    return m


def settle_pose(m: mujoco.MjModel, q_active_sim: np.ndarray) -> np.ndarray:
    """给定 8 个主动关节角，CPU 里用位置保持把被动关节沉淀到闭链一致的姿态，返回全部 qpos。"""
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in NAMES]
    dadr = [m.jnt_dofadr[m.joint(n).id] for n in NAMES]
    for i, a in enumerate(adr):
        d.qpos[a] = q_active_sim[i]
    damp0 = m.dof_damping.copy()
    m.dof_damping[:] = 0.5
    for _ in range(3000):
        d.qfrc_applied[:] = 0
        for i, (a, da) in enumerate(zip(adr, dadr)):
            d.qfrc_applied[da] = 60.0 * (q_active_sim[i] - d.qpos[a]) - 3.0 * d.qvel[da]
        mujoco.mj_step(m, d)
    m.dof_damping[:] = damp0
    return d.qpos.copy()


class Run:
    def __init__(self, d: Path, hang_q0_motor: np.ndarray):
        self.name = d.name
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.rate = float(meta["rate_hz"])
        cmd = fa.read_csv(d / "commands.csv")
        fb = fa.read_csv(d / "feedback.csv")
        n_pre = int(meta.get("n_pre_replies", meta.get("n_pre_replies_all", 0)))
        if meta.get("protocol") == "ms8":
            T = int(max(meta["n_cmds_per_motor"].values()))
            self.kp = np.zeros((T, 8)); self.kd = np.zeros((T, 8)); self.qd = np.zeros((T, 8)); self.ff = np.zeros((T, 8))
            self.qm = np.zeros((T, 8)); self.dqm = np.zeros((T, 8)); self.tm = np.zeros((T, 8)); self.mask = np.ones(8, bool)
            q0m = np.array([meta["q0"][str(i + 1)] for i in range(8)])
            for i in range(8):
                cm = cmd["motor"] == i + 1
                t = int(cm.sum())
                self.kp[:t, i] = cmd["kp"][cm]; self.kd[:t, i] = cmd["kd"][cm]; self.qd[:t, i] = cmd["q_des"][cm]; self.ff[:t, i] = cmd["tau_ff"][cm]
                keep = np.where((fb["motor"] == i + 1) & (np.arange(len(fb["motor"])) >= n_pre) & (fb["state"] == 1))[0][:t]
                self.qm[:len(keep), i] = fb["q"][keep]; self.dqm[:len(keep), i] = fb["dq"][keep]; self.tm[:len(keep), i] = fb["tau"][keep]
            self.T = T
        else:
            j = NAMES.index(meta["name"])
            keep = np.where((np.arange(len(fb["state"])) >= n_pre) & (fb["state"] == 1))[0]
            T = min(len(keep), len(cmd["kp"]))
            self.T = T
            self.kp = np.zeros((T, 8)); self.kd = np.zeros((T, 8)); self.qd = np.zeros((T, 8)); self.ff = np.zeros((T, 8))
            self.qm = np.zeros((T, 8)); self.dqm = np.zeros((T, 8)); self.tm = np.zeros((T, 8)); self.mask = np.zeros(8, bool)
            self.mask[j] = True
            self.kp[:, j] = cmd["kp"][:T]; self.kd[:, j] = cmd["kd"][:T]; self.qd[:, j] = cmd["q_des"][:T]; self.ff[:, j] = cmd["tau_ff"][:T]
            self.qm[:, j] = fb["q"][keep][:T]; self.dqm[:, j] = fb["dq"][keep][:T]; self.tm[:, j] = fb["tau"][keep][:T]
            q0m = hang_q0_motor.copy()
            q0m[j] = self.qm[0, j]
        # 电机坐标 -> 仿真坐标
        for a in (self.qd, self.qm, self.dqm, self.tm, self.ff):
            a *= S2M[None, :]
        self.q0_sim = S2M * q0m
        shift = S2M * ZERO_M
        for a in (self.qd, self.qm):
            a -= shift[None, :]
        self.q0_sim = self.q0_sim - shift
        self.skip = int(1.0 * self.rate)   # 前 1 s 保持阶段不计入


def build_runs(root: Path, which: str):
    ms8 = sorted(d for d in root.iterdir() if "_idall_" in d.name and json.loads((d / "meta.json").read_text(encoding="utf-8")).get("abort") is None)
    hang = np.array([json.loads((ms8[0] / "meta.json").read_text(encoding="utf-8"))["q0"][str(i + 1)] for i in range(8)])
    fit_tags = ("tau_steps", "chirp")
    exam_tags = ("multisine", "ramp")
    runs = []
    if which == "fit":
        runs.append(Run(ms8[0], hang))
        tags = fit_tags
    else:
        runs.append(Run(ms8[1], hang))
        tags = exam_tags
    for n in NAMES:
        for tag in tags:
            ds = [d for d in sorted(root.iterdir()) if f"_id_{n}_{tag}_" in d.name and json.loads((d / "meta.json").read_text(encoding="utf-8")).get("abort") is None]
            if ds:
                runs.append(Run(ds[-1], hang))
    return runs


# ----------------------------------------------------------------------------- Warp 内核
@wp.kernel
def ctrl_kernel(P: wp.array2d(dtype=wp.float32), sidx: wp.array(dtype=wp.int32),
                kp_a: wp.array(dtype=wp.float32), kd_a: wp.array(dtype=wp.float32), qd_a: wp.array(dtype=wp.float32), ff_a: wp.array(dtype=wp.float32),
                run_start: wp.array(dtype=wp.int32), run_len: wp.array(dtype=wp.int32),
                qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
                qadr: wp.array(dtype=wp.int32), dadr: wp.array(dtype=wp.int32),
                ctrl: wp.array2d(dtype=wp.float32), tau_rec: wp.array2d(dtype=wp.float32), n_runs: int, period_ms: float, tmax: float):
    tid = wp.tid()
    w = tid / 8
    j = tid - w * 8
    cand = w / n_runs
    r = w - cand * n_runs
    s = sidx[0]
    delay = P[cand, 0]
    kp_s = P[cand, 1]
    kd_s = P[cand, 2]
    x = float(s) - delay / period_ms
    if x < 0.0:
        x = 0.0
    ia = int(wp.floor(x))
    wgt = x - float(ia)
    n = run_len[r]
    if ia > n - 1:
        ia = n - 1
    ib = ia + 1
    if ib > n - 1:
        ib = n - 1
    base = run_start[r]
    q = qpos[w, qadr[j]]
    dq = qvel[w, dadr[j]]
    ta = kp_s * kp_a[(base + ia) * 8 + j] * (qd_a[(base + ia) * 8 + j] - q) - kd_s * kd_a[(base + ia) * 8 + j] * dq + ff_a[(base + ia) * 8 + j]
    tb = kp_s * kp_a[(base + ib) * 8 + j] * (qd_a[(base + ib) * 8 + j] - q) - kd_s * kd_a[(base + ib) * 8 + j] * dq + ff_a[(base + ib) * 8 + j]
    tau = (1.0 - wgt) * ta + wgt * tb
    tau = wp.clamp(tau, -tmax, tmax)
    ctrl[w, j] = tau
    tau_rec[w, j] = tau


@wp.kernel
def err_kernel(P: wp.array2d(dtype=wp.float32), sidx: wp.array(dtype=wp.int32),
               qm_a: wp.array(dtype=wp.float32), dqm_a: wp.array(dtype=wp.float32), tm_a: wp.array(dtype=wp.float32),
               run_start: wp.array(dtype=wp.int32), run_len: wp.array(dtype=wp.int32), run_skip: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.int32),
               qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32), qadr: wp.array(dtype=wp.int32), dadr: wp.array(dtype=wp.int32),
               tau_rec: wp.array2d(dtype=wp.float32), n_runs: int, err: wp.array2d(dtype=wp.float32)):
    tid = wp.tid()
    w = tid / 8
    j = tid - w * 8
    cand = w / n_runs
    r = w - cand * n_runs
    s = sidx[0]
    if s < run_len[r] and s >= run_skip[r] and mask[r * 8 + j] != 0:
        idx = (run_start[r] + s) * 8 + j
        eq = qpos[w, qadr[j]] - qm_a[idx]
        edq = qvel[w, dadr[j]] - dqm_a[idx]
        et = tau_rec[w, j] + P[cand, 3] - tm_a[idx]
        wp.atomic_add(err, w, 0, eq * eq)
        wp.atomic_add(err, w, 1, edq * edq)
        wp.atomic_add(err, w, 2, et * et)
        wp.atomic_add(err, w, 3, 1.0)


@wp.kernel
def inc_kernel(sidx: wp.array(dtype=wp.int32)):
    sidx[0] = sidx[0] + 1


# ----------------------------------------------------------------------------- 评估器
class Evaluator:
    def __init__(self, m_cpu: mujoco.MjModel, runs: list[Run], n_cand: int, device="cuda:0"):
        self.m_cpu, self.runs, self.R, self.C = m_cpu, runs, len(runs), n_cand
        self.W = self.R * n_cand
        self.device = device
        self.period_ms = 1000.0 / runs[0].rate
        assert abs(m_cpu.opt.timestep * 1000 - self.period_ms) < 1e-6, "MJCF 时间步必须等于数据的发令周期"
        self.Tmax = max(r.T for r in runs)
        cat = lambda key: np.concatenate([getattr(r, key) for r in runs], axis=0).astype(np.float32).ravel()
        with wp.ScopedDevice(device):
            self.kp, self.kd, self.qd, self.ff = (wp.array(cat(k), dtype=wp.float32) for k in ("kp", "kd", "qd", "ff"))
            self.qm, self.dqm, self.tm = (wp.array(cat(k), dtype=wp.float32) for k in ("qm", "dqm", "tm"))
            starts = np.cumsum([0] + [r.T for r in runs[:-1]]).astype(np.int32)
            self.start = wp.array(starts, dtype=wp.int32); self.len = wp.array(np.array([r.T for r in runs], np.int32), dtype=wp.int32)
            self.skip = wp.array(np.array([r.skip for r in runs], np.int32), dtype=wp.int32)
            self.mask = wp.array(np.concatenate([r.mask.astype(np.int32) for r in runs]), dtype=wp.int32)
            m = m_cpu
            self.qadr = wp.array(np.array([m.jnt_qposadr[m.joint(n).id] for n in NAMES], np.int32), dtype=wp.int32)
            self.dadr = wp.array(np.array([m.jnt_dofadr[m.joint(n).id] for n in NAMES], np.int32), dtype=wp.int32)
            self.dadr_np = np.array([m.jnt_dofadr[m.joint(n).id] for n in NAMES])
            # 初始姿态：每个运行一个沉淀好的全姿态
            self.q_init = np.stack([settle_pose(m_cpu, r.q0_sim) for r in runs])        # [R, nq]
            self.mw = mjw.put_model(m)
            nv = m.nv
            self.arm0 = np.tile(m.dof_armature, (self.W, 1)).astype(np.float32)
            self.fr0 = np.tile(m.dof_frictionloss, (self.W, 1)).astype(np.float32)
            self.dp0 = np.tile(m.dof_damping, (self.W, 1)).astype(np.float32)
            self.mw.dof_armature = wp.array(self.arm0, dtype=wp.float32)
            self.mw.dof_frictionloss = wp.array(self.fr0, dtype=wp.float32)
            self.mw.dof_damping = wp.array(self.dp0, dtype=wp.float32)
            self.dw = mjw.make_data(m, nworld=self.W, nconmax=8, njmax=64)
            self.P = wp.zeros((n_cand, NP), dtype=wp.float32)
            self.sidx = wp.zeros(1, dtype=wp.int32)
            self.tau_rec = wp.zeros((self.W, 8), dtype=wp.float32)
            self.err = wp.zeros((self.W, 4), dtype=wp.float32)
            self.q_init_w = np.tile(self.q_init, (n_cand, 1)).astype(np.float32)   # world = cand*R + r
            self._reset()
            for _ in range(3):
                self._step()
            wp.synchronize()
            with wp.ScopedCapture() as cap:
                self._step()
            self.graph = cap.graph

    def _step(self):
        wp.launch(ctrl_kernel, dim=self.W * 8, inputs=[self.P, self.sidx, self.kp, self.kd, self.qd, self.ff, self.start, self.len, self.dw.qpos, self.dw.qvel,
                                                      self.qadr, self.dadr, self.dw.ctrl, self.tau_rec, self.R, self.period_ms, TMAX])
        mjw.step(self.mw, self.dw)
        wp.launch(err_kernel, dim=self.W * 8, inputs=[self.P, self.sidx, self.qm, self.dqm, self.tm, self.start, self.len, self.skip, self.mask,
                                                      self.dw.qpos, self.dw.qvel, self.qadr, self.dadr, self.tau_rec, self.R, self.err])
        wp.launch(inc_kernel, dim=1, inputs=[self.sidx])

    def _reset(self):
        self.dw.qpos.assign(self.q_init_w)
        self.dw.qvel.zero_()
        self.dw.ctrl.zero_()
        self.dw.qacc_warmstart.zero_()
        self.err.zero_()
        self.sidx.zero_()

    def set_params(self, Pnp: np.ndarray):
        """Pnp: [C, NP] 物理参数。更新每个世界的 armature/frictionloss/damping（原地写入，保持 CUDA 图有效）。"""
        arm = self.arm0.copy().reshape(self.C, self.R, -1)
        fr = self.fr0.copy().reshape(self.C, self.R, -1)
        dp = self.dp0.copy().reshape(self.C, self.R, -1)
        for k, da in enumerate(self.dadr_np):
            fr[:, :, da] = Pnp[:, None, 4 + k]
            dp[:, :, da] = Pnp[:, None, 12 + k]
            arm[:, :, da] = ROTOR + Pnp[:, None, 20 + k]
        self.mw.dof_armature.assign(arm.reshape(self.W, -1))
        self.mw.dof_frictionloss.assign(fr.reshape(self.W, -1))
        self.mw.dof_damping.assign(dp.reshape(self.W, -1))
        self.P.assign(Pnp.astype(np.float32))

    def evaluate(self, Pnp: np.ndarray):
        """返回 cost[C]（各运行相加）与 per_run[C,R,3]（均方误差 q,dq,tau）。"""
        self.set_params(Pnp)
        with wp.ScopedDevice(self.device):
            self._reset()
            for _ in range(self.Tmax):
                wp.capture_launch(self.graph)
            wp.synchronize()
            e = self.err.numpy().reshape(self.C, self.R, 4)
        cnt = np.maximum(e[..., 3], 1.0)
        mse = e[..., :3] / cnt[..., None]
        per = mse[..., 0] * W_Q + mse[..., 1] * W_DQ + mse[..., 2] * W_T
        per = np.where(np.isfinite(per), per, 1e9)
        return per.sum(axis=1), mse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--cands", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--zero_offsets", default=None, help="8 个电机坐标零位偏差，逗号分隔")
    ap.add_argument("--out_name", default="leg_fit.json")
    args = ap.parse_args()
    if args.zero_offsets:
        global ZERO_M
        ZERO_M = np.array([float(v) for v in args.zero_offsets.split(",")])
        assert ZERO_M.shape == (8,), "--zero_offsets 需要 8 个值"
        print("零位偏差（电机坐标）", ZERO_M.tolist())
    m = load_model(Path(args.xml))
    root = Path(args.data)
    fit_runs = build_runs(root, "fit")
    exam_runs = build_runs(root, "exam")
    print(f"拟合集 {len(fit_runs)} 个运行，考试集 {len(exam_runs)} 个运行；Tmax {max(r.T for r in fit_runs)}")
    ev = Evaluator(m, fit_runs, args.cands, args.device)
    rng = np.random.default_rng(args.seed)
    z0 = rng.normal(0, 0.5, NP)
    # 以接近 07-K21 的值作为起点：延迟 8 ms，增益 1，摩擦 0.25，粘性小
    start = np.array([8.0, 1.0, 1.0, -0.035] + [0.25] * 8 + [0.01] * 8 + [0.01] * 8)
    u0 = np.clip((start - LO) / (HI - LO), 1e-3, 1 - 1e-3)
    z0 = np.log(u0 / (1 - u0))
    es = fa.CMAES(z0, 0.8, args.cands, args.seed)
    best_f, best_p = np.inf, None
    t0 = time.time()
    # 基线（训练缺省）作为参照
    base_cost, base_mse = ev.evaluate(np.tile(BASE, (args.cands, 1)))
    print(f"基线（训练缺省模型）拟合集代价 {base_cost[0]:.4g}")
    for g in range(args.gens):
        zs = es.ask()
        ps = np.stack([phys(z) for z in zs])
        f, _ = ev.evaluate(ps)
        es.tell(zs, f)
        k = int(np.argmin(f))
        if f[k] < best_f:
            best_f, best_p = float(f[k]), ps[k].copy()
        if g % 10 == 0 or g == args.gens - 1:
            print(f"  gen {g:4d} best {best_f:.4g} sigma {es.sigma:.3f} {time.time() - t0:.0f}s", flush=True)
    # 考试
    ex = Evaluator(m, exam_runs, 2, args.device)
    pair = np.stack([best_p, BASE])
    cost, mse = ex.evaluate(pair)
    res = {"fit_cost": best_f, "fit_cost_baseline": float(base_cost[0]), "params": dict(zip(PNAMES, map(float, best_p))), "baseline": dict(zip(PNAMES, map(float, BASE))),
           "exam_cost": float(cost[0]), "exam_cost_baseline": float(cost[1]), "exam_runs": [r.name for r in exam_runs],
           "exam_rmse_fit": {r.name: [float(np.sqrt(mse[0, i, 0]) * 1e3), float(np.sqrt(mse[0, i, 1])), float(np.sqrt(mse[0, i, 2]))] for i, r in enumerate(exam_runs)},
           "exam_rmse_baseline": {r.name: [float(np.sqrt(mse[1, i, 0]) * 1e3), float(np.sqrt(mse[1, i, 1])), float(np.sqrt(mse[1, i, 2]))] for i, r in enumerate(exam_runs)}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    res["zero_offsets_motor"] = ZERO_M.tolist()
    (Path(args.out) / args.out_name).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("fit_cost", "fit_cost_baseline", "exam_cost", "exam_cost_baseline")}, ensure_ascii=False))
    for k, v in res["params"].items():
        print(f"  {k:14s} {v:9.4f}   (基线 {res['baseline'][k]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
