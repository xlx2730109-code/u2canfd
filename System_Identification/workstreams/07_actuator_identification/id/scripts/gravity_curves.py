"""MuJoCo 整机模型的重力力矩曲线：某个主动关节固定在给定角度（高增益 PD 保持），伙伴关节自由（零力矩），
其余腿自由，底座固定（悬空），摩擦置零；读该关节的保持力矩。用于与 static_stair.py 的实测对照，拟合零位偏差。
单关节固定、伙伴自由时静力学问题是确定的（结果与 PD 增益无关，kp 100 与 400 一致），不同于两个关节同时固定的情形（07-K24）。

用法：python gravity_curves.py --xml <bennett_3.xml> --out <id/results>
"""
import argparse, json, time
from pathlib import Path
import mujoco
import numpy as np

NAMES = ["FL_thigh", "FL_calf", "FR_thigh", "FR_calf", "RL_thigh", "RL_calf", "RR_thigh", "RR_calf"]


def load(xml: Path):
    src = xml.read_text(encoding="utf-8")
    src = src.replace('meshdir="meshes/"', f'meshdir="{(xml.parent / "meshes").as_posix()}/"')
    src = src.replace('<joint name="floating_base_joint" type="free"/>', "")
    src = src.replace('<flag energy="disable" contact="enable"/>', '<flag energy="disable" contact="disable"/>')
    m = mujoco.MjModel.from_xml_string(src)
    m.actuator_gainprm[:, 0] = 0
    m.actuator_biasprm[:, :] = 0
    m.opt.timestep = 0.001
    m.dof_damping[:] = 0.3
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kp", type=float, default=300.0)
    ap.add_argument("--span", type=float, default=0.6)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    m = load(Path(args.xml))
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in NAMES]
    dadr = [m.jnt_dofadr[m.joint(n).id] for n in NAMES]
    for da in dadr:
        m.dof_frictionloss[da] = 0.0
    grid = np.round(np.arange(-args.span, args.span + 1e-9, args.step), 4)
    out = {"grid_q_sim": grid.tolist(), "kp": args.kp, "curves": {}}
    t0 = time.time()
    for j, n in enumerate(NAMES):
        vals = []
        for q in grid:
            mujoco.mj_resetData(m, d)
            d.qpos[adr[j]] = q
            for _ in range(8000):
                d.qfrc_applied[:] = 0
                d.qfrc_applied[dadr[j]] = args.kp * (q - d.qpos[adr[j]]) - 1.0 * d.qvel[dadr[j]]
                mujoco.mj_step(m, d)
            vals.append(float(args.kp * (q - d.qpos[adr[j]])))
        out["curves"][n] = vals
        print(n, [round(v, 3) for v in vals[::4]], f"{time.time() - t0:.0f}s", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "gravity_curves.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
