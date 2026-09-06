# -*- coding: utf-8 -*-
"""DM-J8006 单关节位置阶跃实验 —— 反馈确认 / 编码器映射 / kp/kd 关节端等效刚度.

只驱动【一个】测试电机,其余电机一概不下令(悬空机器人,不用软保持)。
仅当该电机反馈"新鲜"(age<0.1s 且 |pos|<3)才驱动,否则直接拒绝,避免给失联电机下指令。

用法(电机已上电,USB-CANFD 已插好):
    cd E:\\HuanCun\\Desktop\\u2canfd
    # 只读诊断(不下令,看哪些电机在回传):
    python sysid_step.py --diag
    # 正式阶跃(默认 FL_thigh canid1, +0.30 rad 朝向中性):
    python sysid_step.py --canid 1 --channel 0 --step 0.30

输出:sysid_out/step_<时间戳>.csv + step_<时间戳>.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import damiao
from damiao import Control_Mode, DM_Motor_Type, DmActData, Motor_Control
from dmcan import dmcan_device_type

SN = "D6977F56F86C64B77B316E7154FA6DF3"
# canid -> (名称, channel)
MOTORS = {
    1: ("FL_thigh", 0), 2: ("FL_calf", 0),
    3: ("FR_thigh", 1), 4: ("FR_calf", 1),
    5: ("RL_thigh", 0), 6: ("RL_calf", 0),
    7: ("RR_thigh", 1), 8: ("RR_calf", 1),
}
FRESH_AGE = 0.1   # 反馈帧龄 < 0.1s 视为新鲜
POS_BOUND = 3.0   # |pos| > 3 rad 判定为垃圾/失联


def build_init_data():
    init = []
    for canid, (_name, ch) in MOTORS.items():
        init.append(DmActData(
            motorType=DM_Motor_Type.DM8006,
            mode=Control_Mode.MIT_MODE,
            can_id=canid,
            mst_id=0x10 + canid,
            channel=ch,
        ))
    return init


def sample(control, canid):
    name, ch = MOTORS[canid]
    m = control.getMotor(ch, canid)
    pos = m.Get_Position()
    age = m.getTimeInterval()
    return dict(canid=canid, name=name, pos=pos, vel=m.Get_Velocity(),
                tau=m.Get_tau(), err=m.Get_err(), age=age,
                healthy=(age < FRESH_AGE and abs(pos) < POS_BOUND))


def fmt_rows(rows):
    lines = ["canid  name      pos(rad)    vel        tau(Nm)   err   age      ok"]
    for r in rows:
        lines.append(f"{r['canid']:>5}  {r['name']:<10} {r['pos']:>+9.5f} "
                     f"{r['vel'] if isinstance(r['vel'], float) else '?':>9} "
                     f"{r['tau'] if isinstance(r['tau'], float) else '?':>8} "
                     f"{r['err']}  {r['age'] if isinstance(r['age'], float) else '?':>6.4f}  "
                     f"{'YES' if r['healthy'] else 'NO'}")
    return "\n".join(lines)


def save_csv(path, recs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)


def make_plots(path_png, recs, kp_eff, dp, step):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["t"] for r in recs])
    qc = np.array([r["q_cmd"] for r in recs])
    qr = np.array([r["q_real"] for r in recs])
    dq = np.array([r["dq_real"] for r in recs])
    tau = np.array([r["tau"] for r in recs])

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Step response  step={step:+.3f}rad  DP={dp:+.4f}rad  "
                 f"kp_static={kp_eff:+.2f} Nm/rad", fontsize=13)

    ax[0, 0].plot(t, np.degrees(qc), "--", label="q_cmd (deg)")
    ax[0, 0].plot(t, np.degrees(qr), label="q_real (deg)")
    ax[0, 0].set_xlabel("t (s)"); ax[0, 0].set_ylabel("deg")
    ax[0, 0].legend(); ax[0, 0].grid(True); ax[0, 0].set_title("Position (target vs actual)")

    ax[0, 1].plot(t, tau, color="tab:red")
    ax[0, 1].set_xlabel("t (s)"); ax[0, 1].set_ylabel("tau (Nm)")
    ax[0, 1].grid(True); ax[0, 1].set_title("Torque")

    ax[1, 0].plot(qc, qr, ".", ms=1.5)
    ax[1, 0].set_xlabel("q_cmd (rad)"); ax[1, 0].set_ylabel("q_real (rad)")
    ax[1, 0].grid(True); ax[1, 0].set_title("q_real vs q_cmd (slope ~1 => 1:1 in motor units)")

    err = qc - qr
    mask = np.abs(err) > 1e-4
    ax[1, 1].scatter(err[mask], tau[mask], s=5)
    ax[1, 1].set_xlabel("(q_cmd - q_real) rad"); ax[1, 1].set_ylabel("tau (Nm)")
    ax[1, 1].grid(True); ax[1, 1].set_title("Static stiffness kp_static = tau/err (steady)")
    ax[1, 1].text(0.02, 0.95, f"kp_static={kp_eff:+.2f}", transform=ax[1, 1].transAxes,
                  va="top", color="tab:red")

    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canid", type=int, default=1, choices=sorted(MOTORS), help="测试电机 canid(默认1=FL_thigh)")
    ap.add_argument("--channel", type=int, default=None, help="默认按 MOTORS 映射自动推断")
    ap.add_argument("--step", type=float, default=0.30, help="位置阶跃幅值(rad),默认 0.30(~17°)")
    ap.add_argument("--kp", type=float, default=28.0, help="MIT kp,默认 28")
    ap.add_argument("--kd", type=float, default=2.0, help="MIT kd,默认 2")
    ap.add_argument("--ramp_s", type=float, default=0.6, help="k=0->kp 爬升时间")
    ap.add_argument("--pre_s", type=float, default=0.8, help="阶跃前静止采样")
    ap.add_argument("--post_s", type=float, default=2.0, help="阶跃后保持")
    ap.add_argument("--diag", action="store_true", help="只读诊断,不下令")
    ap.add_argument("--diag_s", type=float, default=2.5, help="诊断采样时长")
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sysid_out"))
    args = ap.parse_args()

    ch = args.channel if args.channel is not None else MOTORS[args.canid][1]
    name = MOTORS[args.canid][0]
    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.outdir, f"step_{stamp}.csv")
    png_path = os.path.join(args.outdir, f"step_{stamp}.png")

    print(f"[+] 测试电机 id={args.canid} ({name}) channel={ch}  step={args.step} rad  kp={args.kp} kd={args.kd}")
    print(f"[+] 输出目录: {args.outdir}")

    control = None
    try:
        control = Motor_Control(1_000_000, 5_000_000, SN, build_init_data(),
                                device_type=dmcan_device_type.USB2CANFD_DUAL)
        print("\n[当前电机状态]")
        rows = [sample(control, c) for c in sorted(MOTORS)]
        print(fmt_rows(rows))

        if args.diag:
            # 反复采样,看哪些电机持续新鲜
            print(f"\n[诊断] 连续采样 {args.diag_s}s,统计每电机 pos 波动 / 帧龄 ...")
            t0 = time.perf_counter()
            buckets = {c: [] for c in sorted(MOTORS)}
            while time.perf_counter() - t0 < args.diag_s:
                for c in sorted(MOTORS):
                    s = sample(control, c)
                    buckets[c].append(s["pos"])
                time.sleep(0.005)
            print("\ncanid  name      min_pos      max_pos      span       mean_age_ok")
            for c in sorted(MOTORS):
                a = np.array(buckets[c])
                ok = sum(1 for _ in buckets[c])
                print(f"{c:>5}  {MOTORS[c][0]:<10} {a.min():>+9.5f} {a.max():>+9.5f} {a.max()-a.min():>8.5f}  {ok:>5}")
            return

        test = sample(control, args.canid)
        if not test["healthy"]:
            print(f"\n[拒绝驱动] {name}(canid{args.canid}) 反馈不新鲜/越界: "
                  f"age={test['age']:.4f}s pos={test['pos']:+.4f}. 请检查该电机上电/接线。")
            return

        p0 = test["pos"]
        # 朝向关节中性(0),避免顶限位
        toward = 1.0 if p0 <= 0 else -1.0
        lo, hi = (-0.80, 0.80) if "thigh" in name else (-0.90, 0.55)
        q_target = min(max(p0 + toward * abs(args.step), lo), hi)
        print(f"\n[驱动] p0={p0:+.4f} rad  方向{' +' if toward>0 else ' -'}{abs(args.step):.3f} -> target={q_target:+.4f} rad")

        motor = control.getMotor(ch, args.canid)
        recs = []
        start = time.perf_counter()

        # 爬升自持(kp 0->kp,q=p0)
        n_ramp = int(args.ramp_s * 1000)
        for i in range(n_ramp):
            frac = (i + 1) / n_ramp
            control.control_mit(motor, args.kp * frac, args.kd, p0, 0.0, 0.0)
            time.sleep(1.0 / 1000.0)

        # 阶跃前静止
        n_pre = int(args.pre_s * 1000)
        for _ in range(n_pre):
            recs.append({"t": time.perf_counter() - start, "q_cmd": p0,
                         "q_real": motor.Get_Position(), "dq_real": motor.Get_Velocity(),
                         "tau": motor.Get_tau(), "err": motor.Get_err()})
            control.control_mit(motor, args.kp, args.kd, p0, 0.0, 0.0)
            time.sleep(1.0 / 1000.0)

        # 阶跃
        n_post = int(args.post_s * 1000)
        for _ in range(n_post):
            recs.append({"t": time.perf_counter() - start, "q_cmd": q_target,
                         "q_real": motor.Get_Position(), "dq_real": motor.Get_Velocity(),
                         "tau": motor.Get_tau(), "err": motor.Get_err()})
            control.control_mit(motor, args.kp, args.kd, q_target, 0.0, 0.0)
            time.sleep(1.0 / 1000.0)

        save_csv(csv_path, recs)

        qr = np.array([r["q_real"] for r in recs])
        tau = np.array([r["tau"] for r in recs])
        err_arr = np.array([r["q_cmd"] - r["q_real"] for r in recs])
        q_before = qr[:n_pre].mean()
        q_after = qr[n_pre + int(0.8 * n_post):].mean() if n_post > 0 else qr[-1]
        dp = q_after - q_before
        # 静态刚度:用阶跃后稳定段(后 25%)的 tau/err。切勿用全程瞬态回归(混入阻尼 torque = kp*err + kd*dq,数值会失真)
        st = int(0.75 * len(recs))
        tau_st, err_st = tau[st:].mean(), err_arr[st:].mean()
        kp_eff = tau_st / err_st if abs(err_st) > 1e-5 else float("nan")

        print("\n[结果摘要]")
        print(f"  阶跃前 q_real均值 = {q_before:+.5f} rad")
        print(f"  阶跃后 q_real均值 = {q_after:+.5f} rad")
        print(f"  实际位移 Δ = {dp:+.5f} rad   (指令 {q_target - p0:+.5f})  比值 = "
              f"{dp / (q_target - p0) if q_target != p0 else float('nan'):.3f}")
        print(f"  稳定段 err={err_st:+.5f} rad   tau={tau_st:+.4f} Nm")
        print(f"  静态刚度 kp_static = tau/err = {kp_eff:+.2f} Nm/rad  (对比 sim kp=28;只匹配静态,阻尼 kd 另测)")
        print(f"  已保存: {csv_path}")
        print(f"  已保存: {png_path}")

        make_plots(png_path, recs, kp_eff, dp, q_target - p0)
        # 回到起始位姿并轻保持(悬空腿,别让它自由坠落;也不用 disable_all,避免卡死)
        control.control_mit(motor, args.kp, args.kd, p0, 0.0, 0.0)
        time.sleep(0.2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: hardware exception: {e}", file=sys.stderr)
    finally:
        if control is not None:
            try:
                control.close()
            except Exception:
                pass  # Windows 下 close() 偶发 access violation,忽略
    print("done.")


if __name__ == "__main__":
    main()
