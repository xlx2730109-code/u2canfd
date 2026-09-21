# -*- coding: utf-8 -*-
"""八个关节批量辨识 —— 把 FL_thigh 上跑通的三个实验复制到全部 8 个关节.

它自己不碰 CAN, 只是个调度器: 对每个关节依次调
    1) tau_ff_sweep.py      -> 扫力矩前馈, 解出这个关节自己的 KP + 固定偏置
    2) identify_kp_offset.py -> 从上一步的原始数据解 KP
    3) ramp_velocity.py     -> 匀速斜坡(每速两遍), 解 KD + 摩擦
    4) identify_kd_friction.py --kp <上一步解出的 KP>
       ^ 这一步必须传每个关节**自己的** KP. 该脚本默认 28.41 是 FL_thigh 的值,
         别的关节如果 KP 不同, 直接吃默认值 KD 就会跟着错几个百分点.

为什么必须逐关节做: 摩擦/装配/预紧每个关节都不一样, 共用一组参数没有依据.
07 的 README 也写着"先做一个关节, 流程跑通后再做八个关节悬空".

安全
----
  - 一次只使能**一个**关节(沿用两个实验脚本原有的逻辑, 其余电机不下令);
  - 全程不下地, 不调用 set_zero_position, 不发 0xFE, 零位不动;
  - 每个关节的 |tau|>3N·m / 位置越界 / 反馈超时 保护由子脚本负责;
  - 子脚本前置检查失败(无反馈/位置异常/越界) -> 跳过该关节, 继续下一个;
    运动中途触发保护 -> **立即停整个批次**, 交给人看.

用法
  py -3.13 run_all_joints.py --dry-run              # 只打印计划, 不连硬件
  py -3.13 run_all_joints.py --joints 1,2           # 只做前两个
  py -3.13 run_all_joints.py                        # 全部 8 个
  py -3.13 run_all_joints.py --solve-only           # 不测, 只把已有数据重解一遍
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
WS07 = os.path.join(WS, "07_actuator_identification")

MOTORS = {1: "FL_thigh", 2: "FL_calf", 3: "FR_thigh", 4: "FR_calf",
          5: "RL_thigh", 6: "RL_calf", 7: "RR_thigh", 8: "RR_calf"}

# 前置检查失败 = 还没开始动, 安全, 跳过这个关节接着做下一个.
# 这几种都是子脚本在进入运动循环**之前** return 掉了, 所以不会打印"结束"行.
SKIP_MARKS = ("CTRL_MODE 写入失败", "该电机无反馈", "位置异常", "超出安全范围")

RUN_ID_RE = re.compile(r"\[\+\] run_id:\s*(\S+)")
# 子脚本跑完运动循环一定会打印这行, stop_reason 就是权威结论.
# 非贪婪 + 到"两个以上空格"为止, 把后面那句"  最终位置 +0.0231 rad"甩掉.
END_RE = re.compile(r"\[\+\] 结束:\s*(\S.*?)(?:\s{2,}|$)")


def child_env():
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUNBUFFERED"] = "1"
    return e


def run_step(title, cmd, timeout):
    """跑一个子脚本. 返回 (stdout, 状态) —— 状态 ∈ {ok, timeout, fail}."""
    print(f"\n  ── {title}")
    print(f"     $ {' '.join(cmd[1:])}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=HERE, env=child_env(), timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = r.stdout.decode("utf-8", "replace")
        rc = r.returncode
    except subprocess.TimeoutExpired as e:
        # control.close() 在本机会卡死(内部 disable_all 跑完了但进程不退).
        # 数据在卡死前已经落盘, 所以超时不算失败 —— 由调用方检查数据是否齐.
        out = (e.stdout or b"").decode("utf-8", "replace")
        rc = None
    dt = time.time() - t0
    tail = [l for l in out.splitlines() if l.strip()][-6:]
    for l in tail:
        print(f"     | {l}")
    print(f"     -> 退出 {'超时(数据应已落盘)' if rc is None else rc}  用时 {dt:.1f}s")
    return out, ("timeout" if rc is None else ("ok" if rc == 0 else "fail"))


def classify(out):
    """判定这次子脚本跑下来算不算数.

    别拿关键字去猜 —— "保护"两个字在 --dry-run 的打印里也有, 一猜就假报警.
    子脚本只要进了运动循环, 退出来时必定打印 "[+] 结束: <stop_reason>",
    stop_reason 就是它自己的结论: "完成" / "保护触发: ..." / "使能检查保护" /
    "电机未动(使能未生效?)". 只认这一行.
    """
    m = END_RE.search(out)
    if m:
        reason = m.group(1).strip()
        return "OK" if reason.startswith("完成") else f"STOP:{reason}"
    # 没有"结束"行 = 在进入运动循环之前就 return 了, 电机没动过, 跳过这个关节接着做下一个
    for k in SKIP_MARKS:
        if k in out:
            return "SKIP"
    return "OK"


def stop_msg(name, cls):
    return f"\n[!!] {name} 异常: {cls[5:]}, 停整个批次  ← 电机已经动过, 别自动往下做."


def newest_run(kind, joint_name):
    """按 run_id 里的关节名找最近的 run 目录(kind: tauff / ramp)."""
    base = os.path.join(WS07, "data", "raw")
    hits = []
    for day in sorted(os.listdir(base)):
        d = os.path.join(base, day)
        if not os.path.isdir(d):
            continue
        for r in os.listdir(d):
            if f"_{kind}_{joint_name}_" in r:
                hits.append((os.path.join(d, r), r))
    if not hits:
        return None, None
    hits.sort(key=lambda x: x[1])
    return hits[-1]


def solve_kp(run_dir, canid):
    out, _ = run_step("解 KP", [sys.executable, "identify_kp_offset.py",
                                 "--run", run_dir, "--canid", str(canid)], 120)
    j = os.path.join(WS07, "data", "processed", os.path.basename(run_dir),
                     "kp_offset_ident.json")
    if not os.path.exists(j):
        return None
    with open(j, encoding="utf-8") as f:
        fit = json.load(f)["fits"]["全部合并"]
    return fit


def solve_kd(run_dir, canid, kp):
    out, _ = run_step(f"解 KD + 摩擦 (用本关节 KP={kp:.3f})",
                      [sys.executable, "identify_kd_friction.py", "--run", run_dir,
                       "--canid", str(canid), "--kp", f"{kp:.6f}"], 180)
    j = os.path.join(WS07, "data", "processed", os.path.basename(run_dir),
                     "kd_friction_ident.json")
    if not os.path.exists(j):
        return None, None
    with open(j, encoding="utf-8") as f:
        J = json.load(f)
    return J["kd"]["mean_trusted"], J.get("friction_fit")


def main():
    # 子脚本是直接往终端写的(不捕获), 父进程如果块缓冲, 重定向到日志时两边就会乱序
    try:
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", type=str, default="1,2,3,4,5,6,7,8")
    ap.add_argument("--tag", type=str, default="r01")
    ap.add_argument("--speeds", type=str, default="0.03,0.08,0.13",
                    help="用 r02 那组: 匀速段拉长到 1.8s, 正反两趟位置才重叠, 摩擦解得出来")
    ap.add_argument("--t-cr", type=float, default=1.80)
    ap.add_argument("--kp", type=float, default=28.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solve-only", action="store_true",
                    help="不测硬件, 只把已有数据重解一遍")
    ap.add_argument("--timeout-measure", type=float, default=300.0)
    args = ap.parse_args()

    joints = [int(s) for s in args.joints.split(",") if s.strip()]

    print("=" * 78)
    print("八个关节批量辨识")
    print(f"  关节 {joints}  kp={args.kp} kd={args.kd}")
    print(f"  匀速斜坡 速度 {args.speeds} rad/s, 匀速段 {args.t_cr}s")
    est = len(joints) * (19 + 30 + 2 * (args.t_cr + 1.1) * 3)
    print(f"  预计总时长 ~{est/60:.0f} 分钟 (只驱动一个关节, 全程悬空)")
    print("=" * 78)

    if args.solve_only:
        print("\n[solve-only] 跳过测量, 只重解已有数据\n")

    results = {}
    for cid in joints:
        name = MOTORS[cid]
        print(f"\n{'━'*78}\n■ 关节 {cid}  {name}\n{'━'*78}")

        tauff_run = ramp_run = None
        if args.dry_run:
            # 只打印两个动作序列, 不连硬件 —— 给人过目用
            for title, script in (("① 力矩前馈扫描 (解 KP)", "tau_ff_sweep.py"),
                                  ("② 匀速斜坡 (解 KD + 摩擦)", "ramp_velocity.py")):
                cmd = [sys.executable, script, "--canid", str(cid),
                       "--kp", str(args.kp), "--kd", str(args.kd), "--dry-run"]
                if script == "ramp_velocity.py":
                    cmd += ["--speeds", args.speeds, "--t-cr", str(args.t_cr),
                            "--tag", args.tag]
                print(f"\n  ── {title}")
                subprocess.run(cmd, cwd=HERE, env=child_env())
            continue
        if not args.solve_only:
            cmd = [sys.executable, "tau_ff_sweep.py", "--canid", str(cid),
                   "--kp", str(args.kp), "--kd", str(args.kd)]
            out, st = run_step("① 力矩前馈扫描 (解 KP)", cmd, args.timeout_measure)
            cls = classify(out)
            if cls.startswith("STOP"):
                print(stop_msg(name, cls))
                break
            if cls == "SKIP":
                print(f"[--] {name} 前置检查未通过, 跳过这个关节.")
                results[cid] = {"name": name, "status": "跳过(无反馈/越界)"}
                continue
            tauff_run = RUN_ID_RE.search(out)
            tauff_run = tauff_run.group(1) if tauff_run else None
            if not tauff_run:
                tauff_run, _ = newest_run("tauff", name)
            else:
                tauff_run = os.path.join(WS07, "data", "raw",
                                         time.strftime("%Y-%m-%d"),
                                         tauff_run)
            if not tauff_run or not os.path.isdir(tauff_run):
                print(f"[--] {name} 没找到前馈扫描的数据目录, 跳过.")
                results[cid] = {"name": name, "status": "无数据"}
                continue

            cmd = [sys.executable, "ramp_velocity.py", "--canid", str(cid),
                   "--kp", str(args.kp), "--kd", str(args.kd),
                   "--speeds", args.speeds, "--t-cr", str(args.t_cr),
                   "--tag", args.tag]
            out, st = run_step("② 匀速斜坡 (解 KD + 摩擦)", cmd, args.timeout_measure)
            cls = classify(out)
            if cls.startswith("STOP"):
                print(stop_msg(name, cls))
                break
            r = RUN_ID_RE.search(out)
            ramp_run = (os.path.join(WS07, "data", "raw", time.strftime("%Y-%m-%d"),
                                     r.group(1)) if r else None)
            if not ramp_run or not os.path.isdir(ramp_run):
                ramp_run, _ = newest_run("ramp", name)
        else:
            tauff_run, _ = newest_run("tauff", name)
            ramp_run, _ = newest_run("ramp", name)

        res = {"name": name, "status": "ok",
               # 记下数据出处, 免得回头拿着一张表不知道数是哪次测的
               "run_tauff": os.path.basename(tauff_run) if tauff_run else "",
               "run_ramp": os.path.basename(ramp_run) if ramp_run else ""}
        if tauff_run:
            fit = solve_kp(tauff_run, cid)
            if fit:
                res.update(kp=fit["kp"], kp_se=fit["kp_se"], kp_r2=fit["r2"],
                           kp_n=fit["n"], b=fit["b"])
        if ramp_run and res.get("kp"):
            kd, fr = solve_kd(ramp_run, cid, res["kp"])
            if kd is not None:
                res["kd"] = kd
            if fr:
                res.update(tc=fr["tau_c"], tc_se=fr["tau_c_se"], bv=fr["b_v"],
                           fr_r2=fr["r2"])
        if not res.get("kp"):
            res["status"] = "KP 没解出来"
        elif not res.get("kd"):
            res["status"] = "KD 没解出来"
        results[cid] = res
        print(f"\n  [{name}] KP={res.get('kp', float('nan')):.3f}  "
              f"KD={res.get('kd', float('nan')):.3f}  "
              f"τ_c={res.get('tc', float('nan')):.3f}")

    # ---------------- 汇总 ----------------
    print(f"\n\n{'='*78}\n汇总\n{'='*78}")
    print(f"{'关节':<10}{'KP':>10}{'±':>8}{'R²':>10}{'KD':>9}"
          f"{'τ_c(N·m)':>11}{'±':>8}{'状态':>10}")
    for cid in joints:
        r = results.get(cid)
        if not r:
            continue
        f = lambda k, w=3: (f"{r[k]:.{w}f}" if isinstance(r.get(k), float) else "—")
        print(f"{r['name']:<10}{f('kp'):>10}{f('kp_se'):>8}{f('kp_r2',5):>10}"
              f"{f('kd'):>9}{f('tc'):>11}{f('tc_se'):>8}{r['status']:>10}")

    outdir = os.path.join(WS07, "results", "params")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "all_joints_summary.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({"kp_used": args.kp, "kd_used": args.kd,
                   "speeds": args.speeds, "t_cr": args.t_cr,
                   "joints": {str(k): v for k, v in results.items()}},
                  fp, ensure_ascii=False, indent=1)
    print(f"\n[+] 汇总写入 {os.path.relpath(path, WS)}")
    print("[!] 这些值只在悬空、不下地、单关节条件下测过; 写进仿真配置前先过 10 工作包的新动作验证.")


if __name__ == "__main__":
    main()
