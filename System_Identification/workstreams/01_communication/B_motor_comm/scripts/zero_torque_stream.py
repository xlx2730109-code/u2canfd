"""零力矩命令流：按部署同款负载（1 kHz，每圈给 8 个电机各发一帧）记录反馈到达节奏。

安全：
  - 不使能电机：构造 Motor_Control 时 auto_enable=False，结束时不调用 close()（它会发失能帧）；
  - 发送的 MIT 帧为 kp=0、kd=0、q=0、dq=0、tau=0，电机即使被使能也只是零力矩；
  - 正式发送前先给每个电机发一次状态查询（0x7FF/0xCC），任何电机上报的状态不是"失能"就中止；
  - 会把两路 CAN 都写成 CAN FD 1 M / 5 M（与部署启动时的设置相同）；运行前盒子的设置记在 A_prep/results。

选库（Windows）：环境变量 BENNETT_DM_SDK_PROFILE=original | low_latency。

用法：
    python zero_torque_stream.py --seconds 30 --repeat 1
输出：<B_motor_comm 或 C_loop_timing（带 --policy）>/data/raw/<日期>/<run_id>/ {manifest.yaml, frames.csv, tx.csv, meta.json, policy.csv}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # u2canfd
WS = HERE.parents[1]  # 01_communication
# 试验用：环境变量 BENNETT_DM_CAN_PATH 可指向临时目录里的 dm_can.py（配合临时动态库），默认用仓库里的
DM_CAN = Path(os.environ.get("BENNETT_DM_CAN_PATH") or REPO / "sim2real" / "bennett_deploy" / "dm_can.py")

spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", DM_CAN)
dmc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmc  # dataclass 需要模块已登记
spec.loader.exec_module(dmc)

SN = "D6977F56F86C64B77B316E7154FA6DF3"
# CAN ID -> 通道（关节顺序 FL_thigh, FL_calf, FR_thigh, FR_calf, RL_thigh, RL_calf, RR_thigh, RR_calf）
MOTORS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}
ERR_DISABLED = 0



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def lib_info() -> dict:
    if sys.platform == "win32":
        path, profile = Path(dmc._dm_dll_path), dmc._dm_sdk_profile
    else:
        path, profile = Path(dmc._dm_so_path), "linux_so"
    try:
        shown = str(path.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        shown = str(path)
    return {"profile": profile, "path": shown, "sha256": sha256(path)}


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return "unknown"
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def dlc_to_len(dlc: int) -> int:
    return {9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64}.get(dlc, min(dlc, 8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--rate_hz", type=float, default=1000.0, help="发令环频率，每圈给 8 个电机各发一帧")
    ap.add_argument("--policy", type=str, default=None,
                    help="TorchScript 策略（policy.pt）：在发令环里每 1/policy_hz 秒做一次 CPU 推理，复现部署的单线程结构")
    ap.add_argument("--policy_hz", type=float, default=50.0)
    ap.add_argument("--deploy_clock", action="store_true",
                    help="策略 tick 的调度用 time.monotonic()，与部署的 runtime.py 一致（Python 3.11 的 Windows 上分辨率约 15.6 ms）；"
                         "默认用 perf_counter")
    ap.add_argument("--switch_interval", type=float, default=None,
                    help="sys.setswitchinterval（秒）；不给则保持 Python 缺省 5 ms，与部署一致")
    ap.add_argument("--no_gc", action="store_true", help="发送前 gc.collect() 后 gc.disable()，检验发送尖峰是不是垃圾回收造成的")
    ap.add_argument("--timer_1ms", action="store_true",
                    help="Windows：像部署程序一样调用 timeBeginPeriod(1)，把系统定时器精度从 15.6 ms 提到 1 ms")
    args = ap.parse_args()

    dmcan_device_type = dmc.dmcan_device_type  # dm_can 已处理 dmcan 的导入路径

    if args.switch_interval is not None:
        sys.setswitchinterval(args.switch_interval)
    timer_set = False
    if args.timer_1ms and sys.platform == "win32":
        import ctypes
        timer_set = int(ctypes.WinDLL("winmm").timeBeginPeriod(1)) == 0
        print(f"[定时器] timeBeginPeriod(1) {'成功' if timer_set else '失败'}")

    pol = None
    obs_dim = None
    if args.policy:
        import torch
        pol = torch.jit.load(args.policy, map_location="cpu")
        pol.eval()
        for d in (50, 33, 48, 45, 46, 47, 41):
            try:
                with torch.no_grad():
                    pol(torch.zeros(1, d))
                obs_dim = d
                break
            except Exception:
                continue
        if obs_dim is None:
            raise SystemExit("无法确定策略的观测维度")
        obs = torch.zeros(1, obs_dim)
        with torch.no_grad():
            for _ in range(50):  # 预热
                pol(obs)
        print(f"[策略] {args.policy} 观测维度 {obs_dim}，torch {torch.__version__}，线程数 {torch.get_num_threads()}")

    lib = lib_info()
    stamp = datetime.now()
    tag = ("_t1ms" if timer_set else "") + ("_pol" if pol is not None else "") + ("_dclk" if args.deploy_clock else "")         + (f"_sw{args.switch_interval * 1000:g}" if args.switch_interval is not None else "") + ("_nogc" if args.no_gc else "")
    run_id = f"{stamp.strftime('%Y%m%d_%H%M%S')}_powered_zt{int(args.rate_hz)}hz_{lib['profile']}{tag}_r{args.repeat:02d}"
    # 带策略的运行属于 C_loop_timing（控制程序节拍），其余属于 B_motor_comm
    topic = "C_loop_timing" if pol is not None else "B_motor_comm"
    out_dir = WS / topic / "data" / "raw" / stamp.strftime("%Y-%m-%d") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_id}\n[库] {lib}")

    frames: list = []
    ns = time.perf_counter_ns

    init = [dmc.DmActData(dmc.DM_Motor_Type.DM8006, dmc.Control_Mode.MIT_MODE, cid, 0x10 + cid, ch)
            for cid, ch in MOTORS.items()]
    control = dmc.Motor_Control(1_000_000, 5_000_000, SN, init,
                                device_type=dmcan_device_type.USB2CANFD_DUAL, auto_enable=False)
    orig_cb = control.canframeCallback

    def cb(device, frame):
        t = ns()
        try:
            h = frame.head
            n = dlc_to_len(int(h.dlc))
            frames.append((t, int(h.channel), int(h.can_id), int(h.dlc), int(h.canfd), int(h.brs),
                           int(h.dir), int(h.ack), int(h.esi), int(h.timestamp), bytes(frame.payload[:n])))
        except Exception:
            pass
        orig_cb(device, frame)

    control.getDevice().hook_recv_callback(cb)
    motors = [(cid, control.getMotor(ch, cid)) for cid, ch in MOTORS.items()]
    time.sleep(0.1)

    # 1) 被动监听：没人发命令时不应有帧
    n0 = len(frames)
    time.sleep(0.3)
    passive = len(frames) - n0
    print(f"[监听] 0.3 s 内收到 {passive} 帧")

    # 2) 状态查询，确认全部电机已应答且处于失能状态（刚改写波特率后第一次可能无应答，最多重试 3 次）
    status = {}
    for attempt in range(3):
        i0 = len(frames)
        for cid, m in motors:
            control.refresh_motor_status(m)
            time.sleep(0.002)
        time.sleep(0.5)
        status = {}
        for f in frames[i0:]:
            cid = f[2]
            if 0x11 <= cid <= 0x18 and len(f[10]) >= 1:
                status[cid - 0x10] = (f[10][0] >> 4) & 0x0F
        print(f"[状态 第{attempt + 1}次] 电机状态码（0=失能）: {dict(sorted(status.items()))}")
        if len(status) == len(MOTORS):
            break
        time.sleep(1.0)
    missing = [cid for cid in MOTORS if cid not in status]
    enabled = [cid for cid, e in status.items() if e != ERR_DISABLED]
    if missing or enabled:
        print(f"[中止] 无应答电机 {missing}，状态非失能电机 {enabled}；未发送零力矩帧")
        (out_dir / "ABORTED.txt").write_text(f"missing={missing} not_disabled={enabled} status={status}\n", encoding="utf-8")
        sys.stdout.flush()
        os._exit(2)

    if args.no_gc:
        import gc
        gc.collect()
        gc.disable()
    # 3) 零力矩命令流：每圈给 8 个电机各发一帧
    print(f"[发送] {args.rate_hz:g} Hz × 8 电机，持续 {args.seconds:g} s")
    tx = []
    period_ns = int(1e9 / args.rate_hz)
    t_wall0, t_mono0 = time.time_ns(), ns()
    cpu0 = time.process_time()
    t_start = ns()
    k = 0
    n_fail = 0
    pol_rows = []
    pol_dt_ns = int(1e9 / args.policy_hz)
    next_pol = t_start
    next_pol_m = time.monotonic()
    while True:
        now = ns()
        if now - t_start >= args.seconds * 1e9:
            break
        cyc_t = now
        now_m = time.monotonic()
        due = (now_m >= next_pol_m) if args.deploy_clock else (now >= next_pol)
        if pol is not None and due:
            mono = time.perf_counter()  # dm_can.py 现在用 perf_counter 记 last_time_
            ages = [(mono - m.last_time_) * 1e3 for _, m in motors]  # 与部署的 feedback_age_s 同一定义（主机回调时刻）
            t_a = ns()
            with torch.no_grad():
                pol(obs)
            t_b = ns()
            pol_rows.append((k, now, t_b - t_a, max(ages), sum(ages) / len(ages)))
            next_pol += pol_dt_ns
            if next_pol < now - pol_dt_ns:
                next_pol = now + pol_dt_ns
            next_pol_m += pol_dt_ns / 1e9  # 与 runtime.py 相同的追赶规则
            if next_pol_m < now_m - pol_dt_ns / 1e9:
                next_pol_m = now_m + pol_dt_ns / 1e9
        for cid, m in motors:
            tb = ns()
            ok = control.control_mit(m, 0.0, 0.0, 0.0, 0.0, 0.0)
            ta = ns()
            if not ok:
                n_fail += 1
            tx.append((k, cid, m.GetChannel(), cyc_t, tb, ta, int(bool(ok))))
        k += 1
        nxt = t_start + k * period_ns
        late = nxt - ns()
        if late > 0:
            time.sleep(late / 1e9)
    t_end = ns()
    cpu_s = time.process_time() - cpu0
    time.sleep(0.3)  # 收尾：让在途反馈到达

    # 4) 落盘（不调用 close()，避免发失能帧）
    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_host_mono_ns", "channel", "can_id", "dlc", "canfd", "brs", "dir", "ack", "esi",
                    "dev_timestamp", "data_hex"])
        for r in frames:
            w.writerow([*r[:10], r[10].hex()])
    with open(out_dir / "tx.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "can_id", "channel", "t_cycle_ns", "t_tx_begin_ns", "t_tx_end_ns", "send_ok"])
        w.writerows(tx)
    if pol_rows:
        with open(out_dir / "policy.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cycle", "t_ns", "infer_ns", "age_max_ms", "age_mean_ms"])
            w.writerows(pol_rows)
    meta = {
        "run_id": run_id, "t_start_ns": t_start, "t_end_ns": t_end, "t_mono0_ns": t_mono0, "t_wall0_ns": t_wall0,
        "rate_hz": args.rate_hz, "seconds": args.seconds, "cycles": k, "tx_frames": len(tx), "tx_fail": n_fail,
        "policy": args.policy, "policy_hz": args.policy_hz if pol is not None else None, "policy_ticks": len(pol_rows), "rx_frames_total": len(frames), "process_cpu_s": round(cpu_s, 3), "cpu_over_wall": round(cpu_s / ((t_end - t_start) / 1e9), 3), "passive_frames_before_tx": passive, "initial_status": status,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    manifest = f"""# 零力矩命令流 manifest
run_id: {run_id}
local_start: {stamp.isoformat(timespec='seconds')}
utc_start: {datetime.now(timezone.utc).isoformat(timespec='seconds')}
operator: xu
purpose: 01_communication/B_motor_comm 基线：按部署同款负载记录反馈到达节奏
safety:
  motion_commands: false
  enabled_motors: false
  frame: MIT kp=0 kd=0 q=0 dq=0 tau=0 (零力矩)
  precheck: 0xCC 状态查询，全部电机失能后才发送
  close_called: false
load:
  rate_hz: {args.rate_hz:g}
  motors_per_cycle: 8
  can: 两路 CAN FD, 仲裁 1 Mbps, 数据 5 Mbps (BRS)
  windows_timer_1ms: {timer_set}
  policy_in_loop: {args.policy}
  policy_schedule_clock: {"time.monotonic" if args.deploy_clock else "perf_counter"}
  gc_disabled: {args.no_gc}
  switch_interval_s: {args.switch_interval if args.switch_interval is not None else "default(0.005)"}
dll_profile: {lib['profile']}
dll_path: {lib['path']}
dll_sha256: {lib['sha256']}
software:
  python: {sys.version.split()[0]}
  platform: {sys.platform} {platform.machine()}
  git: {json.dumps(git_info())}
hardware:
  can_device_sn: {SN}
  device_firmware: dual_app v1.0.0.7
battery: unknown
v_start: unknown
v_end: unknown
notes:
  - 时间戳 t_host_mono_ns 为回调第一行的 perf_counter_ns（主机时间，含驱动批量交付）
  - dev_timestamp 为帧头里设备自带的时间戳，单位与含义未验证
"""
    (out_dir / "manifest.yaml").write_text(manifest, encoding="utf-8")
    print(f"[完成] 周期 {k}，发送 {len(tx)} 帧（失败 {n_fail}），接收 {len(frames)} 帧\n[数据] {out_dir.relative_to(REPO)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
