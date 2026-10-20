"""部署端电机看门狗验证（`Go2MotorBus.arm_watchdog` / `close` 的真机行为）：8 台电机、零力矩、不写 flash、不触发看门狗。

背景：部署程序 `sim2real/bennett_deploy` 启动时在电机已使能的状态下写 `TIMEOUT` 并读回，退出时先失能再写回（根 README 01-K98）。
01-K95 与 07 号包的测试都是电机失能时写入，使能状态下写入的行为未实测，本脚本补测。

流程（`--execute`）：
  1. 用部署代码的 `Go2MotorBus` 创建总线（与部署相同，8 台自动使能）；
  2. `arm_watchdog(300 ms, 50 µs)`：读原值、写 6000、保活读回；
  3. 以 500 Hz 发 kp=0、kd=0 的 MIT 帧（零力矩）`--hold_s` 秒，期间每秒读回一次 `TIMEOUT`，记录状态码、反馈时龄、发令节拍、进程 CPU 占比；
  4. `shutdown_motors()`：失能、写回原值（`close()` 在 Jetson 上最后关 USB 设备时 libusb 会 abort，见根 README 01-K98）。
不做"停发观察触发"：触发会使电机锁存状态码 13，需断电（01-K96）。
`--readback`：另起进程只读 8 台 `TIMEOUT` 与状态码，核对 `close()` 的写回（不使能）。

用法（Jetson，`LD_PRELOAD=libusbshim.so BENNETT_DM_SDK_PROFILE=low_latency`）：
    python deploy_watchdog_test.py --execute --hold_s 5
    python deploy_watchdog_test.py --readback
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", REPO / "sim2real" / "bennett_deploy" / "dm_can.py")
dmc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmc
spec.loader.exec_module(dmc)

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}
NAMES = ["FL_thigh", "FL_calf", "FR_thigh", "FR_calf", "RL_thigh", "RL_calf", "RR_thigh", "RR_calf"]
SPECS = [SimpleNamespace(name=NAMES[c - 1], can_id=c, master_id=0x10 + c, channel=ch) for c, ch in MOTORS.items()]
RID_TIMEOUT = 9


def readback() -> int:
    init = [dmc.DmActData(dmc.DM_Motor_Type.DM8006, dmc.Control_Mode.MIT_MODE, c, 0x10 + c, ch) for c, ch in MOTORS.items()]
    ctl = dmc.Motor_Control(1_000_000, 5_000_000, SN, init, device_type=dmc.dmcan_device_type.USB2CANFD_DUAL, auto_enable=False)
    seen = {}
    orig_cb = ctl.canframeCallback

    def cb(device, frame):
        try:
            cid = int(frame.head.can_id)
            if 0x11 <= cid <= 0x18:
                seen[cid - 0x10] = bytes(frame.payload[:1])[0] >> 4
        except Exception:
            pass
        orig_cb(device, frame)

    ctl.getDevice().hook_recv_callback(cb)
    time.sleep(0.3)
    for c in MOTORS:
        ctl.refresh_motor_status(ctl.getMotor(MOTORS[c], c))
        time.sleep(0.002)
    time.sleep(0.4)
    for c in sorted(MOTORS):
        v = ctl.read_motor_param(ctl.getMotor(MOTORS[c], c), RID_TIMEOUT, timeout=1.0)
        print(f"电机 {c} ({NAMES[c - 1]}): 状态码 {seen.get(c, '无应答'):X} TIMEOUT={v}" if c in seen else f"电机 {c}: 无应答 TIMEOUT={v}")
    sys.stdout.flush()
    os._exit(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="真的使能电机并写 TIMEOUT；不加则只打印计划")
    ap.add_argument("--readback", action="store_true", help="只读 TIMEOUT 与状态码")
    ap.add_argument("--hold_s", type=float, default=5.0)
    ap.add_argument("--rate_hz", type=float, default=500.0)
    args = ap.parse_args()
    if args.readback:
        return readback()
    print(f"计划：Go2MotorBus 使能 8 台，arm_watchdog(300 ms)，零力矩 {args.rate_hz:g} Hz × {args.hold_s:g} s，close() 写回。")
    if not args.execute:
        print("未加 --execute，没有动设备。")
        return 0

    bus = dmc.Go2MotorBus(SPECS)
    rc = 0
    try:
        bus.arm_watchdog(300.0, 50.0)
        raw = {n: int(bus.control.read_motor_param(m, RID_TIMEOUT, timeout=1.0)) for n, m in bus.motors.items()}
        print("[读回] 使能状态下 TIMEOUT = " + str(sorted(set(raw.values()))))
        period = 1.0 / args.rate_hz
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        nxt = t0
        n = 0
        stamps = []
        last_report = t0
        sys.setswitchinterval(0.0005)
        while time.perf_counter() - t0 < args.hold_s:
            for m in bus.motors.values():
                bus.control.control_mit(m, 0.0, 0.0, 0.0, 0.0, 0.0)
            now = time.perf_counter()
            stamps.append(now)
            n += 1
            if now - last_report >= 1.0:
                last_report = now
                snap = bus.snapshot(now)
                codes = "".join(f"{snap[nm].error:X}" for nm in NAMES)
                age = max(snap[nm].feedback_age_s for nm in NAMES) * 1000
                tq = max(abs(snap[nm].torque) for nm in NAMES)
                print(f"  t={now - t0:4.1f} s 状态码[{codes}] 最大反馈时龄 {age:5.1f} ms 最大|力矩| {tq:.3f} N·m")
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
        wall = time.perf_counter() - t0
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        gaps.sort()
        print(f"[节拍] {n / wall:.1f} Hz，帧间隔 p50 {gaps[len(gaps) // 2] * 1e3:.2f} ms，p99 {gaps[int(len(gaps) * 0.99)] * 1e3:.2f} ms，最大 {gaps[-1] * 1e3:.2f} ms；"
              f"cpu_ratio {(time.process_time() - cpu0) / wall:.2f}")
        snap = bus.snapshot()
        bad = {nm: snap[nm].error for nm in NAMES if snap[nm].error != 1}
        print("[结果] 全部使能 " if not bad else f"[结果] 非使能状态码 {bad}")
        if bad:
            rc = 3
    except BaseException as exc:
        print(f"[异常] {exc!r}")
        rc = 2
    finally:
        bus.shutdown_motors()  # 与部署退出路径相同；不调用 close()：Jetson 上关闭 USB 设备时 libusb 会 abort（已在第一次测试中复现）
        print("[关闭] shutdown_motors() 已执行（失能、写回 TIMEOUT）")
        sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    sys.exit(main())
