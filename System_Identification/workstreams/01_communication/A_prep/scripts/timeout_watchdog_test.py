"""电机看门狗（寄存器 TIMEOUT，RID 9）单位与行为测试：单个电机、零力矩、不写 flash。

背景：8 台电机 TIMEOUT 都是 0（看门狗关闭，根 README 01-K49）。达妙手册：TIMEOUT 为超时报警时间，uint32；设定周期内收不到
CAN 控制指令，电机自动退出使能（失能），反馈故障码为 D（通讯丢失）。单位没有一致说法（Seeed 文档写 50 µs 周期，其他说 ms），
所以先实测：写入一个已知值，使能一个电机，停止发控制帧，计时到它失能。

安全设计：
  - 默认只打印计划，不动设备；必须加 --execute 才会真的写寄存器、使能电机；
  - 只动一个电机（--motor，默认 1）；使能后只发 kp=0、kd=0、q=0、dq=0、tau=0 的 MIT 帧，零力矩，等同自由状态；
  - 写 TIMEOUT 只写运行期，不发保存命令（0xAA），掉电即恢复；结束时写回原值并读回核对，再失能；
  - 开始前所有电机必须处于失能；任何异常都先失能、写回原值；
  - 整个测试期间最长等待 --max_wait_s 秒（默认 20）。

用法：
    python timeout_watchdog_test.py                       # 只看计划
    python timeout_watchdog_test.py --execute --raw 1000  # 写入 1000，测实际掉线时间
Windows 用环境变量 BENNETT_DM_SDK_PROFILE 选库；运行后会改写盒子的运行期波特率设置，之后用 restore_channel_settings.py 恢复。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
DM_CAN = REPO / "sim2real" / "bennett_deploy" / "dm_can.py"

spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", DM_CAN)
dmc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmc
spec.loader.exec_module(dmc)

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}
RID_TIMEOUT = 9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", type=int, default=1, choices=sorted(MOTORS))
    ap.add_argument("--raw", type=int, default=1000, help="写入 TIMEOUT 的原始值")
    ap.add_argument("--max_wait_s", type=float, default=20.0)
    ap.add_argument("--allow_latched_d", action="store_true",
                    help="被测电机若停在状态码 13（上一次看门狗触发后锁存的通讯丢失，电机本身已失能）也允许开始；其余 7 个电机仍必须为 0")
    ap.add_argument("--execute", action="store_true", help="真的写寄存器、使能电机；不加则只打印计划")
    args = ap.parse_args()

    print(f"计划：电机 {args.motor}（通道 {MOTORS[args.motor]}），写 TIMEOUT={args.raw}（不保存），使能、发零力矩 MIT 帧 1 s，"
          f"停止发帧后最长等 {args.max_wait_s:g} s，记录失能时刻，再写回原值并失能。")
    if not args.execute:
        print("未加 --execute，没有动设备。")
        return 0

    init = [dmc.DmActData(dmc.DM_Motor_Type.DM8006, dmc.Control_Mode.MIT_MODE, c, 0x10 + c, ch) for c, ch in MOTORS.items()]
    ctl = dmc.Motor_Control(1_000_000, 5_000_000, SN, init, device_type=dmc.dmcan_device_type.USB2CANFD_DUAL, auto_enable=False)
    states: dict[int, tuple[float, int]] = {}
    orig_cb = ctl.canframeCallback

    def cb(device, frame):
        try:
            cid = int(frame.head.can_id)
            if 0x11 <= cid <= 0x18:
                states[cid - 0x10] = (time.perf_counter(), (bytes(frame.payload[:1])[0] >> 4) & 0x0F)
        except Exception:
            pass
        orig_cb(device, frame)

    ctl.getDevice().hook_recv_callback(cb)
    time.sleep(0.3)
    m = ctl.getMotor(MOTORS[args.motor], args.motor)

    def query_all():
        states.clear()
        for cid in MOTORS:
            ctl.refresh_motor_status(ctl.getMotor(MOTORS[cid], cid))
            time.sleep(0.002)
        time.sleep(0.3)

    query_all()
    print("[开始前状态码] " + ", ".join(f"{c}:{states[c][1]}" if c in states else f"{c}:无应答" for c in sorted(MOTORS)))
    bad = {c: s[1] for c, s in states.items() if s[1] != 0 and not (args.allow_latched_d and c == args.motor and s[1] == 13)}
    if len(states) != len(MOTORS) or bad:
        print(f"[中止] 开始前要求 8 个电机都应答且失能：应答 {sorted(states)}，非失能 {bad}")
        return 2

    orig = ctl.read_motor_param(m, RID_TIMEOUT, timeout=1.0)
    print(f"[读] TIMEOUT 原值 = {orig}")
    if orig is None:
        print("[中止] 读不到 TIMEOUT")
        return 3

    def restore():
        try:
            ctl.disable_all()
        except Exception:
            pass
        try:
            ctl.write_motor_param(m, RID_TIMEOUT, int(orig).to_bytes(4, "little"))
            time.sleep(0.1)
            back = ctl.read_motor_param(m, RID_TIMEOUT, timeout=1.0)
            print(f"[恢复] TIMEOUT 写回 {orig}，读回 {back}")
        except Exception as exc:
            print(f"[警告] 写回 TIMEOUT 失败：{exc!r}")

    result = None
    try:
        ctl.write_motor_param(m, RID_TIMEOUT, int(args.raw).to_bytes(4, "little"))
        time.sleep(0.1)
        now_val = ctl.read_motor_param(m, RID_TIMEOUT, timeout=1.0)
        print(f"[写] TIMEOUT = {now_val}（目标 {args.raw}）")
        if now_val is None or int(now_val) != int(args.raw):
            print("[中止] 写入没有生效")
            return 4
        # 使能该电机并发零力矩帧 1 s
        ctl.switchControlMode(m, dmc.control_mode_to_code[m.GetMotorMode()])
        time.sleep(0.01)
        for _ in range(3):
            ctl.control_cmd(m.GetCanId() + m.GetMotorMode(), 0xFC, m.GetChannel())
            time.sleep(0.005)
        t_end = time.perf_counter() + 1.0
        while time.perf_counter() < t_end:
            ctl.control_mit(m, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        time.sleep(0.05)
        enabled_state = states.get(args.motor, (0, None))[1]
        print(f"[使能后] 状态码 = {enabled_state}（1 表示已使能）")
        if enabled_state != 1:
            print("[中止] 没有进入使能状态")
            return 5
        # 停发控制帧；稀疏探测状态，避免探测本身喂狗
        t_stop = time.perf_counter()
        probes = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, args.max_wait_s]
        for dt in probes:
            if dt > args.max_wait_s:
                break
            time.sleep(max(0.0, t_stop + dt - time.perf_counter()))
            states.pop(args.motor, None)
            ctl.refresh_motor_status(m)
            time.sleep(0.06)
            st = states.get(args.motor)
            print(f"  停发后 {time.perf_counter() - t_stop:6.3f} s：状态码 {None if st is None else st[1]}")
            if st is not None and st[1] != 1:
                result = (time.perf_counter() - t_stop, st[1])
                break
    finally:
        restore()

    if result:
        print(f"[结果] TIMEOUT={args.raw} 时，停发后约 {result[0]:.2f} s 内失能，状态码 {result[1]}（D=0xD 表示通讯丢失）；"
              f"若单位为 50 µs 应约 {args.raw * 50e-6:.3f} s，若为 ms 应约 {args.raw * 1e-3:.3f} s。")
    else:
        print(f"[结果] {args.max_wait_s:g} s 内没有观察到失能（该值下看门狗未触发，或探测帧本身喂了狗）。")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
