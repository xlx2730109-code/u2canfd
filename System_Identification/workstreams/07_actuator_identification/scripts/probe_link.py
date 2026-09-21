# -*- coding: utf-8 -*-
"""跑实验前的只读链路探针 —— 看电机在不在、现在什么姿态.

只读: 打开设备后只做两件事
  1) refresh_motor_status(0xCC 读状态请求) 让各电机回一帧 —— 只读, 不使能不下令
  2) 被动收帧, 解码 q / dq / tau / err

**不切模式、不使能、不下令、不发 0xFE**. 退出时 close() 会发 0xFD(失能),
那是"关"不是"开", 与所有既有脚本一致.

用法:  py -3.13 probe_link.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)
os.chdir(PARENT)

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from damiao import Control_Mode, DM_Motor_Type, DmActData, Motor_Control  # noqa: E402
from dmcan import dmcan_device_type  # noqa: E402
from identify_kp_kd import P_MAX, V_MAX, T_MAX, decode  # noqa: E402

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: ("FL_thigh", 0), 2: ("FL_calf", 0), 3: ("FR_thigh", 1), 4: ("FR_calf", 1),
          5: ("RL_thigh", 0), 6: ("RL_calf", 0), 7: ("RR_thigh", 1), 8: ("RR_calf", 1)}


def main():
    frames = []
    control = None
    try:
        init = [DmActData(DM_Motor_Type.DM8006, Control_Mode.MIT_MODE, cid, 0x10 + cid, c)
                for cid, (_n, c) in MOTORS.items()]
        control = Motor_Control(1_000_000, 5_000_000, SN, init,
                                device_type=dmcan_device_type.USB2CANFD_DUAL,
                                auto_enable=False)
        orig_cb = control.canframeCallback

        def cb(device, frame):
            try:
                frames.append((time.perf_counter_ns(), int(frame.head.channel),
                               int(frame.head.can_id),
                               bytes(frame.payload[:min(int(frame.head.dlc), 8)]).hex()))
            except Exception:
                pass
            orig_cb(device, frame)

        control.getDevice().hook_recv_callback(cb)
        print("[+] USB-CAN 已打开 (auto_enable=False, 未使能任何电机)")

        for cid in sorted(MOTORS):
            control.refresh_motor_status(control.getMotor(MOTORS[cid][1], cid))
            time.sleep(0.03)
        time.sleep(0.6)

        print(f"[+] 共收到 {len(frames)} 帧")
        seen = {}
        for _t, chn, cid, hexs in frames:
            seen.setdefault(cid, []).append((chn, hexs))

        print(f"\n{'can_id':>7} {'关节':<10} {'ch':>3} {'帧数':>5}  状态")
        print("-" * 72)
        for cid in sorted(MOTORS):
            name, ch = MOTORS[cid]
            mst = 0x10 + cid
            got = seen.get(mst, [])
            if not got:
                print(f"{mst:>7} {name:<10} {ch:>3} {0:>5}  -- 无反馈(未上电/未接?)")
                continue
            # 取最后一帧 MIT 布局的来解码
            best = None
            for _chn, hexs in got:
                b0 = bytes.fromhex(hexs)[0]
                err, q, dq, tau = decode(hexs)
                if (b0 >> 4) & 0x0F in (0, 1) and abs(q) <= 1.0 and abs(dq) <= 10.0:
                    best = (err, q, dq, tau)
            if best is None:
                print(f"{mst:>7} {name:<10} {ch:>3} {len(got):>5}  "
                      f"有帧但都是非 MIT 布局")
            else:
                err, q, dq, tau = best
                print(f"{mst:>7} {name:<10} {ch:>3} {len(got):>5}  "
                      f"q={q:+.4f} rad ({q*57.3:+6.2f}°)  dq={dq:+.4f}  "
                      f"tau={tau:+.3f} N·m  err={err}")

        others = sorted(set(seen) - {0x10 + c for c in MOTORS})
        if others:
            print(f"\n[+] 其它 can_id 的帧: {[(hex(i), len(seen[i])) for i in others]}")
        print("\n[+] 探针结束。imu 未涉及, 零位未涉及。")
    except Exception as exc:
        print(f"[!] 失败: {type(exc).__name__}: {exc}")
    finally:
        if control is not None:
            try:
                control.close()      # 只发 0xFD(失能), 不发 0xFE
            except Exception:
                pass


if __name__ == "__main__":
    main()
