"""只读：查询 8 个电机的状态码、位置、温度字节，不使能、不写寄存器、不发运动命令。

状态码（达妙）：0 失能，1 使能，8 超压，9 欠压，A 过流，B MOS 过温，C 线圈过温，D 通讯丢失，E 过载。
用法：python query_states.py（Jetson 用 BENNETT_DM_SDK_PROFILE=low_latency）
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", REPO / "sim2real" / "bennett_deploy" / "dm_can.py")
dmc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmc
spec.loader.exec_module(dmc)
SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}
init = [dmc.DmActData(dmc.DM_Motor_Type.DM8006, dmc.Control_Mode.MIT_MODE, c, 0x10 + c, ch) for c, ch in MOTORS.items()]
ctl = dmc.Motor_Control(1_000_000, 5_000_000, SN, init, device_type=dmc.dmcan_device_type.USB2CANFD_DUAL, auto_enable=False)
seen = {}
orig = ctl.canframeCallback


def cb(device, frame):
    try:
        cid = int(frame.head.can_id)
        if 0x11 <= cid <= 0x18:
            seen[cid - 0x10] = bytes(frame.payload[:8])
    except Exception:
        pass
    orig(device, frame)


ctl.getDevice().hook_recv_callback(cb)
time.sleep(0.3)
for c in MOTORS:
    ctl.refresh_motor_status(ctl.getMotor(MOTORS[c], c))
    time.sleep(0.002)
time.sleep(0.5)
for c in sorted(MOTORS):
    b = seen.get(c)
    print(f"电机 {c}: " + ("无应答" if b is None else f"状态码 {b[0] >> 4:X}  MOS温度 {b[6]} ℃  线圈温度 {b[7]} ℃"))
sys.stdout.flush()
os._exit(0)
