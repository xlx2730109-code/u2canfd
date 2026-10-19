"""把 USB2CANFD 盒两路 CAN 的设置恢复成 read_device_info.py 记录的值，并读回核对。

只写波特率与采样点，不使能通道、不发送任何 CAN 帧。

用法：
    python restore_channel_settings.py ../results/<时间>_original_device_info.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
DM_CAN = REPO / "sim2real" / "bennett_deploy" / "dm_can.py"

spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", DM_CAN)
dmc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmc
spec.loader.exec_module(dmc)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    saved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    channels = saved["devices"][0]["channels"]

    ctx = dmc.DmCanContext()
    count = ctx.find_devices(dmc.dmcan_device_type.USB2CANFD_DUAL)
    if count < 1:
        print("未找到设备")
        return 2
    dev = ctx.get_device(0)
    if not dev.open():
        print("打开失败")
        return 3

    ok_all = True
    for c in channels:
        ch = c["channel"]
        before = dev.get_channel_baudrate(ch)
        print(f"[ch{ch}] 当前: canfd={bool(before.canfd)} {before.can_baudrate}/{before.canfd_baudrate}")
        info = dmc.dmcan_channel_can_info()
        info.channel = ch
        info.canfd = c["canfd"]
        info.can_baudrate = c["nominal_baud"]
        info.canfd_baudrate = c["data_baud"]
        info.can_sp = c["nominal_sample_point"]
        info.canfd_sp = c["data_sample_point"]
        changed = (bool(before.canfd) != c["canfd"] or before.can_baudrate != c["nominal_baud"]
                   or before.canfd_baudrate != c["data_baud"])
        if changed:
            print(f"[ch{ch}] 写回: canfd={c['canfd']} {c['nominal_baud']}/{c['data_baud']}")
            dev.set_channel_baudrate(ch, info)
        after = dev.get_channel_baudrate(ch)
        same = (bool(after.canfd) == c["canfd"] and after.can_baudrate == c["nominal_baud"]
                and after.canfd_baudrate == c["data_baud"])
        ok_all = ok_all and same
        print(f"[ch{ch}] 核对: canfd={bool(after.canfd)} {after.can_baudrate}/{after.canfd_baudrate} -> {'一致' if same else '不一致'}")
    sys.stdout.flush()
    os._exit(0 if ok_all else 4)


if __name__ == "__main__":
    sys.exit(main())
