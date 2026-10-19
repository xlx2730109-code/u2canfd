"""只读：枚举 USB2CANFD 盒，读固件版本与两路 CAN 的实际速率。

不设置波特率、不使能通道、不注册回调、不发送任何 CAN 帧，也不构造 Motor_Control
（它的缺省会改写波特率并使能电机，见根 README 的 01-K11）。

Windows 用环境变量选择动态库：
    BENNETT_DM_SDK_PROFILE=original | low_latency
Linux 使用 sim2real/sdk/dlls/libdm_device.so（仓库里是 x86-64，Jetson 上需要 arm64 版）。

用法（在仓库任意位置）：
    python read_device_info.py
结果写入 ../results/<时间>_<库>_device_info.json。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # u2canfd
RESULTS = HERE.parent / "results"
DM_CAN = REPO / "sim2real" / "bennett_deploy" / "dm_can.py"

# 只借用 dm_can.py 里的动态库选择与加载，不经过包的 __init__（避免引入 torch 等依赖）
spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", DM_CAN)
dmc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmc  # dataclass 需要模块已登记
spec.loader.exec_module(dmc)

DEVICE_TYPES = (dmc.dmcan_device_type.USB2CANFD_DUAL, dmc.dmcan_device_type.USB2CANFD)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def library_info() -> dict:
    if sys.platform == "win32":
        path = Path(dmc._dm_dll_path)
        profile = dmc._dm_sdk_profile
    else:
        path = Path(dmc._dm_so_path)
        profile = "linux_so"
    return {"profile": profile, "path": str(path.relative_to(REPO)), "sha256": sha256(path)}


def main() -> int:
    stamp = datetime.now()
    out = {
        "time_local": stamp.isoformat(timespec="seconds"),
        "host": {"platform": sys.platform, "machine": platform.machine(), "python": sys.version.split()[0]},
        "library": library_info(),
        "devices": [],
        "note": "只读：未设置波特率，未使能通道，未发送任何帧",
    }
    print(f"[库] {out['library']}")

    ctx = dmc.DmCanContext()
    try:
        count = 0
        used_type = None
        for dtype in DEVICE_TYPES:
            count = ctx.find_devices(dtype)
            if count > 0:
                used_type = dtype
                break
        out["device_count"] = count
        out["device_type"] = used_type.name if used_type is not None else None
        print(f"[设备] 找到 {count} 个 {out['device_type']}")

        for idx in range(max(count, 0)):
            dev = ctx.get_device(idx)
            entry = {"index": idx, "opened": False}
            if dev.open():
                entry["opened"] = True
                entry["version"] = dev.get_version()
                entry["channels"] = []
                for ch in (0, 1):
                    info = dev.get_channel_baudrate(ch)
                    if info is None:
                        entry["channels"].append({"channel": ch, "read": False})
                        continue
                    entry["channels"].append({
                        "channel": ch,
                        "read": True,
                        "canfd": bool(info.canfd),
                        "nominal_baud": int(info.can_baudrate),
                        "data_baud": int(info.canfd_baudrate),
                        "nominal_sample_point": round(float(info.can_sp), 4),
                        "data_sample_point": round(float(info.canfd_sp), 4),
                    })
            out["devices"].append(entry)
            print(f"[设备 {idx}] {entry}")
    finally:
        # Windows 上关闭设备会挂死（见 dm_can.py 的 close），直接写结果后退出进程
        pass

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{stamp.strftime('%Y%m%d_%H%M%S')}_{out['library']['profile']}_device_info.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[结果] {path.relative_to(REPO)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
