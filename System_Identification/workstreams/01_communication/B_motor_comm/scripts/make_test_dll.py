"""在临时目录里生成试验用的 libdm_device.dll 副本（不改仓库里的 sim2real/sdk）。

官方 DLL 与补丁 DLL 只有 4 个字节不同：文件偏移 0x7F11 处的小端 32 位整数，是 `std::this_thread::sleep_for` 的纳秒数
（官方 100 000 000 = 100 ms，补丁 1 000 000 = 1 ms）。本脚本以官方 DLL 为底，把它改成指定的纳秒数，并生成一份配套的
dm_can.py（登记了名为 custom 的库），供 zero_torque_stream.py 通过环境变量试验：

    python make_test_dll.py <输出目录> <睡眠纳秒数>          # 例如 500000（睡不足 1 ms，winpthreads 不睡）
    set BENNETT_DM_CAN_PATH=<输出目录>/bennett_deploy/dm_can.py
    set BENNETT_DM_SDK_PROFILE=custom
    python zero_torque_stream.py --seconds 30 --timer_1ms

仅 Windows 的试验。
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SDK = REPO / "sim2real" / "sdk"
OFFSET = 0x7F11
OFFICIAL = bytes.fromhex("00e1f505")  # 100 000 000 ns


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("仅 Windows（改的是 Windows DLL）")
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    out = Path(sys.argv[1]).resolve()
    ns = int(sys.argv[2])
    dlls = out / "sdk" / "dlls_custom"
    dlls.mkdir(parents=True, exist_ok=True)
    (out / "bennett_deploy").mkdir(parents=True, exist_ok=True)
    for name in ("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libusb-1.0.dll", "libwinpthread-1.dll"):
        shutil.copy2(SDK / "dlls" / name, dlls / name)
    shutil.copy2(SDK / "libusb-1.0.dll", out / "sdk" / "libusb-1.0.dll")

    data = bytearray((SDK / "dlls" / "libdm_device.dll").read_bytes())
    assert bytes(data[OFFSET:OFFSET + 4]) == OFFICIAL, "官方 DLL 的该位置不是 100 ms，文件已变化"
    data[OFFSET:OFFSET + 4] = ns.to_bytes(4, "little")
    (dlls / "libdm_device.dll").write_bytes(bytes(data))
    digest = hashlib.sha256(bytes(data)).hexdigest().upper()

    src = (REPO / "sim2real" / "bennett_deploy" / "dm_can.py").read_bytes().decode("utf-8")
    anchor = '        "low_latency": ('
    assert src.count(anchor) == 1
    src = src.replace(anchor, f'        "custom": ("dlls_custom", "{digest}"),\n' + anchor)
    (out / "bennett_deploy" / "dm_can.py").write_bytes(src.encode("utf-8"))
    print(f"已生成：{out}\n睡眠 {ns} ns，sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
