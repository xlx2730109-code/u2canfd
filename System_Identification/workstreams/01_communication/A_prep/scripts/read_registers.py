"""只读：读 8 个电机的寄存器（0x33 读参数），重复两遍，核对各电机是否相同、两遍是否一致。

不使能电机、不写任何寄存器、不发运动命令；与部署一样把两路 CAN 写成 CAN FD 1 M / 5 M（会改写盒子的运行期设置，
测完用 restore_channel_settings.py 恢复）。

Windows 用环境变量 BENNETT_DM_SDK_PROFILE 选库；建议 low_latency，读得快。
用法：
    python read_registers.py
输出：../results/<时间>_registers.json 与 ../results/<时间>_registers.md（对比表）
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RESULTS = HERE.parent / "results"
DM_CAN = REPO / "sim2real" / "bennett_deploy" / "dm_can.py"

spec = importlib.util.spec_from_file_location("bennett_dm_can_loader", DM_CAN)
dmc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmc
spec.loader.exec_module(dmc)

SN = "D6977F56F86C64B77B316E7154FA6DF3"
MOTORS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}
NAMES = {1: "FL_thigh", 2: "FL_calf", 3: "FR_thigh", 4: "FR_calf", 5: "RL_thigh", 6: "RL_calf", 7: "RR_thigh", 8: "RR_calf"}
# 只读这些寄存器（不含 SN 的字符串类、不含 0x50 以上的实时量）
REGS = [r for r in dmc.DM_REG if r.value <= 36]


def main() -> int:
    init = [dmc.DmActData(dmc.DM_Motor_Type.DM8006, dmc.Control_Mode.MIT_MODE, cid, 0x10 + cid, ch) for cid, ch in MOTORS.items()]
    ctl = dmc.Motor_Control(1_000_000, 5_000_000, SN, init, device_type=dmc.dmcan_device_type.USB2CANFD_DUAL, auto_enable=False)
    time.sleep(0.5)
    motors = {cid: ctl.getMotor(ch, cid) for cid, ch in MOTORS.items()}

    passes = []
    for n in range(2):
        data = {}
        for cid, m in motors.items():
            row = {}
            for reg in REGS:
                v = ctl.read_motor_param(m, int(reg), timeout=0.5)
                row[reg.name] = None if v is None else (float(v) if not isinstance(v, int) else int(v))
            data[cid] = row
        passes.append(data)
        print(f"[第 {n + 1} 遍] 完成，无应答寄存器数：{sum(v is None for r in data.values() for v in r.values())}")

    stamp = datetime.now()
    out = {"time_local": stamp.isoformat(timespec="seconds"), "library": getattr(dmc, "_dm_sdk_profile", "linux"), "motors": NAMES, "passes": passes}
    RESULTS.mkdir(parents=True, exist_ok=True)
    base = RESULTS / f"{stamp.strftime('%Y%m%d_%H%M%S')}_registers"
    base.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # 对比表
    lines = ["# 寄存器读取（只读，" + stamp.isoformat(timespec="seconds") + "）", "",
             "| 寄存器 | " + " | ".join(f"{cid}:{NAMES[cid]}" for cid in MOTORS) + " | 两遍一致 |",
             "|---|" + "---|" * (len(MOTORS) + 1)]
    for reg in REGS:
        vals = [passes[0][cid][reg.name] for cid in MOTORS]
        same = all(passes[0][cid][reg.name] == passes[1][cid][reg.name] for cid in MOTORS)
        def fmt(v):
            return "—" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
        lines.append(f"| {reg.name} | " + " | ".join(fmt(v) for v in vals) + f" | {'是' if same else '否'} |")
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[结果] {base.with_suffix('.md').relative_to(REPO)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
