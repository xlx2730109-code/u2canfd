# libdm_device.dll 睡眠常量补丁说明

对象：`sim2real/sdk/dlls/libdm_device.dll`（官方，Windows x64，204 579 字节）。结论与测量见根 `README.md` 的 01-K36 至 01-K38。

## 三个版本

| 版本 | 文件 | SHA256 | 偏移 `0x7F11` 起 4 字节（小端 32 位） | 睡眠 |
| --- | --- | --- | --- | --- |
| 官方 | `sim2real/sdk/dlls/libdm_device.dll` | `DB7FC43D0D60B847479DCB58CFBE13E524B0CA43C5DFA847D14E760BC0E2A8A7` | `00 E1 F5 05` = 100 000 000 | 100 ms |
| 补丁（现用） | `sim2real/sdk/dlls_lowlatency/libdm_device.dll` | `252FAEFDC70F1C52D61F93E74E29B07F871F00230E75985D03CAE7C43B4AF15E` | `40 42 0F 00` = 1 000 000 | 1 ms |
| 试验 custom | 不进仓库，由脚本生成 | `32361520A4FAFCAC5D47B0F8D71FE285F75DAE68854D6182D003CBF564012212` | `20 A1 07 00` = 500 000 | 0.5 ms |

三者其余字节完全相同。

## 补丁位置与语义

- 位置：`.text` 节，文件偏移 `0x7F11`（虚拟地址 `0x29F3A8911`），指令 `mov edx, imm32`，后接 `xor ecx, ecx` 与 `call` 到 `std::this_thread::sleep_for` 的导入桩；立即数是纳秒，`ecx` 为秒（0）。
- 作用：该睡眠所在循环决定收到的帧多久交给回调一次（推测，由批间隔与睡眠时长吻合推断）。
- 为什么 0.5 ms：DLL 用的 MinGW `sleep_for` 底层是 `libwinpthread` 的 `nanosleep`，实测睡不足 1 ms 几乎不睡，睡 1 ms 以上实际约 15 ms，且 `timeBeginPeriod(1)` 无效（根 README 01-K37）。0.5 ms 落在"不睡"一侧，接收线程不再停顿，代价是多占约 1 个核（01-K38）。
- 没有验证 0.9 ms 与 1.5 ms 两个值；`probe_sleep_for.py` 已直接显示 0.5 ms 以下不睡、1 ms 以上约 15 ms，所以只是未逐个重测，不影响结论。

## 重建

不在仓库里提交二进制。用官方 DLL 重建试验版：

```text
python B_motor_comm/scripts/make_test_dll.py <输出目录> 500000
```

脚本会断言官方 DLL 在偏移 `0x7F11` 处确实是 `00 E1 F5 05`，写入新值，并生成配套的 `dm_can.py`（登记名为 `custom` 的库）。用 `BENNETT_DM_CAN_PATH=<输出目录>/bennett_deploy/dm_can.py`、`BENNETT_DM_SDK_PROFILE=custom` 运行 `zero_torque_stream.py`。重建出的文件与测量时的副本逐字节一致（SHA256 同上）。

## 风险

- 闭源二进制的改动，厂商不认；升级 SDK 或固件后不能沿用。
- 用 custom 做真机部署前，应先做长稳（10 至 30 分钟）、CPU 占用与 2 kHz 余量测试（根 README 待完成）。

## Linux aarch64（Jetson）版

对象：`sim2real/sdk/dlls_linux_aarch64/libdm_device.so`（117 912 字节，SHA256 `4A73C8EC7D64342F0D0A1156D5102F519C5F176FB23696740F254FE6492C3F2A`，来自 Jetson 的 `seeed` 用户目录）。

| 版本 | 偏移 `0x123B8` 起 4 字节（小端 32 位） | 睡眠 | SHA256 |
| --- | --- | --- | --- |
| 官方 | `00 E1 F5 05` = 100 000 000 | 100 ms | `4A73C8EC…3F2A` |
| 0.5 ms 副本 | `20 A1 07 00` = 500 000 | 0.5 ms | `DA573E3B874BBFBEE974323068C36B5CC5FBD14FA91F51E1EF1ED4115F530FEB` |

- 该常量在文件里只出现一次（偏移 `0x123B8`，同一文件 `0x12448` 处是 10 ms 常量），与 Windows 版是同一个设计。Linux 的 `nanosleep` 精确，所以改成 1 ms 也应有效，实测只做了 0.5 ms。
- Jetson 实测（根 README 01-K83、01-K85）：官方版时龄均值约 47 ms，0.5 ms 副本约 0.48 ms，且发令环更稳；仓库不提交 0.5 ms 副本，需要时在 Jetson 上用下面的方法重建。
- 重建：复制 `libdm_device.so`，在偏移 `0x123B8` 写入 `20 A1 07 00`（写前断言原值为 `00 E1 F5 05`）；用同样目录结构放在另一棵目录树里运行，`dm_can.py` 按相对路径找库。
- 加载要求：库用到 `libusb_init_context`，需要 libusb 1.0.27 以上；系统自带 1.0.25 时会报 `undefined symbol`，可在装有 libusb 1.0.29 的 conda 环境里运行，或用只转调 `libusb_init` 的垫片库（只用于测试）。
