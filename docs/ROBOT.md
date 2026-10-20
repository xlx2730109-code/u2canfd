# Bennett 四足机器人总览（ROBOT.md）

这是用户（徐龙翔）的四足机器人的总览。**任何会话里涉及这台机器人，先读本文件全文**，再按需读各工作包 README（路径见第 11 节）。

状态标记：【已确认】读文件或实测得到；【用户确认】用户亲口说明；【推测】我推断，没验证；【未知】还不知道。最后更新：2026-10-17。

## 1. 课题与目标

- 课题：《空间过约束多模式附肢设计与四足机器人行为生成研究》，北京交通大学机电学院，机械工程，学生徐龙翔（25121068），导师武建昫。开题 2026-09 通过。【已确认】
- 对象：Bennett 空间过约束四足机器人（每腿 2 个主动关节，整机 8 个）。
- 研究内容：机构建模与负载匹配、闭链仿真模型、多地形强化学习（Isaac Lab + PPO）、Sim2Sim（MuJoCo）与 Sim2Real、系统辨识；后续用附肢折展面探索步行加划水的多模式。
- 当前阶段：系统辨识。第一个工作包（00 号，模型链校核）已收尾，下一个是通信辨识（02 号）。
- 另一件事（Novelty，新机构探索）与本机器人无关，不在本文件范围。

## 2. 构型与运动学

**整体**【已确认】
- 8 个主动关节，顺序固定：`FL_thigh, FL_calf, FR_thigh, FR_calf, RL_thigh, RL_calf, RR_thigh, RR_calf`。
- 机体约 58 × 43 × 40 cm（长 × 宽 × 站立高）；腿模块化，四条腿前后左右镜像布置。
- 样机的装配方式按**右腿**来，左腿是右腿的镜像【用户确认】。

**一条腿**【已确认】
- thigh 与 calf 两个电机**同轴**，轴线倾斜 45°：轴向 (0, ±0.707, 0.707)（左 +、右 −），calf 轴相对 thigh 沿轴偏移 0.0357 m。
- thigh 绕这根斜轴带动整条腿转动；calf 驱动腿内部的闭环。
- 闭环是空间 Bennett 四杆：thigh、calf、`_2`、`_1` 四个杆件、四个转动副。加 thigh 的主动关节，整腿 **2 个自由度**。
- 三个被动关节 `_3`、`_1`、`_2` 只取决于 calf，与 thigh 无关：calf = −0.85 时 (`_3`, `_1`, `_2`) = (1.044, −1.044, −0.830)，calf = 0 时全 0，calf = 0.5 时 (−0.870, 0.870, 0.479)。
- 名义几何参数（开题报告）：a = 180 mm，α = β = 45°，装配角 σ = 145°，Bennett 条件 a/sinα = b/sinβ。

**足端工作空间**【已确认】
- 理论上是以那根斜轴为轴的圆环面（开题报告 `工作空间-3.png`）；训练用的关节范围只覆盖其中一块。
- 前左腿（thigh −0.8 到 0.8，calf −0.9 到 0.55，机身系，髋在 x 0.211、y 0.068）：x −0.143 到 0.435 m，y 0.009 到 0.392 m，z −0.282 到 −0.068 m；髋到足 0.161 到 0.400 m。
- 默认站姿：thigh（FL、RL 为 +0.08，FR、RR 为 −0.08）、calf −0.16；前左足端 (0.204, 0.184, −0.269) m，几乎在髋正下方（x 差 7 mm），向外撑开约 0.12 m。
- 没有奇异位置问题；关节位置、轴方向无误【用户确认】；关节范围没有独立核对，训练出来的策略一般不别扭【用户确认】。
- 足端运动耦合【已确认，2026-10-16 用 MuJoCo 模型算前左腿，右腿与后腿按镜像、同构推断】：默认站姿下足端相对髋 (−0.007, 0.110, −0.278) m；thigh 每转 1 rad，足端基本只沿 x 前后移 0.274 m；calf 每转 1 rad，足端移 (x −0.157, y +0.258, z −0.111) m，即 calf 同时决定横向位置与高度，没有独立的外展自由度。
- 腿接近伸直【已确认】：足端相对髋最低 z 为 −0.294 m，默认 −0.278 m，机身最多只能再升 16 mm。calf 退 0.1 rad 时足端抬高 12.8 mm，同时向内 24.4 mm、向前 14.4 mm；向外移动几乎不改变高度，向内必须抬脚；默认站姿在“向内”的极限（同一高度下向内只剩 5–11 mm），侧移比前后移动更受机构限制，但已被训练出来（用户确认）。机身越低，足端活动范围越大。详见 `bennett_rl/README.md` 的 K37。
- 训练用关节限位：thigh ±0.8 rad，calf −0.9 到 0.55 rad；被动关节 ±3.14。

**零位**【用户确认】
- 模型（SW、URDF、USD）的零位是 SW 展示的姿态。样机零位是用户肉眼标定的，无法与模型零位完全一致，可能有误差，大小未知。

## 3. 硬件

**电机**【已确认】8 × 达妙 DM-J8006-2EC V1.1
- 减速比 6:1，额定转矩 8 N·m，峰值 20 N·m，双编码器；单个质量 559 g（经销商，与 SW 一致）。
- 寄存器快照（2026-09-16，不使能读取）：Gr 6，NPP 21，PMAX 12.5，VMAX 45，TMAX 20，`KT_Value` 0，`GREF` 1，`Damp` 约 5e-4 到 1.1e-3，`Inertia` 约 1.2e-4，`Rs` 约 0.41，`KP_APR` 54，`I_BW` 1000，`V_BW` 40；**`TIMEOUT` 0（无通信看门狗）**，**`OC_Value` 0.8（含义待查）**，`UV_Value` 15，`OV_Value` 54，`OT_Value` 100；`sub_ver` 51 与 52 混用（`RL_calf`、`RR_thigh`、`RR_calf` 为 52）；`FR_thigh` 与 `RL_thigh` 的 `Rs`、`LS`、`Flux`、`Damp`、`Inertia` 逐位相同（2026-10-19 重读两遍仍相同，不是读错，原因未明）；`can_br` 均为 9（5 Mbps，达妙手册编码表）。
- MIT 模式命令：`τ = Kp·Δq + Kd·Δv + τff`，Kp 12 bit 0–500，Kd 12 bit 0–5，位置 16 bit ±12.5 rad，速度 12 bit ±45，转矩 12 bit ±20；编码用截断。
- 输出端读数就是关节角（1:1，不用乘 6）。

**通信**【已确认】
- USB2CANFD 双路，CAN-FD，仲裁 1 Mbps，数据 5 Mbps（部署代码 `dm_can.py` 启动时写入，CAN FD 与 BRS 都开）。2026-10-19 读到盒子当前设置：ch0 经典 CAN 1 Mbps，ch1 CAN FD 1 M / 5 M，固件 `dual_app v1.0.0.7`；部署启动会把两路都改写成 FD 1 M / 5 M。
- CAN ID：ch0 = id 1,2,5,6（FL_thigh、FL_calf、RL_thigh、RL_calf），ch1 = id 3,4,7,8（FR、RR）；反馈 ID = 0x10 + 发送 ID。
- 达妙官方 `libdm_device.dll`（Windows）与 Linux `.so` 里有 100 ms 批量交付常量：反馈每约 109 ms 一批，控制器拿到的角度平均旧约 55 ms。Windows 有补丁版 `dlls_lowlatency/`（改成 1 ms），Linux 未补丁。2026-10-19 在 Windows 按部署负载（1 kHz、8 电机）实测：官方库反馈时龄均值约 50 ms、最大约 124 ms；补丁库均值约 8–9 ms、最大约 30 ms；部署同款（补丁库加 `timeBeginPeriod(1)`）均值约 7 ms、最大约 16 ms；批间隔约 15.6 ms，不是 1 ms：DLL 里的 `sleep_for(1 ms)` 经 MinGW 的 winpthreads 实际睡约 15 ms，睡不足 1 ms 则不睡；临时副本改成 0.5 ms 后时龄均值约 0.6 ms、最大约 5 ms，代价是多占约 1 个核（只试验、未部署）；部署应使用补丁库（用户已在用 `--dm_sdk_profile low_latency`）。Jetson（aarch64）上 2026-10-20 独立实测【已确认】：未补丁库时龄均值约 47 ms；0.5 ms 副本（仓库 `sdk/dlls_linux_aarch64_lowlatency/`）时龄 p50 0.39 ms、p99 1.25 ms，逐帧 1 kHz 跑 300 s 不丢帧；IMU 时龄最大约 1.6 ms；观测侧时序参数见 `System_Identification/params/timing.yaml`，动作侧延迟未测。

**计算与传感**
- Jetson：用户称 Nano；2026-10-19 经 SSH 读到【已确认】：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super，主机名 seeed，Ubuntu 22.04.5，L4T R36.4.3（JetPack 6.2），aarch64，Python 3.10.12；USB 设备模式地址 192.168.55.1，登录方式见全局规则（claude-lab，无 sudo，只在 dialout 组）。2026-10-19 当时 USB 上有 DM-IMU-L1（`/dev/ttyACM0`）、键盘、鼠标、蓝牙，没有 USB2CANFD 盒；2026-10-20 用户把盒子接到了 Jetson（`34b7:6632`，设备节点权限 666），当天接回 Windows；Jetson 的 `seeed` 用户家目录里有 aarch64 的 1.x `libdm_device.so` 与 pyserial，DeepSeek 已用它跑通 1 kHz × 8 电机零力矩（我的 claude-lab 账户看不到该目录，先前说“Jetson 上没有库”是错的）。SW 里有 Nano 与 DM-ORIN NX 两种文件。+ 手柄接收器；Windows 电脑与 Jetson 都用于部署；Jetson 联网问题仍在解决【用户确认】。
- IMU：DM-IMU，USB 串口 COM6，921600 baud，约 1000 Hz；reg1 加速度、reg2 角速度（rad/s）、reg3 欧拉角。已做安装倾角标定（倾角 1.13° → 0.72°）。【已确认】
- 电子部分的更多细节用户说在通信辨识时讲【未知】。

**供电**：6S 锂电，**5300 mAh 和 4000 mAh 两块换着用**；计算单元独立 12 V，降压模块；XT60 与电源分配板。【已确认】

**材料**：铝合金（腿部连接件、大腿、小腿，密度 2800）和碳纤维（连杆、碳板、碳管，2000）为主；电机连接件 1 为 3D 打印塑料（1250）；螺栓钢。机身是 SW 里的碳板碳管。【已确认】

## 4. 质量

| 模型 | 总质量 kg | 比样机 11.7 |
|---|---:|---:|
| 样机（5300 mAh） | 11.700 | — |
| 样机（4000 mAh） | 11.550 | — |
| SW（5300 mAh） | 11.980 | +0.28 |
| URDF（不含 `X_3`） | 11.968 | +0.27 |
| USD | 11.968 | +0.27 |
| MuJoCo（`_3` 设为 0.001 kg） | 11.972 | +0.27 |

- 样机称重：样机 + 机架 14.5 kg（5300 mAh），机架 2.8 kg。
- SW 偏重 0.28 kg 的原因没定位（SW 没建线缆本应偏轻，说明有零件被高估）；没有小秤，不做分件称重，用域随机化覆盖。
- SW 有 7 种零件覆盖了质量：电机 0.559 × 8 = 4.47 kg（占 37%）、电池 0.663、Jetson 供电 0.339、Jetson 套件 0.182、足 0.072 × 4、碳管连接件、亚克力外框。覆盖后惯量只是按几何等比例缩放，质心不变。
- 各杆件（URDF = USD）：base 5.0514 kg（含机身、电池、Jetson、供电、电子件、四个腿 base），质心 (0.00475, 0.00003, 0.01319) m，主惯量 0.01924 / 0.1432 / 0.1531；thigh 0.99385、calf 0.30971、`_1` 0.16224、`_2` 0.19132、足 0.072，各 × 4。
- 大腿惯量以 SW 右大腿为准（0.000690 / 0.003671 / 0.003934），左大腿取镜像（ixy、iyz 变号）。
- 质心、惯量没有实测（没有条件）。

## 5. 模型文件

链：SW →（手调质量惯量）URDF → USD；URDF → MuJoCo XML（脚本生成）。

| 模型 | 位置 | 状态 |
|---|---|---|
| SW | `E:\HuanCun\Desktop\sw_work\Bennett\Bennett_1\Bennett_test5.SLDASM` | 只用 `Bennett_1`；只读，走 COM API，不点界面 |
| URDF | `bennett_rl/source/bennett_rl/bennett_rl/assets/robots/Urdf_Bennett_3/urdf/Urdf_Bennett_3.urdf` | 25 个 link，每腿多一个闭链用的 `X_3`（`_2` 的复制体，不算质量）；哈希 affac3c8841ccedb |
| USD（**以后训练用**） | 同目录 `Urdf_Bennett_3.usd` | 21 个刚体，无 `_3`；哈希 a4a6ed72c6a287a1 |
| USD（不再用） | `Urdf_Bennett_4/urdf/Urdf_Bennett_3.usd` | 碰撞体做过简化（仿真速度约为 `_3` 的 2.8 倍，2048 环境每步 38–42 ms 对 107–116 ms），已同步更新质量惯量；哈希 8a01726eb20cfd2d |
| MuJoCo | `bennett_rl/sim2sim/models/bennett_3/bennett_3.xml`（同目录 `_genfinal.py` 生成） | 25 个刚体，总质量 11.972 kg；哈希 fe07b42372c534d3；旧 17 刚体模型在 `bennett_3_wuzuduan/`，`bennett_1/` 是 base 仍 4.22 的旧版 |

**闭链处理**【已确认】
- USD：闭合关节是完整转动副（`calf → _2` 与 `thigh → _1`，后者 `excludeFromArticulation = True`）；设计姿态下 4 条腿 × 5 个关节的框架位置、方向间隙全为 0。
- MuJoCo：复制杆 `_3` 加 `<connect>` 点约束（`solref 0.01 1`）；工作空间闭环间隙 ≤ 0.095 mm，重力加接触落地 3 s 最大间隙 0.653 mm。
- MuJoCo 的 `_3` 刚体删不掉（它是点约束所需）；质量降到 0.001 kg。
- **教训**：`_3` 近无质量（1 g）与被动关节 armature 近 0（1e-5）不能同时做，否则重力加接触下闭环张开 21 mm、机身塌；改 MuJoCo 必须做动态验证。
- MuJoCo 生成脚本的旧 bug 已修：特征向量矩阵可能带反射，需要检查行列式（影响 thigh、calf、足 12 个刚体的惯量方向）。
- 默认站姿下被动关节若保持 0，闭环间隙 34 到 38 mm，使用前要先解闭环。
- MuJoCo 所有关节默认 armature 0.01、damping 0.001、frictionloss 0.05（含被动关节），与 Isaac 不一致，不对齐（重点是训练迁移）。
- MuJoCo 打不开含中文的路径。

## 6. 仿真与训练

**Isaac Lab**（仓库 `E:\Project\Isaaclab\bennett_rl`）【已确认】
- 训练的细节、实验记录和待办以 `bennett_rl/README.md` 为准（它是 `bennett_1` 起新训练任务的主记忆）；本节只留摘要，可能滞后。
- 物理步长 0.005 s（200 Hz），decimation 4，策略 50 Hz；4096 并行环境，episode 20 s；PPO（rsl_rl）3000 次迭代，学习率 1e-3。
- 执行器：`DCMotor`，kp 28、kd 2，力矩限 8 N·m、饱和 20、速度限 20 rad/s；**armature、摩擦为空，没有延迟**。
- 动作 = 默认姿态 + 0.2 × action，按关节限位裁剪。
- 旧任务 Trot1 观测 50 维：机体角速度、重力投影、速度指令、关节位置（相对默认）、关节速度、上一步动作、步态相位。观测噪声：角速度 ±0.2、重力 ±0.05、关节位置 ±0.01、关节速度 ±1.5。
- 旧任务 Trot1 的域随机化（`bennett_1` 已另加腿部杆件质量、电枢惯量、零位偏置随机化与观测低通，见其 README）：执行器增益 Kp ×0.8–1.2、Kd ×0.7–1.3；摩擦 0.6–1.3；base 质量 ×0.9–1.1、base 质心 ±0.02 m；推力。**腿部杆件质量、关节摩擦、延迟、关节零位偏置都没有随机化。**
- 任务：`quad_leg_trot`（trot1、trot-motor-ab）、`quad_leg_free_gait`（1 到 4）、`quad_leg_go2`（7、8、10、11、12、13、slope1）、`quad_leg_gravel`、`quad_leg_slope`、`quad_leg_stair`、`quad_leg_standing_1/2`、`quad_leg_track`、`single_leg_rr_trace`、`single_leg_rr_jct` 等。
- **USD 指向**：`BENNETT_CFG_V6`（指向 `_4`）被 trot1、go2-11/12/13、go2-slope1、gravel1、slope3/4、stair1 使用；`go2-8`、`go2-10`、`single_leg_rr_jct` 用 `_3`。用户决定以后都用 `_3`，要训练时再改配置；`_3` 的碰撞体未简化，训练可能变慢。
- 历史问题：free_gait4 把抬腿高度加倍后训练崩溃（std 出现负值），效果差；抬腿高度一直难调。
- **bennett_1（2026-10-16 起的新训练任务，仅仿真）**【已确认】：代码在 `bennett_rl/source/bennett_rl/bennett_rl/tasks/manager_based/bennett/`（2026-10-17 起）：`bennett_1` 冻结为 S1，`bennett_2` 为工作版本，爬行变体在 `bennett_2`；观测先加噪声再做 12 Hz 低通（与部署端一致）。其爬行变体（go2-10 式相位时钟，观测 50 维）训出的 D4，在 Isaac 与 MuJoCo 里都能稳定前进、后退、侧移、转向，四脚各抬 31–55 mm，对 0–100 ms 的观测滞后仍稳（120 ms 失效）；训练用 `_4` 的 USD，D4 与 T9 在 `_3` 下也验证过能走。在 D4 上加横向速度项续训得到的 E2，Isaac 中侧移速度比由 0.63–0.90 升到 0.75–1.07，步态与抗延迟不变（K57）。MuJoCo 中后退、侧移的速度对地面摩擦敏感（摩擦 ×1.6 时降到指令的 0.4–0.65，K58），真机地面摩擦未测。详见 `bennett_rl/README.md` 的 K53、K54、K56–K58。**尚未上真机。**
- go2-10 在真机上前进步态很好、迁移准确；缺点是后退差、转向难操作、加速度大、走路砰砰、抬腿快【用户确认，2026-10-16；真机效果未独立验证】。仿真数据印证了抬腿快与落地冲击大（抬腿最大上升速度 0.65–1.04 m/s，落地前下降速度 0.37–0.67 m/s），后退、侧移、转向只能达到指令的 0.6–0.7。
- 域随机化范围**一律待定，不给具体数值**，后续再分析。候选项：base 质量、腿部杆件质量、base 质心、惯量、关节零位偏置。

**sim2sim（MuJoCo）**：`bennett_rl/sim2sim/sim2sim.py --task <名> --endpoint_trail`；用户验证 sim2sim 没问题。

## 7. 真机部署

代码在 `E:\HuanCun\Desktop\u2canfd`【已确认】
- 运行时 `sim2real/bennett_deploy/`（2026-10-17 起，由 `quad_leg_xu/bennett_deploy/` 迁入，原位置保留；`runtime.py`、`dm_can.py`、`contract.py`、`imu.py` 等），入口 `sim2real/*_deploy.py`，动态库 `sim2real/sdk/`，Windows 与 Linux 通用（Linux 未实测）；根目录 `damiao.py` 等是早期脚本，只加载官方 DLL。
- 主循环：命令 1 kHz，策略 50 Hz；起动状态机：回默认姿态 → 预热 → 等待授权 → 策略。
- 关节速度与 IMU 各加 12 Hz 一阶低通（训练里没有对应滤波）。
- bennett_1 爬行变体的观测是 50 维，前 33 维同上，末 17 维是相位信息：全局相位 sin/cos（2）、各腿相位 sin/cos（8，每条腿的 sin 与 cos 相邻）、应着地标志（4）、步态参数（3：0.55 Hz、占空比 0.78、摆动高度 0.045）；相位 =（策略启动后的时间 × 0.55 Hz）取小数部分，指令为零时全部取站立值；腿序 FL→RR→FR→RL。部署端要照此生成，参考 `bennett_rl/sim2sim/configs/bennett_1_crawl.py` 与 `sim2sim/sim2sim.py` 的 crawl 模式。【已确认训练与 sim2sim 的实现，未部署】 入口已写：`sim2real/bennett_1_crawl_deploy.py`（缺省 T9）、`bennett_1_free_deploy.py`（缺省 S1），`--check_only` 通过，尚未上真机；T9 训练时无反馈滞后，官方动态库（约 109 ms）下可能不稳。
- `--dm_sdk_profile`：缺省 Windows 为 `low_latency`（补丁版）、Linux x86 为 `original`、Linux aarch64（Jetson）为 `low_latency`（2026-10-20 起；此前缺省都是 `original`，不带参数会走官方库，反馈约每 110–120 ms 一批）；部署的时钟统一为 `time.perf_counter()`（此前 `time.monotonic()` 在 Python 3.11 的 Windows 上只有约 15.6 ms 精度，策略调度间隔曾是 15.6/31.2 ms 交替）。
- 软件力矩限 `--software_effort_limit_nm` 默认 0（关闭）；训练限 8 N·m，固件 TMAX 20。
- `sim_to_motor` 符号【已确认，代码注释与用户确认入口已验证】：腿部策略沿用 `bennett_deploy` 的表（FL、RL 的 thigh 为 +1，FR_calf、RR_calf 为 −1）；`free_gait_contract.py` 注释记载 +1 在实机验证过（2026-09-05），`3_quad_leg_track.py` 里的 −1 会使步态镜像、不适用于腿部策略。不改动。
- `技术文档_1.md` 里"关节指令乘 6"已被实测推翻。
- 反馈超时默认 0.5 s，对 109 ms 批量交付不会触发，等于没有保护。

## 8. 已知问题与风险

1. 通信：官方 DLL / Linux `.so` 的 100 ms 批量交付；补丁版实测反馈时龄均值约 8–9 ms、最大约 30 ms（Windows，2026-10-19）；Linux 未测；开题里的"约 36 ms"是批相位，不是常数。
2. 电机 `TIMEOUT` = 0，上位机崩溃时电机会保持最后一条命令。
3. 训练的执行器没有延迟、摩擦、电枢惯量（折算转子惯量约 1.2e-4 × 36 ≈ 4.4e-3，待验证）。【已确认】2026-10-20 实测动作侧延迟（命令到电机力矩出现）小于约 1.6 ms（1 kHz 分辨率上界，真实值未分辨），测了 `FR_calf`、`FR_thigh`、`FL_calf` 三个关节、两个通道，与关节、通道、kp 都无关（根 README 07-K8、07-K9）；首次响应的节拍虽小，但力矩到位是约 10–12 ms 的一阶滞后（10–90% 约 18–24 ms，DeepSeek 开环实测、我核对其原始数据，根 README 07-K11），训练里的执行器要加这个滞后，摩擦、电枢惯量仍待辨识。
4. 训练 USD 指向仍有 `_4`（见第 6 节）。
5. 样机零位肉眼标定，误差未量化。
6. 旧系统辨识数据（2026-09-16、09-17）用的是官方 DLL，只作线索，不进新数据集。
7. 旧辨识结论里"传输延迟约 0.0–0.2 ms"是被批量交付骗了的结果。
8. bennett_1 爬行变体对观测滞后敏感：D4 在 100 ms 以内稳，120 ms 时会摔；官方动态库的批量交付最坏约 109 ms（动作延迟另算，训练里还没有）。
9. bennett_1 爬行变体的后退、侧移速度对地面摩擦敏感（仿真中摩擦越大速度越低），真机地面摩擦未测，实际速度可能低于指令（`bennett_rl/README.md` K58、T34）。
10. 部署默认 1000 Hz 发令，每帧电机回一帧，每路约 8000 帧/s，CAN FD 下负载约 45–50%，若退回经典 CAN 1 Mbps 接近饱和（推测，待实测）。
11. 仓库里 `sdk/dlls/` 的 Linux `.so` 是 x86-64，Jetson 是 arm64，所以另放了 aarch64 版 `sdk/dlls_linux_aarch64/` 与 0.5 ms 副本 `sdk/dlls_linux_aarch64_lowlatency/`（加载需 libusb 1.0.27 以上，系统自带 1.0.25 会报 `undefined symbol`）；未补丁库同样有 100 ms 批量交付（时龄均值约 47 ms，且每 100 ms 让发令环停顿约 14 ms），改成 0.5 ms 后时龄均值约 0.4 ms、发令环更稳（我独立实测；DeepSeek 报告的"抖动变大"是他的回调不做 SDK 解码所致，根 README 01-K92）。
12. 反馈新鲜度用的是主机回调时刻，没有用帧自带时间戳（帧头时间戳单位为 ns，但有跳变）；部署环境 Python 3.11 的 `time.monotonic()` 在 Windows 上分辨率约 15.6 ms，部署日志里的反馈时龄只有 0 或 16 ms 两档，真实时龄要用 `perf_counter` 算（策略 tick 处均值约 6.6 ms）。
13. 12 Hz 观测低通：`bennett_1_*` 训练端与运行时一致；缺省策略 `go2-10` 以及 `trot1`、`free_gait3` 训练端没有，而运行时缺省仍加 12 Hz（约 13.3 ms 滞后），原先不提示，2026-10-20 起启动时打印提示；用户称 `go2-10` 真机迁移很好，是带滤波的情况，是否关闭由用户定。
14. Windows 上发令环有不规则的 10–20 ms 停顿（每 30 s 约 6–9 次，与负载无关，提高进程优先级可减半）；Python 3.11 逐帧发送 8 个电机的上限约 1.37 kHz；`fill_can_queue`/`send_can_queue` 队列发送能把每周期发送耗时降到约 1/5，但 Windows 10 次里 6 次缺帧、Jetson 官方库 6 次里 3 次缺帧（0–100 ppm），且在 `low_latency` 库下没有抖动收益，不采用（2026-10-20）。
15. 看门狗：8 台电机 `TIMEOUT` 都是 0（读两遍确认）；上地面前必须开，单位未核实（可能 50 µs 或 ms），已写好测试脚本（根 README 01-K75、01-T8）。
16. 【用户确认】用户称之前在 Jetson 上复现过并成功迁移（`go2-10` 等）；DeepSeek 称当时 Jetson 只有未补丁的 `.so`（未核实），若属实则那些成功是在平均约 49 ms、最大约 100 ms 的反馈时龄下取得的（根 README 01-K77）。
17. 【已确认】Jetson 上盒子偶发从 USB 总线消失（DLL 内置心跳线程报 `heartbeat transfer failed! error LIBUSB_ERROR_TIMEOUT`，DeepSeek 约 13 次会话 2 次，我 2026-10-20 复现了一次：1 kHz × 600 s 在 61 s 反馈全停、盒子仍在 `lsusb` 里但不转发 CAN；1 kHz 5 分钟以上约一半失败，500 Hz 目前 0/4，建议部署发令降到 500 Hz（01-K94、01-T22）），只能拔插；部署靠反馈超时（缺省 0.5 s）发现，盒子消失后失能命令发不出，电机保持最后一条命令，根本保护是电机看门狗（第 15 项，根 README 01-K91、01-T21）。
18. 【已确认】2026-10-20 实验约 2.5 小时后 8 台电机同时报状态码 9（欠压，寄存器 `UV_Value`=15 V），电池耗尽或供电断开，换电池后恢复；电机状态码：0 失能、1 使能、8 超压、9 欠压、A 过流、B MOS 过温、C 线圈过温、D 通讯丢失（看门狗触发后锁存，需断电）、E 过载；每次使能前后记录电池电压；`A_prep/scripts/query_states.py` 可只读查询状态码与温度（根 README 07-K20）。
19. 【已确认】执行器实测（根 README 07-K11 至 07-K19）：关节库仑摩擦 0.22–0.40 N·m（MuJoCo 缺省 0.05）；开环力矩阶跃上报力矩 10–90% 约 15–30 ms；闭环等效延迟加滞后约 8–10 ms；反馈力矩有约 −0.035 N·m 的常值偏置；8 个电机行为一致。有效 kp、kd 是否低于标称，取决于等效延迟是否单独建模，以根 README 为准。

## 9. 系统辨识

- 工作区：`E:\HuanCun\Desktop\u2canfd\System_Identification\`，总计划 `Plan.md`（关卡 G0–G7），工作包在 `workstreams/`。
- 9 个工作包：00 模型链校核（**已收尾**）、01 硬件固件、02 时序延迟（**下一个**）、03 关节零位、05 电机曲线、06 数据记录、07 执行器辨识（含新动作考试）、08 传感器、11 训练部署。
- 每个工作包有 `claude/`、`codex/`、`comparison/`、`inputs/` 子目录（Claude 与 Codex 各做一份再对比）。
- 规则：原始数据只增不改；每次实验写 manifest（含电池、起止电压、DLL 版本与哈希）；拟合集与考试集分开；**未经用户当次明确同意，不使能电机、不发送运动命令**。
- 通信辨识第一步：不使能电机、零力矩，测反馈周期和延迟；测量时不要同时在同一台电脑上跑训练。

## 10. 已做的决定与已接受的近似

- 训练以后用 `Urdf_Bennett_3` 的 USD；MuJoCo 用 `bennett_3.xml`。
- SW 比样机重 0.28 kg 不再追查，用域随机化覆盖。
- SW 框选 base 漏了亚克力外框和双头螺杆（约 0.0124 kg），不补。
- 质心、惯量不实测；关节位置、轴方向不再跨模型核对。
- MuJoCo 与 Isaac 的被动关节参数不对齐；sim2sim 初始化不用改。
- 对迁移最重要的是执行器模型和延迟，不是 USD 本身（调研结论：执行器与延迟 > 关节摩擦 > 观测设计 > 质量惯量）。

## 11. 路径索引

| 内容 | 路径 |
|---|---|
| 本文件 | `E:\HuanCun\Desktop\u2canfd\docs\ROBOT.md` |
| 系统辨识各工作包进展（总览表 + 每包一节；00 号包最详细，含质量表、闭链、哈希） | `E:\HuanCun\Desktop\u2canfd\README.md` |
| 系统辨识总计划 | `E:\HuanCun\Desktop\u2canfd\System_Identification\Plan.md` |
| 排查记录（通信、寄存器、训练配置） | `E:\HuanCun\Desktop\u2canfd\System_Identification\docs\AUDIT_v0.md`、`BASELINE_AUDIT.md` |
| 训练代码 | `E:\Project\Isaaclab\bennett_rl`（`SIM2REAL_PIPELINE.md` 是 sim2real 全流程清单） |
| 训练工作日志（奖励、PPO、实验、训练待办，约 70 KB） | `E:\Project\Isaaclab\bennett_rl\README.md`；涉及训练、`bennett_1`、奖励、步态、域随机化时读其「基准与结论」和「待完成」，不必通读 |
| 部署代码 | `E:\HuanCun\Desktop\u2canfd\sim2real\`（`quad_leg_xu\` 为迁移前的原位置，保留） |
| 开题材料 | `E:\HuanCun\Desktop\徐龙翔\归档\开题\`（最终版 `材料_提交\20260917_徐龙翔-开题报告.docx`；配图 `材料_配图、视频、Gif\`） |
| 工作区临时文件 | `E:\HuanCun\AI_tmp\Claude\` |

## 12. 未知与待确认

- 电子部分的细节（CAN 盒型号与固件、终端电阻、USB 口分配、供电细节）：用户让我上机时自己读取核对；Jetson 的 JetPack 版本与实际型号未知。
- 补丁版 DLL 的真实反馈延迟、Linux 上的情况。
- 电机寄存器 `OC_Value`（0.8）、`KT_Value`（0）的含义；`FR_thigh`/`RL_thigh` 五个寄存器逐位相同的原因（2026-10-19 重读两遍，确认不是读错）。
- 折算到关节的电枢惯量、摩擦、延迟（执行器辨识要得到）。
- 样机零位偏差的大小；真机的齿隙、松动（旧笔记提过"其他腿有点松动"）。
- 足的形状、半径、软硬、摩擦（仅有 URDF 网格和 0.072 kg）。

## 13. 协作偏好

- 用中文、通俗；做事要快，超过一分钟的操作先说明；只做被要求的范围，别扯远。
- 软件（SolidWorks 等）一律走 API，不点界面，SW 只读。
- 两个仓库（u2canfd、bennett_rl）的代码必须 Windows 与 Linux 都能跑，用 `pathlib`，LF 换行，Windows 专属的东西用 `sys.platform` 判断。
- 临时文件放 `E:\HuanCun\AI_tmp\Claude\任务简称_日期\`。
- 有工作量大的任务，用 README 笔记（基准与结论 / 已完成 / 待完成），格式见全局规则。
