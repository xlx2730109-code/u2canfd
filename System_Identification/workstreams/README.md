# 工作区地图

这里把不同问题分开放。比如测通信延迟的脚本、数据和结果只进入 `02_timing_latency`，不会和电机曲线或 IMU 工作混在一起。

编号同时表示大致顺序，但不是每一步都必须串行。例如 00、01、03、04 可以在不动真机时交叉检查。

| 编号 | 中文作用 | 现在是什么状态 | 下一步要得到什么 |
|---:|---|---|---|
| 00 | [模型链校核 SW→URDF→USD](00_model_chain_sw_urdf_usd/README.md) | 新增（第一步） | 逐 link 质量/质心/惯量四方对表 |
| 01 | [查清硬件和固件](01_hardware_inventory/README.md) | 已列好项目 | 不使能电机的设备快照 |
| 02 | [测通信、周期和延迟](02_timing_latency/README.md) | 已列好项目 | 能区分各种“慢”的时间报告 |
| 03 | [对关节顺序、方向和零位](03_joint_mapping_zero/README.md) | 已列好项目 | 八关节映射表和实物证据 |
| 04 | [核对尺寸、质量、质心和惯量](04_rigid_body_properties/README.md) | 已列好项目 | CAD 与实测参数表 |
| 05 | [达妙 24 V 官方曲线模型](05_motor_datasheet_model/README.md) | 已复制并检查 | 真正使用查表模型的公平对比 |
| 06 | [建立统一的数据记录工具](06_data_acquisition/README.md) | 已列好项目 | 先用假数据完整跑通一次 |
| 07 | [让真机和仿真关节响应对齐](07_actuator_identification/README.md) | 等待前置检查 | 单关节，再到八关节的辨识参数 |
| 08 | [检查编码器和 IMU](08_sensor_identification/README.md) | 已列好项目 | 单位、方向、噪声和时间报告 |
| 09 | [处理脚和地面接触](09_contact_identification/README.md) | 后面再做 | 地面和脚垫参数 |
| 10 | [用新动作检验模型](10_simulation_validation/README.md) | 已列好项目 | 通过、不通过或需要补实验 |
| 11 | [接入训练和真机部署](11_training_deployment/README.md) | 最后做 | 独立训练结果和分级真机验收 |
| 12 | [闭链模型一致性](12_closed_chain_model/README.md) | 新增 | URDF/USD/MuJoCo 与实物闭链的足端轨迹对比 |
| 13 | [电源与温度](13_power_thermal/README.md) | 新增 | 电压/温度记录与对包络的影响 |
| 14 | [对齐控制方式](14_control_contract/README.md) | 已列好项目（原 00 号，为不改其他编号而移到此处） | 一份真机与仿真差异表 |

## 每个工作包里面怎么放

实际开始某项工作时，才创建需要的子文件夹，不提前制造一堆空目录：

```text
<workstream>/
├── README.md
├── config/
├── scripts/
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── params/
│   ├── figures/
│   └── reports/
└── references/
```

这些名字的意思：

- `config/`：这项工作的参数和设置；
- `scripts/`：只服务这项工作的脚本；
- `data/raw/`：刚采到的原始数据，只增加、不覆盖；
- `data/processed/`：对齐、清洗或滤波后的数据；
- `results/`：参数、图片和结论报告；
- `references/`：只与这项工作直接有关的资料。

一个文件只认一个主要归属。其他工作包需要它时，通过链接或实验身份证引用，避免出现几份都能修改、最后不知道哪份是真的。


## 阶段关卡（Plan.md 的 G0–G7）与工作包对照

| 关卡 | 主要工作包 |
|---|---|
| G0 控制方式与安全 | 00、14、01、03、13（寄存器、TIMEOUT、OC_Value） |
| G1 数据记录可靠 | 02、06 |
| G2 单关节打通流程 | 03、07 |
| G3 八关节悬空采集 | 07、13 |
| G4 仿真重放与自动拟合 | 05、07、12 |
| G5 新动作考试 | 10 |
| G6 训练与部署 | 11 |
| G7 扩展（传感器、接触、刚体） | 04、08、09、12 |

刚体参数（04）与 G3 之后的执行器辨识可以并行，因为 04 只需要称重和悬吊，不使能电机。
