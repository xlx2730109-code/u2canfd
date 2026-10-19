# 系统辨识总览

目标：让强化学习策略从仿真迁移到样机时差距尽量小。每个工作包一个一级标题（标题即文件夹路径），下含「基准与结论」「已完成」「待完成」。00 包的结论编号沿用 K1–K29，之后新包的编号带包号，如 `01-K1`。总计划见 `System_Identification/Plan.md`。

| 工作包 | 作用 | 状态 |
|---|---|---|
| 00 模型链校核 | SW → URDF → USD → MuJoCo 与整机称重对齐 | 已收尾（2026-10-15） |
| 01 通信辨识 | 测反馈周期、控制节拍、观测新鲜度，给训练的延迟随机化定范围 | 方案已定（2026-10-19），未采集 |
| 03、05、06、07、08、11 | 零位、电机曲线、数据记录、执行器、传感器、训练部署 | 未开始，开始时在本文件新增一节 |
| sim2real 部署 | 部署运行库、动态库、工具与 bennett_1 的策略入口（Windows、Linux） | 进行中（2026-10-17）：T9、S1 入口已写好，待真机试验 |

# System_Identification/workstreams/00_model_chain_sw_urdf_usd

目标：仿真里的机器人 = SW 里的机器人 = 桌上的样机。全程只读，不改训练在用的模型。

## 基准与结论

### A. 样机总览


| 项目           | 内容                                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| 构型           | 四足；每腿 2 个主动关节（thigh、calf，同轴，轴线倾斜 45°）加内部 Bennett 空间四杆闭环；整机 8 个主动关节      |
| 尺寸           | 约 58 × 43 × 40 cm（长 × 宽 × 站立高）                                                                     |
| 质量           | 11.7 kg（5300 mAh 电池）/ 11.55 kg（4000 mAh）                                                                 |
| 关节顺序       | FL_thigh、FL_calf、FR_thigh、FR_calf、RL_thigh、RL_calf、RR_thigh、RR_calf                                     |
| 电机           | 8 × DM-J8006-2EC（6:1，额定 8 N·m，峰值 20 N·m，双编码器）；CAN ID 1–8，ch0 = id 1,2,5,6，ch1 = id 3,4,7,8 |
| 通信           | USB2CANFD 双路，仲裁 1 Mbps、数据 5 Mbps                                                                       |
| 控制与传感     | Jetson + DM-IMU（USB）+ 手柄；Windows 与 Jetson 都部署；Jetson 具体型号、联网问题待通信辨识时确认              |
| 供电           | 6S 锂电（5300 / 4000 mAh 两块换着用），计算单元独立 12 V                                                       |
| 材料           | 铝合金和碳纤维为主，少量 3D 打印；机身为碳板碳管                                                               |
| 仿真           | Isaac Lab：物理 200 Hz、策略 50 Hz；MuJoCo：步长 0.002 s                                                       |
| 训练用执行器   | DCMotor，kp 28、kd 2，力矩限 8 N·m（峰值 20），速度限 20 rad/s                                                |
| 训练用关节范围 | thigh ±0.8，calf −0.9 到 0.55 rad                                                                            |
| 零位           | 模型零位 = SW 展示的姿态；样机零位肉眼标定，可能有误差                                                         |

### B. 最终版本（2026-10-15，SHA256 前 16 位）


| 文件                                      | 哈希             | 用途           |
| ------------------------------------------- | ------------------ | ---------------- |
| `Urdf_Bennett_3/urdf/Urdf_Bennett_3.urdf` | affac3c8841ccedb | 基准           |
| `Urdf_Bennett_3/urdf/Urdf_Bennett_3.usd`  | a4a6ed72c6a287a1 | 训练用（以后） |
| `Urdf_Bennett_4/urdf/Urdf_Bennett_3.usd`  | 8a01726eb20cfd2d | 已同步，不再用 |
| `sim2sim/models/bennett_3/bennett_3.xml`  | fe07b42372c534d3 | MuJoCo         |

### C. 已接受的近似

1. SW 比样机重 0.28 kg，不再分件称重，用域随机化覆盖。
2. SW 框选 base 漏了亚克力外框和双头螺杆，约 0.0124 kg，不补。
3. MuJoCo 里保留 `_3`，质量 0.001 kg，总质量比 USD 多 0.004 kg。
4. 样机的质心、惯量没有实测。
5. 样机零位是肉眼标定的，与模型零位可能有误差，误差没有量化。
6. 关节位置、轴方向按用户确认无误，不再做足端位置的跨模型对比。

### D. 模型来源

1. SW：`E:\HuanCun\Desktop\sw_work\Bennett\Bennett_1\Bennett_test5.SLDASM`（只用 `Bennett_1`）。
2. URDF：`bennett_rl/source/bennett_rl/bennett_rl/assets/robots/Urdf_Bennett_3/urdf/Urdf_Bennett_3.urdf`；25 个 link，每腿多一个闭链用的 `X_3`。
3. USD：同目录 `Urdf_Bennett_3.usd`；21 个刚体，无 `_3`。（`Urdf_Bennett_4` 的 USD 只简化了碰撞体，质量惯量相同。）训练以后都用 `Urdf_Bennett_3` 的 USD；`Urdf_Bennett_4` 的 USD 已同步更新，但不再使用。
4. MuJoCo：`bennett_rl/sim2sim/models/bennett_3/bennett_3.xml`，由同目录 `_genfinal.py` 读 URDF 生成。旧的 7.53 kg 模型在 `bennett_3_wuzuduan/`，`bennett_1/` 是 base 仍为 4.22 kg 的旧版。
5. 传递链：SW →（手调质量惯量）URDF → USD；URDF → MJCF（脚本生成）。

### E. 质量

总质量（kg）：


| 模型                         | 总质量 | 比样机 11.7 |
| ------------------------------ | -------: | ------------: |
| 样机（5300 mAh）             | 11.700 |          — |
| 样机（4000 mAh）             | 11.550 |          — |
| SW（5300 mAh）               | 11.980 |       +0.28 |
| URDF（不含 X_3）             | 11.968 |       +0.27 |
| USD                          | 11.968 |       +0.27 |
| MuJoCo（`_3` 设为 0.001 kg） | 11.972 |       +0.27 |

各杆件质量（SW、URDF、USD 一致）：


| link          | 单个 kg | 数量 |    合计 kg |
| --------------- | --------: | -----: | -----------: |
| base          | 5.05142 |    1 |     5.0514 |
| thigh         | 0.99385 |    4 |     3.9754 |
| calf          | 0.30971 |    4 |     1.2388 |
| `_1`          | 0.16224 |    4 |     0.6490 |
| `_2`          | 0.19132 |    4 |     0.7653 |
| 足            | 0.07200 |    4 |     0.2880 |
| **合计**      |         |      | **11.968** |
| `X_3`（不计） | 0.19132 |    4 |     0.7653 |

base 参数（与 SW 逐位一致）：


| 项目             | 值                                   |
| ------------------ | -------------------------------------- |
| 质量             | 5.05142 kg                           |
| 质心 x, y, z     | 0.00475, 0.00003, 0.01319 m          |
| 主惯量（小到大） | 0.019243, 0.143200, 0.153101 kg·m² |
| Ixz              | −0.00116 kg·m²                    |

单腿杆件的质心与主惯量（URDF = USD，四条腿相同，FL 腿的值）：


| link  | 质心 x, y, z（m）           | 主惯量（kg·m²，小到大）    |
| ------- | ----------------------------- | ------------------------------ |
| thigh | −0.00998, 0.04023, 0.00855 | 0.000797, 0.003560, 0.003942 |
| calf  | 0.03065, 0.05631, −0.04287 | 0.000065, 0.001548, 0.001579 |
| `_1`  | 0.02246, 0.00258, −0.10122 | 0.000017, 0.000648, 0.000649 |
| `_2`  | 0.03105, 0.00333, 0.06044   | 0.000031, 0.000770, 0.000772 |
| 足    | −0.00097, 0.00021, 0.00363 | 0.000013, 0.000015, 0.000015 |

SW 质量覆盖的 7 种零件：


| 零件                                 | 单个 kg | 数量 | 合计 kg |
| -------------------------------------- | --------: | -----: | --------: |
| DM-J8006 电机（与经销商 559 g 一致） |   0.559 |    8 |   4.472 |
| 电池 5300 mAh                        |   0.663 |    1 |   0.663 |
| Jetson 供电 12V                      |   0.339 |    1 |   0.339 |
| Jetson Nano 套件                     |   0.182 |    1 |   0.182 |
| 足                                   |   0.072 |    4 |   0.288 |
| 机身碳管连接件                       |  0.0057 |   16 |   0.091 |
| 亚克力外框                           |  0.0100 |    1 |   0.010 |

材料密度：


| 材料                 | 密度 kg/m³ | 用在哪                 |
| ---------------------- | ------------: | ------------------------ |
| 铝合金               |        2800 | 大腿、小腿、腿部连接件 |
| 碳纤维               |        2000 | 连杆、碳板、碳管       |
| 塑料（疑似 3D 打印） |        1250 | 电机连接件 1           |
| 钢                   |        8000 | 螺栓                   |

6. 样机实称：5300 mAh 电池 11.7 kg，4000 mAh 电池 11.55 kg。机身是 SW 里的碳板碳管（2020 铝型材只是存放架，不计）。
7. SW 比样机重 0.28 kg（2.4%），与电池无关；SW 没建线缆本应偏轻，说明有零件被高估；原因未定位（无小秤，不做分件称重），决定用训练时的质量域随机化覆盖。
8. 杆件的质量、质心、惯量，SW、URDF、USD 一致；URDF 与 SW 的 0.012 kg 差来自 base 的框选范围（见 10）。
9. base = 机身 + 电池 + Jetson + 供电 + 电子件 + 4×腿 base，与 SW 逐位一致。
10. SW 框选 base 可能漏了亚克力外框和双头螺杆，约 0.0124 kg（0.1%，推测）。
11. SW 有 7 种零件设置了质量覆盖；覆盖后惯量按几何形状等比例缩放，质心不变。
12. 材料密度见上表，铝件 2800、碳纤维 2000。
13. USD 已按 URDF 更新并保存：`Urdf_Bennett_3` 与训练实际使用的 `Urdf_Bennett_4`（只是碰撞体做了简化）两个 USD 的 21 个刚体，质量、质心、惯量与 URDF 全部一致（惯量相对误差 ≤ 2.2e-7）。
14. `X_3` 是 `_2` 的复制体（网格、质量相同），只为处理闭链，不算质量。USD 无它；MuJoCo 里删不掉（它是闭链的点约束所需的刚体），质量设为 0.001 kg（4 个共 0.004 kg）。

### F. 闭链

15. 每腿 thigh、calf 主动，`_1`、`_2` 被动；环路 thigh–calf–`_2`–`_1` 是空间 Bennett 四杆，加 thigh 共 2 个自由度。被动关节只取决于 calf，与 thigh 无关。


|              | USD                                                     | MuJoCo                                                    |
| -------------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| 闭合方式     | 完整转动副：`calf→_2`、`thigh→_1`（后者排除出关节树） | 复制杆 `_3` + `<connect>` 点约束（`solref 0.01 1`）       |
| 是否有 `_3`  | 无                                                      | 有                                                        |
| 闭合精度     | 设计姿态位置、方向间隙 0                                | 工作空间闭环间隙 ≤ 0.095 mm，`_3` 与 `_2` 转角 ≤ 0.56° |
| 被动关节参数 | 未读                                                    | armature 0.01，damping 0.001，frictionloss 0.05           |

16. 默认站姿（thigh ±0.08、calf −0.16）下，被动关节若保持 0，闭环间隙 34–38 mm，必须先解闭环再使用。

### G. 教训与规则

17. `_3` 近无质量（1 g）与被动 armature 近 0（1e-5）不能同时做：闭环张开 21 mm、机身塌；单独做没事。可用：`_3` 1 g + armature ≥ 1e-3，或保留 `_3` + armature 1e-5。所以 `_3` 只把质量降到 0.001 kg，被动关节参数不动。
18. 改 MuJoCo 模型必须做动态验证（重力加接触落地 3 s，看闭环间隙和机身高度）。
19. 当前 `bennett_3.xml` 已验证：总质量 11.972 kg，全 0 间隙 0 mm，重力加接触 3 s 最大间隙 0.653 mm，机身高 0.279 m。
20. MuJoCo 打不开含中文的路径；SW 只用 API 只读；Isaac Sim 通过 8226 端口读取（只读使用）。
21. 脚本：`claude/scripts/sw_read_mass.py`（零件质量）、`claude/scripts/sw_link_inertia.py`（合成各 link 并与 URDF 对比，先用整机自检）；结果：`claude/results/r01|r02|r03/`。
22. 各 link 的 SW 与 URDF 对比（Codex 与我独立各做一次，结果逐位一致；我的自检：整机质心差 0.068 mm、惯量差 0.014%）：除右大腿外，质量、质心、惯量全部一致。


| link                                    | 质心差 mm | 主惯量差（小 / 中 / 大）       | 张量相对差 |
| ----------------------------------------- | ----------: | -------------------------------- | -----------: |
| base                                    |      0.19 | −0.39% / −0.03% / −0.04%    |      0.06% |
| `FL_thigh`、`RL_thigh`                  |     0.041 | 0                              |          0 |
| **`FR_thigh`、`RR_thigh`**（URDF 改前） |     0.000 | **+15.45% / −3.02% / +0.19%** |  **3.47%** |
| 四个 calf                               |         0 | ≤ 0.01%                       |          0 |
| 四个 `_1`                               |         0 | ≤ 0.01%                       |          0 |
| `FL_2`、`FR_2`、`RL_2`                  |         0 | ≤ 0.02%                       |          0 |
| `RR_2`                                  |     0.170 | ≤ 0.03%                       |      0.29% |
| 四个足                                  |         0 | ≤ 0.05%                       |      0.04% |

23. 样机按右腿方式装配，左腿是右腿的镜像，所以大腿惯量以 SW 右大腿（`FR_thigh`、`RR_thigh`，两者相同）为准，左大腿取镜像（ixy、iyz 变号，其余不变）。URDF 原来用的是 SW 左大腿的值（与右大腿主惯量差 +15.45%），已在 2026-10-15 改成右大腿的值（见已完成）。
24. MuJoCo 生成脚本的惯量旧 bug：`eigh` 可能返回带反射的特征向量矩阵，直接转四元数会得到错的主轴方向，导致 `bennett_3.xml` 里 thigh、calf、足共 12 个刚体的惯量方向与 URDF 不符（相对误差最大 0.9）；质量、质心不受影响，`diaginertia` 数值本身也对。已在 `_genfinal.py` 的 `eig_inertia` 里加了行列式检查并重新生成。Isaac 的 USD 不受此影响。
25. 关节的位置、轴方向没有问题（用户确认），本包不再做足端位置对比。
26. 关节零位：模型（SW、URDF、USD）的零位是 SW 展示的姿态；样机零位是用户肉眼标定的，无法和模型零位完全一致，可能有误差，大小未知。
27. 腿的运动：thigh 与 calf 同轴（轴线 (0, ±0.707, 0.707)，倾斜 45°），thigh 绕此轴带动整条腿转动，calf 驱动内部闭环。足端理论工作空间是以这根倾斜轴为轴的环面（圆环状曲面，开题报告 `工作空间-3.png`）；训练用的关节范围只覆盖其中一块。
28. 前左腿足端在 base 系的范围（thigh −0.8 到 0.8、calf −0.9 到 0.55，MuJoCo 无头计算，髋在 x 0.211、y 0.068）：


| 项目                                    | 数值                      |
| ----------------------------------------- | --------------------------- |
| x（前后）                               | −0.143 到 0.435 m        |
| y（向左）                               | 0.009 到 0.392 m          |
| z（向上）                               | −0.282 到 −0.068 m      |
| 髋到足距离                              | 0.161 到 0.400 m          |
| 全 0 姿态足端                           | (0.201, 0.223, −0.280) m |
| 默认站姿（thigh 0.08、calf −0.16）足端 | (0.204, 0.184, −0.269) m |

   足端几乎在髋的正下方（x 相差 7 mm），向外撑开约 0.12 m。被动关节只取决于 calf：calf −0.85 时 (`_3`, `_1`, `_2`) = (1.044, −1.044, −0.830)，calf 0 时全 0，calf 0.5 时 (−0.870, 0.870, 0.479)。图见 `E:/HuanCun/AI_tmp/Claude/kinematics_20261015/workspace_FL.png`。
29. 没有奇异位置问题；关节范围没有独立核对，训练出来的策略一般不别扭。

### H. 需要域随机化的项目（本包检查得出；范围都待后续分析，不定具体数值）


| 项目         | 检查依据                                                               | 现有训练设置（Trot1）          | 建议           |
| -------------- | ------------------------------------------------------------------------ | -------------------------------- | ---------------- |
| base 质量    | SW 比样机重 0.28 kg（2.4%），原因未定位                                | 有                             | 保持，范围待定 |
| 腿部杆件质量 | 偏差可能在电机、铝件，即在腿上                                         | 无                             | 新增，范围待定 |
| base 质心    | 质心未实测                                                             | 有                             | 保持，范围待定 |
| 惯量         | 电机、电池、Jetson 的质量被覆盖，惯量只是按几何等比例缩放，可能有误差  | 改质量时惯量随之重算           | 待定           |
| 关节零位     | 样机零位肉眼标定，与模型零位可能有误差，大小未知；关节位置、轴方向无误 | 无关节位置偏置（只有观测噪声） | 新增，范围待定 |

---

## 已完成

- [X]  第 1 关：用 SW API 只读取全部零件质量（137 种零件、1244 个实例）。（10-14，见 K7、K8、K11）
- [X]  第 2 关（质量）：URDF 25 个 link 与 SW 子装配对表。（10-14，见 K8、K10）
- [X]  通过 Isaac Sim 接口只读取 USD 全部刚体与关节，与 URDF 逐项对比：只有 base 不同（质量 4.2214，差 −0.830 kg；质心差 19.7 mm；主惯量差 7.6%）。（10-14，见 K13）
- [X]  USD base 改成 URDF 值并保存（18:46），从磁盘文件重读确认。（10-14，见 K13）
  - 改前：base 4.2214 kg，质心 (−0.00008, 0, −0.0059)，惯量 0.0151 / 0.1320 / 0.1425。
  - 备份：`E:/HuanCun/AI_tmp/Claude/usd_compare_20261014/`。
- [X]  USD 闭链结构读取：4 条腿 × 5 个关节的框架在设计姿态下全部重合。（10-14，见 K15）
- [X]  MuJoCo 闭链核查：工作空间 25 种姿态，闭环间隙 ≤ 0.095 mm，`_3` 与 `_2` 转角 ≤ 0.56°。（10-14，见 K15、K16）
- [X]  第一次改 MuJoCo（base + `_3` 1 g + 被动 armature 1e-5）失败：viewer 里右前腿小腿断开。已按 git 退回，并复现、定位原因。（10-14，见 K17、K18）
  - 改过的版本留在 `E:/HuanCun/AI_tmp/Claude/rollback_20261014/`。
  - 复现条件：重力加接触落地 3 s，原模型最大闭环间隙 0.54 mm。
  - `_3` 0.001 kg + armature 1e-5：间隙 20 mm，机身塌到 0.175 m（坏）。
  - `_3` 0.001 kg + armature 1e-3：1.19 mm；`_3` 0.001 kg + armature 0.01：0.66 mm。
  - `_3` 0.05 kg + armature 1e-5：1.60 mm；`_3` 0.19 kg（原值）+ armature 1e-5：0.76 mm（可用）。
  - 只改 base 的质量和惯量：0.54 mm，没事。
- [X]  第二次改 MuJoCo：在 `bennett_1 - 副本` 里按新 URDF 只更新 base；用原 `_genfinal.py` 在临时目录重新生成，与副本逐行对比，只有 base 3 行不同。（10-15，见 K19）
- [X]  采用新模型：用户把副本改名 `bennett_3/`（旧模型改名 `bennett_3_wuzuduan/`），验证通过。（10-15，见 K4、K19）
  - `_genfinal.py` 输出名与模型名改为 `bennett_3`（临时目录重跑，与 `bennett_3.xml` 逐行一致）。
  - `quad_leg_trot1.py:52`、`quad_leg_free_gait3.py:73` 改指向 `models/bennett_3/bennett_3.xml`。
- [X]  整机称重：样机放在机架上称（5300 mAh、4000 mAh 各一次），机架 2.8 kg 单称，得样机实重 11.7 / 11.55 kg。（10-15，见 K6）
  - 5300 mAh：样机 + 机架 14.5 kg，减机架 2.8 kg = 11.7 kg。
  - 4000 mAh：读数待补（应为 14.35 kg）。
- [X]  决定：不做分件称重（无小秤），用训练时的质量域随机化覆盖 SW 与样机的 0.28 kg 差。（10-15，见 K7）
- [X]  腿部各 link 的 SW 质心与惯量复核：Codex 用"选中子装配"读，我用叶子零件合成（先还原轻化零件，再用整机自检通过），两种方法结果逐位一致。（10-15，见 K22、K23）
- [X]  URDF 四个大腿的惯量改成 SW 右大腿的值（左大腿取镜像）：只改这四个 `<inertia>`，共 48 行；改后 URDF 能解析，25 个 link、24 个关节、惯量正定，四个大腿主惯量 0.00069014 / 0.00367063 / 0.00393403。（10-15，见 K23）
  - 改前备份：`E:/HuanCun/AI_tmp/Claude/urdf_edit_20261015/Urdf_Bennett_3.urdf.before_thigh_edit`；改动尚未提交 git。
- [X]  MuJoCo `bennett_3.xml` 惯量同步为 URDF 新值，并修好生成脚本的一个旧 bug（见 K24）：`_genfinal.py` 改后在临时目录重新生成，与现有文件对比，只有 12 个刚体（四个 thigh、calf、足）的 `quat` 各一行不同，再复制进去。验证：编译通过，总质量 12.7332 kg，所有刚体由 `diaginertia`、`quat` 重建的惯量与 URDF 相对误差 ≤ 4e-6（改前最大 0.9），重力加接触 3 s 最大闭环间隙 0.537 mm，机身高 0.279 m。（10-15）
- [X]  USD 四个 thigh 惯量已改并保存：从磁盘文件重读，21 个刚体的质量、质心、惯量与 URDF 全部一致（惯量相对误差最大 2.2e-7），总质量 11.968 kg。（10-15）
- [X]  同步 `Urdf_Bennett_4` 的 USD（训练用）：用 Isaac 的 Python 接口把 base 与四个 thigh 的质量、质心、惯量改成 URDF 值并保存；从磁盘重读，21 个刚体与 URDF 一致，且与 `_3` 的 USD 质量惯量完全相同。改前备份：`E:/HuanCun/AI_tmp/Claude/usd4_edit_20261015/`。（10-15）
- [X]  决定不做：MuJoCo 与 Isaac 的被动关节参数对齐（重点是训练迁移）、sim2sim 初始化解闭环（sim2sim 现在表现好）、足端位置对比（关节位置、轴方向无误）。（10-15）
- [X]  决定不做：质心与惯量实测（没有条件）、写入 `params/rigid_body.yaml`（真正的值已经在 URDF、USD 里，不需要再抄一份）。（10-15）
- [X]  不再需要：对表约定与阈值（T10）、读取工具说明（T11）——验收已完成，工具借 Isaac Sim 接口读取即可。（10-15）
- [X]  MuJoCo 的 `_3` 质量降到 0.001 kg：`_genfinal.py` 加了 `PHANTOM_MASS`，在临时目录生成，与现有文件对比只有 4 个 `_3` 的惯量行不同，再复制进去。验证：总质量 11.972 kg，闭环间隙 0 mm，重力加接触 3 s 最大间隙 0.653 mm，机身高 0.279 m。用户随后自己跑 sim2sim 验证。（10-15）
  - 全部 21 个刚体质量对齐；除右大腿惯量外，质心、惯量全部对齐。
- [X]  sim2sim 验证新 MuJoCo 模型：`trot1`、`free_gait3`、`free_gait4`、`go2_10`，用户确认没有问题。（10-15）

---

## 待完成

00 号包已收尾。下面只是用到再做或只记不做的项目。

- [ ]  T4 SW 框选 base 是否补进亚克力外框和双头螺杆（约 0.0124 kg）：你来定。
- [ ]  T6 `quad_leg_track`、`single_leg_rr_trace` 需要 `bennett_3_fixed.xml`：新 `bennett_3/` 里没有，要用时再生成。
- [ ]  T14 训练域随机化（只记不做）：按 H 的项目，范围后续再分析；需要改训练配置，用户要做时再提。
- [ ]  T15 训练配置改用 `Urdf_Bennett_3` 的 USD（只记不做）：现在指向 `Urdf_Bennett_4` 的任务有 `trot1`、`go2-11/12/13`、`go2-slope1`、`gravel1`、`slope3/4`、`stair1`；注意 `_4` 的碰撞体做过简化，换成 `_3` 后训练速度可能变慢。
  改模型的流程：改 URDF → 重新生成 USD 与 MJCF → 再对表；MuJoCo 的改动先在副本上做，并做动态验证。



---

# System_Identification/workstreams/01_communication

目标：测出部署时策略拿到的观测有多旧、动作多久发出，为训练的延迟随机化提供范围（对应 `bennett_rl/README.md` 的 T4a、T4e）。

## 基准与结论

### A. 链路总览

| 环节 | 内容 |
| --- | --- |
| 上位机 | Windows 电脑，或 Jetson Nano（Ubuntu 22.04，用户确认；JetPack 版本未知） |
| 电机链路 | 上位机 → USB-C → USB2CANFD 双路盒 → CAN1：id 1、2、5、6；CAN2：id 3、4、7、8 → 8 × DM-J8006-2EC |
| IMU | DM-IMU，USB 串口，921600 baud，约 1000 Hz |
| 手柄 | ROS 手柄接收器，USB |
| 供电 | 6S 锂电 → 开关 → XT60 → 分电板 → 电机；降压模块 → Jetson 12 V |
| 电机动态库 | Windows：官方 `libdm_device.dll`（缺省）与补丁版 `dlls_lowlatency/`；Linux：`libdm_device.so`（官方，未补丁） |

### B. 部署端实际负载与配置

| 项目 | 值 | 来源 |
| --- | --- | --- |
| 发令环 | 缺省 1000 Hz，每圈给每个活动电机发一帧 | `runtime.py` 的 `control_rate_hz` |
| 策略环 | 50 Hz | `contract` |
| 命令帧 | MIT 模式，8 字节 | `dm_can.py` 的 `control_mit` |
| 帧类型与速率 | CAN FD，BRS 开，仲裁段 1 Mbps、数据段 5 Mbps | `dm_can.py` 第 764–765 行 |
| 反馈 | 电机每收到一帧命令回一帧 | 旧实测（`LEGACY.md`）与社区驱动说明 |
| 每路帧率 | 约 4000 命令 + 4000 反馈 = 8000 帧/s | 计算 |
| 反馈时刻的记法 | 回调里记主机时间 `last_time_`，没有用帧自带的 `timestamp` | `dm_can.py` 的 `canframeCallback` |
| 观测低通 | 关节速度与 IMU 各 12 Hz 一阶，时间常数约 13 ms | `runtime.py` |
| Linux 库 | `sim2real/sdk/dlls/libdm_device.so` 是 x86-64 | `file` 命令 |
| Python 绑定 | `dmcan-sdk` 1.0.4，要求 Python ≥ 3.13；Windows 靠加入 Python313 的 user-site 绕过 | `dm_can.py` |

旧实测（2026-09-16，官方 DLL，1 kHz 命令流，零力矩）：

| 指标 | 数值 |
| --- | --- |
| 每关节反馈频率 | 126 Hz，无丢失 |
| 批内帧间隔 | 约 0.05 ms |
| 批间隔 p50 / 最大 | 108.7 ms / 110.7 ms，8 个关节一致 |

01-K1：旧实测数据来自官方 DLL，只作线索，不进新数据集（`legacy_timing/data/LEGACY.md`）。
01-K2：每路约 8000 帧/s。CAN FD（1M / 5M）一帧约 55–65 µs，负载约 45–50%；若实际退回经典 CAN 1 Mbps，一帧约 110 µs，负载约 90%（推测，需实测）。
01-K3：用户记得速率设为 5M，部署代码里是仲裁 1M、数据 5M，两者大概率指同一设置（推测）；上机时读取设备实际设置核对。
01-K4：当前"反馈新鲜度"是主机回调时刻，包含驱动的批量交付；帧自带 `timestamp` 的单位和含义未验证，验证后才能区分 USB 批量交付与总线时间。
01-K5：Linux `.so` 是 x86-64，Jetson 是 arm64，不能直接加载；需要从达妙 `dmBots/dm-tools` 找 arm64 版。
01-K6：Jetson 为 Nano、Ubuntu 22.04 由用户确认；原版 Nano 官方只到 Ubuntu 20.04，实际型号可能是 Orin 系列（推测）。
01-K7：12 Hz 低通是主动滤波，不算通信延迟，两者分开记录。
01-K8：没有查到达妙对"109 ms 批量延迟"或"补丁后 1 ms"的官方说明，只能以实测为准。
01-K9："不使能"不等于不改变设备状态：构造时会设置 CAN 通道波特率，关闭时会发失能命令；运行前要审查初始化与退出流程。
01-K10：没有外部基准（示波器、逻辑分析仪）测不出绝对延迟，主机侧测量只给下限；端到端延迟（指令到关节动作）留给 07 执行器辨识。
01-K11：`Motor_Control` 缺省 `set_baudrate=True`，会把盒子两路的波特率与采样点改写成 1 M / 5 M / 0.75；缺省 `auto_enable=True`，会使能电机（切模式加 0xFC 各 5 次）。测量脚本必须传 `set_baudrate=False`、`auto_enable=False`。
01-K12：`close()` 一定会给每个电机发 5 次失能帧（0xFD）并关闭通道，Windows 上不关设备（避免挂死）。测试时不能有别的控制程序在驱动同一个盒子；对未使能电机发失能帧应无影响（推测）。
01-K13：SDK 提供 `get_channel_baudrate` 和 `get_version`，只读设备，可用来核对 CAN 实际速率与固件版本；接收帧头有设备时间戳 `timestamp`（64 位），部署代码和旧探测脚本都没有使用。
01-K14：旧探测脚本通过根目录 `damiao.py` 只加载官方 DLL；在回调第一行用 `perf_counter_ns` 记时，包含驱动的批量交付；1 kHz 发令靠 `time.sleep`，实际发令节拍没有记录，需要在新实验里记。

### C. 方案

| 步骤 | 内容 | 是否使能电机 |
| --- | --- | --- |
| A 准备 | 审查初始化与退出流程；验证帧 `timestamp`；核对 CAN 实际速率与 CAN FD 开关；找 arm64 SDK | 否 |
| B 电机通信性能 | 按真实 1 kHz、8 电机负载，记录反馈周期、抖动、8 电机时间差、丢帧、错误、总线负载；Windows 官方库与补丁库对比，再到 Jetson 重复 | 否（零力矩空命令） |
| C 控制程序节拍 | 发令环与策略环的实际周期、超时；单独运行，以及加入推理、IMU、日志后的变化 | 否 |
| D 观测新鲜度 | 策略取观测时，每个关节和 IMU 最新数据有多旧，各路是否相差太多 | 否 |
| 输出 | 观测滞后与动作滞后的分布，折成策略步（20 ms）和物理步（5 ms） | — |

每档先测 10–30 s，发现问题再加对照。采集时不在同一台电脑上跑训练；每次记录电池、DLL profile 与 SHA256（见 `System_Identification/docs/DATA_CONTRACT.md`）。

不做：
- 手拨腿对齐 IMU 与编码器的精确延迟，只看数据率和新鲜度。
- 指令到动作的阶跃延迟，归 07。
- 使能后的掉线与 `TIMEOUT` 行为试验，以后单独评估；现在只审查配置与软件超时逻辑。
- 单独的请求—反馈往返时间，只作电机通信性能的可选诊断（旧版本因批量交付干扰已废弃）。

### D. 文件夹架构

`01_communication/` 自成一体：新采集按 A 至 D 四个步骤分文件夹，旧一轮（官方 DLL 时期）的数据与脚本放在 `legacy_*`，只作线索。

```text
01_communication/
├── A_prep/                 准备：设备信息、初始化审查、帧时间戳验证、CAN 速率核对
│   ├── results/            审查与核对记录
│   └── scripts/            读设备信息的脚本
├── B_motor_comm/           电机通信性能：反馈周期、抖动、8 电机时间差、丢帧、总线负载
│   ├── data/raw/           原始数据，按 日期/实验名 存放
│   ├── results/            汇总表、图
│   └── scripts/            采集与分析脚本
├── C_loop_timing/          控制程序节拍：发令环与策略环的实际周期、超时（结构同 B）
├── D_obs_freshness/        观测新鲜度：策略取观测时关节与 IMU 数据有多旧（结构同 B）
├── legacy_hardware/        2026-09-16 的寄存器快照（原 01_hardware_inventory）
│   ├── data/               raw：两次不使能读寄存器的原始数据，manifest.yaml 为实验身份证
│   ├── scripts/            g0_snapshot.py：不使能读取各电机寄存器
│   └── README.md           旧笔记
├── legacy_timing/          2026-09-16 的通信延迟实测（原 02_timing_latency）
│   ├── data/               raw、processed：1 kHz 命令流探测；LEGACY.md 说明为何不进新数据集
│   ├── results/            figures：fig1_bus_latency.png、fig1b_latency_shift.png
│   ├── scripts/            probe_feedback_age_1khz.py（探测）、plot_bus_latency.py、plot_shift_demo.py（作图）
│   └── README.md           旧笔记，含当时的测法与结论
└── references/             资料
    ├── 硬件/               控制与接线设计图、硬件系统集成图
    └── 调研要点.md         Codex 与自行调研的要点和来源
```

文件夹之外的相关文件：

```text
sim2real/bennett_deploy/dm_can.py                         电机驱动封装：MIT 命令帧、反馈回调、波特率设置
sim2real/bennett_deploy/runtime.py                        部署主循环：1 kHz 发令环、50 Hz 策略环、12 Hz 低通
sim2real/bennett_deploy/imu.py                            DM-IMU 串口读取
sim2real/sdk/dlls/                                        官方 libdm_device：Windows 的 .dll、Linux 的 .so（x86-64）
sim2real/sdk/dlls_lowlatency/                             补丁版 Windows DLL（批量间隔改为 1 ms）
System_Identification/docs/DATA_CONTRACT.md               数据记录规则：实验身份证、命名、记录字段
System_Identification/docs/HARDWARE_SAFETY_CHECKLIST.md   上机前的安全检查清单
docs/ROBOT.md                                             机器人总览
```

新数据放各主题文件夹的 `data/raw/日期/实验名/`，实验名按 `DATA_CONTRACT.md` 命名。空文件夹 git 不会同步，有文件后才会出现在另一台电脑上。

### E. 调研依据

资料见 `System_Identification/workstreams/01_communication/references/调研要点.md`。要点：文献通常在台架上测指令到首个响应，取最小与最大值，作为仿真延迟缓冲的范围；观测延迟与动作延迟分开设置；PACE 把延迟作为全局参数拟合，单驱动通信延迟按可忽略处理。

## 已完成

- [X]  2026-10-19 读两张硬件图（控制与接线、硬件系统集成），明确通信链路与 USB 设备（A）。
- [X]  2026-10-19 核对部署代码的发令率、帧类型、反馈记时方式与 Linux 库架构（B、01-K2 至 01-K5）。
- [X]  2026-10-19 审阅 Codex 的调研，并自行查资料（达妙官方仓库、`dmcan-sdk`、文献），定下方案（C、D）；纠正了"每路 4 电机、50 Hz，总线负载很小"的判断。
- [X]  2026-10-19 清理 `00_…` 与原 `01_…`、`02_…` 下的空文件夹与占位文件。
- [X]  2026-10-19 读 SDK 初始化与退出流程、旧探测脚本和 `dmcan` 接口（01-K11 至 01-K14）；只读代码，没有连设备。
- [X]  2026-10-19 通信辨识定为 01 号包：原 `01_hardware_inventory`、`02_timing_latency` 并入 `01_communication/`，旧数据与脚本移入 `legacy_hardware/`、`legacy_timing/`，硬件图移入 `references/硬件/`；旧脚本里的路径已改，语法检查通过，没有重跑；新建 `A_prep/`、`B_motor_comm/`、`C_loop_timing/`、`D_obs_freshness/`。

## 待完成

按顺序做。每一步完成后把结论写进「基准与结论」。

- [ ]  01-T1 读设备信息：接电不使能，只调用 `get_version`、`get_channel_baudrate`，核对固件版本与 CAN 实际速率、帧类型；记录 USB 口与 CAN 盒型号。卡在：需要上机。
- [ ]  01-T2 基线复现：Windows 官方库，沿用旧探测同款的零力矩 1 kHz 命令流，加记帧 `timestamp` 与实际发令节拍，确认 109 ms 批量是否复现、`timestamp` 的单位与含义。卡在：需要你同意发零力矩帧，并停掉训练。
- [ ]  01-T3 补丁库同测：同样条件换补丁版 DLL，对比批间隔与帧到达分布。
- [ ]  01-T4 控制程序节拍：发令环与策略环的实际周期、超时，单独运行与加入推理、IMU、日志后的变化。
- [ ]  01-T5 观测新鲜度：策略取观测时，每个关节和 IMU 最新数据有多旧，各路是否相差太多。
- [ ]  01-T6 Jetson：找 arm64 SDK，重复 T2 至 T5；卡在 01-K5。
- [ ]  01-T7 把结果折成策略步，写进 `bennett_rl/README.md` 的 T4a、T4e，并更新 `docs/ROBOT.md`。
- [ ]  01-T8 审查电机 `TIMEOUT` 配置与软件反馈超时（现为 0.5 s，对 109 ms 批量交付不起作用）。

---


# sim2real

目标：把 Isaac Lab 训出的策略部署到 Bennett 真机（达妙电机 + DM-IMU），Windows 与 Linux（Jetson）通用。本文件夹汇集部署运行库、动态库与工具，原位置的文件保留不动。

## 基准与结论

### A. 命令

在 `u2canfd/` 下运行（在 `sim2real/` 下运行时去掉 `sim2real/` 前缀）；用装有 torch 的环境（`env_isaaclab`）。**`--check_only` 与 `--dry_run` 不使能电机、不发运动命令；其余命令会连接并驱动真机，须用户当次明确同意。**

不开电机的校验与试运行：

```
python sim2real/bennett_1_crawl_deploy.py --check_only
python sim2real/bennett_1_crawl_deploy.py --dry_run --keyboard
python sim2real/bennett_1_free_deploy.py --check_only
python sim2real/tools/dev_sn.py
```

真机运行（开电机）：

```
python sim2real/bennett_1_crawl_deploy.py --keyboard
python sim2real/bennett_1_free_deploy.py --keyboard
python sim2real/quad_leg_free_gait3_deploy.py --keyboard
python sim2real/bennett_1_crawl_deploy.py --keyboard --dm_sdk_profile low_latency
```

入口脚本：

| 入口 | 策略 |
| --- | --- |
| `bennett_1_crawl_deploy.py` | bennett_1 爬行变体，缺省 T9，与 sim2sim 配置 `bennett_1_crawl` 一致 |
| `bennett_1_free_deploy.py` | bennett_1 自由步态，缺省 S1，与 sim2sim 配置 `bennett_1_free` 一致 |
| `quad_leg_free_gait3_deploy.py`、`quad_leg_free_gait4_deploy.py` | 旧的自由步态策略，取 `quad_leg_free_gait` 下最新导出 |
| `quad_leg_go2-10_deploy.py`、`quad_leg_trot1_deploy.py` | go2-10、trot1 |

常用参数（各入口通用，加在命令后面，顺序不限）：

| 参数 | 作用 |
| --- | --- |
| `--keyboard` | 键盘给速度指令；电机回到默认姿态并预热后，按空格或回车才启动策略 |
| `--check_only` | 只校验训练配置与 TorchScript 策略，不打开 IMU 与电机 |
| `--dry_run` | 读 IMU、跑策略，不打开电机 |
| `--dm_sdk_profile original` / `low_latency` | 选电机动态库：`original`（缺省）为官方版，反馈约每 109 ms 一批；`low_latency` 为补丁版，批量间隔改为 1 ms，真实延迟未测，仅 Windows 有。用 T9 等训练时没加滞后的策略、真机迟钝或摔倒时再试 |
| `--imu_port <串口>` | IMU 串口；Windows 缺省 `COM6`，Linux 缺省 `/dev/ttyACM0`（推测），以实际枚举为准 |
| `--output_scale <0–1>` | 策略动作的缩放，首次试验取小值 |
| `--active_joints rr` | 只驱动 RR 的两个关节，缺省 `all` |
| `--duration_s <秒>` | 运行时长，缺省 0 为不限 |
| `--policy <路径>` | 指定 `exported/policy.pt`；运行目录须含 `params/env.yaml`、`params/agent.yaml` |

其他：
- 键盘：`w`/`s` 前后，`a`/`d` 侧移，`j`/`l` 转向，`0`/`x` 清零，空格或回车授权策略，`q` 或 Esc 退出；指令范围缺省取各策略的训练范围。
- 策略已打包在 `sim2real/policies/`，缺省不需要 `bennett_rl`，整个 `sim2real/` 拷到另一台电脑即可运行；另需 Python（含 torch）、pyserial、pyusb、dmcan-sdk，Linux 另需 libusb（见 `docs/WORKFLOW.md`）。
- 换 bennett_1 的策略：环境变量 `BENNETT1_CRAWL_RUN=T9|D4`、`BENNETT1_FREE_RUN=S1`（打包的别名），或 `bennett_rl/logs/rsl_rl/` 下的完整运行目录名，或 `--policy`。
- 打包的策略找不到时才去 `bennett_rl` 找：环境变量 `BENNETT_RL_ROOT`，否则取同级目录 `bennett_rl`，再否则取本仓库所在盘符下的 `Project/Isaaclab/bennett_rl`。

### B. 文件夹结构

```
sim2real/
├── bennett_deploy/                运行库
│   ├── __init__.py                导出契约类
│   ├── async_csv.py               异步 CSV 日志
│   ├── contract.py                go2-10 与 bennett_1 爬行的 50 维部署契约：校验 env.yaml、关节表与符号、动作缩放
│   ├── dm_can.py                  达妙电机 CAN 通信与 MIT 控制
│   ├── free_gait_contract.py      自由步态（33 维）契约
│   ├── free_gait_policy.py        自由步态观测与推理
│   ├── gait.py                    爬行相位时钟
│   ├── imu.py                     DM-IMU 串口读取与一阶低通
│   ├── input.py                   键盘速度指令（Windows 用 msvcrt，Linux 用 termios）
│   ├── paths.py                   路径解析：打包的策略优先，其次训练仓库
│   ├── policy.py                  50 维观测拼装与策略推理
│   ├── runtime.py                 主程序：参数、安全启动、1 kHz 控制环、50 Hz 策略
│   ├── tests.py                   部署核心单元测试
│   ├── trot1_contract.py          trot1 契约
│   ├── trot1_gait.py              trot1 步态时钟
│   └── trot1_policy.py            trot1 观测与推理
├── docs/
│   └── WORKFLOW.md                达妙 USB-CANFD Python 例程说明（安装与 Ubuntu 环境）
├── policies/                      已打包的策略，只含 exported/policy.pt 与 params/*.yaml，免去引用 bennett_rl
│   ├── bennett_1_crawl_D4/        bennett_1 爬行，D4（滞后 0–5 步训练）
│   ├── bennett_1_crawl_T9/        bennett_1 爬行，T9（缺省）
│   ├── bennett_1_free_S1/         bennett_1 自由步态，S1（缺省）
│   ├── free_gait3/                free_gait3
│   ├── go2-10/                    go2-10
│   └── trot1/                     trot1
├── sdk/                           达妙动态库
│   ├── dlls/                      官方版（反馈每约 109 ms 一批）与 Linux `.so`
│   ├── dlls_lowlatency/           补丁版（批量延迟改为 1 ms，真实延迟未测，无 Linux 版）
│   └── libusb-1.0.dll
├── tools/
│   ├── 0_Return_to_zero.py        8 电机限速安全回零
│   └── dev_sn.py                  列出 USB-CANFD 设备
├── bennett_1_crawl_deploy.py      bennett_1 爬行变体入口（缺省 T9）
├── bennett_1_free_deploy.py       bennett_1 自由步态入口（缺省 S1）
├── quad_leg_free_gait3_deploy.py  自由步态 gait3 入口
├── quad_leg_free_gait4_deploy.py  自由步态 gait4 入口
├── quad_leg_go2-10_deploy.py      go2-10 入口（缺省策略）
├── quad_leg_trot1_deploy.py       trot1 入口
└── README.md                      本文件
```

### C. 迁移对照

| 来源（`u2canfd/` 下） | 去向 | 说明 |
| --- | --- | --- |
| `quad_leg_xu/bennett_deploy/` | `bennett_deploy/` | 复制后有改动，见 D |
| `quad_leg_xu/quad_leg_*_deploy.py`（4 个） | 同名，位于本文件夹根 | 只把写死的路径改为打包策略优先、其次 `bennett_rl` 的解析 |
| `dlls/`、`dlls_lowlatency/`、`libusb-1.0.dll` | `sdk/` | 无改动，`libdm_device.dll` 与原件逐字节相同 |
| `0_Return_to_zero.py` | `tools/` | 动态库目录改为 `sdk/` |
| `dev_sn.py` | `tools/` | 无改动 |
| `WORKFLOW.md` | `docs/` | 无改动 |
| 新写 | `bennett_1_crawl_deploy.py`、`bennett_1_free_deploy.py` | bennett_1 的 T9、S1 入口 |
| `bennett_rl/logs/rsl_rl/` 下的 6 个运行目录 | `policies/` | 只取 `exported/policy.pt`、`params/env.yaml`、`params/agent.yaml`，来源见 E |

### D. 相对原件的改动

| 文件 | 改动 |
| --- | --- |
| `runtime.py` | 动态库目录改为 `sdk/`；缺省策略改为打包的 `policies/go2-10`，没有再取 `bennett_rl`；IMU 缺省串口按平台取值；Linux 上选 `low_latency` 报错；打印训练与运行时的观测低通对比；退出时恢复终端 |
| `dm_can.py` | 动态库目录改为 `sdk/`；Linux 上把 dmcan 的后端库绑定到 `sdk/dlls/libdm_device.so`，不再依赖启动目录 |
| `contract.py`、`free_gait_contract.py` | 关节位置、速度观测同时接受官方函数与 bennett_1 的 `filtered_observation` 包装（内层函数仍须相同） |
| `input.py` | 新增 Linux 键盘读取（termios，按键与 Windows 一致）；`parse_posix_keys` 可单测 |
| `paths.py` | 不再写死盘符：环境变量、同级目录、本仓库所在盘符三级查找；新增 `resolve_policy`，打包的策略优先 |
| `tests.py` | 新增 3 个键盘解析测试；读取不存在的旧脚本的测试改为跳过 |
| 关节符号表 `sim_to_motor` | 未改动 |

### E. 打包的策略

| `policies/` 下的目录 | 来源运行目录（`bennett_rl/logs/rsl_rl/` 下） | 用于 |
| --- | --- | --- |
| `bennett_1_crawl_T9` | `bennett_1/crawl/2026-10-16_13-10-20_t9_same` | `bennett_1_crawl_deploy.py` 缺省 |
| `bennett_1_crawl_D4` | `bennett_1/crawl/2026-10-16_15-13-42_d4_delay5` | `BENNETT1_CRAWL_RUN=D4` |
| `bennett_1_free_S1` | `bennett_1/flat/2026-10-16_05-53-35` | `bennett_1_free_deploy.py` 缺省 |
| `free_gait3` | `quad_leg_free_gait/quad_leg_free_gait3/flat/2026-09-05_07-07-11` | `quad_leg_free_gait3_deploy.py`、`quad_leg_free_gait4_deploy.py` |
| `go2-10` | `quad_leg_go2/quad_leg_go2-10/flat/2026-07-19_18-41-26` | `quad_leg_go2-10_deploy.py` |
| `trot1` | `quad_leg_trot/quad_leg_trot1/flat/2026-07-27_20-08-08` | `quad_leg_trot1_deploy.py` |

每个目录约 0.5 MB，`policy.pt` 与原件逐字节相同；`quad_leg_free_gait4_deploy.py` 在 gait4 没有导出策略时取 `free_gait3`。

### F. 未迁移（原处保留）

| 内容 | 位置 | 原因 |
| --- | --- | --- |
| 早期单腿、四足试验脚本 | 根目录 `1_`、`2_`、`3_`、`4_`、`5_*.py`、`damiao.py`、`damiao_2.py`、`single_leg_policy_udp_runner_2.py` | 已被 `bennett_deploy` 取代 |
| 各策略独立部署脚本 | `quad_leg_xu/quad_leg_go2-*.py`、`quad_leg_crawl*.py`、`quad_leg_walk.py`、`quad_leg_trot/`、`quad_leg_free_gait/` | 早期版本，go2-10 之后统一走 `bennett_deploy` |
| DLL 测试、辨识输出、日志 | `_dlltest/`、`sysid_out/`、`logs/`，以及 `quad_leg_xu/*/logs/` | 数据与测试产物，不是部署代码 |

### G. 结论

- sim2real-K1：迁移与修改后验证（均不打开 IMU 与电机）：6 个入口的 `--check_only` 全部通过；`bennett_deploy.tests` 共 12 个测试，11 个通过、1 个跳过（旧脚本不存在）；动态库解析到 `sim2real/sdk/dlls`，哈希一致。
- sim2real-K2：bennett_1 的观测与训练端逐项一致（读代码对照）：50 维顺序、相位 = 回合时间 × 0.55 Hz 取小数、腿序偏移 (0, 0.5, 0.75, 0.25)、占空比 0.78、指令范数小于 0.025 时取站立值（相位 0、四脚着地、参数 0/1/0）；动作缩放 0.2，裁剪 3.5，kp 28，kd 2；角速度、重力、关节速度的 12 Hz 低通与训练一致（运行时会打印对比），关节位置不滤波。
- sim2real-K3：关节符号表 `sim_to_motor` 沿用原文件（FR_calf、RR_calf 为 −1，其余 +1），不改动。`free_gait_contract.py` 的注释记载：FL、RL 的 thigh 取 +1 在实机验证过（2026-09-05），`3_quad_leg_track.py` 的 −1 会使步态镜像、不适用于腿部策略；入口已由用户验证。
- sim2real-K4：训练配置差异：go2-10 的执行器为力矩限 7 N·m、饱和 12 N·m、速度限 30 rad/s，bennett_1 为 8、20、20。
- sim2real-K5：缺省策略的已知性质（来自 `bennett_rl/README.md`）：T9 训练时没有加反馈滞后，滞后 80 ms 开始变差、100 ms 失效；官方动态库的反馈约 109 ms 一批，所以用 T9 时建议用补丁版（Windows，`--dm_sdk_profile low_latency`），或换 D4（K54）。S1 前进时四脚抬高 27–41 mm，两脚同时着地约占 20%（K53）。
- sim2real-K6：Linux 部分只做了静态检查与键盘解析单测，未在 Linux 上运行：`.so` 绑定、termios 键盘、串口名、`low_latency` 无 Linux 版。
- sim2real-K7：`dm_sdk_profile` 缺省为 `original`（官方版）；补丁版的真实延迟未测。
- sim2real-K8：把训练仓库藏起来（`BENNETT_RL_ROOT` 指向不存在的目录）后，6 个入口的 `--check_only` 与单元测试照常通过，契约指纹与此前一致，说明运行不依赖 `bennett_rl`；未在另一台电脑上验证，那里还需要 Python 环境与依赖。

## 已完成

- [X]  2026-10-16 建 `sim2real/`，复制部署运行库、入口、动态库、工具与文档，改动动态库路径并验证（sim2real-K1）。
- [X]  2026-10-17 写死的 `E:\` 路径改为跨平台解析；加 Linux 键盘与 `.so` 绑定；放宽契约校验（见 D）。
- [X]  2026-10-17 新增 bennett_1 的 T9、S1 入口并通过 `--check_only`（sim2real-K1、K2）。
- [X]  2026-10-17 策略打包进 `policies/`，入口优先用打包的策略（sim2real-K8）。

## 待完成

- [ ]  sim2real-T1：用户在真机上试 `bennett_1_crawl_deploy.py`（T9）与 `bennett_1_free_deploy.py`（S1），结果记入本节；首次上电按用户的流程，须当次明确同意。
- [ ]  sim2real-T2：在 Linux（Jetson）上跑 `--check_only`、`--dry_run`，确认 `.so` 绑定、键盘、IMU 串口名（sim2real-K6）。
- [ ]  sim2real-T3：官方动态库下 T9 对约 109 ms 滞后不稳，真机若出现迟钝或摔倒，改用补丁版或 D4（sim2real-K5）。
- [ ]  sim2real-T4：用户确认文件夹结构后，决定原位置文件是否删除。
- [ ]  sim2real-T5：在另一台电脑上拷入整个 `sim2real/`，运行 `--check_only`，确认 Python 依赖与动态库齐全（sim2real-K8）。
