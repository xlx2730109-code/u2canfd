solidworks模型路径："E:\HuanCun\Desktop\sw_work\Bennett\Bennett_1\Bennett_test5.SLDASM"
导出的URDF模型路径："E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\assets\robots\Urdf_Bennett_3"
URDF转USD模型路径："E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\assets\robots\Urdf_Bennett_3\urdf\Urdf_Bennett_3.usd"
真机重量：System_Identification\workstreams\00_model_chain_sw_urdf_usd\样机重量实拍

我在网上查到的：

* [ ]  1、sw中当有零件质量被覆盖时，导出的urdf的质量、惯量等会出错，需要在urdf中进行修改。

## Claude 排查出的问题（2026-10-14，逐个解决）

**已确认（读文件得出，不是推测）**

- [ ]  2、**MuJoCo 模型和 Isaac Lab 不是同一个机器人**：`sim2sim/models/bennett_3/bennett_3.xml` 总质量约 7.53 kg（base 2.24、thigh 0.63），URDF 是 11.90 kg（base 4.22、thigh 0.99）；MJCF 还缺 `_3` 连杆和足端。来自更早一版导出。
  答：这样的话MuJoCo 模型就错了。
- [ ]  3、**USD 来源不一致**：训练实际用 `Urdf_Bennett_4\urdf\Urdf_Bennett_3.usd`（7-20），不是上面写的 `Urdf_Bennett_3` 那份（7-16）。两份 URDF 逐字相同，USD 不同。
  答：后续会改成真实的USD，即："E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\assets\robots\Urdf_Bennett_3\urdf\Urdf_Bennett_3.usd"
- [ ]  4、**SW 文件有两份**：`sw_work\Bennett\Bennett\`（10-10 更新）和 `Bennett_1\`（10-08，上面写的这份）。要定唯一来源。
  答：按照Bennett_1\来
- [ ]  5、**URDF 可能过时**：URDF 是 7-15 导出，SW 装配体 9-14 之后还在改。要确认导出时用的版本。
  答：模型没什么改动，应该是一样的
  **待补的方法细节**
- [ ]  6、对表约定：单位统一为 kg / m / kg·m²；质心与惯量的参考点；SW 零件 ↔ URDF link ↔ USD 刚体 ↔ MJCF body 的对照表；红线阈值（暂定：质量 >2%，质心 >5 mm，惯量 >10%）。
- [ ]  7、真机称重清单：机架（已称 2.8 kg）、两块电池、Jetson+载板、IMU、USB2CANFD、电源板、降压模块、线材、整机（每块电池各称一次）；腿不可拆，用整机减法。台秤读数 0.01 kg，小件需更小的秤。
- [ ]  8、质心（多点称重/悬吊）和惯量（扭摆）的做法与器材。
- [ ]  9、读 USD 用 Isaac Sim 5.1 的 `pxr`，读 MJCF 用 mujoco 3.6；脚本须 Windows / Linux 通用。
- [ ]  10、足端位置比较需要"同一组关节角"，依赖 `03_joint_zero_calibration` 的零位。
- [ ]  11、发现差异后的修改流程（手改 URDF 还是重新导出；改后重新生成 USD / MJCF；回归再对表）。
- [ ]  12、`bennett_3.xml.bak_flcalf` 说明 MJCF 手改过，要弄清改了什么。

**解决顺序**：先 4、5（定来源）→ 3 → 第 1 关（SW 只读）→ 第 2 关 → 2 → 其余。

# 00 — 模型链校核：SW → URDF → USD → MuJoCo，再到真机称重

整个系统辨识的第一步：先确认仿真里的"机器人"就是 SolidWorks 里的机器人，也是桌上那台真实的机器人。

```text
SW 装配体 ──► URDF ──► USD (Isaac Lab) ──► MuJoCo        （纯软件，不碰真机）
  第 1 关     第 2 关      第 3 关           第 4 关
                                  └──────► 第 5 关：真机称重/质心/惯量实测
```

## 每一关查什么


| 关               | 比什么                                                                                                                                                         | 怎么读                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 1. SW 本身       | 材料/密度、是否有手动质量覆盖、整机与分件质量、质心、惯量；电机/电池/控制板与官方质量核对                                                                      | SolidWorks COM API 只读，不保存、不改模型 |
| 2. SW → URDF    | 每个 link 的质量、质心、惯量张量、坐标系、关节原点/方向/限位                                                                                                   | 解析 URDF（XML），与第 1 关逐 link 对表   |
| 3. URDF → USD   | USD 里每个刚体的 mass/CoM/inertia 有没有被引擎重算或覆盖；**闭环关节**是否补上、主从动关系、关节方向、碰撞体；用同一组关节角做正运动学，比较三个模型的足端位置 | pxr / Isaac Lab 读 USD                    |
| 4. USD → MuJoCo | 同一套质量惯量与关节映射；MuJoCo 的闭环约束与足端轨迹                                                                                                          | 读 MJCF                                   |
| 5. 真机实测      | 整机称重（4000 / 5300 mAh 两块电池分别称）、去电池称重、质心（悬吊/多点秤）、整机扭摆惯量；电机/电池质量用官方值核对                                           | 秤、悬吊、扭摆                            |

## 为什么第 3 关要查足端位置

URDF 是树形结构，闭链要在 USD / MuJoCo 里重新加约束。转换出错（闭环断开、从动关节自由、方向反）不一定报错，但足端会走偏。用同一组主动关节角比较足端位置，一次就能发现。

## 产出

- 逐 link 对表（SW / URDF / USD / MuJoCo / 实测的质量、质心、惯量），差异超阈值标红；
- 整机总质量、质心、惯量的对比，注明用的是哪块电池；
- 需要修改的清单（只提建议，不直接覆盖训练在用的模型）；
- 结论写入 `../../params/rigid_body.yaml`。

## 规则

- SW 只用 API 只读；每个数字写清坐标系、单位、版本和电池型号。
- 数据管理沿用 `claude/`、`codex/`、`comparison/`、`inputs/` 的分工结构。
