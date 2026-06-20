# -*- coding: utf-8 -*-
"""
Bennett 单腿 sim2real 技术文档 2

这个文件不是控制程序，而是一份说明文档。
你可以直接在编辑器里看，也可以运行：

    python 技术文档_2.py

它会把本文档打印到终端。
"""

DOCUMENT = r"""
# Bennett 单腿 sim2real 技术文档 2

## 1. 先说结论

我们最近做了三件事：

1. 先把仿真单腿目标角通过 UDP 发到真实达妙电机，验证角度方向、零位和比例映射。
2. 再把训练好的单腿 policy 通过 UDP 发到 `damiao_2.py`，验证 policy -> 真机动作链路。
3. 最后把 policy 直接集成进 `damiao_3.py`，让 policy 使用真实电机反馈构造 observation，这才是当前的真闭环版本。

简单区分：

```text
damiao_2.py：不是 policy 真闭环
damiao_3.py：是当前的 policy 真闭环最小版本
```

注意：达妙电机 MIT 模式本身内部有控制闭环，这和 RL policy 闭环不是一回事。


## 2. 什么叫开环、闭环？

### 2.1 电机底层闭环

达妙 MIT 控制命令类似：

```python
control.control_mit(motor, kp, kd, q, dq, tau)
```

这里的 `q` 是目标位置，`dq` 是目标速度，`kp/kd` 是刚度和阻尼。
电机驱动器内部会根据编码器反馈去追这个目标，所以从电机驱动器角度，它有底层闭环。

但是这不等于 RL policy 闭环。

### 2.2 policy 开环 / 半开环

`single_leg_policy_udp_runner.py` 或 `single_leg_policy_udp_runner_3.py` 最初版本是这样：

```text
policy runner 自己估计 joint_pos/joint_vel
        ↓
policy 输出动作
        ↓
UDP 发给 damiao_2.py
        ↓
damiao_2.py 控制真实电机
```

问题是：policy runner 没有读取真实电机的 `m7/m8 pos/vel`。

所以如果真实电机因为摩擦、负载、延迟没有跟上，policy runner 不知道。
它还以为真机已经到了目标位置。

因此从 RL policy 角度看，这不是完整闭环。

### 2.3 policy 真闭环

`damiao_3.py` 现在是这样：

```text
damiao_3.py 读取真实 m7/m8 pos/vel
        ↓
换算成 real_deg / real_vel
        ↓
构造 policy observation
        ↓
policy 输出 action
        ↓
转换成 target_deg
        ↓
control_mit 控制真实电机
        ↓
下一帧再读取真实反馈
```

这才叫 policy 真闭环。


## 3. 各个文件分别干什么？

## 3.1 damiao_2.py

作用：

```text
接收 UDP 目标角
控制 canid7/canid8
打印真实电机反馈
做安全保护
```

链路：

```text
single_leg_policy_udp_runner.py
        ↓ UDP 15001
damiao_2.py
        ↓ MIT
canid7 / canid8
```

它会读取电机反馈，但主要用于：

```text
打印
err 检查
tau 力矩保护
```

它没有把真实反馈传回 policy runner。

所以：

```text
damiao_2.py 是 UDP 接收控制桥，不是 policy 真闭环。
```


## 3.2 single_leg_policy_udp_runner.py / single_leg_policy_udp_runner_3.py

作用：

```text
加载 policy.pt
构造 observation
policy(obs) 得到 action
把 action 转成目标角
UDP 发给 damiao_2.py
```

最初问题：

```text
joint_pos = target_offset
```

也就是说 runner 用自己发出去的目标角假装真实关节角。
这只能验证动作链路，不能代表真实闭环。

后来我们给它加了：

```text
output_scale
warmup_s
35 deg/s ramp
duration_s=0 无限运行
```

这让动作安全、平滑，但仍然不是完整闭环。


## 3.3 damiao_3.py

这是当前重点。

作用：

```text
直接加载 policy.pt
直接读取真实电机反馈
直接构造 observation
直接控制 canid7/canid8
```

它不再需要 `single_leg_policy_udp_runner_3.py` 通过 UDP 发目标。

当前运行方式：

```powershell
cd E:\HuanCun\Desktop\u2canfd
D:\Conda\envs\env_isaaclab\python.exe .\damiao_3.py --output_scale 0.05
```

注意：

```text
工作目录必须是 E:\HuanCun\Desktop\u2canfd
不要在 D:\IsaacLab 里运行硬件脚本
```

原因是达妙 SDK 会按当前工作目录找：

```text
.\dlls\libdm_device.dll
```

如果你在 `D:\IsaacLab` 里运行，它会错误地找：

```text
D:\IsaacLab\dlls\libdm_device.dll
```


## 4. 当前真实硬件映射

当前是：

```text
仿真 FR_thigh / FR_calf
        ↓
真实 RR 腿 canid7 / canid8
```

也就是暂时把真实 RR 腿当作仿真 FR 腿来调。

实测默认位置：

```python
m7_default = 0.089
m8_default = 1.433
```

实测映射：

```python
q7_cmd = m7_default + 1.0 * fr_thigh_offset
q8_cmd = m8_default - 1.0 * fr_calf_offset
```

重要结论：

```text
当前不要乘减速比 6。
```

虽然电机机械减速比是 6:1，但我们实际测试发现：

```text
scale = 1.0
```

更符合真实关节角。

所以后续别凭理论减速比直接写：

```python
q_motor = q_default + 6 * q_joint
```

要以实测为准。


## 5. 为什么不用减速比 6？

减速比 6:1 的意思是：

```text
电机轴转 6 圈，输出轴/关节转 1 圈
```

如果电机反馈的是电机转子角，那么理论上关节角和电机角有 6 倍关系。

但达妙 MIT 模式里的 `q` 到底是转子侧还是输出侧，取决于电机固件/型号/配置。

我们真实测试发现：

```text
q7_cmd = m7_default + 1.0 * fr_thigh_offset
q8_cmd = m8_default - 1.0 * fr_calf_offset
```

动作和仿真角度更一致。

因此当前工程里：

```text
真实标定优先级 > 理论减速比
```


## 6. target_deg、desired_deg、real_deg 到底是什么？

在 `damiao_3.py` 里你会看到类似：

```text
target_deg:(+1.0,+0.4)
desired_deg:(+1.0,+0.4)
real_deg:(+0.3,+0.5)
action:(+1.00,+0.39)
```

它们含义不同。

### 6.1 action

policy 网络原始输出，范围大概是：

```text
-1 到 +1
```

例如：

```text
action:(+1.00,-1.00)
```

说明 policy 两个关节都打满了。

### 6.2 desired_deg

policy 输出经过 `output_scale` 缩放后的目标角。

例如训练动作最大是 20 度：

```text
output_scale = 0.05
最大目标 = 20 * 0.05 = 1 度
```

所以你看到 desired_deg 基本在 ±1 度内。

### 6.3 target_deg

真正发给电机的目标角。

它经过了 ramp 限速，避免目标突然跳变。

当前 ramp 速度：

```text
35 deg/s
```

也就是每秒最多变化 35 度。

### 6.4 real_deg

真实电机反馈换算出来的实际关节角。

这个才是硬件实际动到哪里。

如果要看真机是否跟随，重点看：

```text
target_deg vs real_deg
```

如果要看 sim2real 是否一致，应该看：

```text
仿真 sim_real_deg vs 真机 real_deg
```

不是看：

```text
desired_deg vs real_deg
```


## 7. 为什么 output_scale=0.05 时 real_deg 看起来没完全跟上？

因为：

```text
output_scale=0.05
最大动作只有 ±1 度
```

1 度太小。

真实机械里有：

```text
静摩擦
间隙
负载
编码器量化
控制刚度不够
延迟
```

这些都会让 1 度小动作看起来误差比例很大。

例如：

```text
target_deg = +1.0
real_deg = +0.3
```

绝对误差是：

```text
0.7 度
```

它不一定说明系统坏了。

下一步可以逐步试：

```powershell
--output_scale 0.10
--output_scale 0.15
```

不要直接上：

```powershell
--output_scale 0.25
```

因为 policy 原始 action 经常打满，放大太快会冲。


## 8. 当前训练出来的 policy 到底迁移了什么？

要明确：

```text
sim2real 不是把仿真视频逐帧复制到真机。
```

更准确地说：

```text
同一个 policy
在仿真里用仿真 observation 决策
在真机里用真实 observation 决策
然后都输出 action 控制关节
```

宇树/legged_gym 这类部署也是类似流程：

```text
读取真机 q/dq/IMU
构造 observation
policy(obs)
action 转 target joint position
发送给电机 PD 控制器
```

它不是直接回放仿真轨迹。


## 9. 你的 track_reference 没到 5 意味着什么？

如果 reward 里：

```text
track_reference 最大接近 5
```

但训练结果没到 5，说明仿真里策略本身也没有完美跟踪参考轨迹。

这时不能期待真机比仿真更好。

真实情况一般是：

```text
仿真已经很好，真机可能差一点
仿真都不好，真机通常更不好
```

所以后续要验证真正 sim2real，一定要同时导出：

```text
sim_real_deg
sim_target_deg
sim_action
reference_deg
```

再和真机的：

```text
real_deg
target_deg
action
```

做对比。


## 10. 宇树/开源足式通常怎么做？

公开资料里常见流程是：

```text
1. 在仿真训练 policy
2. Play 检查仿真表现
3. Sim2Sim 到另一个仿真器检查是否过拟合
4. 真机部署
5. 真机读取 q/dq/IMU
6. 构造和训练一致的 observation
7. policy 输出 action
8. action 转成目标关节角
9. 电机 PD/MIT 控制执行
```

为了提高迁移成功率，还会做：

```text
domain randomization
摩擦/质量随机化
观测噪声
推扰
执行器模型
延迟建模
系统辨识
安全启动流程
动作限幅
动作低通/ramp
```

我们现在只做了最小版本：

```text
单腿
两个电机
真实 q/vel 闭环
动作限幅
动作 ramp
tau/err 安全保护
```

这只是开始，不是完整四足 locomotion sim2real。


## 11. 当前安全保护有哪些？

`damiao_3.py` 里主要有：

```text
output_scale：缩小 policy 动作
warmup_s：启动先保持 0 目标
fr_limit：目标角限制 ±20 度
speed_deg_s：目标变化速度限制，默认 35 deg/s
tau_limit：力矩保护，默认 1.2 Nm
err 检查：电机错误状态不正常就停
Ctrl+C：手动停止
```

当前建议先从：

```powershell
--output_scale 0.05
```

开始。

稳定后再试：

```powershell
--output_scale 0.10
```


## 12. 当前运行命令

不要在 `D:\IsaacLab` 跑硬件脚本。

正确方式：

```powershell
cd E:\HuanCun\Desktop\u2canfd
D:\Conda\envs\env_isaaclab\python.exe .\damiao_3.py --output_scale 0.05
```

如果要一直运行，默认就是一直运行：

```text
duration_s = 0
```

如果只跑 20 秒：

```powershell
D:\Conda\envs\env_isaaclab\python.exe .\damiao_3.py --output_scale 0.05 --duration_s 20
```


## 13. D:\IsaacLab 不能污染

硬规则：

```text
D:\IsaacLab 只作为 IsaacLab 项目和训练结果目录
硬件脚本不要从 D:\IsaacLab 运行
不要在 D:\IsaacLab 写临时文件
不要为了达妙硬件控制往 IsaacLab 里乱装包
```

当前硬件脚本统一在：

```text
E:\HuanCun\Desktop\u2canfd
```

运行。

policy 文件只是从这里读取：

```text
D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_15-43-00\exported\policy.pt
```


## 14. 当前代码状态总结

```text
damiao_2.py
    UDP 接收版
    runner 发目标，它控制电机
    policy 没有真实反馈
    不是 policy 真闭环

single_leg_policy_udp_runner.py / _3.py
    policy UDP 发送器
    用内部估计构造 observation
    用于早期链路验证

damiao_3.py
    policy 集成版
    读取真实 m7/m8 pos/vel
    构造真实 observation
    policy 直接闭环控制 canid7/canid8
    当前重点文件
```


## 15. 下一步正确路线

### 15.1 先做仿真日志

在 IsaacLab play 里导出：

```text
reference_deg
sim_target_deg
sim_real_deg
sim_action
tracking_error
```

### 15.2 再做真机日志

在 `damiao_3.py` 里导出：

```text
target_deg
real_deg
action
tau
tracking_error
```

### 15.3 对比

真正要比较的是：

```text
sim_real_deg vs real_deg
```

如果：

```text
sim_real_deg 都跟不好 reference
```

就先回到训练和 reward。

如果：

```text
sim_real_deg 很好，但 real_deg 差
```

再调：

```text
真实零位
kp/kd
输出比例
延迟
摩擦
动作尺度
限速
```


## 16. 一句话记住

`damiao_2.py` 是“外部策略发目标，真机执行”的 UDP 桥。

`damiao_3.py` 是“真机反馈进入 policy，policy 直接控制真机”的闭环部署。

真正的 sim2real 不是复制仿真画面，而是让同一个 policy 在真实反馈下也能产生类似仿真的行为。
"""


if __name__ == "__main__":
    print(DOCUMENT)
