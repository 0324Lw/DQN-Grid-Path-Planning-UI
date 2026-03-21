# 🚀 DQN-Grid-Path-Planning-UI

> 基于深度强化学习（DQN/D3QN）的栅格地图路径规划与动态避障算法。
> 本项目不仅实现了底层强化学习算法，还开发了一套基于 PyQt5 的完整可视化交互界面，支持实时训练监控、最优路径动态展示以及模型与数据的自动归档。

## ✨ 核心特性 (Features)

* 🧠 **深度强化学习底座**：自定义基于 `gymnasium` 的栅格导航环境，实现了 DQN（及相关变体）算法，用于解决复杂网格环境下的起点到终点路径寻优。
* 🖥️ **PyQt5 交互式 UI**：告别枯燥的纯代码训练！提供可视化控制面板，可随时一键启动/终止训练，并动态调整环境参数。
* 📈 **实时训练监控 (Matplotlib)**：在界面中实时绘制 Reward、Loss、Success Rate 等关键指标的平滑曲线，训练状态一目了然。
* 🏆 **最优路径动态刷新**：系统会在训练过程中自动捕捉并展示当前探索到的最优无碰撞路径。
* 💾 **全自动归档系统**：
  * `model/`：自动带时间戳保存训练好的 PyTorch `.pth` 模型权重。
  * `data/`：训练彻底结束后，自动导出完整维度的 `training_history.csv`，并生成所有指标的高清 SCI 级别平滑曲线图。
  * `最优路径/`：自动抓拍并保存破纪录时的路线图。

## 🧠 强化学习环境设计 (Environment Design)

本项目的底层环境基于网格地图的马尔可夫决策过程 (MDP) 进行建模，核心的状态空间、动作空间与奖励函数设计如下：

### 1. 状态空间 (Observation Space)

智能体的观测状态为 **15 维的连续向量**，所有维度的数值均在内部进行了归一化处理（映射至 $[-1.0, 1.0]$ 区间），以加速深度 Q 网络 (DQN) 的收敛。具体维度特征如下：

| 维度索引 | 物理含义 (Description) | 数据范围 (Range) |
| :---: | :--- | :---: |
| `0:2` | 智能体当前的二维网格坐标 $(x, y)$ | $[0, 1.0]$ |
| `2:4` | 智能体朝向目标点的相对方向向量 | $[-1.0, 1.0]$ |
| `4` | 智能体到目标点的欧氏距离（除以对角线长度） | $[0, 1.0]$ |
| `5:7` | 智能体上一步执行动作的坐标偏移量 | $[-0.5, 0.5]$ |
| `7:15` | **八向雷达测距**：模拟激光雷达在 8 个方向（上下左右及对角线）探测到的最近障碍物或边界距离 | $[0, 1.0]$ |

### 2. 动作空间 (Action Space)

项目采用了 **9 维离散动作空间 (Discrete 9)**，囊括了二维平面上的 8 个移动方向以及“原地等待”动作：

* `0`: 原地不动 (0, 0)
* `1~4`: 上 (0, -1), 下 (0, 1), 左 (-1, 0), 右 (1, 0)
* `5~8`: 左上 (-1, -1), 右上 (1, -1), 左下 (-1, 1), 右下 (1, 1)

### 3. 奖励函数设计 (Reward Function)

为了在复杂的栅格迷宫中引导智能体进行有效探索，本项目设计了包含动态势场与行为约束的密集型奖励函数。单步总奖励 $R_{raw}$ 由以下几部分累加而成：

$$R_{raw} = r_{step} + r_{fwd} + r_{dir} + r_{rep} + r_{back} + r_{turn} + r_{event}$$

* 🔴 **步数惩罚 ($r_{step}$)**：每走一步产生微小的固定惩罚 ($-0.1$)，鼓励用最短路径到达终点。
* 🟢 **前进奖励 ($r_{fwd}$)**：基于距离差的势能奖励。向终点移动给予正奖励，远离则给予惩罚（缩放系数：`2.0`）。
* 🟢 **方向对齐奖励 ($r_{dir}$)**：智能体移动方向与目标方向向量的余弦相似度奖励（缩放系数：`0.5`）。
* 🔴 **障碍物斥力惩罚 ($r_{rep}$)**：当距离障碍物小于安全距离 (`2.0` 格) 时，基于反比例函数给予排斥惩罚。
* 🔴 **无效行为约束 ($r_{back}, r_{turn}$)**：对无效的“退回上一步”动作 ($-0.5$) 和“频繁改变方向”动作 ($-0.2$) 施加轻微惩罚，提升路径平滑度。
* 🏆 **终局事件 ($r_{event}$)**：
  * **成功到达终点**：给予极大奖励 ($+100.0$)。
  * **发生物理碰撞**：给予致命惩罚 ($-50.0$) 并提前终止回合。

> **💡 奖励值归一化 (Reward Scaling & Clipping):** > 为了防止极端惩罚导致 DQN 训练时的梯度爆炸，环境在输出最终奖励前，会对 $R_{raw}$ 进行线性缩放（除以 **10.0**），并将其硬裁剪 (Clip) 至 **$[-10.0, 10.0]$** 的范围内。

## 🛠️ 环境依赖 (Requirements)

本项目在 Python 3.10 环境下开发与测试。主要依赖如下：

* `torch` (>= 2.2.2)
* `PyQt5` (>= 5.15.10)
* `matplotlib` (>= 3.8.4)
* `gymnasium` (>= 0.29.1)
* `pygame` (>= 2.5.2)
* `numpy`, `pandas`

**快速安装指南：**
```bash
conda create -n dqn_nav python=3.10 -y
conda activate dqn_nav
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install numpy pandas matplotlib pyqt gymnasium pygame -c conda-forge -y
```
## 📂 项目结构 (Project Structure)
├── train_ui.py         # PyQt5 主界面与多线程训练调度入口  
├── plotdata.py         # 训练结束后的数据持久化与高清图表生成模块  
├── model/              # 自动保存的模型权重文件 (.pth)  
├── data/               # 自动保存的训练日志 (.csv) 与分析图表  
└── 最优路径/            # 训练过程中自动抓拍的最优路线截图  

![env_demo_2_size10_complex](https://github.com/user-attachments/assets/1934181c-1f7c-4066-bedb-5f09267a10af)
![env_demo_4_size18_complex](https://github.com/user-attachments/assets/77abcef2-87a7-46da-8032-1b1c9fe86d66)
<img width="1416" height="888" alt="界面1" src="https://github.com/user-attachments/assets/7c343794-7d2b-4ac4-b2fb-a2be38ba155e" />
<img width="1416" height="888" alt="2" src="https://github.com/user-attachments/assets/d0a41ce6-198a-4ab6-8c4b-480444c7cbb8" />
<img width="1416" height="888" alt="3" src="https://github.com/user-attachments/assets/fd0d7dcc-97be-4fd2-bc0f-b16bf11ad505" />
<img width="1416" height="888" alt="4" src="https://github.com/user-attachments/assets/6093ef23-1887-4778-b989-28f1bff4acc0" />
<img width="1416" height="888" alt="5" src="https://github.com/user-attachments/assets/0072c58b-ab58-4800-b462-77f8a19e7200" />
<img width="1422" height="236" alt="路径演变" src="https://github.com/user-attachments/assets/6830b366-2be6-404f-8e7e-7aec8c97077d" />
<img width="500" height="500" alt="损失值" src="https://github.com/user-attachments/assets/fe511cf1-87c0-4303-8372-8ffed122cdf3" />
<img width="500" height="500" alt="奖励值" src="https://github.com/user-attachments/assets/be0f7713-841e-4079-9b1d-b086ce7f927a" />
<img width="500" height="500" alt="碰撞率" src="https://github.com/user-attachments/assets/0f4bad15-ad09-4998-b086-f84f96e17b0e" />
<img width="500" height="500" alt="每轮步数" src="https://github.com/user-attachments/assets/f0bd4232-dc2e-4d1e-9f52-102a447040b4" />

