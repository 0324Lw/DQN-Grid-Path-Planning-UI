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

