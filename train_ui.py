import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
# 导入自定义的网格环境和智能体
from env import GridNavEnv
from train import D3QNAgent


# 1. 后台训练工作线程
class TrainingWorker(QThread):
    # 定义线程间通信的信号，指定信号传递的数据类型
    log_signal = pyqtSignal(str)  # 发送训练日志文本，用于界面实时打印
    metrics_signal = pyqtSignal(dict)  # 发送训练指标数据，用于更新界面曲线
    # 发送最优路径相关信息：路径坐标、发现轮次、步数、环境基础信息
    best_path_signal = pyqtSignal(list, int, int, dict)
    finished_signal = pyqtSignal(dict)  # 训练结束信号，通知主线程恢复界面状态

    def __init__(self, config, env):
        super().__init__()
        self.config = config
        self.env = env
        self.is_running = True
        self.agent = None

    def run(self):
        try:
            # 絕對不能在主執行緒初始化後跨執行緒傳遞，否則會引發 0xC0000409 崩潰
            self.agent = D3QNAgent(input_dim=self.env.observation_space.shape[0],
                                   output_dim=self.env.action_space.n,
                                   config=self.config)

            env = self.env

            # 初始化包含 10 個維度的數據記錄字典
            history = {
                'Reward': [], 'Steps': [], 'Loss': [],
                'Success_Rate': [], 'Collision_Rate': [], 'Epsilon': [],
                'Path_Smoothness': [], 'Final_Distance': [], 'Avg_Step_Reward': [], 'Turn_Count': []
            }

            window_rewards, window_steps, window_losses = [], [], []
            success_count, collision_count = 0, 0

            total_steps = 0
            episodes = 0
            best_success_steps = float('inf')

            self.log_signal.emit(f"🚀 开始 D3QN 训练 (目标步数: {self.config['max_train_steps']})")

            # 訓練主迴圈 (以總步數驅動)
            while total_steps < self.config['max_train_steps'] and self.is_running:
                state, _ = env.reset()
                ep_reward, ep_loss, ep_steps = 0, 0, 0
                ep_turns = 0
                last_action = 0
                path = [list(env.agent_pos)]
                episodes += 1

                # 單局內的步數迴圈
                for step in range(self.config['max_ep_steps']):
                    if not self.is_running: break

                    total_steps += 1
                    ep_steps += 1

                    action = self.agent.select_action(state)

                    # 統計轉彎次數 (用於計算平滑度)
                    if action != 0 and last_action != 0 and action != last_action:
                        ep_turns += 1
                    last_action = action

                    next_state, reward, terminated, truncated, info = env.step(action)
                    self.agent.memory.push(state, action, reward, next_state, terminated)

                    # 降低更新頻率維穩 (每 4 步更新 1 次)
                    if total_steps % 4 == 0:
                        loss = self.agent.update()
                        self.agent.soft_update()
                        ep_loss += loss

                    self.agent.decay_parameters(total_steps)
                    state = next_state
                    ep_reward += reward
                    path.append(list(env.agent_pos))

                    # 終局判定
                    if terminated or truncated:
                        if info.get('is_success'):
                            success_count += 1
                            # 發現更短的成功路徑，觸發最優路徑更新
                            if ep_steps < best_success_steps:
                                best_success_steps = ep_steps
                                env_info = {'size': env.size, 'static': env.static_obstacles, 'start': env.start_pos,
                                            'goal': env.goal_pos}
                                self.best_path_signal.emit(path, episodes, ep_steps, env_info)
                                self.log_signal.emit(f"🌟 发现最优路径！(第 {episodes} 轮出现, 共 {ep_steps} 步)")
                        if info.get('collision'):
                            collision_count += 1
                        break

                if not self.is_running: break

                # 計算進階評價指標
                smoothness = 1.0 - (ep_turns / ep_steps) if ep_steps > 0 else 1.0
                final_dist = max(abs(env.agent_pos[0] - env.goal_pos[0]), abs(env.agent_pos[1] - env.goal_pos[1]))
                avg_step_rew = ep_reward / ep_steps if ep_steps > 0 else 0
                update_times = ep_steps // 4

                # 記錄單輪資料
                history['Reward'].append(ep_reward)
                history['Steps'].append(ep_steps)
                history['Loss'].append(ep_loss / update_times if update_times > 0 else 0)
                history['Epsilon'].append(self.agent.epsilon)
                history['Path_Smoothness'].append(smoothness)
                history['Final_Distance'].append(final_dist)
                history['Avg_Step_Reward'].append(avg_step_rew)
                history['Turn_Count'].append(ep_turns)

                window_rewards.append(ep_reward)
                window_steps.append(ep_steps)
                window_losses.append(ep_loss / update_times if update_times > 0 else 0)

                # 每 10 輪更新一次 UI 介面
                if episodes % 10 == 0:
                    avg_reward = np.mean(window_rewards)
                    avg_steps = np.mean(window_steps)
                    avg_loss = np.mean(window_losses)
                    sr = success_count / 10.0
                    cr = collision_count / 10.0

                    history['Success_Rate'].extend([sr] * 10)
                    history['Collision_Rate'].extend([cr] * 10)

                    lr = self.agent.optimizer.param_groups[0]['lr']

                    log_msg = (f"Steps {total_steps:6d} | Ep {episodes:4d} | AvgRew: {avg_reward:6.1f} | "
                               f"Succ: {sr * 100:3.0f}% | Smooth: {smoothness:.2f} | Dist: {final_dist}")
                    self.log_signal.emit(log_msg)

                    plot_data = {
                        'episodes': list(range(1, episodes + 1)),
                        'rewards': history['Reward'],
                        'success_rates': history['Success_Rate'],
                        'losses': history['Loss']
                    }
                    self.metrics_signal.emit(plot_data)

                    window_rewards, window_steps, window_losses = [], [], []
                    success_count, collision_count = 0, 0

            self.log_signal.emit("⏹️ 訓練已結束或被手動終止。")
            env.close()

            # 補齊成功率資料長度，確保與繪圖對齊
            while len(history['Success_Rate']) < episodes:
                history['Success_Rate'].append(history['Success_Rate'][-1] if history['Success_Rate'] else 0)
                history['Collision_Rate'].append(history['Collision_Rate'][-1] if history['Collision_Rate'] else 0)

            self.finished_signal.emit(history)

        except Exception as e:
            # 捕捉子執行緒中的任何錯誤，防止靜默崩潰，並將錯誤輸出到 UI 控制台
            self.log_signal.emit(f"❌ 训练执行发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """手动终止训练的方法，设置运行标志为False，训练循环会检测并退出"""
        self.is_running = False


# 2. PyQt5 主界面类
class MainUI(QMainWindow):
    def __init__(self):
        """初始化主界面，设置窗口属性并调用界面初始化方法"""
        super().__init__()
        self.setWindowTitle("基于强化学习的动态避障可视化窗口")  # 设置窗口标题
        self.resize(1400, 850)  # 设置窗口初始大小
        self.worker = None  # 初始化训练工作线程，后续启动训练时实例化
        self.init_ui()  # 调用界面初始化方法，构建界面布局和组件

    def init_ui(self):
        """核心界面构建方法，创建所有界面组件并设置布局、样式、信号槽绑定"""
        # 全局主Widget，作为QMainWindow的中心部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)  # 全局主布局：水平布局（左+右）

        # ==================== 左侧面板 (参数与控制) ====================
        # 左侧面板为框架组件，固定宽度380px，包含参数配置、控制按钮、日志输出
        left_panel = QFrame()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)  # 左侧面板布局：垂直布局

        # 1. 超参数配置区：分组框，包含所有算法和环境的超参数输入框
        param_group = QGroupBox("算法与环境超参数")
        param_group.setStyleSheet("QGroupBox { font-weight: bold; }")  # 设置分组框样式
        form_layout = QFormLayout()  # 表单布局，适合“标签+输入框”的组合

        # 默认参数字典：键为参数名，值为QLineEdit输入框，设置默认值
        self.params = {
            'max_train_steps': QLineEdit('40000'),  # 最大全局训练步数
            'max_ep_steps': QLineEdit('150'),       # 单轮最大执行步数
            'epsilon_decay_steps': QLineEdit('10000'),  # 探索率ε的衰减步数
            'learning_rate': QLineEdit('0.0004'),   # 优化器学习率
            'gamma': QLineEdit('0.99'),             # 折扣因子（未来奖励的权重）
            'batch_size': QLineEdit('128'),         # 经验回放的批次大小
            'buffer_size': QLineEdit('50000')       # 经验回放池的最大容量
        }

        # 遍历参数字典，将所有参数的“标签+输入框”添加到表单布局
        for key, widget in self.params.items():
            form_layout.addRow(QLabel(key + ":"), widget)
        param_group.setLayout(form_layout)  # 为分组框设置表单布局
        left_layout.addWidget(param_group)  # 将分组框添加到左侧垂直布局

        # 2. 控制按钮区：水平布局，包含开始训练、终止训练按钮
        btn_layout = QHBoxLayout()
        # 开始训练按钮：设置样式（绿色）、文本，绑定点击事件
        self.btn_start = QPushButton("▶ 开始训练")
        self.btn_start.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.start_training)

        # 终止训练按钮：设置样式（红色）、文本，初始禁用，绑定点击事件
        self.btn_stop = QPushButton("⏹ 终止训练")
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)

        # 将按钮添加到水平布局，再添加到左侧垂直布局
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)

        # 模型保存按钮：设置文本，绑定点击事件，添加到左侧垂直布局
        self.btn_save = QPushButton("💾 导出当前模型权重")
        self.btn_save.clicked.connect(self.save_model)
        left_layout.addWidget(self.btn_save)

        # 3. 日志输出区：添加标签，创建文本浏览器作为日志控制台，设置样式（黑底白字）
        left_layout.addWidget(QLabel("<b>实时日志输出:</b>"))
        self.log_console = QTextBrowser()
        self.log_console.setStyleSheet("background-color: #1e272e; color: #d2dae2; font-family: Consolas;")
        left_layout.addWidget(self.log_console)  # 添加到左侧垂直布局

        # ==================== 右侧面板 (视图与曲线) ====================
        # 右侧面板为垂直分割器，可手动调整上下部分大小，包含最优路径绘图和实时指标曲线
        right_panel = QSplitter(Qt.Vertical)

        # 右上: 最优路径绘图区：创建Matplotlib画布，设置标题，隐藏坐标轴
        self.fig_path, self.ax_path = plt.subplots(figsize=(6, 5))  # 创建绘图图窗和坐标轴
        self.canvas_path = FigureCanvas(self.fig_path)  # 转换为PyQt5可用的画布
        self.ax_path.set_title("Waiting for Training...", fontsize=12)  # 初始标题
        self.ax_path.axis('off')  # 隐藏坐标轴
        right_panel.addWidget(self.canvas_path)  # 添加到垂直分割器

        # 右下: 实时曲线区 (包含三个子图：奖励、成功率、损失)
        # 创建1行3列的绘图图窗，设置画布大小，紧凑布局（避免子图重叠）
        self.fig_metrics, (self.ax_rew, self.ax_sr, self.ax_loss) = plt.subplots(1, 3, figsize=(12, 3))
        self.fig_metrics.tight_layout(pad=3.0)
        self.canvas_metrics = FigureCanvas(self.fig_metrics)  # 转换为PyQt5可用的画布

        # 设置三个子图的初始标题
        self.ax_rew.set_title('Rewards')
        self.ax_sr.set_title('Success Rate')
        self.ax_loss.set_title('Loss')
        right_panel.addWidget(self.canvas_metrics)  # 添加到垂直分割器

        # 设置右侧垂直分割器的上下部分初始大小比例为6:4
        right_panel.setSizes([600, 400])

        # 将左侧面板和右侧分割器添加到全局水平主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def start_training(self):
        try:
            env = GridNavEnv(render_mode=None)

            for widget in self.params.values():
                widget.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.log_console.clear()

            config = {
                'max_train_steps': int(self.params['max_train_steps'].text()),
                'max_ep_steps': int(self.params['max_ep_steps'].text()),
                'epsilon_decay_steps': int(self.params['epsilon_decay_steps'].text()),
                'learning_rate': float(self.params['learning_rate'].text()),
                'gamma': float(self.params['gamma'].text()),
                'batch_size': int(self.params['batch_size'].text()),
                'buffer_size': int(self.params['buffer_size'].text()),
                'tau': 0.001, 'epsilon_start': 1.0, 'epsilon_end': 0.05,
                'per_alpha': 0.6, 'per_beta_start': 0.5
            }

            # 直接实例化 Worker 并传入 config 和 env
            self.worker = TrainingWorker(config, env)
            self.worker.log_signal.connect(self.update_log)
            self.worker.metrics_signal.connect(self.update_metrics)
            self.worker.best_path_signal.connect(self.update_best_path)
            self.worker.finished_signal.connect(self.on_training_finished)
            self.worker.start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"环境初始化失败: {e}")

    def stop_training(self):
        """终止训练的槽函数，绑定到终止训练按钮的点击事件"""
        if self.worker:  # 若训练线程已实例化
            self.worker.stop()  # 调用线程的stop方法，设置运行标志为False
            self.btn_stop.setEnabled(False)  # 禁用终止训练按钮

    def on_training_finished(self):
        """
        训练结束的槽函数，绑定到线程的finished_signal
        逻辑：恢复输入框和按钮的可操作状态
        """
        # 启用所有超参数输入框
        for widget in self.params.values():
            widget.setEnabled(True)
        # 按钮状态恢复：启用开始训练，禁用终止训练
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def save_model(self):
        if self.worker and self.worker.agent:
            import time
            os.makedirs("model", exist_ok=True)  # 创建 model 文件夹

            # 生成带有当前时间戳的文件名 (例如: d3qn_20260313_153022.pth)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"model/d3qn_{timestamp}.pth"

            torch.save(self.worker.agent.policy_net.state_dict(), save_path)
            self.update_log(f"✅ 模型参数已保存至 {save_path}")
        else:
            QMessageBox.warning(self, "警告", "没有正在运行的模型可供保存。")

    def update_log(self, msg):
        """
        更新日志的槽函数，绑定到线程的log_signal
        :param msg: 线程发送的日志文本
        逻辑：在日志控制台添加文本，并将滚动条保持在最底端
        """
        self.log_console.append(msg)
        # 保持滚动条在最底端，实时查看最新日志
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def update_metrics(self, data):
        """
        更新训练指标曲线的槽函数，绑定到线程的metrics_signal
        :param data: 线程发送的绘图数据字典（轮次、奖励、成功率、损失）
        逻辑：清空原有曲线，绘制原始数据和滑动平均数据，刷新画布
        """
        eps = data['episodes']  # 训练轮次

        # 绘制奖励曲线：浅色原始波动 + 10轮滑动平均曲线（深蓝色）
        self.ax_rew.clear()  # 清空原有曲线
        self.ax_rew.plot(eps, data['rewards'], color='#B0C4DE', alpha=0.6)  # 原始奖励曲线
        # 若奖励数据长度大于10，计算并绘制10轮滑动平均曲线，平滑展示趋势
        if len(data['rewards']) > 10:
            smoothed = np.convolve(data['rewards'], np.ones(10) / 10, mode='valid')
            self.ax_rew.plot(eps[9:], smoothed, color='#000080')
        self.ax_rew.set_title('Rewards')  # 重置标题

        # 绘制成功率曲线：绿色，直接绘制原始数据
        self.ax_sr.clear()
        self.ax_sr.plot(eps, data['success_rates'], color='#27ae60')
        self.ax_sr.set_title('Success Rate')

        # 绘制损失曲线：红色，直接绘制原始数据
        self.ax_loss.clear()
        self.ax_loss.plot(eps, data['losses'], color='#c0392b')
        self.ax_loss.set_title('Avg Loss')

        self.canvas_metrics.draw()  # 刷新画布，显示新绘制的曲线

    def update_best_path(self, path, episode, steps, env_info):
        try:
            self.ax_path.clear()
            size = env_info['size']
            grid = np.zeros((size, size))
            for ox, oy in env_info['static']:
                grid[ox, oy] = 1

            cmap = plt.cm.colors.ListedColormap(['white', 'dimgray'])
            self.ax_path.pcolormesh(grid.T, cmap=cmap, edgecolors='lightgray', linewidth=0.5)
            self.ax_path.invert_yaxis()

            start = env_info['start']
            goal = env_info['goal']
            self.ax_path.plot(start[0] + 0.5, start[1] + 0.5, 'go', markersize=12, label='Start')
            self.ax_path.plot(goal[0] + 0.5, goal[1] + 0.5, 'y*', markersize=15, label='Goal')

            if path:
                path_x = [p[0] + 0.5 for p in path]
                path_y = [p[1] + 0.5 for p in path]
                self.ax_path.plot(path_x, path_y, 'b-', linewidth=2, label='Best Path')

            self.ax_path.set_title(f"Best Path (Found at Ep: {episode} | Steps: {steps})", fontsize=14,
                                   color='darkgreen')
            self.ax_path.legend(loc='upper right')
            self.ax_path.axis('equal')

            self.canvas_path.draw()

        except Exception as e:
            print(f"Path Update Error: {e}")


if __name__ == "__main__":
    # 解决高分屏显示器下PyQt5界面缩放模糊的问题
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 创建PyQt5应用程序实例，处理命令行参数
    app = QApplication(sys.argv)
    # 关闭坐标轴负号的unicode显示，避免负号变成方块
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    # 实例化主界面并显示
    window = MainUI()
    window.show()
    # 启动PyQt5应用程序的主循环，直到程序退出，返回退出码
    sys.exit(app.exec_())