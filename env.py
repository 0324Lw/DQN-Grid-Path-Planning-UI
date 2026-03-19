import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import math
import random
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QSpinBox, \
    QPushButton, QButtonGroup, QFrame, QMessageBox
from PyQt5.QtCore import Qt


# 辅助 UI 配置函数
class ConfigDialog(QDialog):
    def __init__(self):
        super().__init__()
        # 设置窗口标题
        self.setWindowTitle("训练环境参数设置")
        # 固定窗口大小
        self.setFixedSize(450, 350)
        # 设置界面样式表，定义各控件的外观
        self.setStyleSheet("""
            QDialog { background-color: #f5f6fa; }
            QLabel { font-size: 14px; font-weight: bold; color: #2f3640; }
            QRadioButton { font-size: 13px; color: #353b48; }
            QPushButton { background-color: #00a8ff; color: white; font-weight: bold; border-radius: 5px; padding: 8px; }
            QPushButton:hover { background-color: #0097e6; }
            QSpinBox { font-size: 13px; padding: 3px; border: 1px solid #dcdde1; border-radius: 3px; }
        """)

        # 初始化配置字典，存储默认的环境参数
        self.config = {
            'size': 20,        # 栅格地图大小，默认20x20
            'obs_mode': 'auto',# 障碍物生成模式，默认自动
            'difficulty': 'simple', # 环境难度，默认简单
            'static_obs': [],  # 静态障碍物坐标集合
            'dynamic_obs': []  # 动态障碍物信息列表
        }
        # 初始化界面控件
        self.init_ui()

    def init_ui(self):
        # 创建垂直主布局，设置控件间距和边距
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # ================= 1. 地图大小设置 =================
        size_label = QLabel("1. 栅格地图大小设置")
        layout.addWidget(size_label)

        # 创建地图大小选择的水平布局
        size_layout = QHBoxLayout()
        # 建立按钮组，实现单选互斥效果
        self.size_group = QButtonGroup(self)
        # 默认20x20单选按钮，默认选中
        self.size_btn_default = QRadioButton("默认 (20x20)")
        self.size_btn_default.setChecked(True)
        # 手动设置单选按钮
        self.size_btn_manual = QRadioButton("手动设置:")

        # 将按钮加入按钮组
        self.size_group.addButton(self.size_btn_default)
        self.size_group.addButton(self.size_btn_manual)

        # 创建数值选择框，设置范围10-40，默认值20，初始禁用
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(10, 40)
        self.size_spinbox.setValue(20)
        self.size_spinbox.setEnabled(False)

        # 绑定手动设置按钮的状态，控制数值框的启用/禁用
        self.size_btn_manual.toggled.connect(lambda: self.size_spinbox.setEnabled(self.size_btn_manual.isChecked()))

        # 将控件加入水平布局
        size_layout.addWidget(self.size_btn_default)
        size_layout.addWidget(self.size_btn_manual)
        size_layout.addWidget(self.size_spinbox)
        size_layout.addStretch()
        layout.addLayout(size_layout)

        # 添加水平分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # ================= 2. 障碍物设置 =================
        obs_label = QLabel("2. 障碍物生成模式")
        layout.addWidget(obs_label)

        # 建立障碍物模式按钮组，实现单选互斥
        self.obs_group = QButtonGroup(self)
        # 自动均匀生成按钮，默认选中
        self.obs_btn_auto = QRadioButton("自动均匀生成")
        self.obs_btn_auto.setChecked(True)
        # 手动绘制地图按钮
        self.obs_btn_manual = QRadioButton("手动绘制地图")

        # 将按钮加入按钮组
        self.obs_group.addButton(self.obs_btn_auto)
        self.obs_group.addButton(self.obs_btn_manual)

        layout.addWidget(self.obs_btn_auto)

        # 难度选择按钮组，建立互斥
        self.diff_group = QButtonGroup(self)
        diff_layout = QHBoxLayout()
        # 设置难度布局的左内边距
        diff_layout.setContentsMargins(20, 0, 0, 0)
        # 简单/一般/复杂环境单选按钮，默认选中简单
        self.diff_simple = QRadioButton("简单环境")
        self.diff_simple.setChecked(True)
        self.diff_medium = QRadioButton("一般环境")
        self.diff_complex = QRadioButton("复杂环境")

        # 将难度按钮加入按钮组
        self.diff_group.addButton(self.diff_simple)
        self.diff_group.addButton(self.diff_medium)
        self.diff_group.addButton(self.diff_complex)

        # 将难度按钮加入水平布局
        diff_layout.addWidget(self.diff_simple)
        diff_layout.addWidget(self.diff_medium)
        diff_layout.addWidget(self.diff_complex)

        # 绑定手动绘制按钮状态，控制难度按钮的启用/禁用
        self.obs_btn_manual.toggled.connect(
            lambda: [btn.setEnabled(not self.obs_btn_manual.isChecked()) for btn in self.diff_group.buttons()]
        )

        layout.addLayout(diff_layout)
        layout.addWidget(self.obs_btn_manual)
        # 添加伸缩项，使下方按钮靠下
        layout.addStretch()

        # 确定按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        # 创建确认按钮，设置最小宽度
        confirm_btn = QPushButton("确认设置并启动环境")
        confirm_btn.setMinimumWidth(150)
        # 绑定按钮点击事件，保存配置并关闭窗口
        confirm_btn.clicked.connect(self.save_and_close)
        btn_layout.addWidget(confirm_btn)
        layout.addLayout(btn_layout)

        # 设置窗口的主布局
        self.setLayout(layout)

    def save_and_close(self):
        # 保存地图大小配置：如果手动设置则取数值框值，否则默认20
        if self.size_btn_manual.isChecked():
            self.config['size'] = self.size_spinbox.value()
        else:
            self.config['size'] = 20

        # 保存障碍物模式配置
        if self.obs_btn_manual.isChecked():
            self.config['obs_mode'] = 'manual'
        else:
            self.config['obs_mode'] = 'auto'
            # 保存难度配置，根据选中的按钮赋值
            if self.diff_simple.isChecked():
                self.config['difficulty'] = 'simple'
            elif self.diff_medium.isChecked():
                self.config['difficulty'] = 'medium'
            else:
                self.config['difficulty'] = 'complex'

        # 确认关闭窗口，返回配置信息
        self.accept()


def get_env_config():
    # 检查是否已有Qt应用实例，避免重复创建
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    # 创建配置窗口并执行
    dialog = ConfigDialog()
    dialog.exec_()
    # 返回用户设置的配置字典
    return dialog.config


# 强化学习环境主类，继承gym的Env基类
class GridNavEnv(gym.Env):
    # 定义环境元数据：渲染模式和渲染帧率
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None):
        super(GridNavEnv, self).__init__()

        # 获取用户从UI设置的环境配置
        self.config = get_env_config()
        # 栅格地图的尺寸
        self.size = self.config['size']
        # 每个栅格的像素大小，总窗口固定600像素
        self.grid_size = int(600 / self.size)
        # 窗口总像素大小
        self.window_size = self.size * self.grid_size

        # 设置智能体起点和终点坐标，避开边界
        self.start_pos = (1, 1)
        self.goal_pos = (self.size - 2, self.size - 2)

        # 定义动作空间：9个离散动作，对应8方向移动+原地不动
        self.action_space = spaces.Discrete(9)
        # 动作字典：键为动作编号，值为坐标偏移量
        self.action_dict = {
            0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0),
            5: (-1, -1), 6: (1, -1), 7: (-1, 1), 8: (1, 1)
        }

        # 定义观测空间：15维连续数组，范围[-1,1]，浮点型
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)

        # 初始化静态障碍物集合
        self.static_obstacles = set()
        # 初始化动态障碍物列表，存储移动相关信息
        self.dynamic_obstacles = []
        # 智能体当前位置，初始为起点
        self.agent_pos = list(self.start_pos)
        # 智能体位置历史，保存最近2步，用于判断后退
        self.pos_history = [list(self.start_pos)] * 2
        # 上一步执行的动作，初始为0（原地不动）
        self.last_action = 0
        # 单局最大步数限制，防止无限循环
        self.max_steps = 500
        # 当前步数计数
        self.current_step = 0

        # 渲染模式，可选human（可视化）/rgb_array（像素数组）
        self.render_mode = render_mode
        # 渲染窗口对象，初始为空
        self.window = None
        # 渲染时钟对象，控制帧率
        self.clock = None

        # 根据障碍物模式，生成对应的障碍物
        if self.config['obs_mode'] == 'auto':
            self._generate_auto_obstacles()
        else:
            self._manual_set_obstacles()  # 手动绘制模式，执行绘图逻辑

    def _is_safe_zone(self, x, y):
        # 计算当前坐标到起点和终点的切比雪夫距离
        dist_start = max(abs(x - self.start_pos[0]), abs(y - self.start_pos[1]))
        dist_goal = max(abs(x - self.goal_pos[0]), abs(y - self.goal_pos[1]))
        # 判断是否在起点/终点的安全区内（距离≤3格），安全区不生成障碍物
        return dist_start <= 3 or dist_goal <= 3

    def _generate_auto_obstacles(self):
        """扫描式候选点生成法：严格保证1格间距，允许贴边，确保生成数量达标"""
        # 难度映射字典：键为难度，值为(静态障碍物占比, 动态障碍物数量)
        diff_map = {'simple': (0.08, 0), 'medium': (0.10, 1), 'complex': (0.12, 2)}
        ratio, dyn_count = diff_map[self.config['difficulty']]

        # 计算静态障碍物的目标生成数量（总栅格数×占比）
        target_static_count = int(self.size * self.size * ratio)

        # 定义1-3格的障碍物形状模板，用于组合生成静态障碍物
        simple_shapes = [
            [(0, 0)],  # 1格
            [(0, 0), (1, 0)], [(0, 0), (0, 1)],  # 2格
            [(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1), (0, 2)],  # 3格直线
            [(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (1, 1)],  # 3格直角
            [(0, 1), (1, 1), (0, 0)], [(1, 0), (1, 1), (0, 1)]   # 3格L型
        ]

        # 已放置的静态障碍物栅格数
        placed_static_count = 0

        # 内部辅助函数：检查某个单独的格子是否可以放置障碍物
        def is_valid_cell(cx, cy):
            # 1. 检查是否越界（坐标在0到size-1之间）
            if not (0 <= cx < self.size and 0 <= cy < self.size):
                return False
            # 2. 检查是否在起终点的安全区内
            if self._is_safe_zone(cx, cy):
                return False
            # 3. 检查该位置是否已经被占用
            if (cx, cy) in self.static_obstacles:
                return False
            # 4. 检查周围8个邻居是否有其他障碍物，保证至少1格间距
            for px in [-1, 0, 1]:
                for py in [-1, 0, 1]:
                    if (cx + px, cy + py) in self.static_obstacles:
                        return False
            return True

        # ==================== 1. 生成静态障碍物 ====================
        # 形状级别的尝试上限，防止后期无空间导致死循环
        attempts_left = 1000

        # 循环生成，直到达到目标数量或尝试次数用尽
        while placed_static_count < target_static_count and attempts_left > 0:
            # 随机选择一个障碍物形状
            shape = random.choice(simple_shapes)

            # 全图扫描：寻找所有能够完整放下当前形状的基准锚点 (bx, by)
            valid_anchors = []
            for bx in range(self.size):
                for by in range(self.size):
                    can_place = True
                    # 检查形状的每个子块是否都合法
                    for dx, dy in shape:
                        if not is_valid_cell(bx + dx, by + dy):
                            can_place = False
                            break
                    if can_place:
                        valid_anchors.append((bx, by))

            # 如果存在合法锚点，随机选择一个放置形状
            if valid_anchors:
                bx, by = random.choice(valid_anchors)
                for dx, dy in shape:
                    self.static_obstacles.add((bx + dx, by + dy))
                    placed_static_count += 1
            else:
                # 无合法锚点，扣除一次尝试机会
                attempts_left -= 1

        # 调试信息：如果因间距限制未达到目标数量，打印提示
        if placed_static_count < target_static_count:
            print(
                f"提示: 维持安全间距的前提下已达空间极限，计划生成 {target_static_count} 格，实际生成了 {placed_static_count} 格。")

        # ==================== 2. 生成动态障碍物 ====================
        # 根据难度生成对应数量的动态障碍物
        for _ in range(dyn_count):
            while True:
                # 随机生成动态障碍物的起点和终点参数，避开边界
                k = random.randint(4, self.size - 5)
                d = random.randint(2, 4)
                sx, sy = k - d, k + d
                ex, ey = k + d, k - d
                # 检查坐标是否越界，且不在安全区内
                if 0 <= sx < self.size and 0 <= sy < self.size and 0 <= ex < self.size and 0 <= ey < self.size:
                    if not self._is_safe_zone(sx, sy) and not self._is_safe_zone(ex, ey):
                        # 添加动态障碍物信息：起点、终点、当前位置、移动方向
                        self.dynamic_obstacles.append({
                            'start': (sx, sy), 'end': (ex, ey),
                            'pos': [sx, sy], 'moving_to_end': True
                        })
                        break

    def _manual_set_obstacles(self):
        """完整的 Pygame 手动绘制逻辑，左键设障碍，右键切换/保存"""
        # 初始化pygame
        pygame.init()
        # 创建绘制窗口
        window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("手动设置模式 - 左键设障碍，右键切换/保存")

        # 绘制阶段：初始为静态障碍物绘制
        phase = 'static'
        # 动态障碍物临时起点，初始为空
        dyn_temp_start = None
        # 绘制循环标志
        running = True

        while running:
            # 窗口背景填充白色
            window.fill((255, 255, 255))
            # 绘制栅格线，遍历所有坐标
            for x in range(self.size):
                for y in range(self.size):
                    rect = pygame.Rect(x * self.grid_size, y * self.grid_size, self.grid_size, self.grid_size)
                    pygame.draw.rect(window, (200, 200, 200), rect, 1)

            # 绘制安全区提示框（起点终点周围4x4区域，绿色边框）
            pygame.draw.rect(window, (200, 255, 200), (0, 0, 4 * self.grid_size, 4 * self.grid_size), 2)
            pygame.draw.rect(window, (200, 255, 200),
                             ((self.size - 4) * self.grid_size, (self.size - 4) * self.grid_size, 4 * self.grid_size,
                              4 * self.grid_size), 2)

            # 绘制已选择的静态障碍物（灰色方块）
            for ox, oy in self.static_obstacles:
                pygame.draw.rect(window, (50, 50, 50),
                                 (ox * self.grid_size, oy * self.grid_size, self.grid_size, self.grid_size))
            # 绘制已设置的动态障碍物（红色起点、浅红终点、红色连线）
            for dyn in self.dynamic_obstacles:
                sx, sy = dyn['start']
                ex, ey = dyn['end']
                pygame.draw.rect(window, (255, 0, 0),
                                 (sx * self.grid_size, sy * self.grid_size, self.grid_size, self.grid_size))
                pygame.draw.rect(window, (255, 100, 100),
                                 (ex * self.grid_size, ey * self.grid_size, self.grid_size, self.grid_size))
                pygame.draw.line(window, (255, 0, 0),
                                 (sx * self.grid_size + self.grid_size / 2, sy * self.grid_size + self.grid_size / 2),
                                 (ex * self.grid_size + self.grid_size / 2, ey * self.grid_size + self.grid_size / 2),
                                 2)

            # 绘制动态障碍物的临时起点（橙色圆圈）
            if dyn_temp_start:
                pygame.draw.circle(window, (255, 165, 0), (int(dyn_temp_start[0] * self.grid_size + self.grid_size / 2),
                                                           int(dyn_temp_start[
                                                                   1] * self.grid_size + self.grid_size / 2)),
                                   self.grid_size // 3)

            # 更新窗口显示
            pygame.display.flip()

            # 事件处理循环
            for event in pygame.event.get():
                # 关闭窗口事件
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # 鼠标点击事件
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # 获取鼠标像素坐标，转换为栅格坐标
                    mx, my = event.pos
                    gx, gy = mx // self.grid_size, my // self.grid_size

                    # 鼠标左键点击：放置障碍物
                    if event.button == 1:
                        # 不在安全区内才允许放置
                        if not self._is_safe_zone(gx, gy):
                            # 静态阶段：添加静态障碍物
                            if phase == 'static':
                                self.static_obstacles.add((gx, gy))
                            # 动态阶段：先选起点，再选终点
                            elif phase == 'dynamic':
                                if not dyn_temp_start:
                                    dyn_temp_start = (gx, gy)
                                else:
                                    # 添加动态障碍物信息
                                    self.dynamic_obstacles.append({
                                        'start': dyn_temp_start, 'end': (gx, gy),
                                        'pos': list(dyn_temp_start), 'moving_to_end': True
                                    })
                                    # 重置临时起点
                                    dyn_temp_start = None
                    # 鼠标右键点击：切换阶段/保存退出
                    elif event.button == 3:
                        if phase == 'static':
                            # 从静态切换到动态，修改窗口标题提示
                            phase = 'dynamic'
                            pygame.display.set_caption("设置动态障碍物: 左键点起点再点终点，右键保存并关闭")
                        elif phase == 'dynamic':
                            # 动态阶段右键，结束绘制
                            running = False
        # 退出pygame绘制窗口
        pygame.quit()

    def reset(self, seed=None, options=None):
        # 调用父类reset方法，设置随机种子
        super().reset(seed=seed)
        # 重置智能体位置为起点
        self.agent_pos = list(self.start_pos)
        # 重置位置历史
        self.pos_history = [list(self.start_pos)] * 2
        # 重置当前步数
        self.current_step = 0
        # 重置上一步动作
        self.last_action = 0
        # 重置所有动态障碍物：回到起点，朝终点移动
        for dyn in self.dynamic_obstacles:
            dyn['pos'] = list(dyn['start'])
            dyn['moving_to_end'] = True

        # 如果是人类可视化模式，执行渲染
        if self.render_mode == "human": self.render()
        # 返回初始观测值和空信息字典
        return self._get_obs(), {}

    def _get_obs(self):
        # 获取智能体和终点的当前坐标
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        # 1. 绝对位置与相对目标信息 (7维)
        # 智能体坐标归一化（0-1）
        norm_x, norm_y = ax / self.size, ay / self.size
        # 智能体到终点的方向向量归一化
        dir_x, dir_y = (gx - ax) / self.size, (gy - ay) / self.size
        # 智能体到终点的欧氏距离归一化（除以对角线最大距离）
        dist_to_goal = math.hypot(gx - ax, gy - ay) / (self.size * 1.414)
        # 上一步动作的偏移量
        vx, vy = self.action_dict[self.last_action]
        # 动作偏移量归一化
        norm_vx, norm_vy = vx / 2.0, vy / 2.0

        # 2. 八向雷达测距 (8维)，模拟激光雷达检测障碍物
        lidar_distances = []
        # 八个探测方向：上, 下, 左, 右, 左上, 右上, 左下, 右下
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0),
                      (-1, -1), (1, -1), (-1, 1), (1, 1)]

        # 预先提取动态障碍物的当前坐标集合，提升检测速度
        dyn_positions = set(tuple(dyn['pos']) for dyn in self.dynamic_obstacles)

        # 遍历每个探测方向，执行射线步进探测
        for dx, dy in directions:
            cx, cy = ax, ay
            # 射线步进，直到碰到边界或障碍物
            while True:
                cx += dx
                cy += dy
                # 碰到地图边界，停止探测
                if not (0 <= cx < self.size and 0 <= cy < self.size):
                    break
                # 碰到静态/动态障碍物，停止探测
                if (cx, cy) in self.static_obstacles or (cx, cy) in dyn_positions:
                    break

            # 计算射线终点到智能体的欧氏距离，并归一化（最大1.0）
            actual_dist = math.hypot(cx - ax, cy - ay)
            norm_dist = min(actual_dist / (self.size * 1.414), 1.0)
            lidar_distances.append(norm_dist)

        # 3. 组合为15维状态向量，作为观测值
        obs = np.array([
            norm_x, norm_y, dir_x, dir_y, dist_to_goal, norm_vx, norm_vy,
            *lidar_distances
        ], dtype=np.float32)

        return obs

    def step(self, action):
        # 步数加1
        self.current_step += 1
        # 根据动作编号获取坐标偏移量
        dx, dy = self.action_dict[action]

        # 记录移动前的位置和到终点的距离，用于计算奖励
        old_pos = list(self.agent_pos)
        old_dist = math.hypot(self.goal_pos[0] - old_pos[0], self.goal_pos[1] - old_pos[1])

        # 计算新位置，限制在地图边界内（0到size-1）
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        new_pos = [new_x, new_y]

        # 更新位置历史：移除最旧的，添加新位置
        self.pos_history.pop(0)
        self.pos_history.append(new_pos)
        # 更新智能体当前位置
        self.agent_pos = new_pos

        # 计算移动后的新距离
        new_dist = math.hypot(self.goal_pos[0] - self.agent_pos[0], self.goal_pos[1] - self.agent_pos[1])

        # 更新所有动态障碍物的位置
        for dyn in self.dynamic_obstacles:
            # 确定当前移动目标（终点/起点）
            target = dyn['end'] if dyn['moving_to_end'] else dyn['start']
            cx, cy = dyn['pos']
            # 如果到达目标，切换移动方向
            if (cx, cy) == target:
                dyn['moving_to_end'] = not dyn['moving_to_end']
                target = dyn['end'] if dyn['moving_to_end'] else dyn['start']
            # 计算x/y方向的移动偏移量（向目标移动1格）
            move_x = 1 if target[0] > cx else (-1 if target[0] < cx else 0)
            move_y = 1 if target[1] > cy else (-1 if target[1] < cy else 0)
            # 更新动态障碍物位置
            dyn['pos'] = [cx + move_x, cy + move_y]

        # 初始化碰撞标志
        collision = False
        # 初始化到障碍物的最小距离，用于斥力惩罚
        min_obs_dist = float('inf')
        # 判断是否与静态障碍物碰撞
        if tuple(self.agent_pos) in self.static_obstacles: collision = True

        # 计算到所有静态障碍物的最小距离
        for ox, oy in self.static_obstacles:
            d = math.hypot(ox - self.agent_pos[0], oy - self.agent_pos[1])
            if d < min_obs_dist: min_obs_dist = d

        # 计算到所有动态障碍物的最小距离，并判断是否碰撞
        for dyn in self.dynamic_obstacles:
            if self.agent_pos == dyn['pos']: collision = True
            d = math.hypot(dyn['pos'][0] - self.agent_pos[0], dyn['pos'][1] - self.agent_pos[1])
            if d < min_obs_dist: min_obs_dist = d

        # ==================== 奖励函数计算 (保持原有逻辑，分离组件) ====================
        # 1. 基础惩罚：每走一步扣少量奖励，鼓励快速到达
        step_p = -0.1

        # 2. 前进奖励：向终点移动则奖励，远离则惩罚
        fwd_r = (old_dist - new_dist) * 2.0

        # 3. 方向奖励：动作方向与目标方向一致则奖励（余弦相似度）
        dir_r = 0.0
        if action != 0:  # 原地不动无方向奖励
                # 动作方向向量和目标方向向量
                vec_action = np.array([dx, dy])
                vec_goal = np.array([self.goal_pos[0] - old_pos[0], self.goal_pos[1] - old_pos[1]])
                # 计算向量模长
                norm_action = np.linalg.norm(vec_action)
                norm_goal = np.linalg.norm(vec_goal)
                # 模长非零时计算余弦相似度
                if norm_action > 0 and norm_goal > 0:
                    cos_sim = np.dot(vec_action, vec_goal) / (norm_action * norm_goal)
                    dir_r = cos_sim * 0.5

        # 4. 斥力惩罚：距离障碍物过近时扣奖励，鼓励保持安全距离
        rep_p = 0.0
        safe_dist = 2.0  # 安全距离阈值
        if min_obs_dist < safe_dist and min_obs_dist > 0:
                rep_p = -0.5 * ((1.0 / min_obs_dist) - (1.0 / safe_dist)) ** 2

            # 5. 后退与转折惩罚：避免智能体往返移动、频繁转向
        back_p = 0.0
        # 回到上一步位置且非原地不动，判定为后退，扣奖励
        if self.pos_history[0] == self.agent_pos and action != 0:
                back_p = -0.5

        turn_p = 0.0
        # 非原地不动且与上一步动作不同，判定为转向，少量扣奖励
        if action != 0 and self.last_action != 0 and action != self.last_action:
                turn_p = -0.2

            # 更新上一步动作
        self.last_action = action

        # 6. 终局事件奖励：碰撞/到达终点/步数用尽
        terminated = False  # 任务完成/失败标志
        truncated = False   # 步数用尽标志
        # 信息字典，记录任务状态
        info = {'is_success': False, 'collision': False}

        # 计算到终点的切比雪夫距离，用于判定是否到达终点
        dist_to_goal_chebyshev = max(abs(self.agent_pos[0] - self.goal_pos[0]),
                                         abs(self.agent_pos[1] - self.goal_pos[1]))

        event_r = 0.0
        # 碰撞：重大惩罚，终止任务
        if collision:
                event_r = -50.0
                terminated = True
                info['collision'] = True
            # 到达终点：重大奖励，终止任务
        elif dist_to_goal_chebyshev <= 1:
            event_r = 100.0
            terminated = True
            info['is_success'] = True
        # 步数用尽：截断任务，无额外奖励/惩罚
        elif self.current_step >= self.max_steps:
                truncated = True

        # ==================== 奖励归一化与裁剪 ====================
        # 将所有原始奖励组件求和，得到总奖励
        raw_reward = step_p + fwd_r + dir_r + rep_p + back_p + turn_p + event_r

        # 1. 线性缩放 (Scaling): 缩小一个数量级，使 Q 网络的输出目标保持在类似 [-5, 10] 的区间，加速神经网络收敛
        scale_factor = 10.0
        scaled_reward = raw_reward / scale_factor

        # 2. 软裁剪 (Clipping): 强行截断可能导致梯度爆炸的极端值 (上下限设为 [-10.0, 10.0])
        final_reward = float(np.clip(scaled_reward, -10.0, 10.0))

        # 人类可视化模式下执行渲染
        if self.render_mode == "human": self.render()

        # 返回step的四要素：新观测、处理后的奖励、任务完成标志、步数用尽标志、信息字典
        return self._get_obs(), final_reward, terminated, truncated, info

    def render(self):
        # 人类可视化模式，初始化窗口和时钟
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # 创建渲染窗口
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("D3QN 导航避障训练环境")
            # 创建时钟对象，控制帧率
            self.clock = pygame.time.Clock()

        # 窗口已初始化时执行绘制
        if self.window is not None:
            # 处理pygame事件，防止窗口卡死
            pygame.event.pump()
            # 窗口背景填充白色
            self.window.fill((255, 255, 255))

            # 绘制栅格线
            for x in range(self.size):
                for y in range(self.size):
                    rect = pygame.Rect(x * self.grid_size, y * self.grid_size, self.grid_size, self.grid_size)
                    pygame.draw.rect(self.window, (230, 230, 230), rect, 1)

            # 绘制终点判定范围（浅黄色背景，3x3区域）
            goal_range_rect = pygame.Rect((self.goal_pos[0] - 1) * self.grid_size,
                                          (self.goal_pos[1] - 1) * self.grid_size, self.grid_size * 3,
                                          self.grid_size * 3)
            pygame.draw.rect(self.window, (255, 250, 205), goal_range_rect)

            # 绘制起点（绿色方块）和终点（黄色方块）
            pygame.draw.rect(self.window, (0, 255, 0),
                             (self.start_pos[0] * self.grid_size, self.start_pos[1] * self.grid_size, self.grid_size,
                              self.grid_size))
            pygame.draw.rect(self.window, (255, 215, 0),
                             (self.goal_pos[0] * self.grid_size, self.goal_pos[1] * self.grid_size, self.grid_size,
                              self.grid_size))

            # 绘制静态障碍物（深灰色方块）
            for ox, oy in self.static_obstacles:
                pygame.draw.rect(self.window, (80, 80, 80),
                                 (ox * self.grid_size, oy * self.grid_size, self.grid_size, self.grid_size))

            # 绘制动态障碍物（红色方块）
            for dyn in self.dynamic_obstacles:
                dx, dy = dyn['pos']
                pygame.draw.rect(self.window, (255, 50, 50),
                                 (dx * self.grid_size, dy * self.grid_size, self.grid_size, self.grid_size))

            # 绘制智能体（蓝色圆圈，居中显示）
            center = (int(self.agent_pos[0] * self.grid_size + self.grid_size / 2),
                      int(self.agent_pos[1] * self.grid_size + self.grid_size / 2))
            pygame.draw.circle(self.window, (0, 0, 255), center, self.grid_size // 2 - 2)

            # 更新窗口显示
            pygame.display.flip()
            # 控制渲染帧率，与metadata中定义的一致
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        # 关闭渲染窗口，释放资源
        if self.window is not None:
            pygame.quit()
            self.window = None