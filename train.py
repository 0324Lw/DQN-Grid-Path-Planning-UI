import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

from env import GridNavEnv
from plotdata import DataPlotter


# 定义Dueling DQN网络模型
class DuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNet, self).__init__()
        # 共享特征提取层：将输入状态转换为高维特征，LayerNorm提升训练稳定性
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),  # 线性层：输入维度→128维隐藏层
            nn.LayerNorm(128),  # 层归一化：缓解梯度消失/爆炸，适配动态环境
            nn.ReLU(),  # ReLU激活函数：引入非线性
            nn.Linear(128, 128),  # 第二层线性层：进一步提取特征
            nn.LayerNorm(128),  # 再次归一化
            nn.ReLU()  # 再次激活
        )

        # 优势函数分支（Advantage）：评估每个动作相对于其他动作的好坏
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # 输出维度=动作数，对应每个动作的优势值
        )

        # 状态价值分支（Value）：评估当前状态本身的价值，与具体动作无关
        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出维度=1，仅表示状态固有价值
        )

        self._orthogonal_init()  # 调用正交初始化函数，优化参数初始值

    def _orthogonal_init(self):
        """正交初始化：强化学习专用参数初始化技巧
        作用：避免梯度消失/爆炸，提升收敛速度，适配控制类任务
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):  # 遍历所有线性层
                # 正交初始化权重，增益√2（ReLU激活的最优增益）
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)  # 偏置项初始化为0

    def forward(self, x):
        """前向传播：计算每个动作的Q值
        核心公式：Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        减去优势值均值：消除“状态价值+优势值”的尺度模糊问题
        """
        features = self.feature_layer(x)  # 提取共享特征
        advantage = self.advantage_layer(features)  # 计算优势值
        value = self.value_layer(features)  # 计算状态价值
        # 聚合得到最终Q值（每个动作对应一个Q值）
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# 定义优先经验回放缓冲区（PER）
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity  # 缓冲区最大容量（经验总数上限）
        self.alpha = alpha  # 优先级指数：0=均匀采样，1=完全按优先级采样
        self.buffer = []  # 经验存储列表，每个元素格式：(state, action, reward, next_state, done)
        self.pos = 0  # 循环存储位置指针（满了之后覆盖最旧经验）
        # 优先级数组：存储每个经验的优先级值
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """添加一条经验到缓冲区，并分配初始最大优先级
        初始最大优先级：确保新经验至少被采样一次（避免新经验优先级太低不被学习）
        """
        # 获取当前缓冲区的最大优先级（无经验时设为1.0）
        max_prio = self.priorities.max() if self.buffer else 1.0

        # 缓冲区未满：直接追加；已满：循环覆盖最旧经验
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio  # 为新经验分配最大优先级
        self.pos = (self.pos + 1) % self.capacity  # 更新存储位置（循环）

    def sample(self, batch_size, beta=0.4):
        """按优先级采样一批经验，并计算重要性采样（IS）权重
        重要性采样权重：修正优先级采样带来的分布偏差，避免训练震荡
        :param batch_size: 采样批次大小（每次训练的经验数量）
        :param beta: 重要性采样系数：随训练进程从0.4逐步提升到1.0
        :return: 采样经验批次、经验索引、IS权重
        """
        # 取有效优先级（缓冲区未满时，仅取已存储经验的优先级）
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # 将优先级转换为采样概率（alpha控制优先级影响程度）
        probs = prios ** self.alpha
        probs /= probs.sum()  # 归一化为概率分布

        # 按概率分布采样batch_size个经验的索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]  # 根据索引获取经验

        # 计算重要性采样权重（修正优先级采样的偏差）
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)  # 权重核心公式
        weights /= weights.max()  # 权重归一化：避免数值过大导致训练不稳定
        weights = np.array(weights, dtype=np.float32)

        # 解包经验：按维度拆分state/action/reward/next_state/done
        states = torch.FloatTensor(np.array([sample[0] for sample in samples]))
        actions = torch.LongTensor(np.array([sample[1] for sample in samples]))
        rewards = torch.FloatTensor(np.array([sample[2] for sample in samples]))
        next_states = torch.FloatTensor(np.array([sample[3] for sample in samples]))
        dones = torch.FloatTensor(np.array([sample[4] for sample in samples]))

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """更新采样经验的优先级（根据TD误差调整）
        :param indices: 采样经验的索引
        :param priorities: 新的优先级值（通常为TD误差的绝对值+小常数）
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """返回当前缓冲区的经验数量"""
        return len(self.buffer)


# 定义Dueling DQN智能体类（整合网络、经验回放、训练逻辑）
class D3QNAgent:
    def __init__(self, input_dim, output_dim, config):
        self.config = config
        self.action_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络与目标网络
        self.policy_net = DuelingNet(input_dim, output_dim).to(self.device)
        self.target_net = DuelingNet(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器（使用固定的学习率，配合权重衰减防止过拟合）
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=config['learning_rate'],
                                    weight_decay=1e-5)

        # 经验回放池
        self.memory = PrioritizedReplayBuffer(config['buffer_size'], alpha=config['per_alpha'])

        self.epsilon = config['epsilon_start']
        self.beta = config['per_beta_start']

    def select_action(self, state):
        """选择动作：ε-贪心策略"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def update(self):
        """核心更新逻辑：包含 Double DQN 和 PER"""
        if len(self.memory) < self.config['batch_size']:
            return 0.0

        # 从经验池采样 (适配你 current buffer 的返回格式)
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.config['batch_size'],
                                                                                            self.beta)

        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Double DQN: 主网络选动作，目标网络算 Q 值
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.config['gamma'] * next_q_values

        current_q = self.policy_net(states).gather(1, actions)

        # 计算 TD 误差用于更新 PER
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors + 1e-6)

        weights = weights / torch.max(weights)
        loss = (weights * nn.SmoothL1Loss(reduction='none')(current_q, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪技巧：防止动态环境下的梯度爆炸
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        return loss.item()

    def soft_update(self):
        """软更新 Target 网络"""
        tau = self.config['tau']
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def decay_parameters(self, total_steps):
        """按全局步数衰减探索率和重要性采样系数"""
        decay_rate = (self.config['epsilon_start'] - self.config['epsilon_end']) / self.config['epsilon_decay_steps']
        self.epsilon = max(self.config['epsilon_end'], self.config['epsilon_start'] - total_steps * decay_rate)

        beta_increase = (1.0 - self.config['per_beta_start']) / self.config['max_train_steps']
        self.beta = min(1.0, self.config['per_beta_start'] + total_steps * beta_increase)


# 配置参数类
class Config:
    def __init__(self):
        self.env_name = "GridNavEnv"  # 环境名称
        self.device = "cuda"  # 设备（cuda/cpu）
        self.lr = 1e-3  # 学习率
        self.gamma = 0.99  # 折扣因子（越接近1，越重视未来奖励）
        self.batch_size = 64  # 批次大小
        self.memory_capacity = 100000  # 经验缓冲区容量
        self.target_update = 100  # 目标网络更新频率
        self.epsilon_start = 1.0  # 初始探索概率
        self.epsilon_end = 0.05  # 最终探索概率
        self.epsilon_decay = 500  # 探索概率衰减步数
        self.alpha = 0.6  # PER优先级指数
        self.beta_start = 0.4  # 重要性采样初始系数
        self.beta_frames = 100000  # beta提升到1.0的总步数
        self.max_episodes = 1000  # 最大训练回合数
        self.max_steps = 200  # 每回合最大步数


# 核心训练函数 (同步了按步数训练和 10 维数据收集)
def train(config, env, agent):
    history = {
        'Reward': [], 'Steps': [], 'Loss': [],
        'Success_Rate': [], 'Collision_Rate': [], 'Epsilon': [],
        'Path_Smoothness': [], 'Final_Distance': [], 'Avg_Step_Reward': [], 'Turn_Count': []
    }

    window_rewards, window_steps, window_losses = [], [], []
    success_count, collision_count = 0, 0
    total_steps = 0
    episodes = 0

    print(f"🚀 开始 D3QN 训练 (目标步数: {config['max_train_steps']})")

    while total_steps < config['max_train_steps']:
        state, _ = env.reset()
        ep_reward, ep_loss, ep_steps, ep_turns = 0, 0, 0, 0
        last_action = 0
        episodes += 1

        for step in range(config['max_ep_steps']):
            total_steps += 1
            ep_steps += 1

            action = agent.select_action(state)
            if action != 0 and last_action != 0 and action != last_action:
                ep_turns += 1
            last_action = action

            next_state, reward, terminated, truncated, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, terminated)

            # 每 4 步更新 1 次网络
            if total_steps % 4 == 0:
                loss = agent.update()
                agent.soft_update()
                ep_loss += loss

            agent.decay_parameters(total_steps)
            state = next_state
            ep_reward += reward

            if terminated or truncated:
                if info.get('is_success'): success_count += 1
                if info.get('collision'): collision_count += 1
                break

        smoothness = 1.0 - (ep_turns / ep_steps) if ep_steps > 0 else 1.0
        final_dist = max(abs(env.agent_pos[0] - env.goal_pos[0]), abs(env.agent_pos[1] - env.goal_pos[1]))
        avg_step_rew = ep_reward / ep_steps if ep_steps > 0 else 0
        update_times = ep_steps // 4

        history['Reward'].append(ep_reward)
        history['Steps'].append(ep_steps)
        history['Loss'].append(ep_loss / update_times if update_times > 0 else 0)
        history['Epsilon'].append(agent.epsilon)
        history['Path_Smoothness'].append(smoothness)
        history['Final_Distance'].append(final_dist)
        history['Avg_Step_Reward'].append(avg_step_rew)
        history['Turn_Count'].append(ep_turns)

        window_rewards.append(ep_reward)
        window_steps.append(ep_steps)
        window_losses.append(ep_loss / update_times if update_times > 0 else 0)

        if episodes % 10 == 0:
            avg_reward = np.mean(window_rewards)
            avg_steps = np.mean(window_steps)
            avg_loss = np.mean(window_losses)
            sr = success_count / 10.0
            cr = collision_count / 10.0
            history['Success_Rate'].extend([sr] * 10)
            history['Collision_Rate'].extend([cr] * 10)

            lr = agent.optimizer.param_groups[0]['lr']
            print(f"Steps {total_steps:6d} | Ep {episodes:4d} | AvgRew: {avg_reward:6.1f} | "
                  f"Succ: {sr * 100:3.0f}% | Smooth: {smoothness:.2f} | Dist: {final_dist}")

            window_rewards, window_steps, window_losses = [], [], []
            success_count, collision_count = 0, 0

    while len(history['Success_Rate']) < episodes:
        history['Success_Rate'].append(history['Success_Rate'][-1] if history['Success_Rate'] else 0)
        history['Collision_Rate'].append(history['Collision_Rate'][-1] if history['Collision_Rate'] else 0)

    print("📊 训练结束，正在保存数据并生成训练曲线...")
    plotter = DataPlotter()
    plotter.save_and_plot(history, window_size=50)

    agent.save(f"d3qn_best_model.pth")
    print("✅ 模型和图表已保存！")


if __name__ == "__main__":
    # 使用与 UI 一致的字典配置
    config = {
        'max_train_steps': 100000,
        'max_ep_steps': 300,
        'epsilon_decay_steps': 30000,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 128,
        'buffer_size': 50000,
        'tau': 0.001,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'per_alpha': 0.6,
        'per_beta_start': 0.5
    }

    env = GridNavEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = D3QNAgent(state_dim, action_dim, config)

    train(config, env, agent)