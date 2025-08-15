import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import os
import time
from collections import deque

from config import device

WEIGHTS_PATH = "agent_model.pth"


# 辅助函数：将输入转换为CUDA张量
def to_cuda_tensor(x):
    """
    自动将列表、numpy数组转换为CUDA张量
    如果已是张量则直接移动到设备
    """
    if isinstance(x, list):
        return torch.tensor(np.array(x), dtype=torch.float32, device=device)
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=device)
    elif isinstance(x, torch.Tensor):
        return x.to(device).float()
    else:
        raise TypeError(f"不支持的类型: {type(x)}，请输入列表、numpy数组或PyTorch张量")


class GNNModule(nn.Module):
    """图神经网络模块，处理空间关系"""

    def __init__(self, input_dim, hidden_dim):
        super(GNNModule, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 聚合图特征
        return x


class LSTMModule(nn.Module):
    """LSTM模块，处理时序关系"""

    def __init__(self, input_dim, hidden_dim):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)  # 返回最后一个时间步的隐藏状态


class ActorCritic(nn.Module):
    """策略网络和价值网络"""

    def __init__(self, gnn_input_dim, lstm_input_dim, hidden_dim, action_dim=2):
        super(ActorCritic, self).__init__()
        self.gnn = GNNModule(gnn_input_dim, hidden_dim)
        self.lstm = LSTMModule(lstm_input_dim, hidden_dim)

        # 策略网络 - 输出动作分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出方向向量(-1,1)范围
        )

        # 价值网络 - 评估状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, gnn_data, lstm_seq):
        # 处理图数据
        gnn_out = self.gnn(gnn_data.x, gnn_data.edge_index, gnn_data.batch)

        # 处理时序数据
        lstm_out = self.lstm(lstm_seq)

        # 融合特征
        combined = torch.cat([gnn_out, lstm_out], dim=1)

        # 输出动作和价值
        action = self.actor(combined)
        value = self.critic(combined)

        return action, value


class AgentEnv:
    """智能体环境，用于模拟和交互（全程使用CUDA张量）"""

    def __init__(self, obs, points_coords, agents_pos, targets_coords):
        self.obs = obs  # 保持原始障碍物结构（Obstacle类实例列表）
        # 强制转换为CUDA张量
        self.points_coords = to_cuda_tensor(points_coords)
        self.agents_pos = to_cuda_tensor(agents_pos)
        self.targets_coords = to_cuda_tensor(targets_coords)
        self.num_agents = self.agents_pos.size(0)

        # 初始化历史位置（使用CUDA张量队列）
        self.agent_history = [deque(maxlen=5) for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.agent_history[i].append(self.agents_pos[i].clone())

    def reset(self, agents_pos=None):
        """重置环境状态"""
        if agents_pos is not None:
            self.agents_pos = to_cuda_tensor(agents_pos)
        for i in range(self.num_agents):
            self.agent_history[i].clear()
            self.agent_history[i].append(self.agents_pos[i].clone())
        return self.get_states()

    def get_states(self):
        """获取所有智能体的当前状态（返回CUDA张量）"""
        states = []
        for i in range(self.num_agents):
            states.append({
                "gnn_data": self.get_gnn_data(i),
                "lstm_seq": self.get_lstm_sequence(i)
            })
        return states

    def get_gnn_data(self, agent_idx):
        """构建GNN输入数据（全程使用CUDA张量）"""
        agent_pos = self.agents_pos[agent_idx]  # 已为CUDA张量
        target_pos = self.targets_coords[agent_idx]  # 已为CUDA张量

        # 节点特征: 智能体(0)、目标(1)、障碍物点(2)
        nodes = [torch.cat([agent_pos, torch.tensor([0.0], device=device)])]  # 智能体节点
        nodes.append(torch.cat([target_pos, torch.tensor([1.0], device=device)]))  # 目标节点

        # 添加障碍物点
        for ob in self.obs:
            start_point = torch.tensor(ob.start_point, device=device, dtype=torch.float32)
            end_point = torch.tensor(ob.end_point, device=device, dtype=torch.float32)

            nodes.append(torch.cat([start_point, torch.tensor([2.0], device=device)]))
            nodes.append(torch.cat([end_point, torch.tensor([2.0], device=device)]))

        # 转换为PyTorch Geometric格式
        x = torch.stack(nodes).to(device)  # 堆叠为CUDA张量
        edges = []
        for i in range(1, len(nodes)):
            edges.append([0, i])  # 智能体到其他节点
            edges.append([i, 0])  # 其他节点到智能体

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)  # 单个图

        return Data(x=x, edge_index=edge_index, batch=batch)

    def get_lstm_sequence(self, agent_idx):
        """获取LSTM输入序列（返回CUDA张量）"""
        history = list(self.agent_history[agent_idx])
        # 确保序列长度固定
        while len(history) < 5:
            history.insert(0, history[0].clone())  # 填充初始位置（使用张量克隆）

        # 直接堆叠为CUDA张量并添加batch维度
        return torch.stack(history).unsqueeze(0).to(device)

    def step(self, actions):
        """执行动作并返回奖励（全程使用CUDA张量）"""
        rewards = []
        old_positions = self.agents_pos.clone()  # 张量克隆

        # 更新所有智能体的位置（张量运算）
        actions_tensor = torch.stack(actions).to(device)  # 动作转为CUDA张量
        self.agents_pos += actions_tensor
        for i in range(self.num_agents):
            self.agent_history[i].append(self.agents_pos[i].clone())

        # 计算每个智能体的奖励（使用张量运算）
        for i in range(self.num_agents):
            old_pos = old_positions[i]
            new_pos = self.agents_pos[i]
            target_pos = self.targets_coords[i]

            # 靠近目标的奖励（张量计算距离）
            distance_old = torch.norm(old_pos - target_pos)
            distance_new = torch.norm(new_pos - target_pos)
            reward = 2.0 * (distance_old - distance_new)  # 权重提高

            # 远离障碍物的奖励
            min_obs_dist = self._get_min_obstacle_distance(i)
            if min_obs_dist < 0.3:  # 过近惩罚
                reward -= 1.0
            elif min_obs_dist < 0.8:  # 较近警告
                reward -= 0.3

            # 动作平滑性奖励
            if len(self.agent_history[i]) > 2:
                prev_dir = self.agent_history[i][-2] - self.agent_history[i][-3]
                curr_dir = actions[i]
                if torch.norm(prev_dir) > 1e-6 and torch.norm(curr_dir) > 1e-6:
                    cos_sim = torch.dot(prev_dir, curr_dir) / (torch.norm(prev_dir) * torch.norm(curr_dir))
                    reward += 0.5 * (cos_sim + 1)  # 方向相似性奖励

            rewards.append(reward.item())  # 转为标量存储

        return rewards, self.get_states()

    def _get_min_obstacle_distance(self, agent_idx):
        """计算智能体到最近障碍物的距离（张量版本）"""
        agent_pos = self.agents_pos[agent_idx]
        min_dist = torch.tensor(float('inf'), device=device)

        # 检查所有障碍物线段
        for ob in self.obs:
            start = torch.tensor(ob.start_point, device=device, dtype=torch.float32)
            end = torch.tensor(ob.end_point, device=device, dtype=torch.float32)

            # 计算点到线段的距离（张量运算）
            dist = self._point_to_segment_distance(agent_pos, start, end)
            min_dist = torch.min(min_dist, dist)

        return min_dist if min_dist != float('inf') else torch.tensor(2.0, device=device)

    @staticmethod
    def _point_to_segment_distance(point, start, end):
        """计算点到线段的最短距离（张量版本）"""
        seg_vec = end - start
        point_vec = point - start
        seg_len_sq = torch.dot(seg_vec, seg_vec)

        if seg_len_sq < 1e-6:  # 线段是一个点
            return torch.norm(point_vec)

        t = torch.clamp(torch.dot(point_vec, seg_vec) / seg_len_sq, 0.0, 1.0)
        projection = start + t * seg_vec
        return torch.norm(point - projection)


class PPO:
    """近端策略优化算法（全程使用CUDA张量）"""

    def __init__(self, gnn_input_dim, lstm_input_dim, hidden_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritic(gnn_input_dim, lstm_input_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(gnn_input_dim, lstm_input_dim, hidden_dim).to(device)

        # 尝试加载已有权重
        self.load_weights()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def load_weights(self):
        """加载权重文件"""
        if os.path.exists(WEIGHTS_PATH):
            try:
                self.policy.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
                print(f"成功加载权重文件: {WEIGHTS_PATH}")
                return True
            except Exception as e:
                print(f"加载权重文件失败: {e}，将使用新模型")
        return False

    def save_weights(self):
        """保存权重文件"""
        try:
            torch.save(self.policy.state_dict(), WEIGHTS_PATH)
            print(f"权重已保存至: {WEIGHTS_PATH}")
            return True
        except Exception as e:
            print(f"保存权重失败: {e}")
            return False

    def select_action(self, states):
        """选择动作（批量处理多个智能体）"""
        actions = []
        with torch.no_grad():
            for state in states:
                action, _ = self.policy_old(state["gnn_data"], state["lstm_seq"])
                actions.append(action)  # 保持CUDA张量
        return actions

    def update(self, memory):
        """更新策略网络（全程使用CUDA张量）"""
        # 计算累积奖励
        rewards = []
        discounted_reward = 0
        for _, _, _, _, reward, is_terminal in reversed(memory):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 标准化奖励（转为CUDA张量）
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 提取记忆中的数据
        old_states_gnn, old_states_lstm, old_actions, old_logprobs = [], [], [], []
        for m in memory:
            old_states_gnn.append(m[0])
            old_states_lstm.append(m[1])
            old_actions.append(m[2])
            old_logprobs.append(m[3])

        # 多次迭代更新
        for _ in range(5):
            # 计算当前策略的动作和价值
            logprobs = []
            state_values = []

            for i in range(len(old_states_gnn)):
                action, state_val = self.policy(old_states_gnn[i], old_states_lstm[i])
                state_values.append(state_val)

                # 计算动作概率的对数（使用CUDA张量运算）
                old_action_tensor = old_actions[i].to(device)
                logprob = -torch.sum(torch.square(action - old_action_tensor))
                logprobs.append(logprob)

            logprobs = torch.stack(logprobs).to(device)
            state_values = torch.stack(state_values).squeeze().to(device)

            # 计算优势
            advantages = rewards - state_values.detach()

            # 计算比率
            ratios = torch.exp(logprobs - torch.tensor(old_logprobs, device=device))

            # 计算PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)

            # 优化
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())


def initialize_agent(obs, points_coords, agents_pos, targets_coords, train_first=True, train_steps=100):
    """初始化智能体模型（包含训练或加载权重）"""
    # 转换输入为CUDA张量
    points_coords = to_cuda_tensor(points_coords)
    agents_pos = to_cuda_tensor(agents_pos)
    targets_coords = to_cuda_tensor(targets_coords)

    # 初始化PPO代理
    gnn_input_dim = 3  # x, y, 节点类型
    lstm_input_dim = 2  # x, y坐标
    hidden_dim = 64
    ppo = PPO(gnn_input_dim, lstm_input_dim, hidden_dim)

    # 如果需要先训练
    if train_first:
        # 初始化环境
        env = AgentEnv(obs, points_coords, agents_pos, targets_coords)
        num_agents = env.num_agents

        print(f"开始训练，共{train_steps}步，设备: {device}")
        start_time = time.time()
        total_rewards = []

        for step in range(train_steps):
            states = env.get_states()
            memory = []

            # 收集经验
            actions = ppo.select_action(states)
            rewards, next_states = env.step(actions)

            # 存储经验到记忆
            for i in range(num_agents):
                # 计算动作概率的对数
                with torch.no_grad():
                    policy_action, _ = ppo.policy(states[i]["gnn_data"], states[i]["lstm_seq"])
                    logprob = -torch.sum(torch.square(actions[i] - policy_action)).item()

                memory.append((
                    states[i]["gnn_data"],
                    states[i]["lstm_seq"],
                    actions[i],
                    logprob,
                    rewards[i],
                    False  # 非终端状态
                ))

            # 更新策略
            ppo.update(memory)

            # 记录奖励
            total_rewards.append(np.mean(rewards))

            # 打印训练进度
            if (step + 1) % 50 == 0:
                avg_reward = np.mean(total_rewards[-50:])
                elapsed = time.time() - start_time
                print(f"步骤: {step + 1}/{train_steps}, 平均奖励: {avg_reward:.4f}, 耗时: {elapsed:.2f}s")

        # 训练结束后保存权重
        ppo.save_weights()
        print(f"训练完成，总耗时: {time.time() - start_time:.2f}s")

    else:
        # 直接使用加载的模型
        if not os.path.exists(WEIGHTS_PATH):
            print("警告: 未找到权重文件，且未进行训练，将使用随机初始化模型")

    return ppo


def get_correction_vectors(ppo, obs, points_coords, agents_pos, targets_coords):
    """获取所有智能体的修正向量（需要已初始化的PPO模型）"""
    # 转换输入为CUDA张量
    points_coords = to_cuda_tensor(points_coords)
    agents_pos = to_cuda_tensor(agents_pos)
    targets_coords = to_cuda_tensor(targets_coords)

    # 初始化环境并获取当前状态
    env = AgentEnv(obs, points_coords, agents_pos, targets_coords)
    states = env.get_states()

    # 生成修正向量
    correction_vectors = ppo.select_action(states)

    return correction_vectors  # 返回CUDA张量列表