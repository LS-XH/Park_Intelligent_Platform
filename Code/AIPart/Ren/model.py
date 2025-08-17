import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import os
import time
from collections import deque

from Ren.config import device, WEIGHTS_PATH, TRAIN_STEP


# 辅助函数：将输入转换为CUDA张量（一次转换，多次使用）
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
    """策略网络和价值网络（支持批量输入）"""

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
        # 处理图数据（批量）
        gnn_out = self.gnn(gnn_data.x, gnn_data.edge_index, gnn_data.batch)

        # 处理时序数据（批量）
        lstm_out = self.lstm(lstm_seq)

        # 融合特征
        combined = torch.cat([gnn_out, lstm_out], dim=1)

        # 输出动作和价值
        action = self.actor(combined)
        value = self.critic(combined)

        return action, value


class AgentEnv:
    """智能体环境，优化后支持批量运算"""

    def __init__(self, obs, points_coords, agents_pos, targets_coords, max_obstacles=30):
        self.obs = obs  # 原始障碍物结构
        self.max_obstacles = max_obstacles  # 限制每个智能体最多考虑的障碍物数量

        # 一次性转换所有静态数据为CUDA张量（避免重复转换）
        self.points_coords = to_cuda_tensor(points_coords)
        self.agents_pos = to_cuda_tensor(agents_pos)
        self.targets_coords = to_cuda_tensor(targets_coords)
        self.num_agents = self.agents_pos.size(0)

        # 预处理障碍物：转换为张量并计算中心点（用于距离排序）
        self.obstacle_segments = []  # 存储所有障碍物线段 (start, end)
        self.obstacle_centers = []  # 存储所有障碍物中心点（用于快速距离计算）
        for ob in self.obs:
            start = torch.tensor(ob.start_point, device=device, dtype=torch.float32)
            end = torch.tensor(ob.end_point, device=device, dtype=torch.float32)
            self.obstacle_segments.append((start, end))
            self.obstacle_centers.append((start + end) / 2.0)  # 中心点

        # 转换为批量张量（形状：[num_obstacles, 2]）
        self.obstacle_centers = torch.stack(self.obstacle_centers) if self.obstacle_centers else None
        self.obstacle_segments = torch.stack(
            [torch.stack(s) for s in self.obstacle_segments]) if self.obstacle_segments else None

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
        """批量获取所有智能体的状态（返回列表）"""
        states = []
        for i in range(self.num_agents):
            states.append({
                "gnn_data": self.get_gnn_data(i),
                "lstm_seq": self.get_lstm_sequence(i)
            })
        return states

    def get_gnn_data(self, agent_idx):
        """构建GNN输入数据（只保留最近的N个障碍物）"""
        agent_pos = self.agents_pos[agent_idx]  # 形状: [2]
        target_pos = self.targets_coords[agent_idx]  # 形状: [2]

        # 节点特征: 智能体(0)、目标(1)
        nodes = [
            torch.cat([agent_pos, torch.tensor([0.0], device=device)]),  # 智能体节点
            torch.cat([target_pos, torch.tensor([1.0], device=device)])  # 目标节点
        ]

        # 只添加最近的N个障碍物（优化点1：限制障碍物数量）
        if self.obstacle_centers is not None and len(self.obstacle_centers) > 0:
            # 计算智能体到所有障碍物中心点的距离（批量运算）
            distances = torch.norm(self.obstacle_centers - agent_pos, dim=1)  # 形状: [num_obstacles]

            # 筛选最近的max_obstacles个障碍物
            _, top_k_indices = torch.topk(distances, k=min(self.max_obstacles, len(distances)), largest=False)

            # 添加选中的障碍物节点
            for idx in top_k_indices:
                start, end = self.obstacle_segments[idx]
                nodes.append(torch.cat([start, torch.tensor([2.0], device=device)]))
                nodes.append(torch.cat([end, torch.tensor([2.0], device=device)]))

        # 转换为PyTorch Geometric格式
        x = torch.stack(nodes).to(device)  # 节点特征
        edges = []
        for i in range(1, len(nodes)):
            edges.append([0, i])  # 智能体到其他节点
            edges.append([i, 0])  # 其他节点到智能体

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)  # 单个图的batch标记

        return Data(x=x, edge_index=edge_index, batch=batch)

    def get_lstm_sequence(self, agent_idx):
        """获取LSTM输入序列（批量兼容）"""
        history = list(self.agent_history[agent_idx])
        # 确保序列长度固定
        while len(history) < 5:
            history.insert(0, history[0].clone())  # 填充初始位置

        # 形状: [1, seq_len, input_dim]（兼容batch处理）
        return torch.stack(history).unsqueeze(0).to(device)

    def step(self, actions):
        """执行动作并返回奖励（批量运算优化）"""
        # 动作张量形状: [num_agents, 2]（确保无冗余维度）
        actions_tensor = torch.stack(actions).to(device)

        # 批量更新位置（优化点3：替代循环）
        old_positions = self.agents_pos.clone()
        self.agents_pos += actions_tensor

        # 批量更新历史位置
        for i in range(self.num_agents):
            self.agent_history[i].append(self.agents_pos[i].clone())

        # 批量计算奖励（优化点3：替代循环）
        rewards = self._calculate_batch_rewards(old_positions, actions_tensor)

        return rewards, self.get_states()

    def _calculate_batch_rewards(self, old_positions, actions_tensor):
        """批量计算所有智能体的奖励（替代循环）"""
        num_agents = self.num_agents

        # 1. 靠近目标的奖励（批量计算）
        distance_old = torch.norm(old_positions - self.targets_coords, dim=1)  # [num_agents]
        distance_new = torch.norm(self.agents_pos - self.targets_coords, dim=1)  # [num_agents]
        rewards = 2.0 * (distance_old - distance_new)  # 基础奖励

        # 2. 障碍物距离惩罚（批量计算）
        if self.obstacle_segments is not None and len(self.obstacle_segments) > 0:
            min_obs_distances = self._batch_get_min_obstacle_distance()  # [num_agents]
            rewards = torch.where(
                min_obs_distances < 0.3,  # 过近惩罚
                rewards - 1.0,
                torch.where(
                    min_obs_distances < 0.8,  # 较近警告
                    rewards - 0.3,
                    rewards  # 安全距离无惩罚
                )
            )

        # 3. 动作平滑性奖励（批量计算）
        if len(self.agent_history[0]) > 2:  # 确保有足够历史
            # 提取前一步和当前步的方向向量
            prev_dirs = torch.stack([self.agent_history[i][-2] - self.agent_history[i][-3]
                                     for i in range(num_agents)])  # [num_agents, 2]
            curr_dirs = actions_tensor  # [num_agents, 2]

            # 计算方向余弦相似度（批量）
            dir_norms = torch.norm(prev_dirs, dim=1) * torch.norm(curr_dirs, dim=1)  # [num_agents]
            valid_mask = dir_norms > 1e-6  # 排除零向量
            cos_sim = torch.zeros(num_agents, device=device)

            if valid_mask.any():
                dot_products = torch.sum(prev_dirs * curr_dirs, dim=1)  # [num_agents]
                cos_sim[valid_mask] = dot_products[valid_mask] / dir_norms[valid_mask]

            rewards += 0.5 * (cos_sim + 1)  # 平滑奖励

        return rewards.cpu().numpy().tolist()  # 转为CPU列表返回

    def _batch_get_min_obstacle_distance(self):
        """批量计算所有智能体到最近障碍物的距离（优化点3：替代循环）"""
        num_agents = self.agents_pos.size(0)
        num_obstacles = self.obstacle_segments.size(0) if self.obstacle_segments is not None else 0
        min_distances = torch.full((num_agents,), float('inf'), device=device)

        if num_obstacles == 0:
            return min_distances.fill_(2.0)

        # 批量计算每个智能体到所有障碍物的距离
        agents_pos = self.agents_pos.unsqueeze(1)  # [num_agents, 1, 2]
        start_points = self.obstacle_segments[:, 0].unsqueeze(0)  # [1, num_obstacles, 2]
        end_points = self.obstacle_segments[:, 1].unsqueeze(0)  # [1, num_obstacles, 2]

        # 计算点到线段的距离（批量向量运算）
        seg_vec = end_points - start_points  # [1, num_obstacles, 2]
        point_vec = agents_pos - start_points  # [num_agents, num_obstacles, 2]
        seg_len_sq = torch.sum(seg_vec ** 2, dim=2)  # [1, num_obstacles]
        seg_len_sq = torch.clamp(seg_len_sq, min=1e-6)  # 避免除以零

        t = torch.sum(point_vec * seg_vec, dim=2) / seg_len_sq  # [num_agents, num_obstacles]
        t = torch.clamp(t, 0.0, 1.0)  # [num_agents, num_obstacles]

        projection = start_points + t.unsqueeze(2) * seg_vec  # [num_agents, num_obstacles, 2]
        distances = torch.norm(agents_pos - projection, dim=2)  # [num_agents, num_obstacles]

        # 取每个智能体的最小距离
        min_distances = torch.min(distances, dim=1).values

        return torch.where(min_distances == float('inf'), torch.tensor(2.0, device=device), min_distances)


class PPO:
    """近端策略优化算法（批量运算优化）"""

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
        """选择动作（批量处理，去除冗余维度）"""
        with torch.no_grad():
            # 批量处理GNN数据
            batch_gnn = Batch.from_data_list([s["gnn_data"] for s in states])
            # 批量处理LSTM序列（形状: [num_agents, seq_len, input_dim]）
            batch_lstm = torch.cat([s["lstm_seq"] for s in states], dim=0)

            # 一次前向传播计算所有动作（优化点3：替代循环）
            actions, _ = self.policy_old(batch_gnn, batch_lstm)
            return [action.squeeze() for action in actions.split(1)]  # 确保形状正确

    def update(self, memory):
        """更新策略网络（批量运算优化）"""
        # 计算累积奖励
        rewards = []
        discounted_reward = 0
        for _, _, _, _, reward, is_terminal in reversed(memory):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 标准化奖励（CUDA张量）
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 提取记忆中的批量数据
        old_gnn_data = [m[0] for m in memory]
        old_lstm_seq = [m[1] for m in memory]
        old_actions = torch.stack([m[2] for m in memory]).to(device)
        old_logprobs = torch.tensor([m[3] for m in memory], device=device)

        # 多次迭代更新
        for _ in range(5):
            # 批量处理状态数据
            batch_gnn = Batch.from_data_list(old_gnn_data)
            batch_lstm = torch.cat(old_lstm_seq, dim=0)

            # 一次前向传播计算所有动作和价值（优化点3：替代循环）
            actions, state_values = self.policy(batch_gnn, batch_lstm)
            state_values = state_values.squeeze()

            # 批量计算动作概率的对数
            logprobs = -torch.sum(torch.square(actions - old_actions), dim=1)

            # 计算优势
            advantages = rewards - state_values.detach()

            # 计算比率和PPO损失
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)

            # 优化
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())


def initialize_agent(obs, points_coords, agents_pos, targets_coords, train_first=True, train_steps=TRAIN_STEP,
                     max_obstacles=10):
    """初始化智能体模型（包含训练或加载权重）"""
    # 转换输入为CUDA张量（一次转换）
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
        # 初始化环境（带障碍物数量限制）
        env = AgentEnv(obs, points_coords, agents_pos, targets_coords, max_obstacles=max_obstacles)
        num_agents = env.num_agents

        print(f"开始训练，共{train_steps}步，设备: {device}")
        start_time = time.time()
        total_rewards = []

        for step in range(train_steps):
            states = env.get_states()
            memory = []

            # 收集经验（批量动作生成）
            actions = ppo.select_action(states)
            rewards, next_states = env.step(actions)

            # 存储经验到记忆
            for i in range(num_agents):
                # 计算动作概率的对数（批量计算的一部分）
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
            # if (step + 1) % 10 == 0:  # 更频繁地打印进度，确认是否卡住
            avg_reward = np.mean(total_rewards[-10:])
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


def get_correction_vectors(ppo, obs, points_coords, agents_pos, targets_coords, max_obstacles=10):
    """获取所有智能体的修正向量（批量处理）"""
    # 转换输入为CUDA张量
    points_coords = to_cuda_tensor(points_coords)
    agents_pos = to_cuda_tensor(agents_pos)
    targets_coords = to_cuda_tensor(targets_coords)

    # 初始化环境并获取当前状态
    env = AgentEnv(obs, points_coords, agents_pos, targets_coords, max_obstacles=max_obstacles)
    states = env.get_states()

    # 生成修正向量（批量处理）
    correction_vectors = ppo.select_action(states)

    return correction_vectors  # 返回CUDA张量列表
