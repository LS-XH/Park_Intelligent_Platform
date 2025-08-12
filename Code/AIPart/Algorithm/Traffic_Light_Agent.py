import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
import pandas as pd
from typing import Tuple
import os
from tqdm import tqdm

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 天气系数
WEATHER_COEFFICIENTS = {
    'sunny': 1.0, 'rainy': 1.2, 'snowy': 1.5, 'foggy': 1.3, 'stormy': 1.6
}

# 动作定义（顺序调整：先增后减，强化增加动作优先级）
ACTION_LARGE_ADD = 20.0
ACTION_SMALL_ADD = 10.0
ACTION_NO_CHANGE = 0.0
ACTION_SMALL_REDUCE = -10.0
ACTION_LARGE_REDUCE = -20.0
ACTIONS = [ACTION_LARGE_ADD, ACTION_SMALL_ADD, ACTION_NO_CHANGE, ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]
ACTION_SIZE = len(ACTIONS)

# 状态参数
MAX_CAR_FLOW = 100.0
MAX_QUEUE_LENGTH = 100
MAX_PEOPLE_FLOW = 50.0
MAX_TRAFFIC_STATUS = 10.0
MAX_GREEN_DURATION = 120
MIN_GREEN_DURATION = 20

# 阈值（更宽松的高流量判断）
THRESHOLD_HIGH = 0.4  # 车流量>40%即判定为高流量
THRESHOLD_LONG_QUEUE = 0.4  # 排队长度>40%即判定为长排队


class TrafficLightEnv:
    def __init__(self):
        self.current_phase = 0
        self.green_duration = 60
        self.car_flow = 0.0
        self.queue_length = 0
        self.people_flow = 0.0
        self.weather = 'sunny'
        self.traffic_status = 0.0
        self.hour = 0
        self.prev_norm_queue = 0.0  # 用于计算排队改善
        self.reset()

    def reset(self) -> np.ndarray:
        peak_hours = list(range(7, 10)) + list(range(17, 20))
        non_peak_hours = list(range(0, 7)) + list(range(10, 17)) + list(range(20, 24))
        self.hour = random.choice(peak_hours * 2 + non_peak_hours)  # 增加高峰期样本比例
        is_peak = 1 if self.hour in peak_hours else 0

        # 强化高峰期拥堵特征
        self.car_flow = random.uniform(60, 100) if is_peak else random.uniform(0, 60)
        self.queue_length = random.randint(50, 100) if is_peak else random.randint(0, 40)
        self.people_flow = random.uniform(30, 50) if is_peak else random.uniform(0, 30)
        self.weather = random.choice(list(WEATHER_COEFFICIENTS.keys()))
        self.traffic_status = random.uniform(6, 10) if is_peak else random.uniform(0, 5)
        self.prev_norm_queue = self.queue_length / MAX_QUEUE_LENGTH
        self.green_duration = random.randint(40, 80)  # 初始绿灯时长更合理
        self.current_phase = random.choice([0, 1])

        return self._get_state()

    def set_state(self, car_flow: float, queue_length: int, people_flow: float,
                  weather: str, traffic_status: float, hour: int):
        """新增：设置环境状态用于推理"""
        self.car_flow = car_flow
        self.queue_length = queue_length
        self.people_flow = people_flow
        self.weather = weather
        self.traffic_status = traffic_status
        self.hour = hour
        self.prev_norm_queue = queue_length / MAX_QUEUE_LENGTH  # 初始化排队长度
        self.green_duration = 60  # 默认绿灯时长
        self.current_phase = 0  # 默认相位

    def _get_state(self) -> np.ndarray:
        norm_car = self .car_flow / MAX_CAR_FLOW
        norm_queue =    self.queue_length / MAX_QUEUE_LENGTH
        norm_people = self.people_flow / MAX_PEOPLE_FLOW
        weather_coeff = WEATHER_COEFFICIENTS[self.weather]
        norm_status = self.traffic_status / MAX_TRAFFIC_STATUS
        norm_green = self.green_duration / MAX_GREEN_DURATION
        is_peak = 1 if self.hour in list(range(7, 10)) + list(range(17, 20)) else 0

        return np.array([
            norm_car, norm_queue, norm_people, weather_coeff,
            norm_status, self.current_phase, norm_green, is_peak
        ])

    def _calculate_reward(self, action: float) -> float:
        is_peak = 1 if self.hour in list(range(7, 10)) + list(range(17, 20)) else 0
        norm_car = self.car_flow / MAX_CAR_FLOW
        norm_queue = self.queue_length / MAX_QUEUE_LENGTH

        # 基础奖励：拥堵越严重，基础奖励越低
        base_reward = -20 * norm_car - 20 * norm_queue  # 放大拥堵惩罚

        # 核心修正：高峰期强制策略
        if is_peak:
            # 高峰期高车流量或长排队时
            if norm_car > THRESHOLD_HIGH or norm_queue > THRESHOLD_LONG_QUEUE:
                # 增加绿灯：大幅奖励
                if action == ACTION_LARGE_ADD:
                    base_reward += 100  # 压倒性奖励
                elif action == ACTION_SMALL_ADD:
                    base_reward += 60
                # 减少绿灯：严厉惩罚
                elif action == ACTION_SMALL_REDUCE:
                    base_reward -= 150  # 惩罚远大于奖励
                elif action == ACTION_LARGE_REDUCE:
                    base_reward -= 200  # 最重惩罚
            # 高峰期低流量时
            else:
                if action in [ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]:
                    base_reward += 30
                elif action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                    base_reward -= 30

        # 非高峰期策略
        else:
            if norm_car < 0.3 and norm_queue < 0.3:
                # 低流量时减少绿灯奖励
                if action == ACTION_LARGE_REDUCE:
                    base_reward += 80
                elif action == ACTION_SMALL_REDUCE:
                    base_reward += 50
                # 低流量时增加绿灯惩罚
                elif action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                    base_reward -= 60
            # 非高峰期高流量
            elif norm_car > 0.6 or norm_queue > 0.6:
                if action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                    base_reward += 50
                elif action in [ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]:
                    base_reward -= 50

        # 排队改善奖励
        queue_improve = self.prev_norm_queue - norm_queue
        if queue_improve > 0:
            base_reward += 30 * queue_improve  # 放大改善奖励
        self.prev_norm_queue = norm_queue

        # 边界惩罚
        if (self.green_duration + action < MIN_GREEN_DURATION and action < 0) or \
                (self.green_duration + action > MAX_GREEN_DURATION and action > 0):
            base_reward -= 50

        return base_reward  # 不设上限，确保奖励差异足够大

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        chosen_action = ACTIONS[action]
        new_duration = self.green_duration + chosen_action
        new_duration = max(MIN_GREEN_DURATION, min(new_duration, MAX_GREEN_DURATION))

        # 计算奖励
        reward = self._calculate_reward(chosen_action)

        # 更新状态（强化动作影响）
        self.green_duration = new_duration
        # 增加绿灯显著减少排队，减少绿灯显著增加排队
        queue_change = -chosen_action * 0.5  # +20→-10米，-20→+10米
        self.queue_length = int(max(0, min(MAX_QUEUE_LENGTH, self.queue_length + queue_change + random.randint(-2, 2))))
        self.car_flow = max(0, min(MAX_CAR_FLOW, self.car_flow + random.uniform(-5, 5)))
        self.hour = (self.hour + 1) % 24
        self.traffic_status = max(0, min(MAX_TRAFFIC_STATUS,
                                         self.traffic_status - chosen_action * 0.3 + random.uniform(-0.5, 0.5)))

        # 相位切换
        done = self.green_duration >= MAX_GREEN_DURATION
        if done:
            self.current_phase = 1 - self.current_phase
            self.green_duration = MIN_GREEN_DURATION

        return self._get_state(), reward, done


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        # 高峰期特征权重（显著放大）
        self.peak_attention = nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        # 提取并放大高峰期特征（第7个特征）
        peak_feature = x[..., 7] * self.peak_attention
        x = torch.cat([x[..., :7], peak_feature.unsqueeze(-1)], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train()
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes=3000, batch_size=64):
    env = TrafficLightEnv()
    state_size = 8
    agent = DQNAgent(state_size, ACTION_SIZE)
    best_reward = -float('inf')
    rewards_history = []

    for e in tqdm(range(episodes), desc="训练进度"):
        state = env.reset()
        total_reward = 0
        for _ in range(200):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if done:
                break
        rewards_history.append(total_reward)

        if e % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"\nEpisode {e}, 平均奖励: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), "Traffic_Light_PeakFix.pth")

    print(f"最佳奖励: {best_reward:.2f}, 最终平均: {np.mean(rewards_history[-100:]):.2f}")
    return agent


def validate_agent(agent, num_tests=200):
    env = TrafficLightEnv()
    total_reward = 0
    peak_results = []
    agent.epsilon = 0.0

    for i in range(num_tests):
        state = env.reset()
        is_peak = 1 if env.hour in list(range(7, 10)) + list(range(17, 20)) else 0
        for _ in range(200):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        if is_peak:
            peak_results.append(ACTIONS[action])  # 记录高峰期最终动作

    # 统计高峰期动作分布
    peak_add_count = sum(1 for a in peak_results if a > 0)
    peak_reduce_count = sum(1 for a in peak_results if a < 0)
    print(f"\n高峰期动作统计: 增加{peak_add_count}次, 减少{peak_reduce_count}次")
    print(f"整体平均奖励: {total_reward / num_tests:.2f}")
    return agent


def Traffic_Control(car_flow, queue_length, people_flow, weather, traffic_status, hour):
    env = TrafficLightEnv()
    """
    红绿灯控制函数，返回‘绿灯’需要增加（正数）或减少（负数）的秒数，相应的红灯就需要减短多少
    """
    try:
        model = DQN(8, ACTION_SIZE).to(device)
        model.load_state_dict(torch.load("Traffic_Light_PeakFix.pth", map_location=device))
        model.eval()
    except:
        print("使用默认策略")
        is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        if is_peak:
            return 20.0 if (car_flow / 100 > 0.4 or queue_length / 100 > 0.4) else -10.0
        else:
            return -20.0 if (car_flow / 100 < 0.3 and queue_length / 100 < 0.3) else 10.0

    env.set_state(car_flow, queue_length, people_flow, weather, traffic_status, hour)
    state = env._get_state()
    print("\n状态详情:")
    print(f"标准化车流量: {state[0]:.2f} (>0.4为高)")
    print(f"标准化排队: {state[1]:.2f} (>0.4为长)")
    print(f"是否高峰期: {state[7]}")

    with torch.no_grad():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_values = model(state_tensor)[0]
        print("动作Q值:")
        for i, a in enumerate(ACTIONS):
            print(f"动作 {a:2.0f}秒: {q_values[i].item():.2f}")
        action_idx = torch.argmax(q_values).item()

    return ACTIONS[action_idx]

# 使用范例
# if __name__ == "__main__":

#     # 高峰期测试（80%车流量，60%排队）

#     peak_result = Traffic_Control(
#         car_flow=80, queue_length=60, people_flow=30,
#         weather='rainy', traffic_status=7.5, hour=8
#     )
#     print(f"\n高峰期控制结果: {peak_result}秒 (预期+20或+10)")
#     print("花费时间：",time.time() - t)
#     tt = time.time()
#     low_result = Traffic_Control(
#         car_flow=25, queue_length=25, people_flow=12,
#         weather='sunny', traffic_status=3, hour=14
#     )
#     print(f"\n低峰期控制结果: {low_result}秒 (预期-20或-10)")
#     print("花费时间：", time.time() - tt)