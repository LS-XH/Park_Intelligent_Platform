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

# 新增：状态区间阈值（基于最大值比例）
LOWEST_THRESHOLD = 1/3  # 极低：<0.333
LOW_THRESHOLD = 0.4     # 偏低：0.333~0.4
MID_LOW = 0.4           # 居中：0.4~0.6
MID_HIGH = 0.6
HIGH_LOW = 0.6          # 偏高：0.6~0.666
HIGH_HIGH = 2/3         # 极高：>0.666
HIGHEST_THRESHOLD = 2/3

# 阈值
THRESHOLD_HIGH = 0.6  # 车流量>60%即判定为高流量
THRESHOLD_LONG_QUEUE = 0.6  # 排队长度>60%即判定为长排队


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
        # 提取基础参数（所有场景共用）
        # 时间特征
        is_peak = 1 if self.hour in list(range(7, 10)) + list(range(17, 20)) else 0
        # 标准化状态特征
        norm_car = self.car_flow / MAX_CAR_FLOW
        norm_queue = self.queue_length / MAX_QUEUE_LENGTH
        norm_people = self.people_flow / MAX_PEOPLE_FLOW
        weather_coeff = WEATHER_COEFFICIENTS[self.weather]
        norm_status = self.traffic_status / MAX_TRAFFIC_STATUS
        norm_green = self.green_duration / MAX_GREEN_DURATION
        # 排队改善计算
        queue_improve = self.prev_norm_queue - norm_queue

        # 基础奖励（核心惩罚项）
        base_reward = -20 * norm_car - 20 * norm_queue  # 车流量和排队越长，基础惩罚越高

        #  分场景核心策略奖励（高峰/非高峰）
        if is_peak:
            # 高峰期策略：优先处理拥堵
            if norm_car > THRESHOLD_HIGH or norm_queue > THRESHOLD_LONG_QUEUE:
                # 拥堵时：鼓励增加绿灯，惩罚减少
                if action == ACTION_LARGE_ADD:
                    base_reward += 60
                elif action == ACTION_SMALL_ADD:
                    base_reward += 30
                elif action == ACTION_SMALL_REDUCE:
                    base_reward -= 80
                elif action == ACTION_LARGE_REDUCE:
                    base_reward -= 100
            else:
                # 非拥堵时：鼓励减少绿灯，惩罚增加
                if action in [ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]:
                    base_reward += 30
                elif action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                    base_reward -= 30

        else:
            # 非高峰期策略：按综合状态（车流量+排队）分区间处理
            combined_state = max(norm_car, norm_queue)

            if combined_state < LOWEST_THRESHOLD:
                # 极低状态：鼓励大幅减少绿灯
                if action == ACTION_LARGE_REDUCE:
                    base_reward += 100
                elif action == ACTION_SMALL_REDUCE:
                    base_reward += 50
                elif action == ACTION_NO_CHANGE:
                    base_reward -= 30
                else:  # 增加动作惩罚
                    base_reward -= 80

            elif LOWEST_THRESHOLD <= combined_state < LOW_THRESHOLD:
                # 偏低状态：鼓励小幅减少绿灯
                if action == ACTION_SMALL_REDUCE:
                    base_reward += 80
                elif action == ACTION_LARGE_REDUCE:
                    pass
                elif action == ACTION_NO_CHANGE:
                    base_reward -= 20
                else:  # 增加动作惩罚
                    base_reward -= 60

            elif MID_LOW <= combined_state <= MID_HIGH:
                # 居中状态：优先不调整，其次小幅调整
                if action == ACTION_NO_CHANGE:
                    base_reward += 80
                elif action in [ACTION_SMALL_ADD, ACTION_SMALL_REDUCE]:
                    pass
                else:  # 大幅调整惩罚
                    base_reward -= 80

            elif HIGH_LOW <= combined_state < HIGH_HIGH:
                # 偏高状态：鼓励小幅增加绿灯
                if action == ACTION_SMALL_ADD:
                    base_reward += 80
                elif action == ACTION_LARGE_ADD:
                    pass
                elif action == ACTION_NO_CHANGE:
                    base_reward -= 20
                else:  # 减少动作惩罚
                    base_reward -= 50

            else:  # combined_state >= HIGHEST_THRESHOLD
                # 极高状态：鼓励大幅增加绿灯
                if action == ACTION_LARGE_ADD:
                    base_reward += 80
                elif action == ACTION_SMALL_ADD:
                    pass
                elif action == ACTION_NO_CHANGE:
                    base_reward -= 30
                else:  # 减少动作惩罚
                    base_reward -= 80

        # 4. 通用奖励因素（所有场景叠加）
        # 4.1 排队改善奖励（动作对排队的影响）
        if queue_improve > 0:
            # 排队减少：增加奖励（正向动作奖励更高）
            base_reward += 20 * queue_improve if action >= 0 else 10 * queue_improve
        elif queue_improve < 0:
            # 排队增加：减少奖励
            base_reward += 15 * queue_improve
        self.prev_norm_queue = norm_queue  # 更新历史排队

        # 4.2 人流量影响（人流量大时抑制绿灯过长）
        if norm_people > 0.5:
            if action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                base_reward -= 80

        # 4.3 天气影响（恶劣天气鼓励延长绿灯）
        if weather_coeff > 1.2:
            if action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                base_reward += 50

        # 4.4 交通状态影响（状态差时避免缩短绿灯）
        if norm_status > 0.6:
            if action in [ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]:
                base_reward -= 40

        # 4.5 绿灯时长边界惩罚（避免过短或过长）
        if norm_green < 0.2:  # 绿灯过短（<24秒）
            if action in [ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]:
                base_reward -= 100
        elif norm_green > 0.8:  # 绿灯过长（>96秒）
            if action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                base_reward -= 100

        # 4.6 动作越界惩罚（超出最小/最大绿灯时长）
        if (self.green_duration + action < MIN_GREEN_DURATION and action < 0) or \
                (self.green_duration + action > MAX_GREEN_DURATION and action > 0):
            base_reward -= 100

        return base_reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        chosen_action = ACTIONS[action]
        new_duration = self.green_duration + chosen_action
        new_duration = max(MIN_GREEN_DURATION, min(new_duration, MAX_GREEN_DURATION))

        # 计算奖励
        reward = self._calculate_reward(chosen_action)

        # 更新状态（强化动作影响）
        self.green_duration = new_duration
        # 增加绿灯显著减少排队，减少绿灯显著增加排队
        self.car_flow = max(0, min(MAX_CAR_FLOW, self.car_flow + random.uniform(-5, 5)))
        self.hour = (self.hour + 1) % 24
        self.traffic_status = max(0, min(MAX_TRAFFIC_STATUS,
                                         self.traffic_status - chosen_action * 0.3 + random.uniform(-0.5, 0.5)))
        # 在step函数中修改排队长度更新逻辑：
        queue_change = -chosen_action * 0.5  # 原逻辑保留
        # 若动作是0，减少随机波动的正向影响（让排队更难随机减少）
        if chosen_action == 0:
            # 随机波动偏向增加（-1~2 → -2~1），削弱无动作时的排队改善
            random_fluct = random.randint(-1, 1)
        else:
            random_fluct = random.randint(-2, 2)
        self.queue_length = int(max(0, min(MAX_QUEUE_LENGTH, self.queue_length + queue_change + random_fluct)))

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
        self.memory = deque(maxlen=30000)  # 增大经验池，存储更多状态样本
        self.gamma = 0.97  # 稍微提高折扣因子，更关注长期奖励
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997  # 减缓探索衰减，延长学习时间
        self.learning_rate = 0.0005  # 降低学习率，提高稳定性

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


def train_agent(episodes=20000, batch_size=128):
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
            torch.save(agent.model.state_dict(), "Traffic_Light5.pth")

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
    """
    params: car_flow: float （需放缩）
    params: queue_length: float 车流量（速度为零的车辆 * 4(假设一辆车的长为4m)）
    params: people_flow: float 人流量（需放缩）
    params: weather: str 天气状况
    params: traffic_status: int 交通状态，通过上面参数计算得到（设计公式自己计算，用来给我们微调用的）
    return: 返回一个绿灯调整的时间，红灯时间 = 周期 - 绿灯时间
    """
    env = TrafficLightEnv()
    try:
        model = DQN(8, ACTION_SIZE).to(device)
        model.load_state_dict(torch.load("Traffic_Light5.pth", map_location=device))
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
    print(f"标准化车流量: {state[0]:.2f} ")
    print(f"标准化排队: {state[1]:.2f}")
    print(f"是否高峰期: {state[7]}")

    with torch.no_grad():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_values = model(state_tensor)[0]
        print("动作Q值:")
        for i, a in enumerate(ACTIONS):
            print(f"动作 {a:2.0f}秒: {q_values[i].item():.2f}")
        action_idx = torch.argmax(q_values).item()

    return ACTIONS[action_idx]


