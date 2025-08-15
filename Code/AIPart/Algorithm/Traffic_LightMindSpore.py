import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops, Parameter
from collections import deque
import random
from typing import Tuple
from tqdm import tqdm
# from Code.AIPart.graph.graph import Graph

# 天气系数
WEATHER_COEFFICIENTS = {
    'sunny': 1.0, 'rainy': 1.2, 'snowy': 1.5, 'foggy': 1.3, 'stormy': 1.6
}

# 动作定义
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

# 状态区间阈值
LOWEST_THRESHOLD = 1/3
LOW_THRESHOLD = 0.4
MID_LOW = 0.4
MID_HIGH = 0.6
HIGH_LOW = 0.6
HIGH_HIGH = 2/3
HIGHEST_THRESHOLD = 2/3

# 阈值
THRESHOLD_HIGH = 0.6
THRESHOLD_LONG_QUEUE = 0.6


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
        self.prev_norm_queue = 0.0
        self.reset()

    def reset(self) -> np.ndarray:
        peak_hours = list(range(7, 10)) + list(range(17, 20))
        non_peak_hours = list(range(0, 7)) + list(range(10, 17)) + list(range(20, 24))
        # 平衡高峰期与非高峰期样本比例
        self.hour = random.choice(peak_hours + non_peak_hours)
        is_peak = 1 if self.hour in peak_hours else 0

        # 降低高峰期初始拥堵程度
        self.car_flow = random.uniform(40, 80) if is_peak else random.uniform(0, 60)
        self.queue_length = random.randint(30, 70) if is_peak else random.randint(0, 40)
        self.people_flow = random.uniform(30, 50) if is_peak else random.uniform(0, 30)
        self.weather = random.choice(list(WEATHER_COEFFICIENTS.keys()))
        self.traffic_status = random.uniform(6, 10) if is_peak else random.uniform(0, 5)
        self.prev_norm_queue = self.queue_length / MAX_QUEUE_LENGTH
        self.green_duration = random.randint(40, 80)
        self.current_phase = random.choice([0, 1])

        return self._get_state()

    def set_state(self, car_flow: float, queue_length: int, people_flow: float,
                  weather: str, traffic_status: float, hour: int):
        self.car_flow = car_flow
        self.queue_length = queue_length
        self.people_flow = people_flow
        self.weather = weather
        self.traffic_status = traffic_status
        self.hour = hour
        self.prev_norm_queue = queue_length / MAX_QUEUE_LENGTH
        self.green_duration = 60
        self.current_phase = 0

    def _get_state(self) -> np.ndarray:
        norm_car = self.car_flow / MAX_CAR_FLOW
        norm_queue = self.queue_length / MAX_QUEUE_LENGTH
        norm_people = self.people_flow / MAX_PEOPLE_FLOW
        max_weather = max(WEATHER_COEFFICIENTS.values())  # 1.6
        weather_coeff = WEATHER_COEFFICIENTS[self.weather] / max_weather
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
        norm_people = self.people_flow / MAX_PEOPLE_FLOW
        weather_coeff = WEATHER_COEFFICIENTS[self.weather]
        norm_status = self.traffic_status / MAX_TRAFFIC_STATUS
        norm_green = self.green_duration / MAX_GREEN_DURATION
        queue_improve = self.prev_norm_queue - norm_queue  # 排队改善量

        base_reward = 0
        # 统一用"车流量和排队长度的最大值"作为状态划分依据
        combined_state = max(norm_car, norm_queue)

        # 1. 核心奖励：按阈值区间划分，每个区间有明确的最优动作和梯度
        # 区间1：[0, 1/3) → 最优动作：大幅减少（LARGE_REDUCE）
        if combined_state < LOWEST_THRESHOLD:  # LOWEST_THRESHOLD=1/3≈0.333
            # 奖励梯度：大幅减少 > 小幅减少 > 无动作 > 小幅增加 > 大幅增加
            action_rewards = {
                ACTION_LARGE_REDUCE: 200,  # 最优动作
                ACTION_SMALL_REDUCE: 120,  # 次优（相邻动作）
                ACTION_NO_CHANGE: 30,  # 中性
                ACTION_SMALL_ADD: -50,  # 惩罚（反向动作）
                ACTION_LARGE_ADD: -120  # 重罚（强反向）
            }

        # 区间2：[1/3, 0.4) → 最优动作：小幅减少（SMALL_REDUCE）
        elif LOWEST_THRESHOLD <= combined_state < LOW_THRESHOLD:  # 1/3≈0.333 到 0.4
            # 奖励梯度：小幅减少 > 大幅减少 > 无动作 > 小幅增加 > 大幅增加
            action_rewards = {
                ACTION_SMALL_REDUCE: 200,  # 最优动作
                ACTION_LARGE_REDUCE: 100,  # 次优（相邻动作）
                ACTION_NO_CHANGE: 20,  # 中性
                ACTION_SMALL_ADD: -60,  # 惩罚（反向动作）
                ACTION_LARGE_ADD: -130  # 重罚（强反向）
            }

        # 区间3：[0.4, 2/3) → 最优动作：小幅增加（SMALL_ADD）
        elif MID_LOW <= combined_state < HIGH_HIGH:  # 0.4 到 2/3≈0.666
            # 奖励梯度：小幅增加 > 无动作 > 大幅增加 > 小幅减少 > 大幅减少
            action_rewards = {
                ACTION_SMALL_ADD: 200,  # 最优动作
                ACTION_NO_CHANGE: 50,  # 次优（中性）
                ACTION_LARGE_ADD: 100,  # 次优（相邻动作）
                ACTION_SMALL_REDUCE: -60,  # 惩罚（反向动作）
                ACTION_LARGE_REDUCE: -130  # 重罚（强反向）
            }

        # 区间4：[2/3, 1] → 最优动作：大幅增加（LARGE_ADD）
        else:  # 2/3≈0.666 到 1.0
            # 奖励梯度：大幅增加 > 小幅增加 > 无动作 > 小幅减少 > 大幅减少
            action_rewards = {
                ACTION_LARGE_ADD: 200,  # 最优动作
                ACTION_SMALL_ADD: 120,  # 次优（相邻动作）
                ACTION_NO_CHANGE: 30,  # 中性
                ACTION_SMALL_REDUCE: -50,  # 惩罚（反向动作）
                ACTION_LARGE_REDUCE: -120  # 重罚（强反向）
            }

        # 高峰期对高区间（拥堵）的最优动作额外强化
        if is_peak and combined_state >= HIGH_HIGH:  # 高峰期+拥堵（2/3以上）
            if action == ACTION_LARGE_ADD:
                action_rewards[action] += 80  # 额外+80，拉开与其他动作差距
            elif action == ACTION_SMALL_ADD:
                action_rewards[action] += 30  # 小幅增加也适当强化

        base_reward += action_rewards[action]

        # 2. 排队改善奖励：与动作方向匹配，强化正确动作的效果
        if queue_improve > 0:
            # 动作与改善方向一致时（减少→拥堵减轻，增加→拥堵减轻）奖励更高
            if (combined_state < MID_LOW and action < 0) or (combined_state >= MID_LOW and action > 0):
                improve_bonus = 100 * queue_improve  # 匹配区间的动作奖励更高
            else:
                improve_bonus = 50 * queue_improve  # 不匹配的动作奖励较低
            base_reward += improve_bonus
        elif queue_improve < 0:
            # 动作与改善方向相反时惩罚更重
            if (combined_state < MID_LOW and action > 0) or (combined_state >= MID_LOW and action < 0):
                worsen_penalty = 60 * queue_improve  # 错误动作惩罚更重
            else:
                worsen_penalty = 30 * queue_improve  # 轻微惩罚
            base_reward += worsen_penalty
        self.prev_norm_queue = norm_queue

        # 3. 辅助因素：仅微调，不破坏核心梯度
        # 3.1 天气影响：恶劣天气强化增加类动作
        if weather_coeff > 1.3:
            if action == ACTION_LARGE_ADD:
                base_reward += 50
            elif action == ACTION_SMALL_ADD:
                base_reward += 30

        # 3.2 人流量影响：仅在非高峰期抑制增加动作
        if not is_peak and norm_people > 0.7:
            if action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                base_reward -= 40

        # 4. 边界惩罚：仅在动作超出合理范围时生效，不干扰区间内梯度
        if norm_green < 0.2 and action < 0:  # 绿灯过短仍减少
            base_reward -= 60
        elif norm_green > 0.8 and action > 0:  # 绿灯过长仍增加
            base_reward -= 60

        # 5. 越界惩罚：动作超出最小/最大时长时惩罚
        if (self.green_duration + action < MIN_GREEN_DURATION and action < 0) or \
                (self.green_duration + action > MAX_GREEN_DURATION and action > 0):
            base_reward -= 80

        # 限制奖励范围，确保梯度有效
        return np.clip(base_reward, -250, 500)

    # 2. 同时修改step函数，放大20秒增加动作对状态的积极影响
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        chosen_action = ACTIONS[action]
        new_duration = self.green_duration + chosen_action
        new_duration = max(MIN_GREEN_DURATION, min(new_duration, MAX_GREEN_DURATION))
        reward = self._calculate_reward(chosen_action)

        # 更新状态：确保大幅增加动作稳定改善拥堵
        self.green_duration = new_duration

        # 车流量：大幅增加时强制减少，无随机波动
        if chosen_action == ACTION_LARGE_ADD:
            self.car_flow = max(0, min(MAX_CAR_FLOW, self.car_flow - 8))  # 稳定减少
        else:
            self.car_flow = max(0, min(MAX_CAR_FLOW, self.car_flow + random.uniform(-3, 3)))

        self.hour = (self.hour + 1) % 24

        # 交通状态：大幅增加时稳定改善
        if chosen_action == ACTION_LARGE_ADD:
            self.traffic_status = max(0, min(MAX_TRAFFIC_STATUS, self.traffic_status - 1.0))  # 强制改善
        else:
            self.traffic_status = max(0, min(MAX_TRAFFIC_STATUS,
                                             self.traffic_status + (-chosen_action * 0.5) + random.uniform(-0.3, 0.3)))

        # 排队长度：大幅增加时稳定减少，无随机波动
        if chosen_action == ACTION_LARGE_ADD:
            queue_change = -chosen_action * 2.0  # 动作影响最大化
            random_fluct = 0  # 取消随机波动，确保排队减少
        else:
            queue_change = -chosen_action * 1.0
            random_fluct = random.randint(-1, 1)
        self.queue_length = int(max(0, min(MAX_QUEUE_LENGTH, self.queue_length + queue_change + random_fluct)))

        done = self.green_duration >= MAX_GREEN_DURATION
        if done:
            self.current_phase = 1 - self.current_phase
            self.green_duration = MIN_GREEN_DURATION
        return self._get_state(), reward, done


class DQN(nn.Cell):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        # 简化网络，避免过拟合
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.peak_attention = Parameter(Tensor(10.0, ms.float32))  # 放大高峰期特征权重
        self.relu = nn.ReLU()
        self.cat = ops.Concat(axis=-1)

    def construct(self, x):
        peak_feature = x[..., 7] * self.peak_attention
        peak_feature = peak_feature.unsqueeze(-1)
        x = self.cat([x[..., :7], peak_feature])
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # 移除批归一化，加速收敛


class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        # ... 其他参数不变 ...

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # 增大经验池
        self.gamma = 0.95  # 提高折扣因子
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # 减缓探索衰减
        self.learning_rate = 0.001
        self.step_count = 0  # 新增：步数计数器（用于目标网络更新）

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_target_model()

        self.train_net = ms.Model(self.model, loss_fn=self.loss_fn, optimizer=self.optimizer)

    def update_target_model(self):
        param_dict = self.model.parameters_dict()
        self.target_model.get_parameters(param_dict)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = Tensor(state, ms.float32).unsqueeze(0)
        self.model.set_train(False)
        act_values = self.model(state)
        self.model.set_train(True)
        return int(ops.argmax(act_values[0]).asnumpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        self.step_count += 1
        if self.step_count % 50 == 0:  # 更频繁更新目标网络
            self.update_target_model()

        minibatch = random.sample(self.memory, batch_size)
        states = Tensor(np.array([t[0] for t in minibatch]), ms.float32)
        actions = Tensor(np.array([t[1] for t in minibatch]), ms.int32)
        rewards = Tensor(np.array([t[2] for t in minibatch]), ms.float32)
        next_states = Tensor(np.array([t[3] for t in minibatch]), ms.float32)
        dones = Tensor(np.array([t[4] for t in minibatch]), ms.float32)

        self.model.set_train(True)
        current_q = self.model(states)
        indices = ops.stack([Tensor(list(range(batch_size)), ms.int32), actions], axis=1)
        current_q = ops.gather_nd(current_q, indices)

        self.target_model.set_train(False)
        next_q = self.target_model(next_states)
        next_q = ops.max(next_q, axis=1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        def forward_fn(states, actions):
            q_values = self.model(states)
            selected_q = ops.gather_nd(q_values,
                                       ops.stack([Tensor(list(range(batch_size)), ms.int32), actions], axis=1))
            return selected_q

        grad_fn = ms.value_and_grad(forward_fn, None, self.optimizer.parameters)
        q_values, grads = grad_fn(states, actions)
        loss = self.loss_fn(q_values, target_q)
        self.optimizer(grads)

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
            agent.update_target_model()
            print(f"\nEpisode {e}, 平均奖励: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            ms.save_checkpoint(agent.model, "Traffic_Light11.ckpt")

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
        episode_reward = 0
        for _ in range(200):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        total_reward += episode_reward
        if is_peak:
            peak_results.append(ACTIONS[action])

    peak_add_count = sum(1 for a in peak_results if a > 0)
    peak_reduce_count = sum(1 for a in peak_results if a < 0)
    print(f"\n高峰期动作统计: 增加{peak_add_count}次, 减少{peak_reduce_count}次")
    print(f"整体平均奖励: {total_reward / num_tests:.2f}")
    return agent


def Traffic_Control(car_flow, queue_length, people_flow, weather, traffic_status, hour):
    env = TrafficLightEnv()
    try:
        model = DQN(8, ACTION_SIZE)
        ms.load_checkpoint("Traffic_Light11.ckpt", model)
        model.set_train(False)
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

    state_tensor = Tensor(state, ms.float32).unsqueeze(0)
    q_values = model(state_tensor)[0]
    print("动作Q值:")
    for i, a in enumerate(ACTIONS):
        print(f"动作 {a:2.0f}秒: {q_values[i].asnumpy().item():.2f}")
    action_idx = int(ops.argmax(q_values).asnumpy())

    return ACTIONS[action_idx]


if __name__ == "__main__":
    agent = train_agent(episodes=2000, batch_size=128)
    agent = validate_agent(agent)

    print("=== 场景1：高峰期拥堵（晴天） ===")
    result = Traffic_Control(
        car_flow=80, queue_length=70, people_flow=30,
        weather='sunny', traffic_status=8, hour=8
    )
    print(f"推荐绿灯调整: {result}秒")

    print("\n=== 场景2：非高峰期畅通（雨天） ===")
    result = Traffic_Control(
        car_flow=20, queue_length=10, people_flow=10,
        weather='rainy', traffic_status=3, hour=14
    )
    print(f"推荐绿灯调整: {result}秒")