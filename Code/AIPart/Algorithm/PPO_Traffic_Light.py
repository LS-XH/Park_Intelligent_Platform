import numpy as np
import random
from typing import Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# 设备配置
device = "cpu"
# print(f"使用设备: {device}")

# 天气系数（保持原样）
WEATHER_COEFFICIENTS = {
    'sunny': 1.0, 'rainy': 1.2, 'snowy': 1.5, 'foggy': 1.3, 'stormy': 1.6
}

# 动作定义（保持原样）
ACTION_LARGE_ADD = 20.0
ACTION_SMALL_ADD = 10.0
ACTION_NO_CHANGE = 0.0
ACTION_SMALL_REDUCE = -10.0
ACTION_LARGE_REDUCE = -20.0
ACTIONS = [ACTION_LARGE_ADD, ACTION_SMALL_ADD, ACTION_NO_CHANGE, ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]
ACTION_SIZE = len(ACTIONS)

# 状态参数（保持原样）
MAX_CAR_FLOW = 100.0
MAX_QUEUE_LENGTH = 100
MAX_PEOPLE_FLOW = 50.0
MAX_TRAFFIC_STATUS = 10.0
MAX_GREEN_DURATION = 120
MIN_GREEN_DURATION = 20

# 状态区间阈值（保持原样）
LOWEST_THRESHOLD = 1 / 3
LOW_THRESHOLD = 0.4
MID_LOW = 0.4
MID_HIGH = 0.6
HIGH_LOW = 0.6
HIGH_HIGH = 2 / 3
HIGHEST_THRESHOLD = 2 / 3

# 阈值（保持原样）
THRESHOLD_HIGH = 0.6
THRESHOLD_LONG_QUEUE = 0.6


class TrafficLightEnv(gym.Env):
    """交通信号灯环境（完全保留原有奖励逻辑，仅添加Gym接口适配）"""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.current_phase = 0
        self.green_duration = 60
        self.car_flow = 0.0
        self.queue_length = 0
        self.people_flow = 0.0
        self.weather = 'sunny'
        self.traffic_status = 0.0
        self.hour = 0
        self.prev_norm_queue = 0.0
        self.render_mode = render_mode

        # 定义动作空间和观测空间（适配Gym接口）
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, min(WEATHER_COEFFICIENTS.values()), 0.0, 0.0, 0.0, 0]),
            high=np.array([1.0, 1.0, 1.0, max(WEATHER_COEFFICIENTS.values()), 1.0, 1.0, 1.0, 1]),
            dtype=np.float32
        )

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """重置环境（仅添加Gym的seed支持，逻辑不变）"""
        super().reset(seed=seed)  # 兼容Gym的随机种子
        peak_hours = list(range(7, 10)) + list(range(17, 20))
        non_peak_hours = list(range(0, 7)) + list(range(10, 17)) + list(range(20, 24))
        self.hour = random.choice(peak_hours * 2 + non_peak_hours)
        is_peak = 1 if self.hour in peak_hours else 0

        # 原有状态初始化逻辑（完全不变）
        self.car_flow = random.uniform(60, 100) if is_peak else random.uniform(0, 60)
        self.queue_length = random.randint(50, 100) if is_peak else random.randint(0, 40)
        self.people_flow = random.uniform(30, 50) if is_peak else random.uniform(0, 30)
        self.weather = random.choice(list(WEATHER_COEFFICIENTS.keys()))
        self.traffic_status = random.uniform(6, 10) if is_peak else random.uniform(0, 5)
        self.prev_norm_queue = self.queue_length / MAX_QUEUE_LENGTH
        self.green_duration = random.randint(40, 80)
        self.current_phase = random.choice([0, 1])

        return self._get_state(), {}

    def set_state(self, car_flow: float, queue_length: int, people_flow: float,
                  weather: str, traffic_status: float, hour: int):
        """设置环境状态用于推理（原有逻辑不变）"""
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
        """获取状态（原有逻辑不变）"""
        norm_car = self.car_flow / MAX_CAR_FLOW
        norm_queue = self.queue_length / MAX_QUEUE_LENGTH
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
        """奖励计算"""
        is_peak = 1 if self.hour in list(range(7, 10)) + list(range(17, 20)) else 0
        norm_car = self.car_flow / MAX_CAR_FLOW
        norm_queue = self.queue_length / MAX_QUEUE_LENGTH
        norm_people = self.people_flow / MAX_PEOPLE_FLOW
        weather_coeff = WEATHER_COEFFICIENTS[self.weather]
        norm_status = self.traffic_status / MAX_TRAFFIC_STATUS
        norm_green = self.green_duration / MAX_GREEN_DURATION
        queue_improve = self.prev_norm_queue - norm_queue

        base_reward = -20 * norm_car - 20 * norm_queue

        if is_peak:
            # 高峰期逻辑保持不变
            if norm_car > THRESHOLD_HIGH or norm_queue > THRESHOLD_LONG_QUEUE:
                if action == ACTION_LARGE_ADD:
                    base_reward += 60
                elif action == ACTION_SMALL_ADD:
                    base_reward += 30
                elif action == ACTION_SMALL_REDUCE:
                    base_reward -= 80
                elif action == ACTION_LARGE_REDUCE:
                    base_reward -= 100
            else:
                if action in [ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]:
                    base_reward += 30
                elif action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                    base_reward -= 30

        else:
            combined_state = max(norm_car, norm_queue)

            # 核心修改：调整LOW区间上限为0.45（原0.4），让0.4落入LOW区间
            if combined_state < LOWEST_THRESHOLD:  # < 0.333
                if action == ACTION_LARGE_REDUCE:
                    base_reward += 100
                elif action == ACTION_SMALL_REDUCE:
                    base_reward += 50
                elif action == ACTION_NO_CHANGE:
                    base_reward -= 30
                else:
                    base_reward -= 80

            # 调整LOW区间：0.333 ≤ 状态 < 0.45（原0.4）
            elif LOWEST_THRESHOLD <= combined_state < 0.45:
                if action == ACTION_SMALL_REDUCE:  # 小幅减少（-10秒）奖励提高
                    base_reward += 100  # 原80，提高奖励权重
                elif action == ACTION_LARGE_REDUCE:
                    pass
                elif action == ACTION_NO_CHANGE:
                    base_reward -= 50  # 原-20，增加不调整的惩罚
                else:
                    base_reward -= 60

            # 中区间调整为0.45 ≤ 状态 ≤ 0.6（原0.4开始）
            elif 0.45 <= combined_state <= MID_HIGH:
                if action == ACTION_NO_CHANGE:
                    base_reward += 80
                elif action in [ACTION_SMALL_ADD, ACTION_SMALL_REDUCE]:
                    pass
                else:
                    base_reward -= 80

            # 剩余区间逻辑保持不变
            elif HIGH_LOW <= combined_state < HIGH_HIGH:
                if action == ACTION_SMALL_ADD:
                    base_reward += 80
                elif action == ACTION_LARGE_ADD:
                    pass
                elif action == ACTION_NO_CHANGE:
                    base_reward -= 20
                else:
                    base_reward -= 50

            else:
                if action == ACTION_LARGE_ADD:
                    base_reward += 80
                elif action == ACTION_SMALL_ADD:
                    pass
                elif action == ACTION_NO_CHANGE:
                    base_reward -= 30
                else:
                    base_reward -= 80

        # 以下逻辑保持不变
        if queue_improve > 0:
            base_reward += 20 * queue_improve if action >= 0 else 10 * queue_improve
        elif queue_improve < 0:
            base_reward += 15 * queue_improve
        self.prev_norm_queue = norm_queue

        if norm_people > 0.5:
            if action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                base_reward -= 80

        if weather_coeff > 1.2:
            if action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                base_reward += 50

        if norm_status > 0.6:
            if action in [ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]:
                base_reward -= 40

        if norm_green < 0.2:
            if action in [ACTION_SMALL_REDUCE, ACTION_LARGE_REDUCE]:
                base_reward -= 100
        elif norm_green > 0.8:
            if action in [ACTION_LARGE_ADD, ACTION_SMALL_ADD]:
                base_reward -= 100

        if (self.green_duration + action < MIN_GREEN_DURATION and action < 0) or \
                (self.green_duration + action > MAX_GREEN_DURATION and action > 0):
            base_reward -= 100

        return base_reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """步骤函数（仅添加Gym要求的truncated返回值，逻辑不变）"""
        chosen_action = ACTIONS[action]
        new_duration = self.green_duration + chosen_action
        new_duration = max(MIN_GREEN_DURATION, min(new_duration, MAX_GREEN_DURATION))

        reward = self._calculate_reward(chosen_action)  # 调用原有奖励函数

        # 原有状态更新逻辑（完全不变）
        self.green_duration = new_duration
        self.car_flow = max(0, min(MAX_CAR_FLOW, self.car_flow + random.uniform(-5, 5)))
        self.hour = (self.hour + 1) % 24
        self.traffic_status = max(0, min(MAX_TRAFFIC_STATUS,
                                         self.traffic_status - chosen_action * 0.3 + random.uniform(-0.5, 0.5)))

        queue_change = -chosen_action * 0.5
        if chosen_action == 0:
            random_fluct = random.randint(-1, 1)
        else:
            random_fluct = random.randint(-2, 2)
        self.queue_length = int(max(0, min(MAX_QUEUE_LENGTH, self.queue_length + queue_change + random_fluct)))

        done = self.green_duration >= MAX_GREEN_DURATION
        if done:
            self.current_phase = 1 - self.current_phase
            self.green_duration = MIN_GREEN_DURATION

        # 适配Gym接口：返回5个值（增加truncated和info）
        return self._get_state(), reward, done, False, {}

    def render(self):
        """渲染函数（保持原样）"""
        if self.render_mode == "human":
            print(
                f"相位: {self.current_phase}, 绿灯时长: {self.green_duration}, 车流量: {self.car_flow:.1f}, 排队长度: {self.queue_length}")

    def close(self):
        """关闭环境（Gym接口要求）"""
        pass


def train_ppo_agent(total_timesteps=2000000):
    """使用Stable Baselines3的PPO训练模型"""
    # 初始化环境并包装Monitor用于记录训练指标
    env = Monitor(TrafficLightEnv())

    # 定义PPO模型（使用库函数，不修改核心逻辑）
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device=device,
        tensorboard_log="./traffic_ppo_logs/"
    )

    # 设置检查点回调，定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./ppo_checkpoints/",
        name_prefix="traffic_light"
    )

    # 开始训练
    print(f"开始PPO训练，总步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 保存最终模型
    model.save("traffic_light_ppo_final1")
    print("模型保存为: traffic_light_ppo_final1.zip")
    return model


def evaluate_ppo_agent(model_path="traffic_light_ppo_final1"):
    """评估训练好的PPO模型"""
    env = TrafficLightEnv()
    model = PPO.load(model_path, env=env, device=device)

    # 评估模型性能
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"评估结果: 平均奖励 = {mean_reward:.2f} ± {std_reward:.2f}")

    # 统计高峰期动作
    peak_add = 0
    peak_reduce = 0
    non_peak_add = 0
    non_peak_reduce = 0

    for _ in range(200):
        obs, _ = env.reset()
        is_peak = 1 if env.hour in list(range(7, 10)) + list(range(17, 20)) else 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        action_value = ACTIONS[action]
        if is_peak:
            if action_value > 0:
                peak_add += 1
            elif action_value < 0:
                peak_reduce += 1
        else:
            if action_value > 0:
                non_peak_add += 1
            elif action_value < 0:
                non_peak_reduce += 1

    print(f"\n高峰期动作统计: 增加 {peak_add} 次, 减少 {peak_reduce} 次")
    print(f"非高峰期动作统计: 增加 {non_peak_add} 次, 减少 {non_peak_reduce} 次")
    return model


def traffic_control_ppo(car_flow, queue_length, people_flow, weather, traffic_status, hour):
    """PPO推理接口（保持原有参数）"""
    env = TrafficLightEnv()
    try:
        # 加载PPO模型
        model = PPO.load("traffic_light_ppo_final1", env=env, device=device)
    except FileNotFoundError:
        print("模型文件未找到，使用默认策略")
        is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        if is_peak:
            return 20.0 if (car_flow / 100 > 0.4 or queue_length / 100 > 0.4) else -10.0
        else:
            return -20.0 if (car_flow / 100 < 0.3 and queue_length / 100 < 0.3) else 10.0

    # 设置当前状态并推理
    env.set_state(car_flow, queue_length, people_flow, weather, traffic_status, hour)
    obs = env._get_state()
    action_idx, _ = model.predict(obs, deterministic=True)
    return ACTIONS[action_idx]


if __name__ == "__main__":
    # # 训练模型（首次运行时执行）
    # model = train_ppo_agent(total_timesteps=2000000)
    #
    # # 评估模型（训练完成后执行）
    # model = evaluate_ppo_agent()

    # 测试推理（示例）
    test_cases = [
        # 早高峰拥堵场景
        (80.0, 70, 35.0, "rainy", 8.0, 8),
        # 平峰畅通场景
        (40.0, 40, 10.0, "rainy", 3.0, 14),
        # 晚高峰中度拥堵
        (50.0, 50, 25.0, "sunny", 6.0, 14)
    ]

    for i, case in enumerate(test_cases):
        car_flow, queue_length, people_flow, weather, traffic_status, hour = case
        action = traffic_control_ppo(*case)
        print(f"\n测试案例 {i + 1}:")
        print(
            f"输入: 车流量={car_flow}, 排队={queue_length}, 人流={people_flow}, 天气={weather}, 状态={traffic_status}, 时间={hour}点")
        print(f"PPO建议调整: {action}秒")
