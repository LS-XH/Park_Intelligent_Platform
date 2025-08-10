import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import matplotlib
from testmap import obstacles_params, targets

from model import *
# 地图参数 - 适配测试地图坐标范围
MAP_SIZE = 700
num_agents = 1000

# 配置交互后端
matplotlib.use('TkAgg')
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建智能体群体
def create_agents(num_agents, map_matrix):
    return [Agent(map_matrix) for _ in range(num_agents)]


# 生成地图障碍物
obstacle_gen = ObstacleGenerator(MAP_SIZE)
obstacle_gen.add_obstacles(obstacles_params)

# 初始化智能体

agents = create_agents(num_agents, obstacle_gen.map_matrix)

# 绘制画布
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, MAP_SIZE)
ax.set_ylim(0, MAP_SIZE)
ax.set_title('基于绝对距离判断的智能体导航（防止远距离目标徘徊）')
ax.set_xlabel('X坐标')
ax.set_ylabel('Y坐标')

# 绘制障碍物
obstacle_x, obstacle_y = np.where(obstacle_gen.map_matrix == 1)
ax.scatter(obstacle_x, obstacle_y, c='black', s=10, label='道路/障碍物')

# 标记路口
if obstacle_gen.intersection_points:
    cross_x, cross_y = zip(*obstacle_gen.intersection_points)
    ax.scatter(cross_x, cross_y, c='yellow', s=50, marker='+', label='路口中心')

# 绘制目标点
targets_plot = ax.scatter(
    targets[:, 0], targets[:, 1],
    c='red', s=150, marker='*',
    edgecolors='black', linewidths=2,
    label='目标点'
)

# 初始化智能体绘制
agent_positions = np.array([agent.position for agent in agents])
agent_sizes = [agent.size for agent in agents]
agent_colors = [agent.color for agent in agents]

agents_plot = ax.scatter(
    agent_positions[:, 0], agent_positions[:, 1],
    s=agent_sizes, c=agent_colors,
    alpha=0.7, edgecolors='white', linewidths=0.5,
    label='智能体'
)

ax.legend()


# 动画更新函数
def update(frame):
    for agent in agents:
        agent.move_towards_target()
    positions = np.array([agent.position for agent in agents])
    agents_plot.set_offsets(positions)
    return agents_plot,


# 运行动画
animation = FuncAnimation(
    fig, update,
    frames=300,
    interval=50,
    blit=True
)

plt.tight_layout()
plt.show()