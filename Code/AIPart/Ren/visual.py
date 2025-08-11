
from model import *

# 配置交互后端
matplotlib.use('TkAgg')
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 创建智能体群体
def create_agents(num_agents, map_matrix, intersection_id_map, obstacle_gen):
    return [Agent(map_matrix, intersection_id_map, obstacle_gen) for _ in range(num_agents)]


# 生成地图障碍物（包含缺口中心与交点ID绑定）
obstacle_gen = ObstacleGenerator(MAP_SIZE)
obstacle_gen.add_obstacles(obstacles_params)

# 初始化智能体
agents = create_agents(num_agents, obstacle_gen.map_matrix, obstacle_gen.intersection_id_map, obstacle_gen)

# 绘制画布
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, MAP_SIZE)
ax.set_ylim(0, MAP_SIZE)
ax.set_title('带红灯缺口斥力场的智能体导航系统')
ax.set_xlabel('X坐标')
ax.set_ylabel('Y坐标')

# 绘制障碍物
obstacle_x, obstacle_y = np.where(obstacle_gen.map_matrix == 1)
ax.scatter(obstacle_x, obstacle_y, c='black', s=10, label='道路/障碍物')

# 标记交点（带ID显示）
for idx, (x, y) in enumerate(points):
    ax.scatter(x, y, c='yellow', s=80, marker='+', label=f'交点 {idx}' if idx == 0 else "")
    ax.text(x + 5, y + 5, f'ID:{idx}', fontsize=9, color='yellow')

# 绘制所有缺口中心（调试用）
for i, obs in enumerate(obstacle_gen.obstacles):
    for idx, gap_center in obs.gap_center_map.items():
        ax.scatter(gap_center[0], gap_center[1], c='blue', s=5, marker='o',
                   alpha=0.5, label='缺口中心' if (i == 0 and idx == 0) else "")

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

# 解析紧急情况列表
eme_pos = []
eme_lev = []
for pos, lev in emergency:
    eme_pos.append(pos)
    eme_lev.append(lev)
eme_pos = np.array(eme_pos)

# 紧急情况标记
eme_sizes = [50 + lev * 30 for lev in eme_lev]
emergency_plot = ax.scatter(
    eme_pos[:, 0], eme_pos[:, 1],
    s=eme_sizes,
    marker='X',
    c='red',
    edgecolors='black',
    linewidths=1.5,
    label='紧急情况',
    visible=False
)

ax.legend()


# 动画更新函数
def update(frame):
    # 模拟交通灯状态（循环变化）
    if frame < 200:
        traffic_light = [1, 1, 1, 1, 1, 1, 1, 1]# ID=0红灯
    elif frame < 400:
        traffic_light = [0, 0, 0, 0, 0, 0, 0, 0]  # ID=1红灯
    elif frame < 600:
        traffic_light = [1, 1, 1, 1, 1, 1, 1, 1]  # ID=2红灯
    else:
        traffic_light = [0, 0, 0, 0, 1, 0, 0, 0]  # ID=4红灯

    # 更新智能体位置
    for agent in agents:
        if 300 <= frame <= 500:
            agent.move_towards_target((eme_pos, eme_lev), traffic_light)
        else:
            agent.move_towards_target(traffic_light=traffic_light)

    # 更新智能体显示
    positions = np.array([agent.position for agent in agents])
    agents_plot.set_offsets(positions)

    # 控制紧急情况显示
    emergency_plot.set_visible(300 <= frame <= 500)

    return agents_plot, emergency_plot


# 运行动画
animation = FuncAnimation(
    fig, update,
    frames=800,
    interval=50,
    blit=True
)

plt.tight_layout()
plt.show()
