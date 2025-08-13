import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import torch

from testmap import obstacles_params, targets, targets_heat, points
from model import AgentGroup, ObstacleGenerator, num_agents, MAP_SIZE, DENSITY_CMAP, DENSITY_ALPHA, emergency

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置交互后端
matplotlib.use('TkAgg')
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建智能体群体
def create_agents(num_agents, map_matrix, intersection_id_map, obstacle_gen):
    return AgentGroup(num_agents, map_matrix, intersection_id_map, obstacle_gen)


# 生成地图障碍物
obstacle_gen = ObstacleGenerator(MAP_SIZE)
obstacle_gen.add_obstacles(obstacles_params)

# 初始化智能体群体
agents = create_agents(num_agents, obstacle_gen.map_matrix, obstacle_gen.intersection_id_map, obstacle_gen)

# 绘制画布
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0.0, MAP_SIZE)
ax.set_ylim(0.0, MAP_SIZE)
ax.set_title('带惯性特性的智能体导航系统')
ax.set_xlabel('X坐标')
ax.set_ylabel('Y坐标')

# 1. 绘制密度热图（作为底层）
density_data = agents.get_density().cpu().numpy()
density_plot = ax.imshow(
    density_data.T,  # 转置以正确显示坐标
    extent=[0, MAP_SIZE, 0, MAP_SIZE],  # 保持原始地图坐标范围
    cmap=DENSITY_CMAP,
    alpha=DENSITY_ALPHA,
    origin='lower',
    vmin=0,
    vmax=min(10, num_agents // 100)  # 限制最大显示值，使颜色更明显
)
# 添加密度颜色条
cbar = fig.colorbar(density_plot, ax=ax)
cbar.set_label('智能体密度')

# 绘制障碍物
obstacle_x, obstacle_y = np.where(obstacle_gen.map_matrix == 1.0)
ax.scatter(obstacle_x, obstacle_y, c='black', s=10, label='道路/障碍物')

# 标记交点
for idx, (x, y) in enumerate(obstacle_gen.intersection_points):
    ax.scatter(x, y, c='yellow', s=80, marker='+', label=f'交点 {idx}' if idx == 0 else "")
    ax.text(x + 5.0, y + 5.0, f'ID:{idx}', fontsize=9, color='yellow')

# 绘制所有缺口中心和斥力线段
for i, obs in enumerate(obstacle_gen.obstacles):
    for idx, gap_center in obs.gap_center_map.items():
        ax.scatter(gap_center[0], gap_center[1], c='blue', s=5, marker='o',
                   alpha=0.5, label='缺口中心' if (i == 0 and idx == 0) else "")
        # 计算并绘制斥力线段
        seg_length = obs.gap_offset_ratio + (obstacle_gen.intersection_gap_size / 2.0) + 2.0
        dir_vec = np.array([obs.dir_x, obs.dir_y], dtype=np.float32)
        seg_start = gap_center - 0.5 * seg_length * dir_vec
        seg_end = gap_center + 0.5 * seg_length * dir_vec
        ax.plot([seg_start[0], seg_end[0]], [seg_start[1], seg_end[1]], 'g--', linewidth=1,
                alpha=0.5, label='斥力线段' if (i == 0 and idx == 0) else "")

# 绘制目标点
targets_np = np.array(targets, dtype=np.float32)
targets_plot = ax.scatter(
    targets_np[:, 0], targets_np[:, 1],
    c='red', s=150, marker='*',
    edgecolors='black', linewidths=2,
    label='目标点'
)

# 初始化智能体绘制
agent_positions = agents.positions.cpu().numpy()
agent_sizes = agents.sizes.cpu().numpy()
agent_colors = agents.colors.cpu().numpy()

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
eme_pos_np = np.array(eme_pos, dtype=np.float32)

# 紧急情况标记
eme_sizes = [50.0 + lev * 30.0 for lev in eme_lev]
emergency_plot = ax.scatter(
    eme_pos_np[:, 0], eme_pos_np[:, 1],
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
    # 模拟交通灯状态
    if frame < 200:
        traffic_light = [1, 0, 0, 0, 0, 0, 0, 0]  # ID=0红灯
    elif frame < 400:
        traffic_light = [0, 1, 0, 0, 0, 0, 0, 0]  # ID=1红灯
    elif frame < 600:
        traffic_light = [0, 0, 1, 0, 0, 0, 0, 0]  # ID=2红灯
    else:
        traffic_light = [0, 0, 0, 0, 1, 0, 0, 0]  # ID=4红灯

    # 更新智能体位置
    if 300 <= frame <= 500:
        agents.move_towards_target((eme_pos, eme_lev), traffic_light)
    else:
        agents.move_towards_target(traffic_light=traffic_light)

    # 更新智能体显示
    positions = agents.positions.cpu().numpy()
    agents_plot.set_offsets(positions)

    if frame % 50 == 0:
        # 更新密度热图
        density_data = agents.get_density().cpu().numpy()
        print(density_data.max(), density_data.min())
        density_plot.set_data(density_data.T)  # 转置以正确显示

        # 动态调整颜色范围以适应密度变化
        current_max = min(density_data.max(), num_agents // 5)
        if current_max > 0:
            density_plot.set_clim(vmin=0, vmax=current_max)

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