from math import log10, pi
import torch
from Code.AIPart.Ren.testmap import obstacles_params, targets, targets_heat, points, MAP_SIZE


# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

PRECISION = 0
# 地图参数
num_agents = 4000
target_radius = 90.0  # 改为float
emergency = [
    ((100.0 * 3, 50.0 * 3), 1),  # 改为float
    ((100.0 * 3, 100.0 * 3), 5),  # 改为float
]
MIN_SPEED, MAX_SPEED = 0.3, 1.0  # 改为float
MAX_STEP = 50
evacuate_rate = log10(MAP_SIZE)  # 疏散速率参数
CHANGE_PROB = 0.002

# 障碍物检测参数
SAFE_DISTANCE = 1.0  # 改为float
ANGLE_RANGE_START = pi / 6  # 尝试角度范围起始值
ANGLE_RANGE_END = pi / 1.5  # 尝试角度范围结束值
NUM_ATTEMPTS = 10  # 尝试次数

# 惯性参数 (可调整范围)
INERTIA_WEIGHT_RANGE = (0.5, 0.7)  # 惯性权重范围，值越大惯性越强

# 智能体间斥力参数
AGENT_REPULSION_RADIUS = 10.0  # 产生斥力的距离阈值
AGENT_REPULSION_STRENGTH = 2.0  # 斥力强度系数

# 密度可视化参数
DENSITY_ALPHA = 0.6  # 密度图透明度
DENSITY_CMAP = 'jet'  # 密度图颜色映射
UPDATE_DENSITY_STEP = 50
DENSITY_MATRIX_SIZE = 200  # 新增：密度矩阵大小