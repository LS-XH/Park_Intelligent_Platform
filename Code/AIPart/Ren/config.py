from math import log10, pi
import torch
# from Code.AIPart.Ren.testmap import obstacles_params, targets, targets_heat, points, MAP_SIZE


# 设备配置（只会在模块首次导入时执行一次）
def _get_device():
    """内部函数：检测并返回可用设备"""
    if torch.cuda.is_available():
        # # 可选：打印GPU信息（如型号）
        # print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("CUDA不可用，使用CPU")
        return torch.device("cpu")

# 全局设备变量（模块导入时自动初始化，后续导入直接使用缓存值）
device = _get_device()

PRECISION = 0
# 地图参数
num_agents = 2000
target_radius = 90.0  # 改为float
MIN_SPEED, MAX_SPEED = 0.3, 1.0  # 改为float
MAX_STEP = 50
evacuate_rate = 3  # 疏散距离参数
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

MODEL_DIR_WEIGHT = 0.2
TRAIN_SET = False
USE_MODEL = True

WEIGHTS_PATH = "./Ren/agent_model.pth"
TRAIN_STEP = 3