import numpy as np
from math import log10, pi
import torch

from testmap import obstacles_params, targets, targets_heat, points, MAP_SIZE

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

PRECISION = 0
num_agents = 1000
target_radius = 90.0  # 改为float
emergency = [
    ((100.0 * 3, 50.0 * 3), 1),  # 改为float
    ((100.0 * 3, 100.0 * 3), 5),  # 改为float
]
MIN_SPEED, MAX_SPEED = 0.3, 1.0  # 改为float
MAX_STEP = 50
evacuate_rate = log10(MAP_SIZE)  # 疏散速率参数

# 障碍物检测参数
SAFE_DISTANCE = 1.0  # 改为float
ANGLE_RANGE_START = pi / 6  # 尝试角度范围起始值
ANGLE_RANGE_END = pi / 1.5  # 尝试角度范围结束值
NUM_ATTEMPTS = 10  # 尝试次数

# 惯性参数 (可调整范围)
INERTIA_WEIGHT_RANGE = (0.5, 0.7)  # 惯性权重范围，值越大惯性越强

# 密度可视化参数
DENSITY_ALPHA = 0.6  # 密度图透明度
DENSITY_CMAP = 'jet'  # 密度图颜色映射
UPDATE_DENSITY_STEP = 50
DENSITY_MATRIX_SIZE = 200  # 新增：密度矩阵大小


class Obstacle:
    """障碍物类，优化边界信息存储"""

    def __init__(self, start_point, end_point, width, gap_offset_ratio=1.0):
        # 转换为float
        self.start_point = tuple(float(x) for x in start_point)
        self.end_point = tuple(float(x) for x in end_point)
        self.width = float(width)
        self.gap_offset_ratio = float(gap_offset_ratio)
        self.base_line = self.bresenham(*self.start_point, *self.end_point)
        self.length = len(self.base_line)
        self.gap_center_map = {}
        self.auto_gaps = []

        # 计算方向向量
        x0, y0 = self.start_point
        x1, y1 = self.end_point
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx ** 2 + dy ** 2) if (dx != 0 or dy != 0) else 1.0

        self.dir_x = dx / length
        self.dir_y = dy / length
        self.perp_dir_x = -self.dir_y  # 垂直于障碍物的方向
        self.perp_dir_y = self.dir_x  # 垂直于障碍物的方向

        # 存储障碍物点集（float类型）
        self.points = np.array([], dtype=np.float32).reshape(0, 2)

    @staticmethod
    def bresenham(x0, y0, x1, y1):
        """生成浮点型的bresenham线段点"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1.0 if x1 > x0 else -1.0
        sy = 1.0 if y1 > y0 else -1.0

        if dx > dy:
            err = dx / 2.0
            while not np.isclose(x, x1):
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
            points.append((x, y))
        else:
            err = dy / 2.0
            while not np.isclose(y, y1):
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
            points.append((x, y))
        return points


class ObstacleGenerator:
    """障碍物生成器，优化障碍物数据结构"""

    def __init__(self, map_size):
        self.map_size = float(map_size)
        # 地图矩阵使用float类型
        self.map_matrix = np.zeros((int(map_size), int(map_size)), dtype=np.float32)
        self.obstacles = []
        # 转换为float
        self.intersection_points = [(float(x), float(y)) for x, y in points]
        self.intersection_id_map = {i: np.array(point, dtype=np.float64) for i, point in
                                    enumerate(self.intersection_points)}
        self.intersection_gap_size = 3.0

        # 预计算障碍物边界张量
        self.obstacle_boundaries = None

    def add_obstacles(self, obstacles_params, intersection_gap_size=3.0):  # 改为float
        self.intersection_gap_size = float(intersection_gap_size)
        temp_obstacles = []
        for params in obstacles_params:
            obstacle = Obstacle(
                start_point=params['start_point'],
                end_point=params['end_point'],
                width=params['width'],
                gap_offset_ratio=max(params.get('gap_offset_ratio', 1.0), 0)
            )
            temp_obstacles.append(obstacle)

        # 预计算交点和自动缺口位置比例
        for i, obstacle in enumerate(temp_obstacles):
            for j, other_obstacle in enumerate(temp_obstacles):
                if i != j:
                    intersection_point, pos_ratio = self.find_intersection(obstacle, other_obstacle)
                    if intersection_point:
                        offset_distance = obstacle.gap_offset_ratio * max(obstacle.width, other_obstacle.width)
                        x0, y0 = obstacle.start_point
                        x1, y1 = obstacle.end_point
                        line_length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) if (x1 != x0 or y1 != y0) else 1.0

                        if line_length > 0:
                            offset_ratio = offset_distance / line_length
                            if pos_ratio + offset_ratio <= 1.0:
                                obstacle.auto_gaps.append(pos_ratio + offset_ratio)
                            if pos_ratio - offset_ratio >= 0.0:
                                obstacle.auto_gaps.append(pos_ratio - offset_ratio)

        # 生成实际障碍、缺口并绑定交点ID
        for obstacle in temp_obstacles:
            self._generate_obstacle(obstacle, self.intersection_gap_size)
            self.obstacles.append(obstacle)

        # 预计算所有障碍物点的张量表示
        self._precompute_obstacle_tensors()

    def _generate_obstacle(self, obstacle, intersection_gap_size):
        obstacle_point_set = set((round(x, PRECISION), round(y, PRECISION)) for x, y in obstacle.base_line)
        contained_ids = []
        for idx, point in self.intersection_id_map.items():
            point_tuple = (round(point[0], PRECISION), round(point[1], PRECISION))
            if point_tuple in obstacle_point_set:
                contained_ids.append(idx)

        # 计算并存储缺口中心
        for gap_pos in obstacle.auto_gaps:
            gap_idx = int(len(obstacle.base_line) * gap_pos)
            gap_idx = max(0, min(len(obstacle.base_line) - 1, gap_idx))
            gap_center = np.array(obstacle.base_line[gap_idx], dtype=np.float64)

            min_dist = float('inf')
            best_id = -1
            for idx in contained_ids:
                dist = np.linalg.norm(gap_center - self.intersection_id_map[idx])
                if dist < min_dist:
                    min_dist = dist
                    best_id = idx

            if best_id != -1:
                obstacle.gap_center_map[best_id] = gap_center

        # 生成带宽度的障碍物点集（float类型）
        obstacle_points = set()
        for (x, y) in obstacle.base_line:
            for i in range(-int(obstacle.width // 2), int(obstacle.width // 2) + 1):
                for j in range(-int(obstacle.width // 2), int(obstacle.width // 2) + 1):
                    if i ** 2 + j ** 2 <= (obstacle.width // 2) ** 2:
                        nx = x + i * obstacle.perp_dir_x - j * obstacle.dir_x
                        ny = y + i * obstacle.perp_dir_y - j * obstacle.dir_y
                        obstacle_points.add((round(nx, PRECISION), round(ny, PRECISION)))

        # 移除缺口点
        for gap_center in obstacle.gap_center_map.values():
            gap_x, gap_y = gap_center
            gap_points = set()

            # 垂直障碍物方向（长度）：障碍物宽度 + 2
            # 计算垂直方向范围（确保覆盖障碍物宽度+2的长度）
            vertical_half = (obstacle.width + 2) / 2
            vertical_start = -int(vertical_half // 1 + 1)
            vertical_end = int(vertical_half // 1 + 1)

            # 沿障碍物方向（宽度）：intersection_gap_size
            # 计算沿方向范围（确保覆盖intersection_gap_size的宽度）
            horizontal_half = intersection_gap_size / 2
            horizontal_start = -int(horizontal_half // 1 + 1)
            horizontal_end = int(horizontal_half // 1 + 1)

            # 生成缺口点（垂直方向用障碍物宽度+2，沿方向用intersection_gap_size）
            for i in range(vertical_start, vertical_end + 1):
                for j in range(horizontal_start, horizontal_end + 1):
                    # i对应垂直障碍物方向（长度），j对应沿障碍物方向（宽度）
                    nx = gap_x + i * obstacle.perp_dir_x - j * obstacle.dir_x
                    ny = gap_y + i * obstacle.perp_dir_y - j * obstacle.dir_y
                    gap_points.add((round(nx, PRECISION), round(ny, PRECISION)))
            obstacle_points -= gap_points

        # 添加到地图
        for x, y in obstacle_points:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self.map_matrix[int(round(x)), int(round(y))] = 1.0

        # 保存障碍物点集（float类型）
        obstacle.points = np.array(list(obstacle_points), dtype=np.float32)

    def _precompute_obstacle_tensors(self):
        """将所有障碍物点转换为张量，加速检测"""
        all_points = []
        for obstacle in self.obstacles:
            if len(obstacle.points) > 0:
                all_points.append(obstacle.points)

        if all_points:
            self.obstacle_boundaries = torch.tensor(
                np.vstack(all_points),
                device=device,
                dtype=torch.float32
            )
        else:
            self.obstacle_boundaries = torch.zeros((0, 2), device=device, dtype=torch.float32)

    def find_intersection(self, obstacle1, obstacle2):
        line1 = obstacle1.base_line
        line2 = obstacle2.base_line
        line1_set = set((round(x, 2), round(y, 2)) for x, y in line1)
        line2_set = set((round(x, 2), round(y, 2)) for x, y in line2)
        intersections = line1_set.intersection(line2_set)

        if intersections:
            intersection_point = next(iter(intersections))
            # 找到最接近的点的索引
            min_dist = float('inf')
            pos_idx = 0
            for i, (x, y) in enumerate(line1):
                rounded_x, rounded_y = round(x, PRECISION), round(y, PRECISION)
                dist = np.hypot(rounded_x - intersection_point[0], rounded_y - intersection_point[1])
                if dist < min_dist:
                    min_dist = dist
                    pos_idx = i
            pos_ratio = pos_idx / len(line1) if len(line1) > 0 else 0.0
            return intersection_point, pos_ratio

        return None, 0.0


class AgentGroup:
    def __init__(self, num_agents, map_matrix, intersection_id_map, obstacle_gen):
        self.num_agents = num_agents
        self.map = torch.tensor(map_matrix, device=device, dtype=torch.float32)
        self.map_size = float(map_matrix.shape[0])
        self.intersection_id_map = intersection_id_map
        self.obstacle_gen = obstacle_gen

        # 密度矩阵缩放因子
        self.density_scale = DENSITY_MATRIX_SIZE / self.map_size  # 新增：计算缩放因子

        # 惯性相关属性
        self.inertia_weights = torch.FloatTensor(num_agents).uniform_(
            INERTIA_WEIGHT_RANGE[0],
            INERTIA_WEIGHT_RANGE[1]
        ).to(device)  # 每个智能体的惯性权重，随机生成
        self.current_directions = self._initialize_directions()  # 当前运动方向

        # 停滞判断参数
        self.previous_distances = torch.zeros((num_agents, MAX_STEP), device=device, dtype=torch.float32)
        self.step_counter = torch.zeros(num_agents, device=device, dtype=torch.int32)
        self.min_movement_threshold = 5.0  # 改为float

        # 目标点信息（float类型）
        self.all_targets = torch.tensor(targets, device=device, dtype=torch.float32)
        self.target_heat = torch.tensor(targets_heat, device=device, dtype=torch.float32)
        self.target_heat = self.target_heat / self.target_heat.sum()

        # 预计算膨胀地图
        self._precompute_expanded_map()

        # 智能体属性
        self.positions = self._initialize_positions()
        self.target_ids = self._initialize_target_ids()
        self.targets = torch.tensor(targets, device=device, dtype=torch.float32)[self.target_ids]
        self.speeds = torch.FloatTensor(num_agents).uniform_(MIN_SPEED, MAX_SPEED).to(device)
        self.randomness = torch.FloatTensor(num_agents).uniform_(0.1, 0.3).to(device)
        self.sizes = torch.randint(15, 30, (num_agents,), device=device)
        self.colors = torch.rand((num_agents, 3), device=device)

        # 密度矩阵相关（修改为200x200）
        self.density = torch.zeros((DENSITY_MATRIX_SIZE, DENSITY_MATRIX_SIZE), device=device, dtype=torch.float32)
        self._update_density()  # 初始化密度矩阵
        self.update_den_step = UPDATE_DENSITY_STEP

    def _update_density(self):
        """高效更新密度矩阵，使用向量化操作，改为200x200大小"""
        # 清零当前密度矩阵
        self.density.zero_()

        # 将智能体位置转换为缩小后的网格坐标（整数索引）
        # 1. 缩放坐标到[0, DENSITY_MATRIX_SIZE)范围
        scaled_x = self.positions[:, 0] * self.density_scale
        scaled_y = self.positions[:, 1] * self.density_scale

        # 2. 转换为整数并钳位到有效范围
        x_coords = torch.clamp(torch.round(scaled_x).long(), 0, DENSITY_MATRIX_SIZE - 1)
        y_coords = torch.clamp(torch.round(scaled_y).long(), 0, DENSITY_MATRIX_SIZE - 1)

        # 3. 使用PyTorch的索引加法高效计算密度
        self.density.index_put_((x_coords, y_coords), torch.ones_like(x_coords, dtype=torch.float32), accumulate=True)

        # 4. 高斯模糊以获得更平滑的密度分布（调整参数适应小矩阵）
        self._gaussian_blur_density()

    def _gaussian_blur_density(self, sigma=5.0):  # 缩小sigma适应小矩阵
        """对密度矩阵进行高斯模糊，使分布更平滑"""
        # 创建高斯核（适应200x200矩阵的大小）
        kernel_size = int(2 * round(2 * sigma) + 1)
        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        # 扩展维度以适应卷积操作
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        density_input = self.density.view(1, 1, DENSITY_MATRIX_SIZE, DENSITY_MATRIX_SIZE)

        # 执行卷积
        blurred = torch.nn.functional.conv2d(
            density_input,
            kernel,
            padding=kernel_size // 2,
            groups=1
        )

        self.density = blurred.view(DENSITY_MATRIX_SIZE, DENSITY_MATRIX_SIZE)

    def get_density(self):
        """获取当前密度矩阵的副本"""
        return self.density.clone()

    def _initialize_directions(self):
        """初始化智能体的初始运动方向（随机方向）"""
        angles = torch.FloatTensor(self.num_agents).uniform_(0, 2 * np.pi).to(device)
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    def _precompute_expanded_map(self):
        """预计算带安全距离的膨胀地图"""
        self.expanded_map = self.map.clone()
        obstacle_points = torch.nonzero(self.map == 1.0).float()

        if len(obstacle_points) > 0:
            # 只膨胀一次，范围为安全距离
            for dx in range(-int(SAFE_DISTANCE // 2), int(SAFE_DISTANCE // 2) + 1):
                for dy in range(-int(SAFE_DISTANCE // 2), int(SAFE_DISTANCE // 2) + 1):
                    if dx == 0 and dy == 0:
                        continue
                    shifted = obstacle_points + torch.tensor([float(dx), float(dy)], device=device, dtype=torch.float32)
                    shifted = shifted.long()
                    valid = (shifted[:, 0] >= 0) & (shifted[:, 0] < self.map_size) & \
                            (shifted[:, 1] >= 0) & (shifted[:, 1] < self.map_size)
                    valid_points = shifted[valid]
                    self.expanded_map[valid_points[:, 0], valid_points[:, 1]] = 1.0

    def _initialize_positions(self):
        positions = torch.zeros((self.num_agents, 2), device=device, dtype=torch.float32)
        valid = torch.zeros(self.num_agents, device=device, dtype=torch.bool)

        while not valid.all():
            # 生成浮点随机位置
            x = torch.rand(self.num_agents, device=device) * self.map_size
            y = torch.rand(self.num_agents, device=device) * self.map_size

            mask = ~valid
            x_valid = x[mask]
            y_valid = y[mask]
            # 检查位置是否有效（转换为整数索引检查地图）
            is_valid = self.expanded_map[x_valid.long(), y_valid.long()] == 0.0

            positions[mask] = torch.stack([x[mask], y[mask]], dim=1)
            valid[mask] = is_valid

        return positions

    def _initialize_target_ids(self):
        return torch.multinomial(self.target_heat, self.num_agents, replacement=True)

    @staticmethod
    def point_to_segment_distance(points, seg_start, seg_end):
        """点到线段距离计算方法"""
        seg_vec = seg_end - seg_start  # (2,)
        point_vec = points - seg_start  # (N, 2)
        seg_len_sq = torch.dot(seg_vec, seg_vec)  # 标量

        # 处理线段退化为点的情况
        if torch.isclose(seg_len_sq, torch.tensor(0.0, device=device)):
            dists = torch.norm(point_vec, dim=1)  # (N,)
            dir_vecs = point_vec / (dists.unsqueeze(1) + 1e-6)  # (N, 2)
            dir_vecs[dists == 0] = 0  # 避免除零
            return dists, dir_vecs

        # 计算投影参数t
        t = torch.clamp(torch.sum(point_vec * seg_vec, dim=1) / seg_len_sq, 0.0, 1.0)  # (N,)
        # 计算投影点
        projection = seg_start + t.unsqueeze(1) * seg_vec.unsqueeze(0)  # (N, 2)
        # 计算距离向量和方向
        distance_vec = points - projection  # (N, 2)
        dists = torch.norm(distance_vec, dim=1)  # (N,)
        dir_vecs = distance_vec / (dists.unsqueeze(1) + 1e-6)  # (N, 2)
        dir_vecs[dists == 0] = 0  # 避免除零

        return dists, dir_vecs

    def check_stagnation(self, current_distances):
        batch_idx = torch.arange(self.num_agents, device=device)
        self.previous_distances[batch_idx, self.step_counter] = current_distances
        self.step_counter = (self.step_counter + 1) % MAX_STEP

        full_history = self.step_counter == 0
        stagnated = torch.zeros(self.num_agents, device=device, dtype=torch.bool)

        if full_history.any():
            initial_distances = self.previous_distances[full_history, 0]
            current = current_distances[full_history]
            distance_reduction = initial_distances - current
            stagnated[full_history] = distance_reduction < self.min_movement_threshold

        return stagnated

    def update_targets(self, stagnated_mask):
        if stagnated_mask.any():
            num_stagnated = stagnated_mask.sum().item()
            new_target_ids = torch.multinomial(self.target_heat, num_stagnated, replacement=True)
            self.target_ids[stagnated_mask] = new_target_ids
            self.targets[stagnated_mask] = self.all_targets[new_target_ids]
            self.previous_distances[stagnated_mask] = 0.0
            self.step_counter[stagnated_mask] = 0

    def fast_obstacle_check(self, positions):
        """快速检查位置是否在障碍物或安全距离内"""
        x = torch.clamp(torch.round(positions[:, 0]).long(), 0, int(self.map_size) - 1)
        y = torch.clamp(torch.round(positions[:, 1]).long(), 0, int(self.map_size) - 1)
        return self.expanded_map[x, y] == 0.0

    def predict_collisions(self, current_pos, move_dir, speed):
        """
        预判式碰撞检测：检查下一步及附近位置是否会碰撞
        遇到障碍物时在原方向的(pi/6, pi/1.5)范围内逐渐增大角度尝试10个方向
        """
        # 计算下一步位置
        next_pos = current_pos + move_dir * speed.unsqueeze(1)

        # 基础检查
        valid = self.fast_obstacle_check(next_pos)

        # 对可能碰撞的智能体进行多方向检查
        if not valid.all():
            # 只处理可能碰撞的智能体，减少计算量
            collision_risk = ~valid
            num_risk = collision_risk.sum().item()

            if num_risk > 0:
                # 提取有碰撞风险的智能体
                risk_pos = current_pos[collision_risk]
                risk_dir = move_dir[collision_risk]
                risk_speed = speed[collision_risk]
                num_risk_agents = risk_pos.shape[0]

                # 初始化新方向为原方向，标记是否找到有效方向
                new_dirs = risk_dir.clone()
                found_valid = torch.zeros(num_risk_agents, device=device, dtype=torch.bool)

                # 计算角度增量（从最小到最大均匀分布）
                angle_increment = (ANGLE_RANGE_END - ANGLE_RANGE_START) / (NUM_ATTEMPTS - 1)

                # 尝试NUM_ATTEMPTS次不同方向，角度逐渐增大
                for attempt in range(NUM_ATTEMPTS):
                    # 当前尝试的角度偏移量（逐渐增大）
                    current_angle = ANGLE_RANGE_START + attempt * angle_increment

                    # 计算原方向的角度
                    original_angles = torch.atan2(risk_dir[:, 1], risk_dir[:, 0])

                    angle_offsets = current_angle

                    # 计算新角度
                    new_angles = original_angles + angle_offsets

                    # 计算新方向向量
                    attempt_dirs = torch.stack([
                        torch.cos(new_angles),
                        torch.sin(new_angles)
                    ], dim=1)

                    # 计算尝试位置
                    attempt_pos = risk_pos + attempt_dirs * risk_speed.unsqueeze(1)

                    # 检查该位置是否有效
                    attempt_valid = self.fast_obstacle_check(attempt_pos)

                    # 更新尚未找到有效方向的智能体
                    not_found = ~found_valid
                    if not_found.any():
                        # 对尚未找到有效方向的智能体更新方向
                        new_dirs[not_found] = torch.where(
                            attempt_valid[not_found].unsqueeze(1),
                            attempt_dirs[not_found],
                            new_dirs[not_found]
                        )
                        # 更新找到有效方向的标记
                        found_valid[not_found] = found_valid[not_found] | attempt_valid[not_found]

                    # 如果所有智能体都找到有效方向，提前退出循环
                    if found_valid.all():
                        break

                # 更新方向向量
                move_dir[collision_risk] = new_dirs

                # 计算新位置
                next_pos[collision_risk] = risk_pos + move_dir[collision_risk] * risk_speed.unsqueeze(1)

                # 最终检查，确保不会进入障碍物
                final_valid = self.fast_obstacle_check(next_pos[collision_risk])
                # 只有所有尝试都失败的智能体才保持原地
                next_pos[collision_risk] = torch.where(
                    final_valid.unsqueeze(1),
                    next_pos[collision_risk],
                    risk_pos  # 所有方向都无效，保持原地
                )

        return next_pos, move_dir

    def move_towards_target(self, emergency: tuple = (None, None), traffic_light: list = None):
        """恢复红绿灯斥力场和原始疏散处理算法，并加入惯性特性"""
        # 事故疏散处理
        eme_pos, eme_lev = emergency
        SEM = False
        if eme_lev and max(eme_lev) >= 4:
            SEM = True

        # 计算基础方向和距离
        direction = self.targets - self.positions
        current_distances = torch.norm(direction, dim=1)

        # 检查停滞并切换目标
        stagnated = self.check_stagnation(current_distances)
        far_from_target = current_distances > self.speeds * target_radius
        need_new_target = stagnated & far_from_target
        self.update_targets(need_new_target)

        # 重新计算方向
        direction = self.targets - self.positions
        current_distances = torch.norm(direction, dim=1)

        # 到达目标附近处理
        near_target = current_distances < self.speeds * 5.0
        if near_target.any():
            if SEM:
                new_target_ids = torch.multinomial(self.target_heat, near_target.sum().item(), replacement=True)
                self.target_ids[near_target] = new_target_ids
                self.targets[near_target] = self.all_targets[new_target_ids]
                self.previous_distances[near_target] = 0.0
                self.step_counter[near_target] = 0
            else:
                change_prob = 0.002
                n_near = near_target.sum().item()
                temp_mask = torch.rand(n_near, device=device) < change_prob
                change_mask = torch.zeros_like(near_target, dtype=torch.bool)
                change_mask[near_target] = temp_mask

                if change_mask.any():
                    new_target_ids = torch.multinomial(self.target_heat, change_mask.sum().item(), replacement=True)
                    self.target_ids[change_mask] = new_target_ids
                    self.targets[change_mask] = self.all_targets[new_target_ids]
                    self.previous_distances[change_mask] = 0.0
                    self.step_counter[change_mask] = 0

        # 基础方向向量
        direction_normalized = direction / (current_distances.unsqueeze(1) + 1e-6)

        # 红灯缺口斥力场算法
        if traffic_light and not SEM:
            for idx, state in enumerate(traffic_light):
                if state == 1:  # 红灯状态
                    for obstacle in self.obstacle_gen.obstacles:
                        if idx in obstacle.gap_center_map:
                            gap_center = torch.tensor(obstacle.gap_center_map[idx], device=device, dtype=torch.float32)
                            road_width = obstacle.width
                            seg_length = (self.obstacle_gen.intersection_gap_size / 2.0) + 2.0
                            dir_vec = torch.tensor([obstacle.dir_x, obstacle.dir_y], device=device, dtype=torch.float32)

                            seg_start = gap_center - seg_length * dir_vec
                            seg_end = gap_center + seg_length * dir_vec

                            dist_to_seg, repel_dir = self.point_to_segment_distance(
                                self.positions, seg_start, seg_end
                            )

                            repel_radius = road_width
                            in_range = (dist_to_seg > 0.0) & (dist_to_seg < repel_radius)
                            if in_range.any():
                                repel_strength = 2.5 * (1.0 - dist_to_seg[in_range] / repel_radius)
                                direction_normalized[in_range] += repel_dir[in_range] * repel_strength.unsqueeze(1)

        # 疏散处理算法
        if eme_lev and eme_pos is not None and len(eme_pos) > 0:
            eme_pos_tensor = torch.tensor(eme_pos, device=device, dtype=torch.float32)
            eme_lev_tensor = torch.tensor(eme_lev, device=device, dtype=torch.float32)

            for i in range(len(eme_pos)):
                pos = eme_pos_tensor[i]
                lev = eme_lev_tensor[i]

                dist_to_eme = torch.norm(self.positions - pos, dim=1)  # (N,)
                safe_dist = dist_to_eme > 0.0
                if safe_dist.any():
                    evacuate_dir = (self.positions[safe_dist] - pos) / dist_to_eme[safe_dist].unsqueeze(1)
                    evacuate_strength = (lev * 2.0 + 0.1) ** evacuate_rate / (dist_to_eme[safe_dist] ** 2 + 1e-6)
                    direction_normalized[safe_dist] += evacuate_dir * evacuate_strength.unsqueeze(1)

        # 归一化方向向量（融合了目标方向和斥力等）
        dir_norms = torch.norm(direction_normalized, dim=1, keepdim=False)
        non_zero_dir = dir_norms > 0.0
        direction_normalized[non_zero_dir] = direction_normalized[non_zero_dir] / dir_norms[non_zero_dir].unsqueeze(1)

        # 添加随机偏移
        angle_offset = torch.FloatTensor(self.num_agents).uniform_(
            -np.pi * 0.5, np.pi * 0.5
        ).to(device) * self.randomness

        cos_theta = torch.cos(angle_offset)
        sin_theta = torch.sin(angle_offset)

        # 旋转方向向量
        rotated_dir_x = direction_normalized[:, 0] * cos_theta - direction_normalized[:, 1] * sin_theta
        rotated_dir_y = direction_normalized[:, 0] * sin_theta + direction_normalized[:, 1] * cos_theta
        new_direction = torch.stack([rotated_dir_x, rotated_dir_y], dim=1)

        # 核心：加入惯性 - 融合当前方向和新计算方向
        # 惯性权重越高，当前方向的影响越大
        inertia_weight = self.inertia_weights.unsqueeze(1)  # 形状变为 [N, 1] 以便广播
        move_direction = inertia_weight * self.current_directions + (1.0 - inertia_weight) * new_direction

        # 归一化最终方向向量
        move_dir_norms = torch.norm(move_direction, dim=1, keepdim=True)
        move_direction = move_direction / (move_dir_norms + 1e-6)

        # 预判碰撞并修正路径（包含逐渐增大角度尝试）
        next_positions, move_direction = self.predict_collisions(
            self.positions,
            move_direction,
            self.speeds
        )

        # 更新当前方向（用于下一帧的惯性计算）
        self.current_directions = move_direction

        # 最终位置更新
        self.positions = next_positions

        # 更新密度矩阵
        if self.update_den_step < 0:
            self._update_density()
            self.update_den_step = UPDATE_DENSITY_STEP
        else:
            self.update_den_step -= 1

