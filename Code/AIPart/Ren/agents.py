import numpy as np
import torch
from Ren.config import device, DENSITY_MATRIX_SIZE, MAX_STEP, UPDATE_DENSITY_STEP, MIN_SPEED, MAX_SPEED, \
    INERTIA_WEIGHT_RANGE, SAFE_DISTANCE, AGENT_REPULSION_RADIUS, AGENT_REPULSION_STRENGTH, ANGLE_RANGE_END, \
    ANGLE_RANGE_START, NUM_ATTEMPTS, target_radius, CHANGE_PROB, evacuate_rate, TRAIN_SET, MODEL_DIR_WEIGHT, USE_MODEL
from Ren.model import get_correction_vectors, initialize_agent


class AgentGroup:
    def __init__(self,
                 num_agents,
                 obstacle_gen,
                 targets,
                 targets_heat
                 ):
        self.num_agents = num_agents
        self.map = torch.tensor(obstacle_gen.map_matrix, device=device, dtype=torch.float32)
        self.map_size = float(obstacle_gen.map_matrix.shape[0])
        # self.intersection_id_map = intersection_id_map
        self.obstacle_gen = obstacle_gen

        # 密度矩阵缩放因子
        self.density_scale = DENSITY_MATRIX_SIZE / self.map_size  # 新增：计算缩放因子

        # 初始化智能体属性为空
        self.positions = torch.zeros((0, 2), device=device, dtype=torch.float32)
        self.target_ids = torch.zeros(0, device=device, dtype=torch.long)
        self.targets = torch.zeros((0, 2), device=device, dtype=torch.float32)
        self.speeds = torch.zeros(0, device=device, dtype=torch.float32)
        self.randomness = torch.zeros(0, device=device, dtype=torch.float32)
        self.sizes = torch.zeros(0, device=device, dtype=torch.int32)
        self.colors = torch.zeros((0, 3), device=device, dtype=torch.float32)

        # 惯性相关属性
        self.inertia_weights = torch.zeros(0, device=device, dtype=torch.float32)
        self.current_directions = torch.zeros((0, 2), device=device, dtype=torch.float32)

        # 停滞判断参数
        self.previous_distances = torch.zeros((0, MAX_STEP), device=device, dtype=torch.float32)
        self.step_counter = torch.zeros(0, device=device, dtype=torch.int32)
        self.min_movement_threshold = 5.0  # 改为float

        # 目标点信息（float类型）
        self.all_targets = torch.tensor(targets, device=device, dtype=torch.float32)
        self.target_heat = torch.tensor(targets_heat, device=device, dtype=torch.float32)
        self.target_heat = self.target_heat / self.target_heat.sum()

        # 预计算膨胀地图
        self.expanded_map = self._precompute_expanded_map()

        # 密度矩阵相关（修改为200x200）
        self.density = torch.zeros((DENSITY_MATRIX_SIZE, DENSITY_MATRIX_SIZE), device=device, dtype=torch.float32)
        self.update_den_step = UPDATE_DENSITY_STEP

        # 根据节点热度初始化智能体
        self.initialize_by_heat(num_agents)

        self.model = initialize_agent(
            obs=self.obstacle_gen.obstacles,
            points_coords=self.obstacle_gen.intersection_points,
            agents_pos=self.positions,
            targets_coords=self.targets,
            train_first=TRAIN_SET
        )

        self.model_direction = get_correction_vectors(
            self.model,
            obs=self.obstacle_gen.obstacles,
            points_coords=self.obstacle_gen.intersection_points,
            agents_pos=self.positions,
            targets_coords=self.targets,
        )


    def initialize_by_heat(self, num_agents):
        """根据目标点热度分布智能体"""
        # 根据热度选择目标点
        target_indices = torch.multinomial(self.target_heat, num_agents, replacement=True)

        # 为每个目标点分配智能体数量
        counts = torch.bincount(target_indices, minlength=len(self.all_targets))

        # 在每个目标点周围生成相应数量的智能体
        for i, count in enumerate(counts):
            if count > 0:
                target_pos = self.all_targets[i]
                # 根据目标热度设置生成半径，热度高的目标周围半径更大
                radius = target_radius / 2.0 + self.target_heat[i] * 10.0
                self.birth(target_pos[0], target_pos[1], radius, count.item())

        # 初始化密度矩阵
        self._update_density()

    def birth(self, x, y, radius, num):
        """在(x, y)位置周围radius半径范围内生成num个智能体"""
        if num <= 0:
            return

        new_positions = torch.zeros((num, 2), device=device, dtype=torch.float32)
        valid = torch.zeros(num, device=device, dtype=torch.bool)

        # 生成有效的智能体位置
        while not valid.all():
            # 在圆形区域内生成随机点 - 修复设备参数问题
            angles = torch.empty(num, device=device).uniform_(0, 2 * np.pi)
            radii = torch.empty(num, device=device).uniform_(0, radius)

            new_x = x + radii * torch.cos(angles)
            new_y = y + radii * torch.sin(angles)

            # 确保在地图范围内
            new_x = torch.clamp(new_x, 0, self.map_size)
            new_y = torch.clamp(new_y, 0, self.map_size)

            mask = ~valid
            x_valid = new_x[mask]
            y_valid = new_y[mask]

            # 检查位置是否有效（不在障碍物上）
            is_valid = self.expanded_map[x_valid.long(), y_valid.long()] == 0.0

            new_positions[mask, 0] = x_valid
            new_positions[mask, 1] = y_valid
            valid[mask] = is_valid

        # 为新智能体生成属性 - 修复设备参数问题
        new_target_ids = torch.multinomial(self.target_heat, num, replacement=True)
        new_targets = self.all_targets[new_target_ids]
        new_speeds = torch.empty(num, device=device).uniform_(MIN_SPEED, MAX_SPEED)
        new_randomness = torch.empty(num, device=device).uniform_(0.1, 0.3)
        new_sizes = torch.randint(15, 30, (num,), device=device)
        new_colors = torch.rand((num, 3), device=device)

        # 惯性相关属性 - 修复设备参数问题
        new_inertia_weights = torch.empty(num, device=device).uniform_(
            INERTIA_WEIGHT_RANGE[0], INERTIA_WEIGHT_RANGE[1]
        )
        new_directions = self._initialize_directions(num)

        # 停滞判断相关
        new_previous_distances = torch.zeros((num, MAX_STEP), device=device, dtype=torch.float32)
        new_step_counter = torch.zeros(num, device=device, dtype=torch.int32)

        # 将新智能体添加到现有群体
        self.positions = torch.cat([self.positions, new_positions], dim=0)
        self.target_ids = torch.cat([self.target_ids, new_target_ids], dim=0)
        self.targets = torch.cat([self.targets, new_targets], dim=0)
        self.speeds = torch.cat([self.speeds, new_speeds], dim=0)
        self.randomness = torch.cat([self.randomness, new_randomness], dim=0)
        self.sizes = torch.cat([self.sizes, new_sizes], dim=0)
        self.colors = torch.cat([self.colors, new_colors], dim=0)

        self.inertia_weights = torch.cat([self.inertia_weights, new_inertia_weights], dim=0)
        self.current_directions = torch.cat([self.current_directions, new_directions], dim=0)

        self.previous_distances = torch.cat([self.previous_distances, new_previous_distances], dim=0)
        self.step_counter = torch.cat([self.step_counter, new_step_counter], dim=0)

        # 更新智能体数量
        self.num_agents = self.positions.shape[0]

    def kill(self, x, y, radius, num):
        """在(x, y)位置周围radius半径范围内删除num个智能体"""
        if num <= 0 or self.num_agents <= 0:
            return

        # 计算所有智能体到目标点的距离
        dx = self.positions[:, 0] - x
        dy = self.positions[:, 1] - y
        distances = torch.sqrt(dx ** 2 + dy ** 2)

        # 找到在范围内的智能体
        in_range = distances < radius
        if not in_range.any():
            return

        # 选择要删除的智能体（最多num个）
        num_to_kill = min(num, in_range.sum().item())
        to_kill = torch.where(in_range)[0][:num_to_kill]

        # 保留不删除的智能体
        mask = torch.ones(self.num_agents, device=device, dtype=torch.bool)
        mask[to_kill] = False

        self.positions = self.positions[mask]
        self.target_ids = self.target_ids[mask]
        self.targets = self.targets[mask]
        self.speeds = self.speeds[mask]
        self.randomness = self.randomness[mask]
        self.sizes = self.sizes[mask]
        self.colors = self.colors[mask]

        self.inertia_weights = self.inertia_weights[mask]
        self.current_directions = self.current_directions[mask]

        self.previous_distances = self.previous_distances[mask]
        self.step_counter = self.step_counter[mask]

        # 更新智能体数量
        self.num_agents = self.positions.shape[0]

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

    def _initialize_directions(self, num):
        """初始化智能体的初始运动方向（随机方向）"""
        # 修复设备参数问题
        angles = torch.empty(num, device=device).uniform_(0, 2 * np.pi)
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    def _precompute_expanded_map(self):
        """预计算带安全距离的膨胀地图"""
        expanded_map = self.map.clone()
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
                    expanded_map[valid_points[:, 0], valid_points[:, 1]] = 1.0
        return expanded_map


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

    def calculate_agent_repulsion(self):
        """计算智能体之间的斥力"""
        repulsion = torch.zeros_like(self.positions, device=device)

        # 如果智能体数量很少，直接返回
        if self.num_agents < 2:
            return repulsion

        # 计算所有智能体之间的相对位置
        pos_diff = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)  # 形状: [N, N, 2]
        distances = torch.norm(pos_diff, dim=2)  # 形状: [N, N]

        # 只考虑距离小于阈值的智能体对，且排除自身
        mask = (distances < AGENT_REPULSION_RADIUS) & (distances > 0)

        if mask.any():
            # 计算归一化方向和斥力强度（与距离成反比）
            directions = pos_diff[mask] / (distances[mask].unsqueeze(1) + 1e-6)
            strengths = AGENT_REPULSION_STRENGTH * (1 - distances[mask] / AGENT_REPULSION_RADIUS)

            # 累加斥力
            repulsion.index_add_(0,
                                 torch.where(mask)[0],
                                 directions * strengths.unsqueeze(1))

        return repulsion

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
        """恢复红绿灯斥力场和原始疏散处理算法，并加入惯性特性和智能体间斥力"""
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
                n_near = near_target.sum().item()
                temp_mask = torch.rand(n_near, device=device) < CHANGE_PROB
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

        # 计算智能体间斥力并添加到方向向量
        agent_repulsion = self.calculate_agent_repulsion()
        direction_normalized += agent_repulsion

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

                            repel_radius = road_width + 2
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

        # 归一化方向向量（融合了目标方向、斥力等）
        dir_norms = torch.norm(direction_normalized, dim=1, keepdim=False)
        non_zero_dir = dir_norms > 0.0
        direction_normalized[non_zero_dir] = direction_normalized[non_zero_dir] / dir_norms[non_zero_dir].unsqueeze(1)

        # 添加随机偏移 - 修复设备参数问题
        angle_offset = torch.empty(self.num_agents, device=device).uniform_(
            -np.pi * 0.5, np.pi * 0.5
        ) * self.randomness

        cos_theta = torch.cos(angle_offset)
        sin_theta = torch.sin(angle_offset)

        # 旋转方向向量
        rotated_dir_x = direction_normalized[:, 0] * cos_theta - direction_normalized[:, 1] * sin_theta
        rotated_dir_y = direction_normalized[:, 0] * sin_theta + direction_normalized[:, 1] * cos_theta
        new_direction = torch.stack([rotated_dir_x, rotated_dir_y], dim=1)

        # 模型预测方向
        if self.update_den_step < 0:
            self.model_direction = get_correction_vectors(
                self.model,
                obs=self.obstacle_gen.obstacles,
                points_coords=self.obstacle_gen.intersection_points,
                agents_pos=self.positions,
                targets_coords=self.targets,
            )
            self.update_den_step = UPDATE_DENSITY_STEP
        else:
            self.update_den_step -= 1

        # 加入模型预测
        if USE_MODEL:
            # 确保model_direction是CUDA张量并与new_direction在同一设备上
            if isinstance(self.model_direction, list):
                self.model_direction = torch.stack(self.model_direction).to(device)
            # print("model_direction shape:", self.model_direction.shape)  # 应是 [N, 2]
            # print("new_direction shape before fusion:", new_direction.shape)  # 应是 [N, 2]
            if self.model_direction.shape[0] == new_direction.shape[0]:
                new_direction = new_direction * (1 - MODEL_DIR_WEIGHT) + MODEL_DIR_WEIGHT * self.model_direction
            # print("DEBUG1 PASS")
            # print("new_direction shape after fusion:", new_direction.shape)  # 必须是 [N, 2]




        # 加入惯性 - 融合当前方向和新计算方向
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
            # 更新训练模型
            if TRAIN_SET:
                self.model = initialize_agent(
                    obs=self.obstacle_gen.obstacles,
                    points_coords=self.obstacle_gen.intersection_points,
                    agents_pos=self.positions,
                    targets_coords=self.targets,
                    train_first=TRAIN_SET
                )
        else:
            self.update_den_step -= 1


