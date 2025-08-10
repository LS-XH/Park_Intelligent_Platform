import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import matplotlib
from math import log10
from testmap import obstacles_params, targets, targets_heat, points

# 配置交互后端
matplotlib.use('TkAgg')
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 地图参数
MAP_SIZE = 700
num_agents = 300
target_radius = 30
emergency = [
    ((100 * 3, 50 * 3), 1),
    ((100 * 3, 100 * 3), 2),
]
MIN_SPEED, MAX_SPEED = 0.3, 1
MAX_STEP = 50
evacuate_rate = log10(MAP_SIZE)  # 疏散速率参数



class Obstacle:
    """障碍物类，存储单个障碍物信息及预计算的缺口中心（与交点ID绑定）"""

    def __init__(self, start_point, end_point, width, gap_offset_ratio=1.0):
        self.start_point = start_point
        self.end_point = end_point
        self.width = width
        self.gap_offset_ratio = gap_offset_ratio
        self.base_line = self.bresenham(*start_point, *end_point)
        self.length = len(self.base_line)
        # 存储缺口中心与对应交点ID的映射 {id: (x, y)}
        self.gap_center_map = {}
        self.auto_gaps = []  # 存储缺口位置比例

        # 计算方向向量
        x0, y0 = start_point
        x1, y1 = end_point
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx ** 2 + dy ** 2) if (dx != 0 or dy != 0) else 1

        # 单位方向向量与垂直向量
        self.dir_x = dx / length
        self.dir_y = dy / length
        self.perp_dir_x = -self.dir_y
        self.perp_dir_y = self.dir_x

    @staticmethod
    def bresenham(x0, y0, x1, y1):
        """Bresenham直线算法，生成两点间的栅格点"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
            points.append((x, y))
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
            points.append((x, y))
        return points


class ObstacleGenerator:
    """障碍物生成器，管理交点ID与坐标映射"""

    def __init__(self, map_size):
        self.map_size = map_size
        self.map_matrix = np.zeros((map_size, map_size), dtype=int)  # 0:可通行 1:障碍物
        self.obstacles = []
        self.intersection_points = points  # 使用预设的交点列表
        # 建立交点ID到坐标的映射（ID为points的索引0-7）
        self.intersection_id_map = {i: np.array(point, dtype=np.float64) for i, point in enumerate(points)}

    def add_obstacles(self, obstacles_params, intersection_gap_size=12):
        """批量添加障碍物，预计算缺口中心并与交点ID绑定"""
        # 创建临时障碍物对象
        temp_obstacles = []
        for params in obstacles_params:
            obstacle = Obstacle(
                start_point=params['start_point'],
                end_point=params['end_point'],
                width=params['width'],
                gap_offset_ratio=params.get('gap_offset_ratio', 1.0)
            )
            temp_obstacles.append(obstacle)

        # 预计算交点和自动缺口位置比例
        for i, obstacle in enumerate(temp_obstacles):
            for j, other_obstacle in enumerate(temp_obstacles):
                if i != j:
                    intersection_point, pos_ratio = self.find_intersection(obstacle, other_obstacle)
                    if intersection_point:
                        # 计算偏移比例
                        offset_distance = obstacle.gap_offset_ratio * max(obstacle.width, other_obstacle.width)
                        x0, y0 = obstacle.start_point
                        x1, y1 = obstacle.end_point
                        line_length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) if (x1 != x0 or y1 != y0) else 1

                        if line_length > 0:
                            offset_ratio = offset_distance / line_length
                            if pos_ratio + offset_ratio <= 1.0:
                                obstacle.auto_gaps.append(pos_ratio + offset_ratio)
                            if pos_ratio - offset_ratio >= 0.0:
                                obstacle.auto_gaps.append(pos_ratio - offset_ratio)

        # 生成实际障碍、缺口并绑定交点ID
        for obstacle in temp_obstacles:
            self._generate_obstacle(obstacle, intersection_gap_size)
            self.obstacles.append(obstacle)

    def _generate_obstacle(self, obstacle, intersection_gap_size):
        """生成单个障碍物并绑定缺口中心与交点ID"""
        # 找出当前障碍物包含的交点ID
        obstacle_point_set = set(obstacle.base_line)
        contained_ids = []
        for idx, point in self.intersection_id_map.items():
            point_tuple = (int(round(point[0])), int(round(point[1])))
            if point_tuple in obstacle_point_set:
                contained_ids.append(idx)

        # 计算并存储缺口中心（与交点ID绑定）
        for gap_pos in obstacle.auto_gaps:
            gap_idx = int(len(obstacle.base_line) * gap_pos)
            gap_idx = max(0, min(len(obstacle.base_line) - 1, gap_idx))
            gap_center = np.array(obstacle.base_line[gap_idx], dtype=np.float64)

            # 找到最近的交点ID并绑定
            min_dist = float('inf')
            best_id = -1
            for idx in contained_ids:
                dist = np.linalg.norm(gap_center - self.intersection_id_map[idx])
                if dist < min_dist:
                    min_dist = dist
                    best_id = idx

            if best_id != -1:
                obstacle.gap_center_map[best_id] = gap_center

        # 生成带宽度的障碍物点集
        obstacle_points = set()
        for (x, y) in obstacle.base_line:
            for i in range(-obstacle.width // 2, obstacle.width // 2 + 1):
                for j in range(-obstacle.width // 2, obstacle.width // 2 + 1):
                    if i ** 2 + j ** 2 <= (obstacle.width // 2) ** 2:  # 圆形扩展
                        nx = int(x + i * obstacle.perp_dir_x - j * obstacle.dir_x)
                        ny = int(y + i * obstacle.perp_dir_y - j * obstacle.dir_y)
                        obstacle_points.add((nx, ny))

        # 移除缺口点
        for gap_center in obstacle.gap_center_map.values():
            gap_x, gap_y = gap_center
            gap_points = set()
            for i in range(-intersection_gap_size // 2, intersection_gap_size // 2 + 1):
                for j in range(-int(intersection_gap_size * 0.6), int(intersection_gap_size * 0.6) + 1):
                    nx = int(gap_x + i * obstacle.perp_dir_x - j * obstacle.dir_x)
                    ny = int(gap_y + i * obstacle.perp_dir_y - j * obstacle.dir_y)
                    gap_points.add((nx, ny))
            obstacle_points -= gap_points

        # 添加到地图
        for x, y in obstacle_points:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self.map_matrix[x, y] = 1

    def find_intersection(self, obstacle1, obstacle2):
        """检测两个障碍物是否相交，并返回交点和在第一个障碍物上的位置比例"""
        line1 = obstacle1.base_line
        line2 = obstacle2.base_line
        line1_set = set(line1)
        line2_set = set(line2)
        intersections = line1_set.intersection(line2_set)

        if intersections:
            intersection_point = next(iter(intersections))
            pos_idx = line1.index(intersection_point)
            pos_ratio = pos_idx / len(line1) if len(line1) > 0 else 0
            return intersection_point, pos_ratio

        return None, 0


class Agent:
    def __init__(self, map_matrix, intersection_id_map, obstacle_gen):
        self.map = map_matrix
        self.map_size = map_matrix.shape[0]
        self.intersection_id_map = intersection_id_map  # 交点ID->坐标映射
        self.obstacle_gen = obstacle_gen  # 障碍物生成器引用

        # 随机初始位置（非障碍物区域）
        while True:
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            if self.map[x, y] == 0:
                self.position = np.array([x, y], dtype=np.float64)
                break

        # 初始参数
        self.target_ids = [idx for idx in range(len(targets))]
        self.target_id = self.select_random_target_by_heat(self.target_ids, targets_heat)
        self.target = targets[self.target_id]
        self.speed = random.uniform(MIN_SPEED, MAX_SPEED)
        self.randomness = random.uniform(0.1, 0.3)
        self.size = random.randint(15, 30)
        self.color = (random.random(), random.random(), random.random())

        # 停滞判断参数
        self.previous_distances = []
        self.max_history = MAX_STEP
        self.min_movement_threshold = 5.0
        self.distance_ratio_factor = 0.1

    def select_random_target_by_heat(self, target_ids, heat_weights):
        """根据热度权重随机选择目标点"""
        if len(target_ids) != len(heat_weights):
            raise ValueError("目标点数量与热度权重数量不匹配")
        total = sum(heat_weights)
        heat_weights = [w / total for w in heat_weights]
        return np.random.choice(target_ids, p=heat_weights)

    def is_valid_position(self, x, y):
        """检查位置是否合法"""
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            grid_x = int(round(x))
            grid_y = int(round(y))
            return self.map[grid_x, grid_y] == 0
        return False

    def check_stagnation(self, current_distance):
        """判断是否停滞"""
        self.previous_distances.append(current_distance)
        if len(self.previous_distances) > self.max_history:
            self.previous_distances.pop(0)
        if len(self.previous_distances) == self.max_history:
            initial_distance = self.previous_distances[0]
            distance_reduction = initial_distance - current_distance
            if distance_reduction < self.min_movement_threshold:
                return True
        return False

    def move_towards_target(self, emergency: tuple = (None, None), traffic_light: list = None):
        """移动逻辑，包含红灯缺口斥力场（使用预绑定的缺口中心）"""
        # 事故疏散处理
        eme_pos, eme_lev = emergency
        SEM = False
        if eme_lev and max(eme_lev) > 4:
            SEM = True

        # 计算基础方向和距离
        direction = self.target - self.position
        current_distance = np.linalg.norm(direction)

        # 检查停滞并切换目标
        if self.check_stagnation(current_distance) and current_distance > self.speed * target_radius:
            possible_indices = [idx for idx in range(len(targets)) if idx != self.target_id]
            possible_heat = [targets_heat[idx] for idx in possible_indices]
            self.target_id = self.select_random_target_by_heat(possible_indices, possible_heat)
            self.target = targets[self.target_id]
            self.previous_distances = []
            direction = self.target - self.position
            current_distance = np.linalg.norm(direction)

        # 到达目标附近处理
        if current_distance < self.speed * 5.0:
            if random.random() < 0.002 or SEM:
                self.target_id = self.select_random_target_by_heat(range(len(targets)), targets_heat)
                self.target = targets[self.target_id]
                self.previous_distances = []
            return

        # 基础方向向量
        direction_normalized = direction / current_distance if current_distance != 0 else np.array([0, 0])

        # 红灯缺口斥力场（使用预绑定的缺口中心）
        if traffic_light is not None:
            for idx, state in enumerate(traffic_light):
                if state == 1:  # 红灯状态
                    # 快速找到该交点ID对应的所有缺口中心
                    for obstacle in self.obstacle_gen.obstacles:
                        if idx in obstacle.gap_center_map:
                            # 直接获取预计算的缺口中心
                            gap_center = obstacle.gap_center_map[idx]
                            road_width = obstacle.width
                            repel_radius = np.sqrt(2) * road_width

                            # 计算斥力
                            dist_to_gap = np.linalg.norm(self.position - gap_center)
                            if 0 < dist_to_gap < repel_radius:
                                repel_dir = (self.position - gap_center) / dist_to_gap
                                repel_strength = 2.5 * (1 - dist_to_gap / repel_radius)
                                direction_normalized += repel_dir * repel_strength

        # 疏散向量
        if eme_lev and eme_pos.size > 0:
            for pos, lev in zip(eme_pos, eme_lev):
                dist_to_eme = np.linalg.norm(pos - self.position)
                if dist_to_eme > 0:
                    evacuate_dir = (self.position - pos) / dist_to_eme
                    evacuate_strength = (lev * 2 + 0.1) ** evacuate_rate / (dist_to_eme ** 2 + 1e-6)
                    direction_normalized += evacuate_dir * evacuate_strength

        # 归一化方向向量
        if np.linalg.norm(direction_normalized) > 0:
            direction_normalized = direction_normalized / np.linalg.norm(direction_normalized)

        # 添加随机偏移
        angle_offset = random.uniform(-np.pi * self.randomness, np.pi * self.randomness) * 0.5
        cos_theta = np.cos(angle_offset)
        sin_theta = np.sin(angle_offset)
        rotated_dir_x = direction_normalized[0] * cos_theta - direction_normalized[1] * sin_theta
        rotated_dir_y = direction_normalized[0] * sin_theta + direction_normalized[1] * cos_theta

        move_direction = np.array([rotated_dir_x, rotated_dir_y])
        move_direction = move_direction / np.linalg.norm(move_direction) if np.linalg.norm(
            move_direction) > 0 else np.array([0, 0])

        # 计算下一步位置
        next_position = self.position + move_direction * self.speed

        # 避障逻辑
        if self.is_valid_position(next_position[0], next_position[1]):
            self.position = next_position
        else:
            current_angle = np.arctan2(move_direction[1], move_direction[0]) if np.linalg.norm(
                move_direction) > 0 else 0
            for _ in range(5):
                relative_angle = random.uniform(-np.pi * 3 / 5, 0)
                angle = current_angle + relative_angle
                avoid_dir = np.array([np.cos(angle), np.sin(angle)])
                new_next_pos = self.position + avoid_dir * self.speed
                if self.is_valid_position(new_next_pos[0], new_next_pos[1]):
                    self.position = new_next_pos
                    return
