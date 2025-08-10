import numpy as np
import random
from testmap import obstacles_params, targets



class Obstacle:
    """障碍物类，存储单个障碍物的信息"""
    def __init__(self, start_point, end_point, width, gap_offset_ratio=1.0):
        self.start_point = start_point
        self.end_point = end_point
        self.width = width
        self.gap_offset_ratio = gap_offset_ratio
        self.base_line = self.bresenham(*start_point, *end_point)
        self.length = len(self.base_line)

        # 计算方向向量
        x0, y0 = start_point
        x1, y1 = end_point
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx **2 + dy** 2) if (dx != 0 or dy != 0) else 1

        # 单位方向向量与垂直向量
        self.dir_x = dx / length
        self.dir_y = dy / length
        self.perp_dir_x = -self.dir_y
        self.perp_dir_y = self.dir_x

        self.auto_gaps = []

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
    """障碍物生成器，支持批量生成障碍物并预计算所有交点"""
    def __init__(self, map_size):
        self.map_size = map_size
        self.map_matrix = np.zeros((map_size, map_size), dtype=int)  # 0:可通行 1:障碍物
        self.obstacles = []
        self.intersection_points = []

    def add_obstacles(self, obstacles_params, intersection_gap_size=12,
                      gap_positions=None, gap_widths=None):
        """批量添加障碍物，预先计算所有交点后再生成所有障碍物"""
        if gap_positions is None:
            gap_positions = [[] for _ in range(len(obstacles_params))]
        if gap_widths is None:
            gap_widths = [[20] * len(gaps) for gaps in gap_positions]

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

        # 预计算交点和自动缺口
        for i, obstacle in enumerate(temp_obstacles):
            for j, other_obstacle in enumerate(temp_obstacles):
                if i != j:
                    intersection_point, pos_ratio = self.find_intersection(obstacle, other_obstacle)
                    if intersection_point:
                        offset_distance = obstacle.gap_offset_ratio * max(obstacle.width, other_obstacle.width)
                        x0, y0 = obstacle.start_point
                        x1, y1 = obstacle.end_point
                        line_length = np.sqrt((x1 - x0) **2 + (y1 - y0)** 2) if (x1 != x0 or y1 != y0) else 1

                        if line_length > 0:
                            offset_ratio = offset_distance / line_length
                            if pos_ratio + offset_ratio <= 1.0:
                                obstacle.auto_gaps.append(pos_ratio + offset_ratio)
                            if pos_ratio - offset_ratio >= 0.0:
                                obstacle.auto_gaps.append(pos_ratio - offset_ratio)

                        if intersection_point not in self.intersection_points:
                            self.intersection_points.append(intersection_point)

        # 生成实际障碍和缺口
        for idx, obstacle in enumerate(temp_obstacles):
            self._generate_obstacle(
                obstacle,
                gap_positions[idx],
                gap_widths[idx],
                intersection_gap_size
            )
            self.obstacles.append(obstacle)

    def _generate_obstacle(self, obstacle, gap_positions, gap_widths, intersection_gap_size):
        """生成单个障碍物的实际障碍和缺口"""
        all_gap_positions = gap_positions + obstacle.auto_gaps
        all_gap_widths = gap_widths + [intersection_gap_size] * len(obstacle.auto_gaps)

        # 生成带宽度的障碍物点集
        obstacle_points = set()
        for (x, y) in obstacle.base_line:
            for i in range(-obstacle.width // 2, obstacle.width // 2 + 1):
                for j in range(-obstacle.width // 2, obstacle.width // 2 + 1):
                    if i **2 + j** 2 <= (obstacle.width // 2) **2:  # 圆形扩展
                        nx = int(x + i * obstacle.perp_dir_x - j * obstacle.dir_x)
                        ny = int(y + i * obstacle.perp_dir_y - j * obstacle.dir_y)
                        obstacle_points.add((nx, ny))

        # 移除缺口点
        for gap_pos, gap_width in zip(all_gap_positions, all_gap_widths):
            gap_idx = int(len(obstacle.base_line) * gap_pos)
            gap_idx = max(0, min(len(obstacle.base_line) - 1, gap_idx))
            gap_x, gap_y = obstacle.base_line[gap_idx]

            gap_points = set()
            for i in range(-gap_width // 2, gap_width // 2 + 1):
                for j in range(-int(gap_width * 0.6), int(gap_width * 0.6) + 1):
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


# # 目标点坐标（测试地图关键位置）
# targets = np.array([
#     [40, 50],    # 靠近陆路
#     [116, 50],   # 靠近小路
#     [220, 220],  # 靠近凤路
#     [40, 61],    # 靠近希路（修正为整数坐标）
#     [116, 61],   # 靠近文路
#     [220, 61],   # 靠近锐路
#     [116, 128],  # 靠近争路
#     [220, 98]    # 靠近英路
# ])
#
# # 测试地图的障碍物参数（1/10缩放后）
# obstacles_params =


# 智能体类（核心修改：使用绝对距离判断停滞）
class Agent:
    def __init__(self, map_matrix):
        self.map = map_matrix
        self.map_size = map_matrix.shape[0]

        # 随机初始位置（非障碍物区域）
        while True:
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            if self.map[x, y] == 0:
                self.position = np.array([x, y], dtype=np.float64)
                break

        # 初始参数
        self.target = random.choice(targets)
        self.speed = random.uniform(0.3, 1.0)
        self.randomness = random.uniform(0.1, 0.3)
        self.size = random.randint(15, 30)
        self.color = (random.random(), random.random(), random.random())

        # 停滞判断参数（绝对距离版）
        self.previous_distances = []  # 存储最近的距离记录
        self.max_history = 30  # 检测窗口大小（步数）
        self.min_movement_threshold = 5.0  # 最小移动距离阈值（绝对距离）
        # 目标过远时放宽阈值（按距离比例动态调整）
        self.distance_ratio_factor = 0.1


    def is_valid_position(self, x, y):
        """检查位置是否合法（在地图内且非障碍物）"""
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            grid_x = int(round(x))
            grid_y = int(round(y))
            return self.map[grid_x, grid_y] == 0
        return False


    def check_stagnation(self, current_distance):
        """
        绝对距离判断停滞：
        若在检测窗口内，向目标移动的距离小于阈值，则判定为停滞
        阈值会根据初始距离动态调整（远距离目标允许更大的最小移动距离）
        """
        self.previous_distances.append(current_distance)

        # 保持窗口大小
        if len(self.previous_distances) > self.max_history:
            self.previous_distances.pop(0)

        # 窗口填满后才判断
        if len(self.previous_distances) == self.max_history:
            initial_distance = self.previous_distances[0]  # 窗口起始时的距离
            distance_reduction = initial_distance - current_distance  # 向目标移动的距离

            # 动态计算阈值：基础阈值 + 初始距离的一定比例（适应远距离目标）
            dynamic_threshold = self.min_movement_threshold + (initial_distance * self.distance_ratio_factor)

            # 若移动距离小于动态阈值，判定为停滞
            if distance_reduction < dynamic_threshold:
                return True
        return False


    def move_towards_target(self, target_radius=30):
        # 计算方向和距离
        direction = self.target - self.position
        current_distance = np.linalg.norm(direction)

        # 检查停滞：若停滞且距离目标较远，则切换目标
        if self.check_stagnation(current_distance) and current_distance > self.speed * target_radius:
            # 选择不同的新目标
            possible_targets = [t for t in targets if not np.array_equal(t, self.target)]
            self.target = random.choice(possible_targets)
            self.previous_distances = []  # 重置记录
            direction = self.target - self.position
            current_distance = np.linalg.norm(direction)

        # 到达目标附近时，小概率切换目标
        if current_distance < self.speed * 5.0:
            if random.random() < 0.002:
                self.target = random.choice(targets)
                self.previous_distances = []
            return

        # 基础移动方向（朝向目标）
        direction_normalized = direction / current_distance

        # 添加随机偏移
        angle_offset = random.uniform(-np.pi * self.randomness, np.pi * self.randomness) * 0.5
        cos_theta = np.cos(angle_offset)
        sin_theta = np.sin(angle_offset)
        rotated_dir_x = direction_normalized[0] * cos_theta - direction_normalized[1] * sin_theta
        rotated_dir_y = direction_normalized[0] * sin_theta + direction_normalized[1] * cos_theta

        move_direction = np.array([rotated_dir_x, rotated_dir_y])
        move_direction = move_direction / np.linalg.norm(move_direction)

        # 计算下一步位置
        next_position = self.position + move_direction * self.speed

        # 避障逻辑
        if self.is_valid_position(next_position[0], next_position[1]):
            self.position = next_position
        else:
            # 尝试当前方向附近的随机方向避障
            current_angle = np.arctan2(move_direction[1], move_direction[0])
            for _ in range(5):
                relative_angle = random.uniform(-np.pi * 3/5, 0 * np.pi / 2)
                angle = current_angle + relative_angle
                avoid_dir = np.array([np.cos(angle), np.sin(angle)])
                new_next_pos = self.position + avoid_dir * self.speed
                if self.is_valid_position(new_next_pos[0], new_next_pos[1]):
                    self.position = new_next_pos
                    return

