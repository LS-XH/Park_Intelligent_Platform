import numpy as np
import torch

from Code.AIPart.Ren.config import PRECISION, device


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

    def __init__(self, map_size, points):
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





