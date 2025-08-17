import json
import math
import os
import random
from enum import Enum

import numpy as np

from Interface.graph import GraphBase, PointType, GateStatus
from graph.common_func import generate_cars_list, generate_people_list

CARS_NUM_THREADS = 30  # 车辆阈值
PEOPLE_NUM_THREADS = 3000  # 人阈值
GREEN_LIGHT_TIMES = 27  # 绿灯时间
YELLOW_LIGHT_TIMES = 3  # 黄灯时间
ALLOW_LIGHT_TIMES = GREEN_LIGHT_TIMES + YELLOW_LIGHT_TIMES  # 允许通行时间
ROAD_WIDTH = 3 * 6
LANE_WIDTH = 3


class TrafficLightStatue(Enum):
    red = 0,
    green = 1,
    yellow = 2


class Graph(GraphBase):
    def __init__(self):

        """
        初始化图类，继承自GraphBase
        """
        super().__init__()  # 调用父类的初始化方法

        # 位置矩阵
        self.__positions = np.empty((0, 2), dtype=float)
        # 长度矩阵
        self.__length = np.empty((0, 0), dtype=float)
        # 权重矩阵
        self.__weight = np.empty((0, 0), dtype=float)
        # 度矩阵
        self.__degree = np.empty((0, 0))
        # 限速矩阵
        self.__limit_speed = np.empty((0, 0), dtype=float)
        # 路口半径
        self.__crossing_radius = np.empty((0, 0), dtype=float)
        # 截断距离
        self.__cut_length = np.empty((0, 0), dtype=float)
        # 路口转向路径矩阵
        self.__crossing_turn = np.empty((0, 0), dtype=float)
        # 路口坐标基底矩阵
        self.__road_basic = np.empty((0, 0, 0, 0), dtype=float)
        #红绿灯矩阵
        self.__traffic_light = np.empty((0, 0), dtype=float)

        self.__tick = 0

        # 存储密度数据
        self.__density_data = {}

        # 存放节点风险数据
        self.__point_risk_history = {}  # 格式: {point_id: [历史风险1, 历史风险2, ...]}

        # 存放边风险数据
        self.__road_risk_history = {}  # 新增道路风险历史：{road_id: [历史风险1, 历史风险2, ...]}

        # 天气信息
        self.__weather = [
            random.uniform(-20, 40),  # temperature
            random.uniform(0, 100),  # wind
            random.uniform(0, 100),  # airQuality
            random.uniform(0, 100)  # rain
        ]
        # 初始化&载入点边
        self.load_json("%s/graph/data.json" % os.getcwd())
        # 加载密度数据

    def point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        """
        :param px: 车的x坐标
        :param py: 车的y坐标
        :param x1: 起点x坐标
        :param y1: 起点y坐标
        :param x2: 终点x坐标
        :param y2: 终点y坐标
        :return: 返回车到边的最短距离
        计算点(px, py)到线段(x1,y1)-(x2,y2)的最短距离
        """
        # 线段向量
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:  # 线段为点时，直接返回点到点的距离
            return math.hypot(px - x1, py - y1)
        # 计算点在 segment 上的投影比例（0~1之间为线段上的投影）
        t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2)
        t = max(0, min(1, t))  # 限制t在[0,1]范围内
        # 投影点坐标
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        # 返回点到投影点的距离
        return math.hypot(px - proj_x, py - proj_y)

    def initialize_car(self, num_cars):
        """
        初始化车的位置
        """
        cars_list = generate_cars_list(num_cars)
        return cars_list

    def initialize_crowd(self, num_people=None):
        """
        初始化人的位置
        :return:
        """
        people_list = generate_people_list(num_people)
        return people_list

    def get_road_density(self, start_id: int, end_id: int, density_matrix, density_scale: float) -> float:
        """
        根据起点ID和终点ID获取对应路段的人流密度
        :param start_id: 路段起点ID
        :param end_id: 路段终点ID
        :param density_matrix: AgentGroup计算的人流密度矩阵
        :param density_scale: 密度缩放率
        :return: 路段区域的平均人流密度
        """
        # 1. 根据start_id和end_id获取路段的起点和终点坐标
        start_point = self.points[start_id]
        end_point = self.points[end_id]
        start_x, start_y = start_point.x, start_point.y
        end_x, end_y = end_point.x, end_point.y

        # 2. 如果起点和终点相同，返回0密度
        if start_x == end_x and start_y == end_y:
            return 0.0

        # 3. 将坐标转换为密度矩阵中的网格坐标
        grid_start_x = int(round(start_x / density_scale))
        grid_start_y = int(round(start_y / density_scale))
        grid_end_x = int(round(end_x / density_scale))
        grid_end_y = int(round(end_y / density_scale))

        # 4. 使用Bresenham算法获取线段经过的所有网格点
        def bresenham_line(x0, y0, x1, y1):
            """生成从(x0,y0)到(x1,y1)的线段经过的所有网格点"""
            points = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x, y = x0, y0
            sx = 1 if x1 > x0 else -1
            sy = 1 if y1 > y0 else -1

            # 处理水平或垂直线的情况
            if dx == 0:  # 垂直线
                while y != y1:
                    points.append((x, y))
                    y += sy
                points.append((x, y))
            elif dy == 0:  # 水平线
                while x != x1:
                    points.append((x, y))
                    x += sx
                points.append((x, y))




            else:  # 斜线
                err = dx - dy
                while True:
                    points.append((x, y))
                    if x == x1 and y == y1:
                        break
                    e2 = 2 * err
                    # （1）判断是否沿 x 方向移动：if e2 > -dy
                    # 当e2 > -dy时，说明x 方向的累积误差已经足够大，需要沿 x 方向移动一步（否则线段会偏离 x 方向太远）。
                    if e2 > -dy:
                        # 移动后，误差需要更新：err -= dy（减去 y 方向总距离，平衡误差累积）。
                        # 同时，x 坐标沿sx方向移动一步：x += sx。
                        err -= dy
                        x += sx
                        # （2）判断是否沿 y 方向移动：if e2 < dx
                        # 当e2 < dx时，说明y 方向的累积误差已经足够大，需要沿 y 方向移动一步（否则线段会偏离 y 方向太远）。
                    if e2 < dx:
                        # 移动后，误差需要更新：err += dx（加上 x 方向总距离，平衡误差累积）。
                        err += dx
                        # 同时，y 坐标沿sy方向移动一步：y += sy。
                        y += sy
            return points

        # 获取线段经过的所有网格点
        line_points = bresenham_line(grid_start_x, grid_start_y, grid_end_x, grid_end_y)

        # 5. 过滤掉超出密度矩阵范围的点
        valid_points = []
        max_row, max_col = density_matrix.shape
        for (x, y) in line_points:
            if 0 <= x < max_row and 0 <= y < max_col:
                valid_points.append((x, y))

        # 6. 如果没有有效点，返回0密度
        if not valid_points:
            return 0.0

        # 7. 计算所有有效网格点的平均密度
        total_density = 0.0
        for (x, y) in valid_points:
            total_density += density_matrix[x, y].item()

        return total_density / len(valid_points)

    def get_road_density_matrix(self, density_matrix, density_scale: float) -> np.ndarray:
        """
        生成点i到点j的边密度矩阵
        第i行第j列的元素表示从点i到点j的边的人群密度
        若不存在从i到j的边，则值为0
        """
        # 初始化矩阵，大小为点的数量×点的数量，初始值为0
        density_matrix_ij = np.zeros((self.num_points, self.num_points), dtype=np.float32)

        # 遍历所有边，计算密度并填充矩阵
        for edge in self.edges:
            start_id = edge["start_id"]
            end_id = edge["end_id"]
            # 计算该边的密度
            density = self.get_road_density(start_id, end_id, density_matrix, density_scale)
            # 填充矩阵
            density_matrix_ij[start_id, end_id] = density

        return density_matrix_ij

    @property
    def nodes_position(self) -> np.ndarray:
        return np.array(self.__positions)

    @property
    def length(self) -> np.ndarray:
        return np.array(self.__length)

    @property
    def weight(self) -> np.ndarray:
        return np.array(self.__weight)

    @property
    def degree(self) -> np.ndarray:
        return np.array(self.__degree)

    @property
    def limit_speed(self) -> np.ndarray:
        return np.array(self.__limit_speed)

    @property
    def traffic_light(self) -> np.ndarray:
        return np.array(self.__traffic_light)

    @property
    def corssing_radius(self) -> np.ndarray:
        return np.array(self.__crossing_radius)

    @property
    def cut_length(self) -> np.ndarray:
        return np.array(self.__cut_length)

    @property
    def road_basic(self) -> np.ndarray:
        return np.array(self.__road_basic)

    @property
    def crossing_turn(self) -> np.ndarray:
        return np.array(self.__crossing_turn)

    def now_light(self, start_id: int, end_id: int) -> (TrafficLightStatue, float):
        """
        获取指定路段当前的红绿灯状态及剩余时间
        :param start_id: 起始节点id，即为道路的方向，如果输入-1，则为询问人行道
        :param end_id: 目标节点id，即为红绿灯所在路口id
        :return: 一个元组，包含:（红绿灯当前状态颜色，剩余时间）
        """
        if end_id != start_id: self._check_edge(start_id, end_id)
        light_time = self.traffic_light[start_id, end_id] % (
                self.traffic_light[end_id, end_id] - self.__tick)  # 截断求余数后，窗口期时间，如果为负数，就是红灯（露头时间长度）
        # 判断是否处于窗口（去下不取上）
        allow_t = self.get_light(start_id, end_id)
        if allow_t <= light_time: return TrafficLightStatue.red, (self.traffic_light[
                                                                      end_id, end_id] - self.__tick) - light_time
        if 0 <= light_time < allow_t - YELLOW_LIGHT_TIMES: return TrafficLightStatue.green, allow_t - YELLOW_LIGHT_TIMES - light_time
        if allow_t - YELLOW_LIGHT_TIMES <= light_time < allow_t: return TrafficLightStatue.yellow, allow_t - light_time

    def control_light(self, start_id: int, end_id: int, add_green_time: float):
        # 检查输入参数
        if end_id != start_id:
            self.__traffic_light[start_id:, end_id] += add_green_time
        else:
            self.__traffic_light[start_id, end_id] += add_green_time

    def get_light(self, start_id: int, end_id: int) -> float:
        """
        获取指定路段当前的红绿灯的通行时间
        :param start_id: 起始节点id，即为道路的方向，如果输入-1，则为询问人行道
        :param end_id: 目标节点id，即为红绿灯所在路口id
        :return:
        """
        if end_id != start_id: self._check_edge(start_id, end_id)
        end = self.__traffic_light[start_id, end_id].item()
        column = np.hstack((np.sort(
            ((np.where(self.__degree != 0, 1, 0) + np.diag(np.diag(self.__degree) + 1)) * self.__traffic_light)[
                :, end_id], axis=0)[::-1], 0))
        return (end - column[np.where(column == end)[0][0] + 1]).item()
        # accord = [l for l in reversed((np.where(self.__degree != 0, 1, 0) * self.__traffic_light)[:start_id,end_id]) if l != 0]
        # return self.__traffic_light[start_id,end_id].item() - (0 if len(accord)==0 else accord[0])

    def simulate_light(self, dt=0.1) -> None:
        self.__tick += dt
        self.__traffic_light += dt

    def leave_gate(self):
        """
        处理门点的状态：若门关闭则先打开，然后清除地图上的车辆
        """
        # 遍历所有点，找到类型为门的点
        tolerance = 2
        for point_id, point in enumerate(self.points):
            if point.type == PointType.gate:
                # 检查门的当前状态
                if point.gate_status == GateStatus.close:
                    # 切换门状态为打开
                    point.gate_transition()
                    i, j = point.x, point.y
                    if abs(self.car.x - i) < tolerance and abs(self.car.y - j) < tolerance:
                        pass

    def enter_gate(self, cars: list, limit_speed: float, density_matrix: np.ndarray, density_scale: float,
                   density_threshold: float = 0.8) -> dict:
        """
        车辆准入算法：仅允许在门点处于打开状态时，车辆进入门控区域
        :param cars: 车辆字典，格式为[car1, car2, car3...]
        :param limit_speed:限速
        :param density_matrix:人流密度矩阵
        :param density_threshold:人流密度阈值，超过时限制车辆入内
        :return: 处理后的车辆字典，不符合准入条件的车辆被限制进入
        """
        tolerance = 2
        updated_cars = []

        # 获取所有门点信息
        gate_points = [
            (point_id, point)
            for point_id, point in enumerate(self.points)
            if point.type == PointType.gate
        ]

        for car in cars:
            car_x, car_y = car.x, car.y
            is_blocked = False

            if not is_blocked and hasattr(car, 'target_road_ids'):
                for road_id in car.target_road_ids:
                    if road_id < 0 or road_id >= len(self.edges):
                        continue  # 无效路段ID跳过
                    edge = self.edges[road_id]  # 获取路段对象
                    start_id = edge.start_id  # 提取起点ID
                    end_id = edge.end_id  # 提取终点ID

                    road_density = self.get_road_density(start_id, end_id, density_matrix, density_scale)
                    if road_density > density_threshold:
                        is_blocked = True
                        print(f"车辆{car.id}被限制进入：路段{road_id}人流密度过高（{road_density:.2f}）")
                        break

            # 检查车辆是否试图进入任何门控区域
            for gate_id, gate_point in gate_points:
                gate_x, gate_y = gate_point.x, gate_point.y
                # 计算车辆到门点的距离
                distance = math.hypot(car_x - gate_x, car_y - gate_y)

                # 若车辆在门控区域内且门处于关闭状态，阻止进入
                # 速度超过限速，不准许车辆进入
                if car_x ** 2 + car.y ** 2 > limit_speed ** 2:
                    if distance < tolerance and gate_point.gate_status == GateStatus.close:
                        is_blocked = True
                        break

            if is_blocked:
                # 阻止车辆进入：将车辆位置调整到门控区域外，速度置零
                for gate_id, gate_point in gate_points:
                    gate_x, gate_y = gate_point.x, gate_point.y
                    if math.hypot(car_x - gate_x, car_y - gate_y) < tolerance:
                        # 计算区域外偏移位置（沿远离门点方向偏移tolerance）
                        dx = car_x - gate_x
                        dy = car_y - gate_y
                        if dx == 0 and dy == 0:
                            # 若车辆正好在门点位置，随机偏移
                            dx, dy = random.uniform(-1, 0), random.uniform(-1, 0)
                        norm = math.hypot(dx, dy)
                        new_x = gate_x + dx / norm * tolerance
                        new_y = gate_y + dy / norm * tolerance
                        updated_cars[car.id] = {
                            'p_x': new_x,
                            'p_y': new_y,
                            'v_x': 0,
                            'v_y': 0,
                            'angle': car.angle
                        }
                        break

            else:
                # 允许进入，保持车辆原有状态
                updated_cars.append(car)

        return updated_cars

    def upgrade_weight(self):
        pass

    def get_weather(self) -> list:
        """
        用户获取天气，输出一个列表
        :return: [temperature, wind, airQuality, rain]
        """
        return self.__weather

    def set_weather(self, weather: dict):
        """
        用户修改天气，输入一个字典
        :return: bool
        """
        param_rules = {
            "temperature": (-20, 40),
            "wind": (0, 100),
            "airQuality": (0, 100),
            "rain": (0, 100)
        }
        try:
            for key in param_rules:
                if key not in weather:
                    raise KeyError(f"缺少必要天气参数：{key}")
            for key, value in weather.items():
                if not isinstance(value, (int, float)):
                    raise TypeError(f"{key} 必须是数值类型（float），当前值：{value}")
                min_val, max_val = param_rules[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} 超出有效范围 [{min_val}, {max_val}]，当前值：{value}")
            # 更新存储的天气数据
            self.__weather[0] = weather["temperature"]
            self.__weather[1] = weather["wind"]
            self.__weather[2] = weather["airQuality"]
            self.__weather[3] = weather["rain"]
            return True
        except (KeyError, TypeError, ValueError) as e:
            print(f"修改天气失败：{e}")
            return False

    def matrix_to_map(self, density_matrix: np.ndarray):
        max_x = np.max(self.__positions[:, 0]) + 100
        max_y = np.max(self.__positions[:, 1]) + 100
        MAP_SIZE = int(max(max_x, max_y))
        DENSITY_MATRIX_SIZE = 200
        scale = MAP_SIZE / DENSITY_MATRIX_SIZE

        # 生成地图坐标（x对应列，y对应行）
        x = np.arange(MAP_SIZE)  # 水平方向（列）
        y = np.arange(MAP_SIZE)  # 垂直方向（行）
        # 映射到密度矩阵的索引（x→列，y→行）
        col_idx = np.clip(np.round(x / scale).astype(int), 0, DENSITY_MATRIX_SIZE - 1)
        row_idx = np.clip(np.round(y / scale).astype(int), 0, DENSITY_MATRIX_SIZE - 1)

        # 用meshgrid生成网格，行对应y，列对应x
        Y, X = np.meshgrid(y, x, indexing='ij')  # Y是行索引网格，X是列索引网格
        dst_matrix = density_matrix[row_idx[Y], col_idx[X]]  # 行用y映射，列用x映射
        dst_matrix /= scale
        return dst_matrix

    def get_all_information(self, cal_cars: list, cal_people: list, hot_data: float):
        """
        获取全局车流、人流、拥挤程度数据（通过遍历所有边汇总）
        :param cal_cars: 所有车辆的坐标列表，格式为[(x1,y1), (x2,y2), ...]
        :param cal_people: 所有行人的坐标列表，格式同上
        :param hot_data: 用于计算拥挤度的热数据
        :return: [总车流量, 总人流量, 平均拥挤度]
        """
        total_cars = 0  # 全局总车流量
        total_people = 0  # 全局总人流量
        total_crowding = 0.0  # 所有边的拥挤度总和
        edge_count = 0  # 有效边的数量（用于计算平均拥挤度）

        # 遍历所有边
        for road_id, edge in enumerate(self.edges):
            try:
                # 调用get_road_information获取当前边的信息
                # 边的start_id和end_id从edge对象中获取
                road_info = self.get_road_information(
                    road_id=road_id,
                    start_id=edge.start_id,
                    end_id=edge.end_id,
                    cal_cars=cal_cars,
                    cal_people=cal_people,
                    hot_data=hot_data
                )
                # 解析返回结果：[cars_num, people_num, crowding, emergency, trafficLight]
                cars_num, people_num, crowding = road_info[0], road_info[1], road_info[2]

                # 累加数据
                total_cars += cars_num
                total_people += people_num
                total_crowding += crowding
                edge_count += 1

            except ValueError as e:
                # 跳过无效边（如ID错误），打印警告
                continue

        # 计算平均拥挤度（避免除以0）
        avg_crowding = total_crowding / edge_count if edge_count > 0 else 0.0

        return [total_cars, total_people, avg_crowding]

    def get_point_id(self, cal_x, cal_y):
        shortest_distance = float("inf")
        for point_id in range(len(self.__positions)):
            x, y = self.__positions[point_id]  # 获取节点的x,y坐标
            distance = np.linalg.norm(np.array([cal_x, cal_y]) - np.array([x, y]))
            if distance < shortest_distance:
                shortest_distance = distance
                result_point_id = point_id
        return result_point_id

    def get_point_information(self, point_id: int, cal_cars: list, cal_people: list, hot_data: dict):
        """
        获取指定节点的车流、人流、拥挤程度数据
        :param point_id: 节点id,
        :return:[cars, people, crowding, emergency, trafficLight]
        """
        cars_point_id = []
        people_point_id = []
        if point_id < 0 or point_id >= len(self.points):  # 检查节点ID是否有效
            raise ValueError("节点id无效")

        for i in range(len(cal_cars)):
            if self.get_point_id(cal_cars[i][0], cal_cars[i][1]) == point_id:
                cars_point_id.append(self.get_point_id(cal_cars[i][0], cal_cars[i][1]))
            else:
                pass
        for i in range(len(cal_people)):
            if self.get_point_id(cal_people[i][0], cal_people[i][1]) == point_id:
                people_point_id.append(self.get_point_id(cal_people[i][0], cal_people[i][1]))
            else:
                pass

        cars_num = len(cars_point_id)
        people_num = len(people_point_id)
        scale = hot_data[self.__getitem__(point_id)] / 0.1
        crowding = hot_data[self.__getitem__(point_id)] * scale

        emergency = ""

        traffic_light = []
        light = self.now_light(point_id, point_id)[0]
        if light == TrafficLightStatue.green:
            traffic_light.append('green')
        elif light == TrafficLightStatue.yellow:
            traffic_light.append('yellow')
        else:
            traffic_light.append('red')
        traffic_light.append(self.now_light(point_id, point_id)[1].item())
        traffic_light.append(60)

        return [cars_num, people_num, crowding, emergency, traffic_light]

    def get_road_id(self, px: float, py: float) -> int:
        """
        找到距离点(px, py)最近的边的ID
        :param px: 点的x坐标
        :param py: 点的y坐标
        :return: 最近边的ID（索引），若无边则返回None
        """
        if not self.edges:  # 没有边时返回None
            return None

        min_distance = float('inf')
        nearest_edge_id = 0

        for edge_id, edge in enumerate(self.edges):
            # 获取边的起点和终点坐标
            start_point = self.points[edge.start_id]
            end_point = self.points[edge.end_id]

            # 计算点到当前边的距离
            distance = self.point_to_segment_distance(
                px, py,
                start_point.x, start_point.y,
                end_point.x, end_point.y
            )

            # 更新最近边信息
            if distance < min_distance:
                min_distance = distance
                nearest_edge_id = edge_id

        return nearest_edge_id

    def get_road_information(self, road_id: int, start_id: int, end_id: int, cal_cars: list, cal_people: list,
                             hot_data: dict):
        """
        获取指定路段的车流、人流、拥挤程度数据
        :param road_id: int
        :param start_id: int
        :param end_id: int
        :return:[cars, people, crowding, emergency, trafficLight]
        """
        cars_road_id = []
        people_road_id = []
        if road_id < 0 or road_id >= len(self.edges):
            raise ValueError("边id无效")

        for i in range(len(cal_cars)):
            if self.get_road_id(cal_cars[i][0], cal_cars[i][1]) == road_id:
                cars_road_id.append(self.get_road_id(cal_cars[i][0], cal_cars[i][1]))
            else:
                pass
        for i in range(len(cal_people)):
            if self.get_road_id(cal_people[i][0], cal_people[i][1]) == road_id:
                people_road_id.append(self.get_road_id(cal_people[i][0], cal_people[i][1]))
            else:
                pass

        cars_num = len(cars_road_id)
        people_num = len(people_road_id)
        point_id = end_id
        scale = hot_data[self.__getitem__(point_id)] / 0.1
        crowding = hot_data[self.__getitem__(point_id)] * scale

        emergency = ""

        traffic_light = []
        light = self.now_light(start_id, point_id)[0]
        if light == TrafficLightStatue.green:
            traffic_light.append('green')
        elif light == TrafficLightStatue.yellow:
            traffic_light.append('yellow')
        else:
            traffic_light.append('red')
        traffic_light.append(self.now_light(start_id, point_id)[1].item())
        traffic_light.append(60)

        return [cars_num, people_num, crowding, emergency, traffic_light]

    def get_point_risk_data(self, point_id: int, cal_cars: list, cal_people: list, hot_data: dict):
        """
        获取指定节点的风险数据（结合历史风险、拥挤度、天气、紧急事件）
        :param hot_data:
        :param cal_people:
        :param cal_cars:
        :param point_id: 节点ID

        :return: [历史风险列表 + 当前综合风险], 短期风险预测（0-10）
        """
        # 初始化当前节点的历史风险列表（首次调用时）
        if point_id not in self.__point_risk_history:
            self.__point_risk_history[point_id] = []

        # 1. 获取基础数据
        cars, people, crowding, emergency, traffic_light = self.get_point_information(point_id, cal_cars, cal_people,
                                                                                      hot_data)
        temperature, wind, air_quality, rain = self.get_weather()
        crowding = np.mean(crowding) if isinstance(crowding, np.ndarray) else crowding

        # 2. 定义风险因素权重
        weights = {
            "crowding": 0.5,
            "weather": 0.3,
            "emergency": 0.2
        }

        # 3. 各风险因素量化（同之前逻辑）
        crowd_ratio = min(1.0, (cars / CARS_NUM_THREADS + people / PEOPLE_NUM_THREADS) / 2)
        crowd_risk = 10 * (crowd_ratio ** 2)

        rain_risk = min(10, rain / 10)
        wind_risk = min(10, wind / 10)
        air_risk = max(0, 10 - air_quality / 10)
        weather_risk = (rain_risk + wind_risk + air_risk) / 3

        emergency_risk = 10 if emergency else 0

        # 4. 计算当前综合风险
        current_total_risk = (
                weights["crowding"] * crowd_risk +
                weights["weather"] * weather_risk +
                weights["emergency"] * emergency_risk
        )
        current_total_risk = min(10, max(0, current_total_risk))

        # 5. 将当前综合风险加入历史缓存
        self.__point_risk_history[point_id].append(current_total_risk)

        # 6. 短期风险预测（当前综合风险的基础上调整）
        predict_risk = min(10, current_total_risk + 1) if (crowd_ratio > 0.7 or rain > 70) else max(0,
                                                                                                    current_total_risk - 1)

        # 7. 构建返回的风险列表（历史数据 + 当前数据）
        total_risk_list = self.__point_risk_history[point_id].copy()

        return total_risk_list, predict_risk

    def get_road_risk_data(self, road_id: int, cal_cars: list, cal_people: list,
                           hot_data: dict):
        """
        获取指定路段的风险数据（结合历史风险、道路密度、交通灯、天气、限速）
        :param road_id: 路段ID
        :param cal_cars: 车辆数据列表
        :param cal_people: 行人数据列表
        :param hot_data: 热点数据
        :return: [历史风险列表 + 当前综合风险], 短期风险预测（0-10）
        """
        # 1. 初始化当前道路的历史风险列表（首次调用时）
        if road_id not in self.__road_risk_history:
            self.__road_risk_history[road_id] = []

        # 2. 校验路段ID有效性
        if road_id < 0 or road_id >= len(self.edges):
            raise ValueError("路段ID无效")
        edge = self.edges[road_id]
        start_id = edge.start_id
        end_id = edge.end_id

        cars, people, crowding, emergency, light_time = self.get_road_information(
            road_id, start_id, end_id, cal_cars, cal_people, hot_data
        )
        temperature, wind, air_quality, rain = self.get_weather()
        limit_speed = self.limit_speed[start_id, end_id]  # 路段限速
        road_density = crowding  # 复用用拥挤拥挤度作为道路负荷指标

        # 4. 定义风险因素权重
        weights = {
            "road_load": 0.4,  # 道路负荷（车流密度）
            "traffic_light": 0.3,  # 交通灯等待风险
            "weather": 0.2,  # 天气影响
            "speed_deviation": 0.1  # 车速与限速的偏差
        }

        # 5. 各风险因素量化（归一化到0-10）
        # 5.1 道路负荷风险：密度越高风险越大
        road_load_ratio = min(1.0, road_density)  # 假设road_density已归一化到0-1
        road_load_risk = 10 * road_load_ratio

        # 5.2 交通灯风险：红灯灯时间越长风险越高
        light_status, remaining_time = self.now_light(start_id, end_id)
        light_risk = 0
        if light_status == TrafficLightStatue.red:
            light_risk = min(10, remaining_time / 5)  # 红灯剩余时间每5秒增加2点风险
        elif light_status == TrafficLightStatue.yellow:
            light_risk = 3  # 黄灯有一定风险

        # 5.3 天气风险：侧重降雨对路面的影响
        weather_risk = min(10, rain / 10 + wind / 20)

        # 5.4 车速偏差风险：实际车速与限速偏差差越大风险越高
        estimated_speed = limit_speed * (1 - road_load_ratio)  # 拥挤时车速降低
        speed_deviation = abs(estimated_speed - limit_speed) / limit_speed  # 偏差比例
        speed_risk = 10 * speed_deviation

        # 6. 计算当前综合风险
        current_total_risk = (
                weights["road_load"] * road_load_risk +
                weights["traffic_light"] * light_risk +
                weights["weather"] * weather_risk +
                weights["speed_deviation"] * speed_risk
        )
        current_total_risk = min(10, max(0, current_total_risk))  # 约束在0-10之间

        # 7. 将当前综合风险加入历史缓存
        self.__road_risk_history[road_id].append(current_total_risk)

        # 8. 短期风险预测：若道路负荷>70%且为红灯，风险上升
        predict_risk = min(10, current_total_risk + 2) if (
                road_load_ratio > 0.7 and light_status == TrafficLightStatue.red
        ) else max(0, current_total_risk - 1)  # 否则略降

        # 9. 构建返回的风险列表（历史数据 + 当前数据）
        total_risk_list = self.__road_risk_history[road_id].copy()

        return total_risk_list, predict_risk

    def load_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for point in data['points']: self._add_point(point["name"], point["x"], point["y"],
                                                         point_type=point["type"])

            for edge in data['edges']:
                self._add_edge(
                    start_id=edge['start_id'],
                    end_id=edge['end_id'],
                    degree=edge['degree'],
                    limit_speed=float(edge['limit_speed'])
                )

        # 初始化position
        for point in self.points: self.__positions = np.vstack((self.__positions, np.array([point.x, point.y])))

        # 初始化degree，length，weight
        self.__degree = np.zeros((len(self.points), len(self.points)), dtype=int)
        self.__length = np.zeros((len(self.points), len(self.points)), dtype=float)
        self.__weight = np.ones((len(self.points), len(self.points)), dtype=float)
        self.__limit_speed = np.zeros((len(self.points), len(self.points)), dtype=float)
        for edge in self.edges:
            # 度矩阵
            self.__degree[edge.start_id, edge.end_id] += edge.lane_degree
            # 行二范数
            self.__length[edge.start_id, edge.end_id] = np.linalg.norm(
                self.__positions[edge.start_id] - self.__positions[edge.end_id])
            # 限速矩阵
            self.__limit_speed[edge.start_id, edge.end_id] = edge.limit_speed

        self.__degree += self.__degree.T
        self.__length += self.__length.T
        self.__limit_speed += self.__limit_speed.T

        # 初始化radius
        self.__crossing_radius = np.zeros(len(self.points), dtype=float)
        for cross_id, cross_point in enumerate(self.points):
            # 遍历cross列，即为到达这个路口的点
            for from_point in [self.points[point_id] for point_id, degree in enumerate(self.degree[:, cross_id]) if
                               degree != 0]:
                # 遍历cross行，即为到从这个路口出发的点
                for to_point in [self.points[point_id] for point_id, degree in enumerate(self.degree[cross_id, :]) if
                                 degree != 0]:
                    l3 = np.linalg.norm(from_point.position - to_point.position)
                    if l3 == 0: continue

                    l1 = np.linalg.norm(cross_point.position - from_point.position)
                    l2 = np.linalg.norm(cross_point.position - to_point.position)
                    cos = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2)
                    radius = ROAD_WIDTH / (2 * np.sqrt((1 - cos) / 2))
                    if self.__crossing_radius[cross_id] == 0 or self.__crossing_radius[cross_id] > radius:
                        self.__crossing_radius[cross_id] = radius

        # 初始化cut_length
        self.__cut_length = np.array(self.__length.copy())
        for start_id, row in enumerate(self.__cut_length):
            for end_id, item in enumerate(row):
                self.__cut_length[start_id, end_id] -= self.__crossing_radius[start_id] + self.__crossing_radius[end_id]

        # 初始化turn
        self.__crossing_turn = np.empty((len(self.points), len(self.points), len(self.points)), dtype=dict)
        self.__road_basic = np.zeros((len(self.points), len(self.points), 2, 2), dtype=float)
        for cross_id, cross_point in enumerate(self.points):
            # 遍历cross列，即为到达这个路口的点
            for from_point in [self.points[point_id] for point_id, degree in enumerate(self.degree[:, cross_id]) if
                               degree != 0]:
                # from法向量
                vector_from = (cross_point.position - from_point.position) / self.length[
                    self.point_name2id[from_point.name], cross_id]
                # 垂直逆时针法向量
                vertical_from = np.array([vector_from[1], -vector_from[0]])
                vertical_from *= np.cross(vector_from, vertical_from)  # 用于更为逆时针

                # 遍历cross行，即为到从这个路口出发的点，用from，to两向量正角度来按照角度排序（to的逆时针排序，即车道从lane0到lane2，先右转再左转）（掉头为lane3，默认直接转换为lane2）
                for lane, to_point in enumerate(
                        sorted([self.points[point_id] for point_id, degree in enumerate(self.degree[cross_id, :]) if
                                degree != 0], key=lambda to_point: np.cross(cross_point.position - from_point.position,
                                                                            to_point.position - cross_point.position).item())):
                    # to法向量
                    vector_to = (to_point.position - cross_point.position) / self.length[
                        cross_id, self.point_name2id[to_point.name]]
                    # 垂直逆时针法向量
                    vertical_to = np.array([vector_to[1], -vector_to[0]]) / np.linalg.norm(vector_to)
                    vertical_to *= np.cross(vector_to, vertical_to)  # 用于更为逆时针

                    # 方向系数，如果为顺时针，则圆心在下，即叉乘为负，取正值（让点+y）
                    direction_c = 1 if np.cross(vector_from, vector_to) < 0 else -1

                    # 在路口上的入节点和出节点（即在路口拐弯的from和to点）
                    point_s = cross_point.position - self.corssing_radius[cross_id] * vector_from + (
                            lane + 0.5) * LANE_WIDTH * vertical_from * direction_c
                    point_e = cross_point.position + self.corssing_radius[cross_id] * vector_to + (
                            lane + 0.5) * LANE_WIDTH * vertical_to * direction_c

                    if np.cross(vector_from, vector_to) == 0:
                        centre = (point_s + point_e) / 2
                    else:
                        centre = np.linalg.solve(np.vstack((vector_from, vector_to)),
                                                 np.array([np.dot(vector_from, point_s), np.dot(vector_to, point_e)]))

                    radius = np.linalg.norm(centre - point_s)
                    self.__crossing_turn[
                        cross_id, self.point_name2id[from_point.name], self.point_name2id[to_point.name]] = {
                        "centre": centre if centre is not None else None,
                        "radius": radius,
                        "from": (-vertical_from) * radius,
                        "to": (-vertical_to) * radius,
                        "lane": lane if lane < 3 else 2
                    }
                    self.__road_basic[self.point_name2id[from_point.name], cross_id] = np.array(
                        [[vector_from[0], vertical_from[0]],
                         [vector_from[1], vertical_from[1]]])
                    self.__road_basic[cross_id, self.point_name2id[to_point.name]] = np.array(
                        [[vector_to[0], vertical_to[0]],
                         [vector_to[1], vertical_to[1]]])
        # 初始化traffic_light
        self.__traffic_light = np.zeros((len(self.points), len(self.points)), dtype=float)
        for end_id, end_point in enumerate(self.points):
            index = 1
            if end_id == 30:
                a = 1
            for start_id, start_point in enumerate(self.points):
                if self.degree[start_id, end_id] != 0:
                    self.__traffic_light[start_id, end_id] = index * ALLOW_LIGHT_TIMES
                    index += 1
            # 人行道
            self.__traffic_light[end_id, end_id] = index * ALLOW_LIGHT_TIMES
        return True
