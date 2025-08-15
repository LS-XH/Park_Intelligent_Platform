from numpy.ma.extras import column_stack

from sympy.physics.units import temperature
from Interface.graph import GraphBase, PointType,Point, Edge, GateStatus

import math
from enum import Enum
import os
import numpy as np
import json

import random
from AIPart.graph.common_func import generate_cars_list, generate_people_list



#from .information import weather,traffic_flow,human_flow,accident_rate


CARS_NUM_THREADS = 30 # 车辆阈值
PEOPLE_NUM_THREADS = 3000 # 人阈值
GREEN_LIGHT_TIMES = 27 # 绿灯时间
YELLOW_LIGHT_TIMES = 3 # 黄灯时间
ALLOW_LIGHT_TIMES = GREEN_LIGHT_TIMES+YELLOW_LIGHT_TIMES # 允许通行时间


class TrafficLightStatue(Enum):
    red=0,
    green=1,
    yellow=2

class Graph(GraphBase):
    def __init__(self):

        """
        初始化图类，继承自GraphBase
        """
        super().__init__()  # 调用父类的初始化方法


        # 位置矩阵
        self.__positions = np.empty((0,2),dtype=float)
        # 长度矩阵
        self.__length = np.empty((0,0),dtype=float)
        # 权重矩阵
        self.__weight = np.empty((0,0),dtype=float)
        # 度矩阵
        self.__degree = np.empty((0,0))
        # 限速矩阵
        self.__limit_speed = np.empty((0,0),dtype=float)
        #红绿灯矩阵
        self.__traffic_light = np.empty((0,0),dtype=float)

        self.__tick = 0

        # 天气信息
        self.__weather =[
            random.uniform(-20, 40), # temperature
            random.uniform(0, 100), # wind
            random.uniform(0, 100), #airQuality
            random.uniform(0, 100) #rain
        ]
        #初始化&载入点边
        self.load_json("%s\\graph\\data.json"%os.getcwd())


    def initialize_car(self, num_cars=None):
        """
        初始化车的位置，根据固定的热点数据随机生成
        :param num_cars:    生成的车的数量
        :return:            字典，键表示车的ID，值表示车的坐标
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

    def get_road_density(self, road_id: int, density_matrix, AgentGroup) -> float:
        """
        根据路段ID获取对应区域的人流密度
        :param road_id: 路段ID
        :param density_matrix: AgentGroup计算的人流密度矩阵
        :return: 路段区域的平均人流密度
        """
        # 1. 根据road_id获取路段的起点和终点坐标（需结合现有edges数据）
        edge = self.edges[road_id]
        start_point = self.points[edge.start_id]
        end_point = self.points[edge.end_id]
        start_x, start_y = start_point.x, start_point.y
        end_x, end_y = end_point.x, end_point.y

        # 2. 计算路段在密度矩阵中的覆盖范围（基于density_scale缩放）
        density_scale = AgentGroup.density_scale  # 从AgentGroup获取缩放因子
        min_x = min(start_x, end_x) * density_scale
        max_x = max(start_x, end_x) * density_scale
        min_y = min(start_y, end_y) * density_scale
        max_y = max(start_y, end_y) * density_scale

        # 3. 截取密度矩阵中路段对应的区域并计算平均密度
        x1, x2 = int(min_x), int(max_x)
        y1, y2 = int(min_y), int(max_y)
        road_density = density_matrix[x1:x2, y1:y2].mean().item()
        return road_density

    @property
    def nodes_position(self)->np.ndarray:
        return np.array(self.__positions)

    @property
    def length(self)->np.ndarray:
        return np.array(self.__length)

    @property
    def weight(self)->np.ndarray:
        return np.array(self.__weight)

    @property
    def degree(self)->np.ndarray:
        return np.array(self.__degree)

    @property
    def limit_speed(self)->np.ndarray:
        return np.array(self.__limit_speed)

    @property
    def traffic_light(self)->np.ndarray:
        return np.array(self.__traffic_light)

    def now_light(self, start_id: int, end_id: int) -> (TrafficLightStatue,float):
        """
        获取指定路段当前的红绿灯状态及剩余时间
        :param start_id: 起始节点id，即为道路的方向，如果输入-1，则为询问人行道
        :param end_id: 目标节点id，即为红绿灯所在路口id
        :return: 一个元组，包含:（红绿灯当前状态颜色，剩余时间）
        """
        if end_id!=start_id: self._check_edge(start_id, end_id)
        light_time = self.traffic_light[start_id,end_id]%(self.traffic_light[end_id,end_id]-self.__tick)        # 截断求余数后，窗口期时间，如果为负数，就是红灯（露头时间长度）
        # 判断是否处于窗口（去下不取上）
        allow_t=self.get_light(start_id,end_id)
        if allow_t <= light_time:return TrafficLightStatue.red,(self.traffic_light[end_id,end_id]-self.__tick)-light_time
        if 0 <= light_time < allow_t-YELLOW_LIGHT_TIMES:return TrafficLightStatue.green, allow_t-YELLOW_LIGHT_TIMES-light_time
        if allow_t-YELLOW_LIGHT_TIMES <= light_time < allow_t:return TrafficLightStatue.yellow, allow_t-light_time

    def control_light(self,start_id: int, end_id: int, add_green_time:float):
        # 检查输入参数
        if end_id!=start_id: self.__traffic_light[start_id:,end_id]+=add_green_time
        else: self.__traffic_light[start_id,end_id]+=add_green_time

    def get_light(self, start_id: int, end_id: int) -> float:
        """
        获取指定路段当前的红绿灯的通行时间
        :param start_id: 起始节点id，即为道路的方向，如果输入-1，则为询问人行道
        :param end_id: 目标节点id，即为红绿灯所在路口id
        :return:
        """
        if end_id!=start_id: self._check_edge(start_id, end_id)
        end = self.__traffic_light[start_id,end_id].item()
        column = np.hstack((np.sort(((np.where(self.__degree != 0, 1, 0)+np.diag(np.diag(self.__degree)+1)) * self.__traffic_light)[:, end_id],axis = 0)[::-1],0))
        return (end-column[np.where(column == end)[0][0]+1]).item()
        # accord = [l for l in reversed((np.where(self.__degree != 0, 1, 0) * self.__traffic_light)[:start_id,end_id]) if l != 0]
        # return self.__traffic_light[start_id,end_id].item() - (0 if len(accord)==0 else accord[0])


    def simulate_light(self, dt=0.1)->None:
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
                    if abs(Graph.car.x - i) < tolerance and abs(Graph.car.y - j) < tolerance:
                        pass

    def enter_gate(self, cars: list, limit_speed:float) -> dict:
        """
        车辆准入算法：仅允许在门点处于打开状态时，车辆进入门控区域
        :param cars: 车辆字典，格式为[car1, car2, car3...]
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

            # 检查车辆是否试图进入任何门控区域
            for gate_id, gate_point in gate_points:
                gate_x, gate_y = gate_point.x, gate_point.y
                # 计算车辆到门点的距离
                distance = math.hypot(car_x - gate_x, car_y - gate_y)

                # 若车辆在门控区域内且门处于关闭状态，阻止进入
                # 速度超过限速，不准许车辆进入
                if car_x**2 + car.y**2 > limit_speed**2:
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

    def set_weather(self,weather: dict):
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
        

    def get_all_information(self, time:int):
        """
        获取全局车流、人流、拥挤程度数据
        :return: [cars,people,crowding]
        """



        pass

    def get_point_information(self, point_id: int, time:int):
        """
        获取指定节点的车流、人流、拥挤程度数据
        :param time:
        :param point_id: 节点id,
        :return:[cars, people, crowding, emergency, trafficLight]
        """
        pass

    def get_road_information(self, road_id: int, time:int):
        """
        获取指定路段的车流、人流、拥挤程度数据
        :param road_id: int
        :param time: int
        :return:[cars, people, crowding, emergency, trafficLight]
        """
        pass

    def get_point_risk_data(self, point_id: int, time:int):
        """
        获取指定节点的风险数据
        :param point_id: int
        :param time: int
        :return: [riskData1, riskData2, ..., riskDataN],predict
        """

        pass

    def get_road_risk_data(self, road_id: int, time:int):
        """
        获取指定路段的风险数据
        :param road_id: int
        :param time: int
        :return: [riskData1, riskData2, ..., riskDataN],predict
        """
        pass

    def load_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for point in data['points']:self._add_point(point["name"] , point["x"],point["y"], point_type=point["type"])

            for edge in data['edges']:
                self._add_edge(
                    start_id = edge['start_id'],
                    end_id = edge['end_id'],
                    degree= edge['degree'],
                    limit_speed = float(edge['limit_speed'])
                )

        # 初始化position
        for point in self.points:self.__positions = np.vstack((self.__positions,np.array([point.x,point.y])))

        # 初始化degree，length，weight
        self.__degree = np.zeros((len(self.points), len(self.points)), dtype=int)
        self.__length = np.zeros((len(self.points), len(self.points)), dtype=float)
        self.__weight = np.ones((len(self.points), len(self.points)), dtype=float)
        self.__limit_speed = np.zeros((len(self.points), len(self.points)), dtype=float)
        for edge in self.edges:
            # 度矩阵
            self.__degree[edge.start_id, edge.end_id] += edge.lane_degree
            # 行二范数
            self.__length[edge.start_id, edge.end_id] = np.linalg.norm(self.__positions[edge.start_id]-self.__positions[edge.end_id])
            # 限速矩阵
            self.__limit_speed[edge.start_id, edge.end_id] = edge.limit_speed

        self.__degree += self.__degree.T
        self.__length += self.__length.T
        self.__limit_speed += self.__limit_speed.T

        # 初始化traffic_light
        self.__traffic_light = np.zeros((len(self.points), len(self.points)), dtype=float)
        for end_id,end_point in enumerate(self.points):
            index = 1
            for start_id,start_point in enumerate(self.points):
                if self.degree[start_id, end_id] != 0:
                    self.__traffic_light[start_id, end_id] = index * ALLOW_LIGHT_TIMES
                    index += 1
            # 人行道
            self.__traffic_light[end_id, end_id] = index * ALLOW_LIGHT_TIMES
        return True

#graph=Graph()
#graph.load_json("data.json")
#print("\n度的邻接矩阵")
# graph = Graph()
# graph.load_json("data02.json")
# print("\n长度的邻接矩阵")
# print(graph.length)
# graph.initialize_traffic_light()