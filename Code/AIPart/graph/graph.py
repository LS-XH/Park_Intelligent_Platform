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

        # 存储密度数据
        self.__density_data = {}

        # 天气信息
        self.__weather =[
            random.uniform(-20, 40), # temperature
            random.uniform(0, 100), # wind
            random.uniform(0, 100), #airQuality
            random.uniform(0, 100) #rain
        ]
        #初始化&载入点边
        self.load_json("%s\\graph\\data.json"%os.getcwd())
        #加载密度数据


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
        for point_id, point in enumerate(self.points):
            if point.type == PointType.gate:
                # 检查门的当前状态
                if point.gate_status == GateStatus.close:
                    # 切换门状态为打开
                    point.gate_transition()
                    # 这里假设存在清除车辆的方法，实际实现需根据车辆存储方式调整
                    # 示例：若车辆信息存储在某个属性中，如self.cars，则清空
                    pass

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

    def matrix_to_map(self, density_matrix:np.ndarray):
        # 计算最大坐标并确定地图尺寸
        max_x = np.max(self.__positions[:, 0]) + 100
        max_y = np.max(self.__positions[:, 1]) + 100
        MAP_SIZE = int(max(max_x, max_y))
        DENSITY_MATRIX_SIZE = 200 # 密度矩阵尺寸
        scale = MAP_SIZE / DENSITY_MATRIX_SIZE # 缩放比例

        # 生成目标矩阵的网格坐标（I, J），并反向映射到原始矩阵坐标（i, j）
        I = np.arange(MAP_SIZE)
        J = np.arange(MAP_SIZE)
        i = np.clip(np.round(I / scale).astype(int), 0, DENSITY_MATRIX_SIZE - 1)
        j = np.clip(np.round(J / scale).astype(int), 0, DENSITY_MATRIX_SIZE - 1)

        # 利用广播机制快速填充目标矩阵
        dst_matrix = density_matrix[i[:, np.newaxis], j]  # i[:, np.newaxis]将I转为列向量，与J广播

        return dst_matrix


    def get_all_information(self, density_matrix:np.ndarray):
        """
        获取全局车流、人流、拥挤程度数据

        :param density_matrix: 密度矩阵，用于计算各项指标
        :return: [cars,people,crowding] 返回包含车流量、人流量和拥挤程度的列表
        """
        # 将密度矩阵转换为地图形式
        dst_matrix = self.matrix_to_map(density_matrix)
        # 计算地图上的总密度值
        total_density = np.sum(dst_matrix)
        # 计算地图的总面积
        map_area = dst_matrix.shape[0] * dst_matrix.shape[1]
        # 计算车流量
        cars = total_density * CARS_NUM_THREADS
        # 计算人流量
        people = total_density * PEOPLE_NUM_THREADS
        # 计算拥挤程度
        crowding = total_density
        # 返回计算结果
        return [cars, people, crowding]

    def get_point_information(self, point_id: int, density_matrix:np.ndarray):
        """
        获取指定节点的车流、人流、拥挤程度数据
        :param point_id: 节点id,
        :return:[cars, people, crowding, emergency, trafficLight]
        """
        dst_matrix = self.matrix_to_map(density_matrix) # 将密度矩阵转换为地图数据
        MAP_SIZE = dst_matrix.shape[0] # 获取地图尺寸
        if point_id < 0 or point_id >= len(self.points): # 检查节点ID是否有效
            raise ValueError("节点id无效")
        x, y = self.__positions[point_id] # 获取节点的x,y坐标
        # 计算地图的x,y坐标范围
        min_x_map = np.min(self.__positions[:, 0])
        max_x_map = np.max(self.__positions[:, 0]) + 100
        min_y_map = np.min(self.__positions[:, 1])
        max_y_map = np.max(self.__positions[:, 1]) + 100
        # 计算x,y方向的归一化坐标
        norm_x = (x - min_x_map) / (max_x_map - min_x_map) if (max_x_map - min_x_map) != 0 else 0
        norm_y = (y - min_y_map) / (max_y_map - min_y_map) if (max_y_map - min_y_map) != 0 else 0
        # 将归一化坐标转换为地图矩阵的索引
        I = int(np.clip(round(norm_x * (MAP_SIZE - 1)), 0, MAP_SIZE - 1))  # x方向索引
        J = int(np.clip(round(norm_y * (MAP_SIZE - 1)), 0, MAP_SIZE - 1))  # y方向索引
        point_density = dst_matrix[I, J] # 获取该点的密度值
        # 根据密度值计算车流量、人流量和拥挤程度
        cars = point_density * CARS_NUM_THREADS
        people = point_density * PEOPLE_NUM_THREADS
        crowding = point_density
        # 突发事件
        emergency = None
        # trafficLight红绿灯是边的数据
        return [cars, people, crowding, emergency]
        # 返回包含车流、人流、拥挤程度、紧急情况和交通灯状态的列表

    def get_road_information(self, road_id: int, density_matrix:np.ndarray):
        """
        获取指定路段的车流、人流、拥挤程度数据
        :param road_id: int
        :return:[cars, people, crowding, emergency, trafficLight]
        """
        dst_matrix = self.matrix_to_map(density_matrix) # 将密度矩阵转换为地图数据
        MAP_SIZE = dst_matrix.shape[0] # 获取地图尺寸
        if road_id < 0 or road_id >= len(self.edges): # 检查节点ID是否有效
            raise ValueError("节点id无效")
    # return [cars, people, crowding, emergency, trafficLight]


    def get_point_risk_data(self, point_id: int, density_matrix:np.ndarray):
        """
        获取指定节点的风险数据,跟天气，拥挤程度，紧急状况
        :param point_id: int
        :return: [riskData1, riskData2, ..., riskDataN],predict
        """
        cars, people, crowding, emergency = self.get_point_information(point_id, density_matrix)

        pass

    def get_road_risk_data(self, road_id: int, density_matrix:np.ndarray):
        """
        获取指定路段的风险数据
        :param road_id: int
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

# graph=Graph()
# print("\n度的邻接矩阵")
# print(graph.length)
# graph.load_json("data02.json")
# print("\n长度的邻接矩阵")
# print(graph.length)
# graph.initialize_traffic_light()
