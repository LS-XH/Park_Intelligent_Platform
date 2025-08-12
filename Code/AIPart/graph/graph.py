from numpy.ma.extras import column_stack
from sympy.physics.units import temperature
from Interface.graph import GraphBase, PointType,Point, Edge
import math
from enum import Enum
import os
import numpy as np
import json
import random
from graph.common_func import generate_cars_list, generate_people_list
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


        # 位置矩阵，存储图中各节点的坐标信息
        self.__positions = np.empty((0,2))
        # 长度矩阵，存储节点间的距离
        self.__length = np.empty((0,0))
        # 权重矩阵，存储节点间的权重
        self.__weight = np.empty((0,0))
        # 度矩阵，存储节点的度数
        self.__degree = np.empty((0,0))
        # 限速矩阵，存储路段的限速信息
        self.__limit_speed = np.empty((0,0))
        #红绿灯矩阵，存储红绿灯的状态信息
        self.__traffic_light = np.empty((0,0))
        # 时间计数器，用于模拟红绿灯变化
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
        # 检查输入参数
        self._check_edge(start_id, end_id)

        # 此路口灯数
        light_count = np.count_nonzero(self.__degree[:, end_id])
        # 截断求余数后，窗口期时间，如果为负数，就是红灯（露头时间长度）
        light_time = self.traffic_light[start_id,end_id]%(self.traffic_light[end_id,end_id]-self.__tick)
        # 判断是否处于窗口（去下不取上）
        if ALLOW_LIGHT_TIMES < light_time:return TrafficLightStatue.red,-light_time
        if 0 <= light_time < GREEN_LIGHT_TIMES:return TrafficLightStatue.green, GREEN_LIGHT_TIMES-light_time
        elif GREEN_LIGHT_TIMES <= light_time < ALLOW_LIGHT_TIMES:return TrafficLightStatue.yellow, ALLOW_LIGHT_TIMES-light_time

    def control_light(self,start_id: int, end_id: int, add_green_time:float):
        # 检查输入参数
        self._check_edge(start_id, end_id)
        self.__traffic_light[start_id:,end_id]+=add_green_time

    def get_light(self, start_id: int, end_id: int) -> float:
        """
        获取指定路段当前的红绿灯的通行时间
        :param start_id: 起始节点id，即为道路的方向，如果输入-1，则为询问人行道
        :param end_id: 目标节点id，即为红绿灯所在路口id
        :return:
        """
        np.where(self.__degree != 0, 1, 0) * self.__traffic_light
        column = self.__traffic_light[:start_id,end_id]
        accord = [l for l in reversed(column) if l != 0]
        return self.__traffic_light[start_id,end_id].item() - (0 if len(accord)==0 else accord[0])


    def simulate_light(self, dt=0.1)->None:
        self.__tick += dt
        self.__traffic_light -= dt

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
        self.__traffic_light = np.zeros((len(self.points), len(self.points)), dtype=int)
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