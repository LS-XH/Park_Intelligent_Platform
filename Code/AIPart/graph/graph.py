from numpy.ma.extras import column_stack

from Interface.graph import GraphBase, PointType,Point, Edge
import math
from enum import Enum
import os
import numpy as np
import json
from graph.common_func import generate_cars_list, generate_people_list
#from .information import weather,traffic_flow,human_flow,accident_rate

CARS_NUM_THREADS = 30
PEOPLE_NUM_THREADS = 3000
GREEN_LIGHT_TIMES = 27
YELLOW_LIGHT_TIMES = 3
ALLOW_LIGHT_TIMES = GREEN_LIGHT_TIMES+YELLOW_LIGHT_TIMES

class TrafficLightStatue(Enum):
    red=0,
    green=1,
    yellow=2

class Graph(GraphBase):
    def __init__(self):
        super().__init__()


        # 位置矩阵
        self.__positions = np.empty((0,2))
        # 长度矩阵
        self.__length = np.empty((0,0))
        # 权重矩阵
        self.__weight = np.empty((0,0))
        # 度矩阵
        self.__degree = np.empty((0,0))
        # 限速矩阵
        self.__limit_speed = np.empty((0,0))
        #红绿灯矩阵
        self.__traffic_light = np.empty((0,0))
        self.__tick = 0

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

    def load_json(self, path: str, start_name=None, end_name=None):
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


# # graph=Graph()
# # graph.load_json("data02.json")
# # print("\n长度的邻接矩阵")
# # print(graph.degree)
# graph = Graph()
# graph.load_json("data02.json")
# print("\n长度的邻接矩阵")
# print(graph.length)
# graph.initialize_traffic_light()