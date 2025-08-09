from Interface.graph import GraphBase, PointType,Point, Edge
import math
import numpy as np
import json
from .common_func import generate_cars_list, generate_people_list

CARS_NUM_THREADS = 30
PEOPLE_NUM_THREADS = 3000

class Graph(GraphBase):
    def __init__(self):
        super().__init__()
        self.__points=[]
        self.__points_id={}
        self.__edges=[]

        self.__positions = np.empty((0,2))
        self.__length = np.empty((0,0))
        self.__weight = np.empty((0,0))
        self.__degree = np.empty((0,0))
        self.__limit_speed = np.empty((0,0))


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
        """
        节点矩阵，行为序号，列0为x，列1为y
        :return: dim=2
        """
        return np.array(self.__positions)

    @property
    def length(self)->np.ndarray:
        """
        道路长度邻接矩阵
        :return: dim=2
        """
        return np.array(self.__length)

    @property
    def weight(self)->np.ndarray:
        """
        权重邻接矩阵
        :return: dim=2
        """
        return np.array(self.__weight)

    @property
    def degree(self)->np.ndarray:
        """
        道路数量和方向的度矩阵
        :return: dim=2
        """
        return np.array(self.__degree)

    @property
    def limit_speed(self)->np.ndarray:

        return np.array(self.__limit_speed)

    @property
    def traffic_light(self)->np.ndarray:
        pass


    @property
    def get_light(self,start_id:str="",end_id:str="")->float:
        """
        获取红绿灯所剩的时间，如果为负数，则是绿灯，其绝对值为所剩的时间
        :param start_id: 起始点的名称id
        :param end_id: 末端点（路口）的名称id
        :return:
        """
        pass
        # self.traffic_light[self.__points_id[start_id],self.__points_id[end_id]]

    def add_point(self,id:str,x:float,y:float,type:PointType=PointType.crossing):
        if id in self.__points_id:
            raise Exception(f"节点 {id} 已存在")
        self.__points.append(Point(id,x,y,type=type))
        self.__points_id[id]=len(self.__points_id)

        n = len(self.__points)
        self.__positions = np.vstack([self.__positions,[x,y]]) - 1

        new_length=np.zeros((n,n))
        new_degree=np.zeros((n,n))
        new_limit_speed=np.zeros((n,n))
        new_weight=np.zeros((n,n))
        if n>1:
            old_n=n-1
            new_length[0:old_n,0:old_n]=self.__length
            new_degree[0:old_n,0:old_n]=self.__degree
            new_limit_speed[0:old_n,0:old_n]=self.__limit_speed

        self.__length = new_length
        self.__degree = new_degree
        self.__limit_speed = new_limit_speed

        return True

    def add_edge(self,start_id:str,end_id:str,length:float,degree:int,limit_speed:float):
        if not (start_id in [p.id for p in self.__points] and end_id in [p.id for p in self.__points]):
            raise Exception(f"起始点 {start_id} 或终点 {end_id} 不存在于已添加的节点中")

        i = self.__points_id[start_id]
        j = self.__points_id[end_id]
        self.__length[i,j] = length
        self.__degree[i,j] = degree
        self.__limit_speed[i,j] = limit_speed

        #self.__weight[i,j] =

        self.__edges.append(Edge(start_id,end_id,length,degree,limit_speed))


    def load_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for point_id, point_info in data['points'].items():
            id = str(point_id)
            x = float(point_info['x'])
            y = float(point_info['y'])
            type = PointType(point_info['type'])
            self.add_point(id, x, y, type=type)

        for edge_key, edge_info in data['edges'].items():
            start_id, end_id = edge_key.split('->')
            start_id = str(start_id)
            end_id = str(end_id)

            degree = int(edge_info['degree'])
            limit_speed = float(edge_info['limit_speed'])

            start_point = self.__points[self.__points_id[start_id]]
            end_point = self.__points[self.__points_id[end_id]]
            length = math.sqrt(
                (end_point.x - start_point.x) ** 2 +
                (end_point.y - start_point.y) ** 2
            )

            self.add_edge(
                start_id = start_id,
                end_id = end_id,
                length = length,
                degree = degree,
                limit_speed = limit_speed
            )

graph=Graph()
graph.load_json("data.json")
print("\n长度的邻接矩阵")
print(graph.length)