import json
from abc import abstractmethod
import numpy as np
from enum import Enum

class PointType(Enum):
    crossing=0
    station=1


class Point:
    def __init__(self,id:str,x:float,y:float,type:PointType=PointType.crossing):
        self.__id=id
        self.__x=x
        self.__y=y
        self.__type=type

    @property
    def type(self)->PointType:
        return self.__type

    @property
    def id(self)->str:
        return self.__id




class Edge:
    def __init__(self,start_id:str,end_id:str,length:float,degree:int):
        self.start = start_id
        self.end = end_id
        self.__length = length
        self.__degree = degree

    @property
    def length(self)->float:
        return self.__length
    @property
    def degree(self)->float:
        return self.__degree




class GraphBase:
    def __init__(self):
        self.__points=[]
        self.__points_id={}
        self.__edges=[]

    @abstractmethod
    def initialize_car(self):
        """
        初始化车的位置
        :return:
        """
        pass
    @abstractmethod
    def initialize_crowd(self):
        """
        初始化人的位置
        :return:
        """

    @property
    @abstractmethod
    def nodes_position(self)->np.matrix:
        """
        节点矩阵，行为序号，列0为x，列1为y
        :return: dim=2
        """
        pass

    @property
    @abstractmethod
    def length(self)->np.matrix:
        """
        道路长度邻接矩阵
        :return: dim=2
        """
        pass


    @property
    @abstractmethod
    def weight(self)->np.matrix:
        """
        权重邻接矩阵
        :return: dim=2
        """
        pass

    @property
    @abstractmethod
    def degree(self)->np.matrix:
        """
        道路数量和方向的度矩阵
        :return: dim=2
        """
        pass

    @property
    @abstractmethod
    def limit_speed(self)->np.matrix:
        pass

    @property
    @abstractmethod
    def traffic_light(self)->np.matrix:
        pass

    @property
    @abstractmethod
    def get_light(self,start_id:str="",end_id:str="")->float:
        """
        获取红绿灯所剩的时间，如果为负数，则是绿灯，其绝对值为所剩的时间
        :param start_id: 起始点的名称id
        :param end_id: 末端点（路口）的名称id
        :return:
        """
        pass
        # self.traffic_light[self.__points_id[start_id],self.__points_id[end_id]]

    @abstractmethod
    def _simulate_light(self,dt=0.1):
        pass

    @abstractmethod
    def simulate(self,dt=0.1):
        self._simulate_light(dt=dt)
        pass

    def add_point(self,id:str,x:float,y:float,type:PointType=PointType.crossing):
        self.__points.append(Point(id,x,y,type=type))
        self.__points_id[id]=len(self.__points_id)
        return True

    def add_edge(self,start_id:str,end_id:str,length:float,degree:int):
        if not (start_id in [p.id for p in self.__points] and end_id in [p.id for p in self.__points]):
            raise Exception("")

        self.__edges.append(Edge(start_id,end_id,length,degree))

    def load_json(self,path:str):
        with open(path,'r') as f:
            data=json.load(f)













