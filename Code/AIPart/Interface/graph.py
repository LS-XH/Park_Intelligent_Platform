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
        self.__edges=[]


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
    def nodes(self)->np.matrix:
        """
        节点矩阵，行为序号，列0为x，列1为y
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

    def add_point(self,id:str,x:float,y:float,type:PointType=PointType.crossing):
        self.__points.append(Point(id,x,y,type=type))
        return True

    def add_edge(self,start_id:str,end_id:str,length:float,degree:int):
        if not (start_id in [p.id for p in self.__points] and end_id in [p.id for p in self.__points]):
            raise Exception("")

        self.__edges.append(Edge(start_id,end_id,length,degree))











