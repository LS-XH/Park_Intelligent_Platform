from abc import abstractmethod, ABC
from enum import Enum

import numpy as np

class PointType(Enum):
    crossing = 0
    station = 1


class Point:
    def __init__(self, id: int, name:str, x: float, y: float, degree: int, type: PointType = PointType.crossing):
        self.__id = id
        self.__name = name
        self.__x = x
        self.__y = y
        self.__degree = degree
        self.__type = type

    @property
    def type(self) -> PointType:
        # 返回点的类型
        return self.__type

    @property
    def name(self) -> str:
        return self.__name

    @property
    def id(self) -> int:
        # 返回点的id
        return self.__id

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    def degree(self) -> int:
        return self.__degree


class Edge:
    def __init__(self, start_name: str, end_name: str, length: float,  limit_speed: float, start_id: int, end_id: int, car_num: list = None):
        self.__start_id = start_id
        self.__end_id = end_id
        self.__length = length
        self.__length = length
        self.__limit_speed = limit_speed
        self.__start_name = start_name
        self.__end_name = end_name

    @property
    def length(self) -> float:
        return self.__length

    @property
    def start_id(self) -> int:
        return self.__start_id

    @property
    def end_id(self) -> int:
        return self.__end_id

    @property
    def start_name(self) -> str:
        return self.__start_name

    @property
    def end_name(self) -> str:
        return self.__end_name

    @property
    def limit_speed(self) -> float:
        return self.__limit_speed


class GraphBase(ABC):
    def __init__(self):
        self.__points = []
        self.__points_id = {}
        self.__edges = []

    @property
    @abstractmethod
    def nodes_position(self) -> np.matrix:
        """
        节点矩阵，行为序号，列0为x，列1为y
        :return: dim=2
        """
        pass

    @property
    @abstractmethod
    def length(self) -> np.matrix:
        """
        道路长度邻接矩阵
        :return: dim=2
        """
        pass

    @property
    @abstractmethod
    def weight(self) -> np.matrix:
        """
        权重邻接矩阵
        :return: dim=2
        """
        pass

    @property
    @abstractmethod
    def degree(self) -> np.matrix:
        """
        道路数量和方向的度矩阵
        :return: dim=2
        """
        pass

    @property
    @abstractmethod
    def limit_speed(self) -> np.matrix:
        n = len(self.__points)
        speed_mat = np.zeros((n, n))
        for edge in self.__edges:
            i = self.__points_id[edge.start_id]
            j = self.__points_id[edge.end_id]
            speed_mat[i, j] = edge.limit_speed
        return np.matrix(speed_mat)

    @property
    @abstractmethod
    def traffic_light(self) -> np.matrix:
        pass

    @abstractmethod
    def get_light(self, start_id, end_id) -> float:
        """
        获取红绿灯所剩的时间，如果为负数，则是绿灯，其绝对值为所剩的时间
        :param start_id: 起始点的名称id
        :param end_id: 末端点（路口）的名称id
        :return:
        """
        pass
        # self.traffic_light[self.__points_id[start_id],self.__points_id[end_id]]

    def simulate_light(self, dt=0.1):
        pass

    def simulate(self, dt=0.1):
        self.simulate_light(dt=dt)
        pass

    def add_point(self, id: int, name: str , x: float, y: float, degree:int, type: PointType = PointType.crossing):
        self.__points.append(Point(id,name, x, y, degree, type=type))
        self.__points_id[id] = len(self.__points_id)
        return True

    def add_edge(self, start_id: int, end_id: int, length: float,  limit_speed: float):
        if not (start_id in [p.id for p in self.__points] and end_id in [p.id for p in self.__points]):
            raise Exception("")

        self.__edges.append(Edge(start_id, end_id, length, limit_speed))

    def load_json(self, path: str):
        pass

