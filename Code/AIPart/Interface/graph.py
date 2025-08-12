from abc import abstractmethod, ABC
from enum import Enum

import numpy as np

class PointType(Enum):
    crossing = 0
    station = 1


class Point:
    def __init__(self, name:str, x: float, y: float, point_type: PointType = PointType.crossing):
        self.__name = name
        self.__x = x
        self.__y = y
        self.__type = point_type

    @property
    def type(self) -> PointType:
        # 返回点的类型
        return self.__type
    @property
    def name(self) -> str:
        return self.__name
    @property
    def x(self) -> float:
        return self.__x
    @property
    def y(self) -> float:
        return self.__y

class Edge:
    def __init__(self, start_id: int, end_id: int,degree: int,  limit_speed: float,  car_num: list = None):
        self.__start_id = start_id
        self.__end_id = end_id
        self.__degree = degree
        self.__limit_speed = limit_speed

    @property
    def start_id(self) -> int:
        """
        起始节点id
        :return:
        """
        return self.__start_id
    @property
    def end_id(self) -> int:
        """
        边的终止节点id
        :return:
        """
        return self.__end_id
    @property
    def limit_speed(self) -> float:
        """
        边的限速
        :return:
        """
        return self.__limit_speed
    @property
    def lane_degree(self):
        """
        道路度数，即为此道路单方向的车道数（假设来回车道数相等）
        :return:
        """
        return self.__degree



class GraphBase(ABC):
    def __init__(self):
        self.__points:list[Point] = []
        self.__edges:list[Edge] = []

        # name -> id
        self.__points_id:dict[str, int] = {}

    def __getitem__(self, point_id):
        """
        通过点的id，获取点的名称name
        :param point_id: 点的id
        :return: 点的名称
        """
        return self.__points[point_id].name

    @property
    def points(self)->list[Point]:
        """
        返回此图所有点Point类型的list，索引即为点的id
        :return:
        """
        return self.__points
    @property
    def edges(self)->list[Edge]:
        """
        返回此图所有的边Edge类型的list
        :return:
        """
        return self.__edges
    @property
    def point_name2id(self)->dict[str, int]:
        """
        点的名字到数字id（列表索引）的映射
        :return:
        """
        return self.__points_id

    @property
    @abstractmethod
    def nodes_position(self) -> np.ndarray:
        """
        节点矩阵，行为序号，列0为x，列1为y
        :return: dim=2
        """
        pass
    @property
    @abstractmethod
    def length(self) -> np.ndarray:
        """
        道路长度邻接矩阵
        :return: dim=2
        """
        pass
    @property
    @abstractmethod
    def weight(self) -> np.ndarray:
        """
        权重邻接矩阵
        :return: dim=2
        """
        pass
    @property
    @abstractmethod
    def degree(self) -> np.ndarray:
        """
        道路数量和方向的度矩阵
        :return: dim=2
        """
        pass
    @property
    @abstractmethod
    def limit_speed(self) -> np.ndarray:
        """
        限速邻接矩阵
        :return:
        """
        pass
    @property
    @abstractmethod
    def traffic_light(self) -> np.ndarray:
        """
        红绿灯矩阵，最好别用，建议用get_light

        储存格式:
            路口：[start_id->end_id]

            人行道：[end_id->end_id]


        操作方法:
            每一tick上市一格，判断0~green范围内的元素，要用最大值-总时间戳tick求余数


        时长控制:
            完成取余(截断）操作后，即每一个元素都在0~max_0(取余后的值，为$$x%(max_t-tick=max_0)$$)

            行数更小的上一个元素，到本元素的差值，为本元素位置的红绿灯时长

            如果上一个元素不存在（即本元素为首行），即以tick（相对0点）作差
        :return:
        """
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


    def _add_point(self, name: str , x: float, y: float, point_type: PointType = PointType.crossing)->int:
        """
        新增一个节点
        :param name: 节点名称
        :param x: 节点x坐标
        :param y: 节点y坐标
        :param point_type: 节点类型
        :return: 节点的id
        """
        self.__points.append(Point(name, x, y, point_type=point_type))
        self.__points_id[name] = len(self.__points_id)
        return len(self.__points_id)-1
    def _add_edge(self, start_id: int, end_id: int,degree:int,  limit_speed: float):
        """
        连接两条现有节点，新增一个边
        :param start_id: 起始节点id
        :param end_id: 终止节点id
        :param degree: 道路度数，即为此道路单方向的车道数（假设来回车道数相等）
        :param limit_speed: 道路限速
        :return: None
        """
        if start_id >= len(self.__points) or end_id >= len(self.__points):
            raise Exception("未找到输入的节点，请先添加节点")

        self.__edges.append(Edge(start_id, end_id,degree, limit_speed))

    def _check_edge(self,start_id: int, end_id: int):
        """
        判断节点的id是否在__points_id的值域中
        :param start_id: 起始节点id
        :param end_id: 终止节点id
        :return:
        """
        if start_id >= len(self.points) or end_id >= len(self.points):
            raise ValueError(f"节点 {self[start_id]} 或 {self[end_id]} 不存在")
        if self.degree[start_id, end_id] == 0:
            raise ValueError(f"节点 {self[start_id]} 到 {self[end_id]} 之间没有路段")

