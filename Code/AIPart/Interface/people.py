from abc import abstractmethod, ABC


class PersonBase(ABC):
    @abstractmethod
    def __init__(self, position:list, graph:any, *kwargs):
        """

        :param position: [(x,y), (x,y), ]
        :param graph: 地图
        :param kwargs:
        """
        pass

    @abstractmethod
    def simulate(self, happened:list=None):
        """
        模拟一帧
        :param happened:[((x, y), accident:str), ...]
        """
        pass

    @property
    @abstractmethod
    def position(self)->list:
        """
        all position
        :return: [(x, y), (x, y), ...] in all
        """
        pass

    @abstractmethod
    def get_pos(self, node_id:int, ranges:int=30)->list:
        """
        获取节点周围的人
        :param node_id:
        :param ranges:
        :return: [(x, y), (x, y), ...] in range
        """
        pass

    @abstractmethod
    def kill(self, node_id:int, ranges:int=20):
        """
        remove people in range
        :param node_id:
        :param ranges:
        """
        pass

