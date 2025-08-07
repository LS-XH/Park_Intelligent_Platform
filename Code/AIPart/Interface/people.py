from abc import abstractmethod


class PersonBase:

    @abstractmethod
    def __init__(self, total_number:int, graph:any, *kwargs):
        """

        :param total_number: 总人数
        :param graph: 地图
        :param kwargs:
        """
        pass

    @abstractmethod
    def simulate(self, happened:list=None):
        """
        模拟一帧，获取人的属性
        :param happened:[{"position", "type"}, ...]
        """
        pass

    @property
    @abstractmethod
    def position(self)->dict:
        """
        all position
        :return: dict{id: (x, y), id: (x, y), ...} in all
        """
        pass

    @abstractmethod
    def get_pos(self, node_id:int, ranges:int=20)->dict:
        """
        获取节点周围的人
        :param node_id:
        :param ranges:
        :return: dict{id: (x, y), id: (x, y), ...} in range
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