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

    @property
    @abstractmethod
    def density(self):
        """
        :return: np.array(DENSITY_MATRIX_SIZE * DENSITY_MATRIX_SIZE)
        """
        pass


    @abstractmethod
    def kill(self, node_id:int, radius:int=20, num:int=30):
        """
        remove people in range
        :param node_id:
        :param radius:
        :param num:
        """
        pass

    @abstractmethod
    def birth(self, node_id: int, radius: int = 20, num: int = 30):
        """
        makeup people in range
        :param node_id:
        :param radius:
        :param num:
        """
        pass
