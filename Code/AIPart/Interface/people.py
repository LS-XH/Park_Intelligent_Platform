from abc import abstractmethod, ABC


class PersonBase(ABC):
    @abstractmethod
    def simulate(self, dt: float = 0):
        """
        模拟一帧，获取人的属性
        :return:
        """
        pass
