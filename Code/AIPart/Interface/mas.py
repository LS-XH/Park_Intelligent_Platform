from abc import abstractmethod, ABC


class MasBase(ABC):
    def __init__(self, start_id, end_id):
        self.start_id = start_id
        self.end_id = end_id

    @abstractmethod
    def control(self):
        """
        获取当前帧的信息，检查此智能体是否需要调控
        :return:
        """
        pass

    @abstractmethod
    def simulate(self):
        """
        此智能体对环境做出的影响
        :return:
        """
