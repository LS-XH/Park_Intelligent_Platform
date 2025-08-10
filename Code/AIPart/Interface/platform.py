from abc import abstractmethod, ABC


class PlatformBase(ABC):
    def __init__(self):
        return

    @property
    @abstractmethod
    def dt(self):
        """
        模拟时一帧经过的时间
        :return:
        """
        pass

    @property
    @abstractmethod
    def time_stamp(self):
        """
        全局时间戳
        :return:
        """
        pass

    @abstractmethod
    def run(self):
        """
        开始模拟进程的主函数
        :return:
        """
        pass

    @abstractmethod
    def pause(self):
        """
        预留的函数，客户端暂停模拟时
        :return:
        """
        pass
