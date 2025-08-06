from abc import abstractmethod


class CarBase:
    @abstractmethod
    def generate_a(self)->[float,float]:
        pass

    @abstractmethod
