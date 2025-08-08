from abc import abstractmethod, ABC


class RigidBody(ABC):
    @property
    @abstractmethod
    def p_x(self):
        pass

    @property
    @abstractmethod
    def p_y(self):
        pass

    @property
    @abstractmethod
    def v_x(self):
        pass

    @property
    @abstractmethod
    def v_y(self):
        pass

    @property
    @abstractmethod
    def simulate(self, dt: float = 0):
        pass
