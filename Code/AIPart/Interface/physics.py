from abc import abstractmethod, abstractproperty


class RigidBody:
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
    def simulate(self,dt:float=0,a_x:float=0,a_y:float=0):
        pass



