from abc import abstractmethod
import numpy as np


def transform_to(vectors: np.ndarray,curt_ref:np.ndarray,obj_ref:np.ndarray) -> np.ndarray:
    return np.linalg.inv(obj_ref)@curt_ref@vectors

class RigidBody:
    def __init__(
            self,
            p_x: float = 0,
            p_y: float = 0,
            v_x: float = 0,
            v_y: float = 0,
            a_x: float = 0,
            a_y: float = 0
        ):
        self.__position = np.array([[p_x], [p_y]])
        self.__velocity = np.array([[v_x],[v_y]])
        self.__acceleration = np.array([[a_x],[a_y]])
        self.__vector_basis = np.array([[1,0],[0,1]])

    def __lt__(self, other: 'RigidBody'):
        """
        复制实例
        :param other:
        :return:
        """
        self.__position = other.position
        self.__velocity = other.__velocity
        self.__acceleration = other.__acceleration
        self.__vector_basis= other.__vector_basis

    def __add__(self, other: 'RigidBody'):
        res = RigidBody()
        res.__position = self.position + other.position
        res.__velocity = self.velocity+other.velocity
        res.__acceleration = self.acceleration+other.acceleration

        return res

    @property
    @abstractmethod
    def position(self)->np.array:
        return self.__position

    @property
    @abstractmethod
    def velocity(self)->np.array:
        return self.__velocity

    @property
    @abstractmethod
    def acceleration(self)->np.array:
        return self.__acceleration

    @property
    @abstractmethod
    def vector_basis(self)->np.array:
        return self.__vector_basis

    @vector_basis.setter
    @abstractmethod
    def vector_basis(self,obj_ref):
        self.transform(obj_ref)
        self.__vector_basis=obj_ref


    @property
    @abstractmethod
    def p_x(self):
        return self.position[0,0]
    @property
    @abstractmethod
    def p_y(self):
        return self.position[1,0]


    @property
    @abstractmethod
    def v_x(self):
        return self.velocity[0,0]
    @property
    @abstractmethod
    def v_y(self):
        return self.velocity[1,0]



    @property
    @abstractmethod
    def a_x(self):
        return self.acceleration[0,0]
    @a_x.setter
    @abstractmethod
    def a_x(self,obj_ref):
        self.__acceleration[0,0] = obj_ref
    @property
    @abstractmethod
    def a_y(self):
        return self.acceleration[1,0]
    @a_y.setter
    @abstractmethod
    def a_y(self,obj_ref):
        self.__acceleration[1,0] = obj_ref

    def transform(self,obj_ref:np.ndarray):
        """
        对此对象的物理量转移基底向量至直角坐标系，本质为一组线性变换
        :param obj_ref: 转换的目标参考系
        :return:
        """
        self.__position = transform_to(self.position, self.__vector_basis, obj_ref)
        self.__velocity = transform_to(self.velocity,self.__vector_basis,obj_ref)
        self.__acceleration = transform_to(self.acceleration,self.__vector_basis,obj_ref)

        self.__vector_basis = obj_ref


    @property
    @abstractmethod
    def simulate(self, dt: float = 0):
        pass
