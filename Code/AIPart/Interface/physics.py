from abc import abstractmethod
import numpy as np
from sympy.abc import delta


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
            a_y: float = 0,
            vector_basis:np.ndarray = np.array([[1,0],[0,1]],dtype=np.float64)
        ):
        self.__position = np.array([[p_x], [p_y]],dtype=np.float64)
        self.__velocity = np.array([[v_x],[v_y]],dtype=np.float64)
        self.__acceleration = np.array([[a_x],[a_y]],dtype=np.float64)
        self.__vector_basis = np.array([[1,0],[0,1]],dtype=np.float64) if vector_basis is None else vector_basis

    def __lt__(self, other: 'RigidBody')->'RigidBody':
        """
        减法抑制
        :param other: 用于抑制的增量
        :return: 抑制后的增量
        """
        res = RigidBody()
        def minabs(obj,adj):
            if adj == 0:
                return obj

            #不能同号，只能异号（相减抑制）
            if obj * adj > 0:
                return obj
            #抑制效果不能大于本身（改变原有方向）
            if abs(obj) < abs(adj):
                return 0

            return obj+adj

        res.p_x = minabs(self.p_x,other.p_x)
        res.p_y = minabs(self.p_y,other.p_y)
        res.v_x = minabs(self.v_x,other.v_x)
        res.v_y = minabs(self.v_y,other.v_y)
        res.a_x = minabs(self.a_x,other.a_x)
        res.a_y = minabs(self.a_y,other.a_y)
        return res
    def __gt__(self, other: 'RigidBody')->'RigidBody':
        """
        加法增强
        :param other: 用于增强的增量
        :return: 增强后的增量
        """
        res = RigidBody()
        def minabs(obj, adj):
            if adj == 0:
                return obj

            # 不能异号，只能同号（相加增强）
            if obj * adj <= 0:
                return obj

            return obj + adj

        res.p_x = minabs(self.p_x, other.p_x)
        res.p_y = minabs(self.p_y, other.p_y)
        res.v_x = minabs(self.v_x, other.v_x)
        res.v_y = minabs(self.v_y, other.v_y)
        res.a_x = minabs(self.a_x, other.a_x)
        res.a_y = minabs(self.a_y, other.a_y)
        return res

    def __mul__(self, other: 'RigidBody')->'RigidBody':
        """
                加法增强
                :param other: 用于增强的增量
                :return: 增强后的增量
                """
        res = RigidBody()

        def minabs(obj, adj):
            if adj == 0:
                return obj

            # 不能异号，只能同号（相加增强）
            if obj * adj <= 0:
                return obj

            return obj * adj

        res.p_x = minabs(self.p_x, other.p_x)
        res.p_y = minabs(self.p_y, other.p_y)
        res.v_x = minabs(self.v_x, other.v_x)
        res.v_y = minabs(self.v_y, other.v_y)
        res.a_x = minabs(self.a_x, other.a_x)
        res.a_y = minabs(self.a_y, other.a_y)
        return res
    def __add__(self, other: 'RigidBody')->'RigidBody':
        res = RigidBody()
        res.__position = self.position + other.position
        res.__velocity = self.velocity + other.velocity
        res.__acceleration = self.acceleration + other.acceleration
        return res
    def __iadd__(self, other: 'RigidBody'):
        """
        原地操作，相当于+=，会改变原来的变量
        :param other:
        :return:
        """
        self.__position = self.position + other.position
        self.__velocity = self.velocity+other.velocity
        self.__acceleration = self.acceleration+other.acceleration

        return self

    def __repr__(self):
        return f"Points({self.p_x},{self.p_y}),Velocity({self.v_x}，{self.v_y},Acceleration({self.a_x},{self.a_y}))"
    @property
    @abstractmethod
    def position(self)->np.ndarray:
        return self.__position

    @property
    @abstractmethod
    def velocity(self)->np.ndarray:
        return self.__velocity

    @property
    @abstractmethod
    def acceleration(self)->np.ndarray:
        return self.__acceleration

    @property
    @abstractmethod
    def vector_basis(self)->np.ndarray:
        return self.__vector_basis
    @vector_basis.setter
    @abstractmethod
    def vector_basis(self,obj_ref):
        self.transform(obj_ref)
        self.__vector_basis=obj_ref




    @property
    @abstractmethod
    def p_x(self)->float:
        return self.position[0,0].item()
    @p_x.setter
    @abstractmethod
    def p_x(self,p_x):
        self.__position[0,0] = p_x

    @property
    @abstractmethod
    def p_y(self)->float:
        return self.position[1,0].item()
    @p_y.setter
    @abstractmethod
    def p_y(self,p_y):
        self.__position[1,0] = p_y


    @property
    @abstractmethod
    def v_x(self)->float:
        return self.velocity[0,0].item()
    @v_x.setter
    @abstractmethod
    def v_x(self,v_x):
        self.__velocity[0,0] = v_x

    @property
    @abstractmethod
    def v_y(self)->float:
        return self.velocity[1,0].item()
    @v_y.setter
    @abstractmethod
    def v_y(self,v_y):
        self.__velocity[1,0] = v_y



    @property
    @abstractmethod
    def a_x(self)->float:
        return self.acceleration[0,0].item()
    @a_x.setter
    @abstractmethod
    def a_x(self,a_x):
        self.__acceleration[0,0] = a_x

    @property
    @abstractmethod
    def a_y(self)->float:
        return self.acceleration[1,0].item()
    @a_y.setter
    @abstractmethod
    def a_y(self,a_y):
        self.__acceleration[1,0] = a_y

    def transform(self,obj_ref:np.ndarray=np.array([[1,0],[0,1]])):
        """
        对此对象的物理量转移基底向量至直角坐标系，本质为一组线性变换
        :param obj_ref: 转换的目标参考系
        :return:
        """
        self.__position = transform_to(self.position, self.__vector_basis, obj_ref)
        self.__velocity = transform_to(self.velocity,self.__vector_basis,obj_ref)
        self.__acceleration = transform_to(self.acceleration,self.__vector_basis,obj_ref)

        self.__vector_basis = obj_ref

        return self


    def simulate(self, dt: float = 0):
        self.__velocity += self.__acceleration * dt
        self.__position += self.__velocity * dt
