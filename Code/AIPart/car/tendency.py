from abc import abstractmethod,ABC
from typing import Callable
import numpy as np
import random
import math

from Interface.physics import RigidBody

__all__=[
    'Tendency',
    'Suppression',

]




class Tendency:
    """
    趋势模块，要求override increment函数，返回增量
    """
    def __init__(self):
        return
    def __call__(self,obj:RigidBody,*args)->RigidBody:
        return self.increment(obj,*args)

    @abstractmethod
    def increment(self,obj:RigidBody,*args)->RigidBody:
        pass




class Suppression(Tendency):
    """
    抑制模块，要求init中有tendency参数，要求override increment函数，增量不会改变原tendency的方向和最大值
    如果tendency为None，则功能等同于他的父类Tendency
    如果tendency为一个模块，则会反向链式调用所有suppression的抑制效果，直到调用到tendency
    """
    def __init__(
            self,
            tendency:Tendency
    ):
        super().__init__()
        self.tendency=tendency

    @abstractmethod
    def increment(self,obj:RigidBody,*args)->RigidBody:
        pass

    def __call__(self,obj:RigidBody,*args)->RigidBody:
        if self.tendency is None:
            return super().__call__(obj)
        else:
            adjust = self.increment(obj)
            delta = self.tendency(obj)
            return delta < adjust






class Forward(Tendency):
    def __init__(
            self,
            scale:float=1,
            bx:float=1,
            cruise_speed:float=1
    ):
        """
        前行趋势
        :param scale: 前行趋势的规模
        :param cruise_speed: 巡航速度
        :return:
        """
        self.__scale = scale
        self.__bx = bx
        self.__cruise_speed = cruise_speed

        super().__init__()
        return

    def increment(self,obj:RigidBody,*args):
        return RigidBody(a_x=self.__scale-self.__bx*obj.v_x)

class AlignCar(Tendency):
    def __init__(
            self,
            r_h,
            ah: float = 1,
            bh: float = 1
    ):
        """
        横向对齐前方的一个目标
        :param r_h: 期望横向偏移量
        :param ah: 邻接权重系数
        :param bh: 速度阻尼系数
        :return:
        """
        super().__init__()
        
        self.__ah = ah
        self.__bh = bh
        self.__r_h = r_h

    def increment(self,obj:RigidBody,target_car:RigidBody = None):
        if target_car is None:
            raise Exception("please add target_car")
        return RigidBody(
            a_x = self.__ah * ((target_car.p_y - obj.p_y - self.__r_h) + self.__bh * (target_car.v_y - obj.v_y))
        )

class CircleAvoidance(Tendency):
    def __init__(
            self,
            k: float = 1,
            ax: float = 1,
            bx: float = 1,
            ay: float = 1,
            by: float = 1,
            safe_distance: float = 3,
            crash_distance: float = 1,
            epsilon : float = 0.01
    ):
        """

        :param k:
        :param ax:
        :param bx:
        :param ay:
        :param by:
        :param safe_distance:
        :param crash_distance:
        :param epsilon:
        """
        super().__init__()
        self.__k = k
        self.__ax = ax
        self.__bx = bx
        self.__ay = ay
        self.__by = by
        self.__safe_distance = safe_distance
        self.__crash_distance = crash_distance
        self.__epsilon = epsilon


    def increment(self, obj: RigidBody, target_cars: list[RigidBody] = None):
        if target_cars is None:
            raise Exception("please add target_car")

        a_x, a_y = 0, 0
        for target_car in target_cars:
            distance = max(np.linalg.norm(target_car.position - obj.position).item(),self.__epsilon)
            if (target_car.p_x != obj.p_x or target_car.p_y != obj.p_y or obj!=target_car) and distance < self.__safe_distance:
                distance_k = (1 / (max(distance - self.__crash_distance, self.__epsilon)) - 1 / self.__safe_distance)
                distance_i = (self.__safe_distance**2/distance-self.__safe_distance)
                distance_xi = (self.__safe_distance**2/max(abs(obj.p_x-target_car.p_x),self.__epsilon)-self.__safe_distance)
                distance_l = (self.__safe_distance/distance-1)
                distance_in=(1/(max(distance,self.__epsilon)))

                a_x = max(self.__ax*distance_xi*(obj.p_x-target_car.p_x)/distance-self.__bx*(obj.v_x-target_car.v_x),a_x,key=abs)

                a_y = max(self.__ay*(obj.p_y-target_car.p_y)/distance-self.__by*(obj.v_y-target_car.v_y),a_y,key=abs)
        return RigidBody(
            a_x=self.__k * a_x,
            a_y=self.__k * a_y,
        )


class Avoidance(Tendency):
    def __init__(
            self,
            k: float = 1,
            ax: float = 1,
            bx: float = 1,
            ay: float = 1,
            by: float = 1,
            safe_distance: float = 3,
            crash_distance: float = 2,
            epsilon: float = 0.01
    ):
        """
        躲避障碍物
        :param k: 整体权重
        :param ax: 横向邻接权重系数
        :param bx: 横向速度阻尼系数
        :param ay: 纵向邻接权重系数
        :param by: 纵向速度阻尼系数
        :param safe_distance: 安全距离
        :param crash_distance: 碰撞距离
        :param epsilon: 防止分母为0的增量
        :return:
        """
        Tendency.__init__(self)

        self.__k = k
        self.__ax = ax
        self.__bx = bx
        self.__ay = ay
        self.__by = by
        self.__safe_distance = safe_distance
        self.__crash_distance = crash_distance
        self.__epsilon = epsilon
        return
    def increment(self,obj:RigidBody,target_cars:list[RigidBody] = None):
        if target_cars is None:
            raise Exception("please add target_car")

        a_x, a_y = 0, 0
        for target_car in target_cars:
            if target_car.p_x!=obj.p_x and abs(target_car.p_x-obj.p_x)<self.__safe_distance and target_car.p_y!=obj.p_y and abs(target_car.p_y-obj.p_y)<self.__safe_distance:
                # a_x += self.__ax*((target_car.p_x-obj.p_x-self.__safe_distance)-self.__bx*(target_car.v_x-obj.v_x))
                a_x += self.__ax*((1/(max(target_car.p_x-obj.p_x-self.__crash_distance,self.__epsilon))-1/self.__safe_distance)
                                  -self.__bx*(target_car.v_x-obj.v_x))

                a_y += self.__ay*((1/(max(target_car.p_y-obj.p_y-self.__crash_distance,self.__epsilon))-1/self.__safe_distance)
                                  -self.__by*(target_car.v_y-obj.v_y))

        return RigidBody(
            a_x=self.__k*a_x,
            a_y=self.__k*a_y,
        )



class AlignLane(Tendency):
    def __init__(
            self,
            lane_length:float=3,
            ay: float = 1,
            by: float = 1
    ):
        """
        变道趋势，即为对齐车道中心
        :param ay: 邻接权重系数
        :param by: 速度阻尼系数
        :return:
        """
        super().__init__()
        self.__lane_length = lane_length
        self.__ay = ay
        self.__by = by
        return

    def increment(self,obj:RigidBody,obj_lane:int = 0):
        r_lane = (obj_lane+0.5)*self.__lane_length
        a_y = self.__ay * ((r_lane-obj.p_y) + self.__by * (0 - obj.v_y))
        return RigidBody(
            a_y=a_y
        )

class Chatter(Tendency):
    def __init__(
            self,
            amplitude=0.01
    ):
        """
        引入随机震荡
        :param amplitude: 震荡幅度
        :return:
        """
        super().__init__()
        self.__amplitude = amplitude

    def increment(self,obj:RigidBody,*args) ->RigidBody:
        rd_x = random.uniform(-abs(self.__amplitude), abs(self.__amplitude))
        rd_y = random.uniform(-abs(self.__amplitude), abs(self.__amplitude))
        return RigidBody(
            a_x=rd_x,
            a_y=rd_y
        )

class Brake(Tendency):
    def __init__(
            self,
            max_speed:float=1,
            k:float=1,
    ):
        """

        :param max_speed: 最大速度
        :param k: 刹车阻尼系数
        """
        super().__init__()
        self.__max_speed = max_speed
    def increment(self,obj:RigidBody,*args) ->RigidBody:
        a_x = 0
        if abs(obj.v_x)>self.__max_speed:
            a_x = (abs(obj.v_x) - self.__max_speed)*(-1 if obj.v_x>0 else 1)

        return RigidBody(
            a_x=a_x,
        )
class Cruise(Tendency):
    def __init__(
            self,
            bx:float=1,
            cruise_speed:float=1,
    ):
        """

        :param bx: 速度阻尼系数
        :param cruise_speed: 巡航速度
        """
        super().__init__()
        self.__bx = bx
        self.__cruise_speed = cruise_speed

    def increment(self,obj:RigidBody,*args) ->RigidBody:
        return RigidBody(
            a_x=0 if abs(obj.v_x)<self.__cruise_speed else -self.__bx*(abs(obj.v_x)-self.__cruise_speed)**2,
        )



