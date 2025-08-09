from typing_extensions import Optional

from Interface.physics import RigidBody
import numpy as np
import torch
import random



class Car(RigidBody):
    car_length=1
    min_distance=1
    lane_length=2


    def __init__(self,lane=0,x=0,y=0,id=""):
        super(Car,self).__init__(p_x=x,p_y=y)
        self.__obj_lane=lane
        self.id=id




    def catch_av(
        self,
        obj_car:RigidBody,
        r_v:float,
        a:float=1,
        b:float=1
        ):
        """
        纵向追赶目标车辆，线性提升加速度，直到保持预期距离
        :param obj_car:
        :param r_v:
        :param a:
        :param b:
        :return:
        """
        self.__a_v += a * ((obj_car.p_x - self.p_x - r_v) + b * (obj_car.v_x - self.v_x))



    def lane_centre_ah(self,a:float=1,b:float=1):
        """
        变道趋势，即为对齐车道中心
        :param a: 邻接权重系数
        :param b: 速度阻尼系数
        :return:
        """
        r_lane = (self.obj_lane+0.5)*self.lane_length
        a_y = a*((r_lane-self.p_y)+b*(0-self.v_y))
        self.__a_h += a_y

    def chatter_a(self,amplitude=0.01):
        """
        引入随机震荡
        :param amplitude: 震荡幅度
        :return:
        """
        self.__a_v += random.uniform(-abs(amplitude), abs(amplitude))
        self.__a_h += random.uniform(-abs(amplitude), abs(amplitude))

    def limrate_av(self,k=1,limit_rate = 20):
        if self.v_x>limit_rate:
            self.__a_v += -k * self.v_x



    @property
    def obj_lane(self)->int:
        return self.__obj_lane

    def get_lane(self):
        return int(self.p_y/Car.lane_length)

    def zero_a(self):
        self.__a_v=0
        self.__a_h=0

    def simulate(self,dt:float=0.1):
        self.__v_v+= self.a_x * dt
        self.__v_h+= self.a_y * dt

        self.__p_v+= self.v_x * dt
        self.__p_h+= self.v_y * dt

        self.zero_a()

class CavGroup:
    def __init__(self, leader:Car, follows:list[Car]):
        self.leader = leader
        self.follows = follows
    def simulate(self):
        return


class Cars(Interface.CarsBase):
    def __init__(self,graph:Optional[Interface.GraphBase],cars:list[Car]):
        self.graph = graph
        self.cars = cars

    def simulate(self,dt:float=0.1):
        for car in self.cars:
            car.zero_a()
            car.chatter_a(0.1)
            car.forward_av(scale=10)
            car.limrate_av(limit_rate=2)
            for obj_car in self.cars:
                if obj_car !=car:
                    if car.get_lane() != car.obj_lane:
                        car.lane_centre_ah(a=0.1,b=5)
                        car.avoidance_a(obj_car,safe_distance=10,k=1,av=0.5,bv=3,ah=0.1,bh=3)
                    else:
                        car.lane_centre_ah(a=0.1,b=5)
                    continue

            car.simulate(dt=dt)





