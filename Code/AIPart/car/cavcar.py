from typing_extensions import Optional

import Interface
import numpy as np
import torch



class Car(Interface.RigidBody):
    car_length=1
    min_distance=1
    lane_length=2

    def __init__(self,lane=0,x=0,y=0):
        self.__p_x=0
        self.__p_y=0

        self.__v_x=0
        self.__v_y=0

        self.__a_x=0
        self.__a_y=0

        self.init_pos(x,y)
        self.lane=lane

    def init_pos(self,x,y):
        self.__p_x=x
        self.__p_y=y



    def forward_a(self,scale:float=1):
        self.__a_x+=1

    def align_a(self,obj_car:Interface.RigidBody,r_y,a:float=1,b:float=1):
        self.__a_y += a*((obj_car.p_y-self.p_y-r_y)+b*(obj_car.v_y-self.v_y))
    def catch_a(self,obj_car:Interface.RigidBody,r_x,a=1,b=1):
        self.__a_x += a*((obj_car.p_x-self.p_x-r_x)+b*(obj_car.v_x-self.v_x))


    def avoidance_a(self,obj_car:Interface.RigidBody,safe_distance:float=min_distance,k=1,b=1):
        a_x,a_y=0,0
        if abs(obj_car.p_x-self.p_x)<safe_distance:
            a_x = k*((obj_car.p_x-self.p_x)-b*(obj_car.v_x-self.v_x))
        if abs(obj_car.p_y-self.p_y)<safe_distance:
            a_y = k*((obj_car.p_y-self.p_y)-b*(obj_car.v_y-self.v_y))
        self.__a_x += a_x
        self.__a_y += a_y

    def lane_a(self,a:float=1,b:float=1):
        """
        变道趋势
        :param a: 邻接权重系数
        :param b: 速度阻尼系数
        :return:
        """
        # r_lane = (self.lane+0.5)*self.lane_length
        r_lane = 1
        a_y = a*((r_lane-self.p_y)+b*(0-self.v_y))
        self.__a_y += a_y


    @property
    def p_x(self)->float:
        return self.__p_x
    @property
    def p_y(self)->float:
        return self.__p_y
    @property
    def v_x(self)->float:
        return self.__v_x
    @property
    def v_y(self)->float:
        return self.__v_y
    @property
    def a_x(self)->float:
        return self.__a_x
    @property
    def a_y(self)->float:
        return self.__a_y

    def zero_a(self):
        self.__a_x=0
        self.__a_y=0

    def simulate(self,dt:float=0.1):
        self.__v_x+=self.a_x*dt
        self.__v_y+=self.a_y*dt

        self.__p_x+=self.v_x*dt
        self.__p_y+=self.v_y*dt

        self.zero_a()

class CavGroup:
    def __init__(self, leader:Car, follows:list[Car]):
        self.leader = leader
        self.follows = follows
    def simulate(self):
        self.leader.zero_a()
        self.leader.lane_a()

        for follow in self.follows:
            follow.align_a(self.leader,0)


class Cars(Interface.CarsBase):
    def __init__(self,graph:Optional[Interface.GraphBase],cars:list[Car]):
        self.graph = graph
        self.cars = cars

    def simulate(self,dt:float=0.1):
        for car in self.cars:
            car.zero_a()
            car.lane_a(a=0.3,b=5)
            car.forward_a()
            for obj_car in self.cars:
                if obj_car !=car:
                    car.avoidance_a(obj_car)

            car.simulate(dt=dt)





