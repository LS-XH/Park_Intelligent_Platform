from sympy.physics.units.definitions import curie
from typing_extensions import Optional
import numpy as np
import torch
import random

from Interface.physics import RigidBody
from Interface.car import CarsBase
import Interface
import car.tendency as td



class Car(RigidBody):
    car_length=4
    min_distance=1
    lane_length=3


    def __init__(self,lane=0,x=0,y=0,id=""):
        super(Car,self).__init__(p_x=x,p_y=y)
        self.__obj_lane=lane
        self.id=id

        self.v_x = 0.01

        self.text = ""




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
        self.a_x += a * ((obj_car.p_x - self.p_x - r_v) + b * (obj_car.v_x - self.v_x))

    @property
    def obj_lane(self)->int:
        return self.__obj_lane

    def get_lane(self):
        return int(self.p_y/Car.lane_length)

    def zero_a(self):
        self.a_x=0
        self.a_y=0

    def simulate(self, dt: float = 0):
        super(Car,self).simulate(dt)



class CavGroup:
    def __init__(self, leader:Car, follows:list[Car]):
        self.leader = leader
        self.follows = follows
    def simulate(self):
        return


class Cars:
    def __init__(self,graph:Optional[Interface.GraphBase],cars:list[Car]):
        self.graph = graph
        self.cars = cars

        self.chatter = td.Chatter(0.1)
        self.forward = td.Forward(scale=10,bx=0.1)
        self.to_lane = td.AlignLane(ay=0.1,by=6)
        self.avoid = td.CircleAvoidance(k=1,ax=0.3,bx=1.5,ay=0.5,by=1.5,safe_distance=10)

        self.brake = td.Brake(max_speed=5)
        self.cruise = td.Brake(max_speed=10)

    def simulate(self,dt:float=0.1):
        for car in self.cars:

            if car.id == "0":
                a = 1
            if car.id == "1":
                a = 1
            if car.id == "4":
                a = 1
            if car.id == "8":
                a = 1

            car.zero_a()

            avoid = self.avoid.increment(car,self.cars)


            forward = self.forward(car)
            brake = self.brake(car)
            cruise = self.cruise(car)
            c_lane = self.to_lane.increment(car,obj_lane = car.obj_lane)
            chatter = self.chatter(car)

            if car.get_lane()==car.obj_lane:
                forward = forward > avoid
                forward = forward < avoid
                # car + cruise
            else:
                forward = forward < RigidBody(a_x=-0.1)
                forward = forward < avoid
                c_lane = c_lane < avoid
                # car + brake

            car + forward
            car + c_lane

            # car + chatter
            car.simulate(dt=dt)

            # car.text = "%.1f,%.1f"%(car.a_x,car.a_y)
            car.text = car.id

            # car.text = "%.1f,%.1f\n%.1f,%.1f\n" % (car.v_x,car.v_y,car.a_x, car.a_y)
