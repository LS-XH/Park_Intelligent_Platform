from typing import Optional

from sympy.physics.units.definitions import curie
from typing_extensions import Optional
import numpy as np
import torch
import random

from Interface.physics import RigidBody
from car.cavcar import Car
from car.road import Road
from Interface.car import CarsBase,Delegation
from Algorithm.bidirectional_dijkstra import bidirectional_dijkstra as find_road


import Interface
import car.tendency as td
from graph import Graph


class Crossing(Delegation):
    def __init__(
            self,
            point_id,
            graph: Graph,
            cars:list[Car]
    ):
        Delegation.__init__(self,cars)
        self.point_id = point_id

        self.graph = graph

        # 位置与self.cars一一对应
        self.step_tick = []


    def append(self,transfer:Car):
        self.step_tick.append(0)
        Delegation.append(self,transfer)

    def back(self,transfer:Car):
        del self.step_tick[self.cars.index(transfer)]
        Delegation.back(self,transfer)



    def simulate(self,dt=0.01):
        """

        :param ds: 步过弧长
        :return:
        """
        res = []
        for i,car in enumerate(self.cars):
            circle = self.graph.crossing_turn[self.point_id,car.from_id,car.to_id]
            # 旋转圆弧圆心
            centre = circle["centre"]

            # 起始向量
            begin = circle["from"]
            # 终止向量
            end = circle["to"]

            # 步过时间戳*速度绝对值*dt/半径 = 弧度增量
            dv = np.linalg.norm(car.velocity.flatten())
            dtheta = dt*dv*self.step_tick[i]/circle["radius"]
            sin = np.sin(dtheta)
            cos = np.cos(dtheta)

            # 将起始向量begin旋转角
            now = np.array([[cos,-sin],[sin,cos]]) @ begin.reshape((2,1)).flatten()



            car.p_x = now[0]+centre[0]
            car.p_y = now[1]+centre[1]
            self.step_tick[i] += 1

            # 旋转后不能超过end，超过则为结束此委托，会改变速度方向到与当前弧相同
            if np.cross(begin, end) * np.cross(now, end) < 0:
                new_v = self.graph.road_basic[self.point_id,car.to_id][:,0] * dv
                car.a_x = 0
                car.a_y = 0

                car.v_x = new_v[0]
                car.v_y = new_v[1]

                self.send(car)
                res.append(car)


        return res








