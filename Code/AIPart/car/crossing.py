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



    def simulate(self,ds=0.1):
        """

        :param ds: 步过弧长
        :return:
        """
        res = []
        for i,car in enumerate(self.cars):
            circle = self.graph.crossing_turn[self.point_id,car.from_id,car.to_id]
            centre = circle["centre"]
            begin = circle["from"]
            end = circle["to"]

            dtheta = ds*self.step_tick[i]/circle["radius"]
            sin = np.sin(dtheta)
            cos = np.cos(dtheta)

            now = np.array([[cos,-sin],[sin,cos]]) @ begin.reshape((2,1))

            if np.cross(begin,end)*np.cross(now,end)<0:
                self.send(car)
                res.append(car)

            car.p_x = now[0,0]
            car.p_y = now[1,0]


        return res








