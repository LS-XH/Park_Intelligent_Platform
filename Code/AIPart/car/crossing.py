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
            cars:list[Car]
    ):
        Delegation.__init__(self,cars)
        self.point_id = point_id




    def simulate(self,dt=0.1):
        for car in self.cars:
            return








