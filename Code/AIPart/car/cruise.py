from abc import ABC, abstractmethod
from car.tendency import Tendency
from Interface.physics import RigidBody




class Avoidence(Suppression):
    def __init__(
            self,
            tendency:Tendency
    ):
        super().__init__(tendency)




