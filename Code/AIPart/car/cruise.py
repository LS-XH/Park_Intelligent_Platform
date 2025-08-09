from car.tendency import Tendency
from Interface.physics import RigidBody


class Suppression:
    """
    抑制模块
    """
    def __init__(self,tend:Tendency):
        self.increment = tend.increment
        return
    def __call__(self, obj:RigidBody):
        return

    def operate(self, obj:RigidBody):
        return

