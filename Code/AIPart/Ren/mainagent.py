import numpy as np

from Code.AIPart.Interface.people import PersonBase





class Crowd(PersonBase):


    def __init__(self, position:list, graph:any ):
        self.pos = position
        self.graph=graph

    def simulate(self, happened: list = None):

        pass


    def get_pos(self, node_id: int, ranges: int = 30) -> list:
        pass

    @property
    def position(self) -> list:
        return []


    def kill(self, node_id: int, ranges: int = 20):
        pass


    @staticmethod
    def get_emergency(happened: list = None):
        if happened:
            for happened in happened:
                e=666
        return []

    @staticmethod
    def deal_trafficlight(trafficlight: np.array = None)->list:
        """

        :param trafficlight:
        :return:
        """
        light = []
        try:
            for i in range(len(trafficlight[0])):
                if max(trafficlight[:,i]) < 1:
                    light.append(1)
                else:
                    light.append(0)
            return light
        except IndexError:
            print("Traffic Light Matrix Is Empty")
            return []