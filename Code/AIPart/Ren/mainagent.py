import numpy as np
import torch

from Code.AIPart.Interface.people import PersonBase
from model import ObstacleGenerator, AgentGroup

map_path = "testmap.py"



class Crowd(PersonBase):

    @staticmethod
    def create_agents(num_agents, map_matrix, intersection_id_map, obstacle_gen):
        return AgentGroup(num_agents, map_matrix, intersection_id_map, obstacle_gen)

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

    @staticmethod
    def make_obs():
        map_size = None
        obs_params = None
        obs = ObstacleGenerator(map_size)
        obs.add_obstacles(obs_params)
        return obs
    @staticmethod
    def deal_map(graph:any) -> (list, np.array, list):
        """

        :return: obstacles_params, targets, points
        """
        return None, None, None

    def __init__(self, position:list, graph:any, num_agents=1000):
        self.pos = position
        self.graph=graph
        self.obstacle = self.make_obs()
        self.num_agents = num_agents
        self.agents = self.create_agents(
            num_agents,
            self.obstacle.map_matrix,
            self.obstacle.intersection_id_map,
            self.obstacle
        )



    def simulate(self, happened: list = None, trafficlight: list = None):
        """

        :param happened: [((x,y), str), ...]
        :param trafficlight:
        :return:
        """
        emergency = self.get_emergency(happened=happened)
        eme_pos = []
        eme_lev = []
        if emergency:
            for pos, lev in emergency:
                eme_pos.append(pos)
                eme_lev.append(lev)
        self.agents.move_towards_target((eme_pos, eme_lev), trafficlight)
        pass


    def get_pos(self, node_id: int, ranges: int = 30) -> list:

        pass

    @property
    def density(self) -> float:
        return self.agents.get_density().T

    @property
    def position(self) -> list:
        positions = self.agents.positions.cpu().numpy()
        return []


    def kill(self, node_id: int, ranges: int = 20):
        pass





