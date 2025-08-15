import numpy as np
import torch

from Code.AIPart.Interface.people import PersonBase
from Code.AIPart.Ren.csxiaofeng import crowd_evacuation
from Code.AIPart.Ren.agents import AgentGroup
from Code.AIPart.Ren.obstacles import ObstacleGenerator
from Code.AIPart.Ren.config import *

map_path = "testmap.py"



class Crowd(PersonBase):

    @staticmethod
    def create_agents(num_agents, map_matrix, intersection_id_map, obstacle_gen, targets, targets_heat):
        return AgentGroup(num_agents, map_matrix, intersection_id_map, obstacle_gen, targets, targets_heat)

    def get_emergency(self, happens: list = None):
        emergency = []
        if happens:
            dm = self.agents.density.T
            for happen in happens:
                (x, y), desc = happen
                x, y = x * DENSITY_MATRIX_SIZE / MAP_SIZE, y * DENSITY_MATRIX_SIZE / MAP_SIZE
                dens = dm[x][y]
                level = crowd_evacuation(dens, desc)
        return emergency

    # @staticmethod
    # def deal_trafficlight(trafficlight: np.array = None)->list:
    #     """
    #
    #     :param trafficlight:original trafficlight matrix
    #     :return:
    #     """
    #     light = []
    #     try:
    #         for i in range(len(trafficlight[0])):
    #             if max(trafficlight[:,i]) < 1:
    #                 light.append(1)
    #             else:
    #                 light.append(0)
    #         return light
    #     except IndexError:
    #         print("Traffic Light Matrix Is Empty")
    #         return []

    @staticmethod
    def make_obs():
        map_size = None
        obs_params = None
        obs = ObstacleGenerator(map_size)
        obs.add_obstacles(obs_params)
        return obs

    @staticmethod
    def deal_map(graph:any) -> (list, np.ndarray, list):
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

        :param happened: [((x,y), str), ...]  (pos, describe)
        :param trafficlight: [1, 0, 1, 0, ...]  id - statu
        :return: no return
        """
        emergency = self.get_emergency(happened)
        eme_pos = []
        eme_lev = []
        if emergency:
            for pos, lev in emergency:
                eme_pos.append(pos)
                eme_lev.append(lev)
        self.agents.move_towards_target((eme_pos, eme_lev), trafficlight)

    # def get_pos(self, node_id: int, radius: int = 30) -> list:
    #     pass

    @property
    def density(self) -> np.ndarray:
        """
        :return: density (DENSITY_MATRIX_SIZE * DENSITY_MATRIX_SIZE)
        """
        return self.agents.get_density().cpu().numpy().T

    @property
    def position(self) -> list:
        """
        :return: [(x, y), ...]
        """
        positions = self.agents.positions.cpu().numpy().tolist()
        return positions

    def kill(self, node_id: int, radius: int = 20, num: int = 30):
        """
        be same to name
        :param node_id:
        :param radius:
        :param num:
        :return:
        """
        x, y = points[node_id]
        self.agents.kill(x, y, radius, num)


    def birth(self, node_id: int, radius: int = 20, num: int = 30):
        """
        be same to name
        :param node_id:
        :param radius:
        :param num:
        :return:
        """
        x, y = points[node_id]
        self.agents.birth(x, y, radius, num)






