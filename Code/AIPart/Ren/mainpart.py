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
    def create_agents(num_agents, obstacle, targets, targets_heat):
        return AgentGroup(num_agents, obstacle, targets, targets_heat)

    def get_emergency(self, happens: list = None):
        emergency = []
        if happens:
            dm = self.agents.density.T
            for happen in happens:
                (x, y), desc = happen
                x, y = x * DENSITY_MATRIX_SIZE / self.MAP_SIZE, y * DENSITY_MATRIX_SIZE / self.MAP_SIZE
                dens = dm[x][y]
                level = crowd_evacuation(dens, desc)
                emergency.append((dens,level))
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
    def make_obs(map_size, obs_params, points):
        obs = ObstacleGenerator(map_size, points)
        obs.add_obstacles(obs_params)
        return obs

    @staticmethod
    def deal_map(edges:list, points_coords) -> (list, int):
        """
        :return: obstacles_params, map_size
        """
        # 生成道路参数
        obstacles_params = []
        for edge in edges:
            start_idx = edge["start_id"]
            end_idx = edge["end_id"]
            start_point = points_coords[start_idx]
            end_point = points_coords[end_idx]
            obstacles_params.append({
                "start_point": start_point,
                "end_point": end_point,
                "width": 12,
                "gap_offset_ratio": 1.0
            })
        max_x = max(coord[0] for coord in points_coords) + 100
        max_y = max(coord[1] for coord in points_coords) + 100
        MAP_SIZE = max(max_x, max_y)
        return obstacles_params, MAP_SIZE

    def __init__(self, points:list, edges:list, targets:list, targets_heat:list, num_agents:int=3000):
        """
        :param points:       [(x, y), ...]                        ->  vertex(node) in map graph
        :param edges:        [{"start_id":1, "end_id":2}, ... ]   ->  edges in map graph         (list of dict of id)
        :param targets:      [(x, y), ...]                        -> targets' positions         # NOT AS LONGER AS POINTS #
        :param targets_heat: [0.01, 0.02, ...]                    -> targets' heat              # AS LONGER AS TARGETS #
        :param num_agents:    default 3000
        """
        self.points = points
        self.targets_heat = np.array(targets_heat)
        self.targets = targets
        obstacle_params, self.MAP_SIZE = self.deal_map(edges, points)
        self.obstacle = self.make_obs(self.MAP_SIZE, obstacle_params, points)
        self.num_agents = num_agents
        self.agents = self.create_agents(
            num_agents,
            self.obstacle,
            targets=self.targets,
            targets_heat=self.targets_heat
        )



    def simulate(self, happened: list = None, trafficlight: list = None):
        """
        模拟人群一步
        :param happened: [((x,y), str), ...]  (coord, describe)
        :param trafficlight: [1, 0, 1, 0, ...]  id - label
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
        密度矩阵: density per point = 人数 / ((1*1)*缩放比)
        :return: density (DENSITY_MATRIX_SIZE * DENSITY_MATRIX_SIZE)
        """
        return self.agents.get_density().cpu().numpy().T

    @property
    def position(self) -> list:
        """
        position of points (id to coord)
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
        """
        x, y = self.points[node_id]
        self.agents.kill(x, y, radius, num)


    def birth(self, node_id: int, radius: int = 20, num: int = 30):
        """
        be same to name
        :param node_id:
        :param radius:
        :param num:
        """
        x, y = self.points[node_id]
        self.agents.birth(x, y, radius, num)
