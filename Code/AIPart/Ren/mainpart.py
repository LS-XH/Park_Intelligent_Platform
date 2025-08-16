import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
from PIL import Image
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time

from Interface.people import PersonBase
from Ren.csxiaofeng import crowd_evacuation
from Ren.agents import AgentGroup
from Ren.obstacles import ObstacleGenerator
from Ren.config import *

# 配置matplotlib后端和中文显示
matplotlib.use('Agg')  # 非交互式后端，适合生成图像
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Crowd(PersonBase):

    @staticmethod
    def create_agents(num_agents, obstacle, targets, targets_heat):
        return AgentGroup(num_agents, obstacle, targets, targets_heat)

    def get_emergency(self, happens: list = None):
        emergency = []
        if happens:
            for happen in happens:
                (x, y), level = happen
                emergency.append(((x, y), level))
        return emergency

    @staticmethod
    def make_obs(map_size, obs_params, points):
        obs = ObstacleGenerator(map_size, points)
        obs.add_obstacles(obs_params)
        return obs

    @staticmethod
    def deal_map(edges: list, points_coords) -> (list, int):
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

    def __init__(self, points: list, edges: list, targets: list, targets_heat: list, num_agents: int = 3000):
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
        # 初始化图形和轴，避免重复创建
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self._init_plot_elements()

    def _init_plot_elements(self):
        """初始化绘图元素，只执行一次"""
        # # 设置坐标轴范围
        # self.ax.set_xlim(0.0, self.MAP_SIZE)
        # self.ax.set_ylim(0.0, self.MAP_SIZE)

        # 1. 绘制密度热图（作为底层）
        density_data = self.agents.get_density().cpu().numpy()
        self.density_plot = self.ax.imshow(
            density_data.T,  # 转置以正确显示坐标
            extent=[0, self.MAP_SIZE, 0, self.MAP_SIZE],
            cmap=DENSITY_CMAP,
            alpha=DENSITY_ALPHA,
            origin='lower',
            vmin=0,
            vmax=min(10, self.num_agents // 100)
        )


        # 绘制障碍物
        obstacle_x, obstacle_y = np.where(self.obstacle.map_matrix == 1.0)
        self.ax.scatter(obstacle_x, obstacle_y, c='black', s=10, label='道路/障碍物')

        # 标记交点
        for idx, (x, y) in enumerate(self.obstacle.intersection_points):
            self.ax.scatter(x, y, c='yellow', s=80, marker='+', label=f'交点 {idx}' if idx == 0 else "")
            self.ax.text(x + 5.0, y + 5.0, f'ID:{idx}', fontsize=9, color='yellow')

        # # 绘制所有缺口中心和斥力线段
        # for i, obs in enumerate(self.obstacle.obstacles):
        #     for idx, gap_center in obs.gap_center_map.items():
        #         self.ax.scatter(gap_center[0], gap_center[1], c='blue', s=5, marker='o',
        #                         alpha=0.5, label='缺口中心' if (i == 0 and idx == 0) else "")
        #         # 计算并绘制斥力线段
        #         seg_length = obs.gap_offset_ratio + (self.obstacle.intersection_gap_size / 2.0) + 2.0
        #         dir_vec = np.array([obs.dir_x, obs.dir_y], dtype=np.float32)
        #         seg_start = gap_center - 0.5 * seg_length * dir_vec
        #         seg_end = gap_center + 0.5 * seg_length * dir_vec
        #         self.ax.plot([seg_start[0], seg_end[0]], [seg_start[1], seg_end[1]], 'g--', linewidth=1,
        #                      alpha=0.5, label='斥力线段' if (i == 0 and idx == 0) else "")

        # 绘制目标点
        targets_np = np.array(self.targets, dtype=np.float32)
        self.ax.scatter(
            targets_np[:, 0], targets_np[:, 1],
            c='red', s=150, marker='*',
            edgecolors='black', linewidths=2,
            label='Targets'
        )

        # 初始化智能体绘制
        agent_positions = self.agents.positions.cpu().numpy()
        agent_sizes = self.agents.sizes.cpu().numpy()
        agent_colors = self.agents.colors.cpu().numpy()

        self.agents_plot = self.ax.scatter(
            agent_positions[:, 0], agent_positions[:, 1],
            s=agent_sizes, c=agent_colors,
            alpha=0.7, edgecolors='white', linewidths=0.5,
            label='Agents'
        )

        # 初始化紧急情况标记
        self.emergency_plot = self.ax.scatter(
            [], [],  # 初始为空
            s=[],
            marker='X',
            c='red',
            edgecolors='black',
            linewidths=1.5,
            label='Emergency'
        )

        self.ax.legend()
        self.ax.axis('off')
        plt.tight_layout()

    def simulate(self, happened: list = None, trafficlight: list = None):
        """
        模拟人群一步
        :param happened: [((x,y), int), ...]  (coord, level)
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
        移除指定位置的智能体
        :param node_id:
        :param radius:
        :param num:
        """
        x, y = self.points[node_id]
        self.agents.kill(x, y, radius, num)

    def birth(self, node_id: int, radius: int = 20, num: int = 30):
        """
        在指定位置生成新的智能体
        :param node_id:
        :param radius:
        :param num:
        """
        x, y = self.points[node_id]
        self.agents.birth(x, y, radius, num)

    def generate_frame(self, happened: list = None) -> str:
        """
        生成当前状态的可视化图像，并返回Base64编码字符串
        :param frame: 当前帧编号，用于动画效果控制
        :param happened: 紧急情况列表
        :return: Base64编码的图像字符串
        """
        # 更新智能体显示
        positions = self.agents.positions.cpu().numpy()
        self.agents_plot.set_offsets(positions)

        # 更新智能体大小
        agent_sizes = self.agents.sizes.cpu().numpy()
        self.agents_plot.set_sizes(agent_sizes)

        # 更新智能体颜色
        agent_colors = self.agents.colors.cpu().numpy()
        self.agents_plot.set_facecolors(agent_colors)

        # 更新密度热图
        density_data = self.agents.get_density().cpu().numpy()
        self.density_plot.set_data(density_data.T)

        # 动态调整颜色范围
        current_max = min(density_data.max(), self.agents.num_agents // 10)
        if current_max > 0:
            self.density_plot.set_clim(vmin=0, vmax=current_max)

        # 处理紧急情况显示
        emergency = self.get_emergency(happened)
        if emergency:
            eme_pos = []
            eme_sizes = []
            for (pos, lev) in emergency:
                eme_pos.append(pos)
                eme_sizes.append(50.0 + lev * 30.0)

            eme_pos_np = np.array(eme_pos, dtype=np.float32)
            self.emergency_plot.set_offsets(eme_pos_np)
            self.emergency_plot.set_sizes(eme_sizes)
            self.emergency_plot.set_visible(True)
        else:
            self.emergency_plot.set_visible(False)

        # 将图像保存到内存缓冲区
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)

        # 转换为Base64编码
        img = Image.open(buf)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        return img_base64
