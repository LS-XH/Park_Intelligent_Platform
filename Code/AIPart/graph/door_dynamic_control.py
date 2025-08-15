import json
from Code.AIPart.graph.graph import Graph
import numpy as np

class DoorController:
    def __init__(self, hot_data_path: str):
        # 初始化Graph和热度数据
        self.graph = Graph()
        with open(hot_data_path, 'r', encoding='utf-8') as f:
            self.hot_district = json.load(f)  # 当前热度值
        self.original_hot_district = self.hot_district.copy()  # 初始热度值
        # 记录当前门的状态（避免重复操作）
        self.current_closed_points = set()  # 存储当前处于关闭状态的点ID
        self.current_open_points = set()  # 存储当前处于打开状态的点ID

    def get_point_by_xy(self, target_x, target_y, tolerance=1.1):
        """根据坐标查找Point对象（封装为类方法）"""
        for point in self.graph.points:
            x_match = abs(point.x - target_x) < tolerance
            y_match = abs(point.y - target_y) < tolerance
            if x_match and y_match:
                return point
        return None

    def door_dynamic_control(self, map_square_rate: float, person_matrix: np.ndarray, total_num: int):
        """
        若超越人群阈值，则将热点地区降为0
        :param map_square_rate: 地图缩放率
        :param person_matrix: 人数矩阵
        :param total_num: 人数总量
        :return: hot_district 热点区域图
        """
        # 本次调用需要关闸和开闸的点
        new_close_points = []
        new_open_points = []

        # 遍历人口密度矩阵，判断每个点的状态
        for i in range(len(person_matrix)):
            for j in range(len(person_matrix[i])):
                current_density = person_matrix[i][j] * map_square_rate * total_num
                actual_x = i * map_square_rate
                actual_y = j * map_square_rate
                point = self.get_point_by_xy(actual_x, actual_y)
                if not point:
                    continue

                # 根据密度判断是否需要关闸/开闸
                if current_density >= 350:
                    # 密度过高，需要关闸（且之前未关闭）
                    if point.id not in self.current_closed_points:
                        new_close_points.append(point)
                else:
                    # 密度正常，需要开闸（且之前未打开）
                    if point.id not in self.current_open_points:
                        new_open_points.append(point)

        # 执行关闸操作并更新状态
        for point in new_close_points:
            district_name = self.graph.__getitem__(point.id)
            if district_name in self.hot_district:
                self.hot_district[district_name] = 0  # 热度设为0
            # 切换门状态（如果不是gate且当前是打开的）
            if point.type != point.PointType.gate and point.DoorStatus.open:
                point.door_transition()
            # 更新状态记录
            self.current_closed_points.add(point.id)
            self.current_open_points.discard(point.id)  # 从打开集合中移除

        # 执行开闸操作并更新状态
        for point in new_open_points:
            district_name = self.graph.__getitem__(point.id)
            # 恢复初始热度（如果存在）
            if district_name in self.hot_district and district_name in self.original_hot_district:
                self.hot_district[district_name] = self.original_hot_district[district_name]
            # 切换门状态（如果不是gate且当前是关闭的）
            if point.type != point.PointType.gate and point.DoorStatus.close:
                point.door_transition()
            # 更新状态记录
            self.current_open_points.add(point.id)
            self.current_closed_points.discard(point.id)  # 从关闭集合中移除

        return True
