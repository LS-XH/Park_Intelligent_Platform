from Interface.graph import GraphBase, PointType,Point, Edge
import math
import numpy as np
import json
from  common_func import generate_cars_list, generate_people_list
#from .information import weather,traffic_flow,human_flow,accident_rate

CARS_NUM_THREADS = 30
PEOPLE_NUM_THREADS = 3000

class Graph(GraphBase):


    def __init__(self,green_light=17,yellow_light=3,red_light=20):
        super().__init__()
        self.__points=[]
        self.__points_id={}
        self.__edges=[]

        self.__positions = np.empty((0,2))
        self.__length = np.empty((0,0))
        self.__weight = np.empty((0,0))
        self.__degree = np.empty((0,0))
        self.__limit_speed = np.empty((0,0))

        self.__traffic_light = np.empty((0,0))
        self.__red_light = red_light
        self.__yellow_light = yellow_light
        self.__green_light = green_light

        self.current_red = red_light
        self.current_yellow = yellow_light
        self.current_green = green_light


    def initialize_car(self, num_cars=None):
        """
        初始化车的位置，根据固定的热点数据随机生成

        :param num_cars:    生成的车的数量
        :return:            字典，键表示车的ID，值表示车的坐标
        """

        cars_list = generate_cars_list(num_cars)
        return cars_list

    def initialize_crowd(self, num_people=None):
        """
        初始化人的位置
        :return:
        """
        people_list = generate_people_list(num_people)
        return people_list

    def initialize_traffic_light(self):
        """
        初始化红绿灯
        :return:
        """


    @property
    def nodes_position(self)->np.ndarray:
        """
        节点矩阵，行为序号，列0为x，列1为y
        :return: dim=2
        """
        return np.array(self.__positions)

    @property
    def length(self)->np.ndarray:
        """
        道路长度邻接矩阵
        :return: dim=2
        """
        return np.array(self.__length)

    @property
    def weight(self)->np.ndarray:
        """
        权重邻接矩阵
        :return: dim=2
        """
        return np.array(self.__weight)

    @property
    def degree(self)->np.ndarray:
        """
        道路数量和方向的度矩阵
        :return: dim=2
        """
        return np.array(self.__degree)

    @property
    def limit_speed(self)->np.ndarray:

        return np.array(self.__limit_speed)


    def traffic_light(self)->np.ndarray:
        return np.array(self.__traffic_light)

    def get_red_light(self):
        return self.__red_light

    def get_yellow_light(self):
        return self.__yellow_light

    def get_green_light(self):
        return self.__green_light


    def set_green_light(self, value):
        # 可以添加参数校验（如确保是正数）
        if isinstance(value, int) and value > 0:
            self.current_green = value
        else:
            raise ValueError("绿灯时间必须是正整数")


    def set_yellow_light(self, value):
        if isinstance(value, int) and value > 0:
            self.current_yellow = value
        else:
            raise ValueError("黄灯时间必须是正整数")


    def set_red_light(self, value):
        if isinstance(value, int) and value > 0:
            self.current_red = value
        else:
            raise ValueError("红灯时间必须是正整数")

#点击节点，返回红绿灯数据
#[]


    def get_light(self, start_id: int, end_id: int) -> dict:
        """
        获取指定路段的红绿灯状态及剩余时间
        :param start_id: 起始节点ID
        :param end_id: 目标节点ID
        :return: 包含状态和剩余时间的字典
        """
        if start_id not in self.__points_id or end_id not in self.__points_id:
            raise ValueError(f"节点 {start_id} 或 {end_id} 不存在")

        i = self.__points_id[start_id]
        j = self.__points_id[end_id]

        if self.__length[i, j] == 0:
            raise ValueError(f"节点 {start_id} 到 {end_id} 之间没有路段")

        remaining_time = self.__traffic_light[i, j]

        if remaining_time < 0:
            status = "green"
            remaining = abs(remaining_time)
        else:
            status = "red"
            remaining = remaining_time

        return {
            "status": status,
            "remaining_time": float(f"{remaining:.2f}"),
            "start_id": start_id,
            "end_id": end_id
        }

    def get_traffic_light_status_all(self):
        return self.__traffic_light


    def _simulate_light(self, dt=0.1):
        n=len(self.__points)
        for i in range(n):
            for j in range(n):
                if self.__length[i,j] == 0:
                    continue
                current = self.__traffic_light[i,j]
                if current < 0:
                    current += dt
                    if current >= 0:
                        self.__traffic_light[i,j] = self.__red_light
                    else:
                        self.__traffic_light[i,j] = current
                elif current >0:
                    current -= dt
                    if current <= 0:
                        self.__traffic_light[i,j] = -(self.__green_light + self.__yellow_light)
                    else:
                        self.__traffic_light[i,j] = current

    def simulate(self, dt=0.1):
        self._simulate_light(dt=dt)



    def upgrade_weight(self):
        pass



    def add_point(self, id: int, x: float, y: float, degree :int,type: PointType = PointType.crossing):
        if id in self.__points_id:
            raise Exception(f"节点 {id} 已存在")
        self.__points.append(Point(id, x, y, degree, type=type))
        self.__points_id[id] = len(self.__points_id)

        n = len(self.__points)
        self.__positions = np.vstack([self.__positions, [x, y]])

        new_length = np.zeros((n, n))
        new_degree = np.zeros((n, n))
        new_limit_speed = np.zeros((n, n))
        new_weight = np.zeros((n, n))
        new_traffic_light = np.zeros((n, n))

        if n > 1:
            old_n = n - 1
            new_length[0:old_n, 0:old_n] = self.__length
            new_degree[0:old_n, 0:old_n] = self.__degree
            new_limit_speed[0:old_n, 0:old_n] = self.__limit_speed
            new_weight[0:old_n, 0:old_n] = self.__weight
            new_traffic_light[0:old_n, 0:old_n] = self.__traffic_light

        self.__length = new_length
        self.__degree = new_degree
        self.__limit_speed = new_limit_speed
        self.__weight = new_weight
        self.__traffic_light = new_traffic_light

        return True

    def add_edge(self,start_id:int,end_id:int,length:float,limit_speed:float):
        if not (start_id in [p.id for p in self.__points] and end_id in [p.id for p in self.__points]):
            raise Exception(f"起始点 {start_id} 或终点 {end_id} 不存在于已添加的节点中")

        i = self.__points_id[start_id]
        j = self.__points_id[end_id]
        self.__length[i,j] = length
        self.__limit_speed[i,j] = limit_speed

        self.__weight[i,j] = length/limit_speed
        self.__traffic_light[i,j] = -(self.__green_light + self.__yellow_light)

        self.__edges.append(Edge(start_id,end_id,length,limit_speed))


    def load_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for point_id, point_info in data['points'].items():
            id = int(point_id)
            x = float(point_info['x'])
            y = float(point_info['y'])
            degree = int(point_info['degree'])
            type = PointType(point_info['type'])
            self.add_point(id, x, y, degree, type=type)

        for edge_group in data['edges'].values():
            start_id = edge_group['start_id']
            end_id = edge_group['end_id']

            limit_speed = float(edge_group['limit_speed'])

            start_point = self.__points[self.__points_id[start_id]]
            end_point = self.__points[self.__points_id[end_id]]
            length = math.sqrt(
                (end_point.x - start_point.x) ** 2 +
                (end_point.y - start_point.y) ** 2
            )

            self.add_edge(
                start_id = start_id,
                end_id = end_id,
                length = length,
                limit_speed = limit_speed
            )


graph=Graph()
graph.load_json("data.json")
print("\n长度的邻接矩阵")
print(graph.degree)