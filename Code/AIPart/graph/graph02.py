from Interface.graph import GraphBase, PointType,Point, Edge
import math
import numpy as np
import json
from  common_func import generate_cars_list, generate_people_list
#from .information import weather,traffic_flow,human_flow,accident_rate

CARS_NUM_THREADS = 30
PEOPLE_NUM_THREADS = 3000

class Graph(GraphBase):

    def __init__(self):
        super().__init__()
        # 节点集合
        self.__points=[]
        # 节点名字id映射表
        self.__points_id={}
        # 边集合
        self.__edges=[]

        # 位置矩阵
        self.__positions = np.empty((0,2))
        # 长度矩阵
        self.__length = np.empty((0,0))
        # 权重矩阵
        self.__weight = np.empty((0,0))
        # 度矩阵
        self.__degree = np.empty((0,0))
        # 限速矩阵
        self.__limit_speed = np.empty((0,0))
        """
        红绿灯智能调控需要用到的参数
        """
        self.__traffic_light = np.empty((0,0))
        # 新增：每个独立红绿灯的参数（i,j为方向）
        self.__original_params = {}  # 原始时长：{(i,j): (red, yellow, green)}
        self.__current_params = {}  # 当前生效时长：{(i,j): (red, yellow, green)}
        self.__is_adjusted = {}  # 是否处于调整状态：{(i,j): bool}


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

    def initialize_traffic_light(self, node_name=None):
        """
        根据节点的度数设置不同数量的红绿灯，并初始化红绿灯参数
        :param node_name: 节点名称（可选参数，用于错误提示）
        :return: 无返回值
        """
        # 遍历所有节点
        for node in self.__points:
            # 获取当前节点的名称和索引
            nodes_name = node.name
            i= self.__points_id[nodes_name]
            node_degree = node.degree  # 获取节点的度数
            x_i, y_i = self.__positions[i]  # 获取节点的坐标

            # 收集所有出边（从当前节点出发的边）
            out_edges = []
            for j in range(len(self.__points)):
                if self.__length[i, j] > 0:  # 存在从i到j的道路
                    out_edges.append(j)

            # 根据节点的度设置红绿灯
            # 度=4，无红绿灯，清空相关参数
            if node_degree == 4:
                for j in out_edges:
                # 从三个参数字典中删除相关参数
                    for param_dict in [self.__original_params, self.__current_params, self.__is_adjusted]:
                        if (i, j) in param_dict:
                            del param_dict[(i, j)]
                    self.__traffic_light[i, j] = 0  # 0表示无红绿灯
            elif node_degree in (6, 8):
                # 度=6→3个红绿灯；度=8→4个红绿灯
                num_lights = 3 if node_degree == 6 else 4
                # 检查出边数量是否与度数匹配
                if len(out_edges) != num_lights:
                    raise ValueError(f"节点{node_name}的出边数量与度不匹配（预期{num_lights}，实际{len(out_edges)}）")

                # 按顺时针排序出边方向
                # 计算每个出边的方向角度（与x轴正方向的夹角，范围0~2π）
                edge_angles = []
                for j in out_edges:
                    x_j, y_j = self.__positions[j]  # 获取目标节点的坐标
                    dx = x_j - x_i  # x方向向量
                    dy = y_j - y_i  # y方向向量
                    angle = math.atan2(dy, dx)  # 计算角度（弧度）
                    if angle < 0:
                        angle += 2 * math.pi  # 转换为0~2π范围
                    edge_angles.append((angle, j))

                # 按角度降序排序（逆序即为顺时针方向）
                edge_angles.sort(reverse=True, key=lambda x: x[0])
                sorted_j = [j for (angle, j) in edge_angles]  # 顺时针排序的目标节点索引

                # 初始化红绿灯参数（红20s、黄3s、绿17s）
                red, yellow, green = 20, 3, 17
                cycle = red + yellow + green  # 一个完整周期的时长

                # 按顺时针顺序设置相位差（依次延迟启动，实现交替变换）
                for idx, j in enumerate(sorted_j):
                    # 存储原始参数和当前参数
                    self.__original_params[(i, j)] = (red, yellow, green)
                    self.__current_params[(i, j)] = (red, yellow, green)
                    self.__is_adjusted[(i, j)] = False

                    # 初始状态：按顺序延迟启动，形成顺时针交替
                    delay = (cycle / num_lights) * idx  # 每个红绿灯的延迟时间
                    green_yellow = green + yellow  # 绿灯+黄灯总时长

                    if delay < green_yellow:
                        # 延迟后仍在绿灯阶段（负值表示绿灯）
                        self.__traffic_light[i, j] = -(green_yellow - delay)
                    else:
                        # 延迟后进入红灯阶段（正值表示红灯）
                        self.__traffic_light[i, j] = cycle - delay

            else:
                # 其他度数的节点默认无红绿灯
                for j in out_edges:
                    self.__traffic_light[i, j] = 0

    @property
    def current_params(self):# 当前生效时长
        return self.__current_params

    @property
    def original_params(self): # 原始时长
        return self.__original_params

    @property
    def is_adjusted(self): # 是否处于调整状态
        return self.__is_adjusted

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

    @property
    def traffic_light(self)->np.ndarray:
        return np.array(self.__traffic_light)

    # 通过值得到键
    def get_name_by_index(self, index: int) -> str:
        """
        根据节点索引（__points_id的值）获取对应的节点名称（键）
        :param index: 节点索引（整数）
        :return: 节点名称（字符串）
        :raises ValueError: 若索引不存在
        """
        # 通过值映射回键
        for name, idx in self.__points_id.items():
            if idx == index:
                return name
        raise ValueError(f"不存在索引为 {index} 的节点")

    # 获取红绿灯时间
    def get_light(self, start_id: int, end_id: int) -> dict:
        """
        获取指定路段的红绿灯状态及剩余时间
        :param start_id: 起始节点ID
        :param end_id: 目标节点ID
        :return: 包含状态和剩余时间的字典
        """
        # 判断节点的id是否在__points_id的值域中
        if start_id not in self.__points_id.values() or end_id not in self.__points_id.values():
            raise ValueError(f"节点 {self.get_name_by_index(start_id)} 或 {self.get_name_by_index(end_id)} 不存在")

        i = start_id
        j = end_id

        if self.__length[i, j] == 0:
            raise ValueError(f"节点 {self.get_name_by_index(start_id)} 到 {self.get_name_by_index(end_id)} 之间没有路段")

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
            "start_name": self.get_name_by_index(start_id),
            "end_name": self.get_name_by_index(end_id)
        }

    # 获取红绿灯矩阵
    def get_traffic_light_status_all(self):
        return self.__traffic_light

    def simulate_light(self, dt=0.1):
        n = len(self.__points)
        for i in range(n):
            for j in range(n):
                if self.__length[i, j] == 0:
                    # 跳过无路段的方向
                    continue
                    # 获取当前生效的红绿灯参数（red, yellow, green）
                red, yellow, green = self.current_params.get((i, j), (20, 3, 17))  # 默认值
                current = self.__traffic_light[i, j]

                if current < 0:
                    # 绿灯阶段（包含黄灯，总时长为 green + yellow）
                    current += dt
                    if current >= 0:
                        # 绿灯结束，切换为红灯，倒计时设为当前生效的 red 时长
                        self.__traffic_light[i, j] = red
                    else:
                        self.__traffic_light[i, j] = current
                elif current > 0:
                    # 红灯阶段
                    current -= dt
                    if current <= 0:
                        # 红灯结束，切换为绿灯（包含黄灯），倒计时设为 -(green + yellow)
                        self.__traffic_light[i, j] = -(green + yellow)
                    else:
                        self.__traffic_light[i, j] = current

    def upgrade_weight(self):
        pass

    """
    在图中加载点
    """
    def add_point(self, id: int,name: str, x: float, y: float, degree :int, type: PointType = PointType.crossing): # add the point in json into graph
        if name in self.__points_id:
            raise Exception(f"节点 {name} 已存在")
        self.__points.append(Point(id, name, x, y, degree, type=type))
        self.__points_id[name] = len(self.__points_id)

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

    """
    在图中加载边
    """

    def add_edge(self, start_name: str, end_name: str, length: float, limit_speed: float):
        if not (start_name in [p.name for p in self.__points] and end_name in [p.name for p in self.__points]):
            raise Exception(f"起始点 {start_name} 或终点 {end_name} 不存在于已添加的节点中")

        i = self.__points_id[start_name]
        j = self.__points_id[end_name]
        self.__length[i,j] = length
        self.__limit_speed[i,j] = limit_speed


        self.__weight[i,j] = length/limit_speed

        # 初始化红绿灯
        # 默认时长（可根据实际需求调整，这里设为红20s、黄3s、绿17s）
        #default_red = 20
        #default_yellow = 3
        #default_green = 17

        #self.original_params[(i, j)] = (default_red, default_yellow, default_green)
        # 当前参数初始化为原始参数
        #self.current_params[(i, j)] = (default_red, default_yellow, default_green)
        # 初始未调整
        #self.is_adjusted[(i, j)] = False
        # 初始化倒计时：绿灯+黄灯总时长（负值表示绿灯）
        #self.__traffic_light[i, j] = -(default_green + default_yellow)

        self.__edges.append(Edge(start_name, end_name, length, limit_speed, self.__points_id[start_name], self.__points_id[end_name]))
    """
    红绿灯 
    """

    def get_direction(self, start_name: str, end_name: str) -> tuple:
        """获取start_id→end_id对应的i,j索引"""
        i = self.__points_id[start_name]
        j = self.__points_id[end_name]
        return (i, j)

    def get_original_params(self, i: int, j: int) -> tuple:
        """获取(i,j)方向的原始时长参数"""
        return self.__original_params.get((i, j), (20, 3, 17))  # 默认值

    def set_current_params(self, i: int, j: int, red: int, yellow: int, green: int):
        """设置(i,j)方向的当前生效时长"""
        if not (isinstance(red, int) and red > 0 and
                isinstance(yellow, int) and yellow > 0 and
                isinstance(green, int) and green > 0):
            raise ValueError("红绿灯时长必须为正整数")
        self.current_params[(i, j)] = (red, yellow, green)
        self.is_adjusted[(i, j)] = True  # 标记为已调整

    def restore_original_params(self, i: int, j: int):
        """恢复(i,j)方向的原始参数"""
        if (i, j) in self.__original_params:
            self.current_params[(i, j)] = self.__original_params[(i, j)]
            self.is_adjusted[(i, j)] = False  # 取消调整标记



    def load_json(self, path: str, start_name=None, end_name=None):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for point_id, point_info in data['points'].items():
            id = int(point_id)
            name = point_info['name']
            x = float(point_info['x'])
            y = float(point_info['y'])
            degree = int(point_info['degree'])
            type = PointType(point_info['type'])
            self.add_point(id, name , x, y, degree, type=type)

        for edge_group in data['edges'].values():
            start_id = str(edge_group['start_id'])
            end_id = str(edge_group['end_id'])

            limit_speed = float(edge_group['limit_speed'])

            start_point = self.__points[self.__points_id[start_id]]
            end_point = self.__points[self.__points_id[end_id]]
            length = math.sqrt(
                (end_point.x - start_point.x) ** 2 +
                (end_point.y - start_point.y) ** 2
            )

            self.add_edge(
                start_name = start_name,
                end_name = end_name,
                length = length,
                limit_speed = limit_speed
            )
        return True


# graph=Graph()
# graph.load_json("data02.json")
# print("\n长度的邻接矩阵")
# print(graph.degree)
graph = Graph()
graph.load_json("data02.json")
print("\n长度的邻接矩阵")
print(graph.length)
graph.initialize_traffic_light()