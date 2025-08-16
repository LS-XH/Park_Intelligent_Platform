import heapq    # 导入优先模块，用于高效获取距离最小的节点，构建小顶堆
# heappush(heap, item): 将元素item添加到heap中，并保持堆的顺序。
# heappop(heap): 弹出并返回堆中的最小元素。
import numpy as np
import random
from Code.AIPart.graph.graph import Graph


"""

adj 邻接矩阵
start 起点
end 终点
"""

def normalize_min_max(matrix):
    """
    归一化
    """
    min_val = matrix.min()
    max_val = matrix.max()
    if max_val - min_val < 1e-9:  # 避免除以0（矩阵所有元素相同）
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)

def bidirectional_dijkstra(car_matrix: ndarray ,start: int, end: int, graph: Graph):
    """
    最短路径算法
    :param adj:长度矩阵
    :param start: 起点id
    :param end: 终点id
    :param weight_matrix: 权重矩阵
    :return: path 最短路径列表
    """
    length_adj = graph.length  # 假设Graph类用adj_matrix存储邻接矩阵
    length_adj = normalize_min_max(length_adj)
    road_density = graph.get_road_density_matrix()
    road_density = normalize_min_max(road_density)
    road_density *= 0.3
    car_matrix = normalize_min_max(car_matrix)
    car_matrix *= 0.7
    adj = np.add(road_density, np.add(length_adj, car_matrix))



    # 创建正向距离字典：起点到每个结点的初始距离设为无穷大
    forward_dist = {node: float('inf') for node in range(len(adj))}
    # 起点距离设置为 0，放入“优先队列”
    forward_dist[start] = 0 # forward_dist[start]即为起点距离的值
    # 正向优先队列：存储（距离、结点），初始放入起点
    forward_heap = [(0, start)]
    # 各节点的上游节点
    # 正向前驱字典：记录每个结点的上游结点，用于回溯路径
    forward_prev = {}

    # 创建方向距离字典：终点到每个结点的初始距离设为无限大
    reverse_dist = {node: float('inf') for node in range(len(adj))}
    # 终点到自身的距离设为 0
    reverse_dist[end] = 0
    # 反向优先队列：存储（距离，结点），初始放入终点
    reverse_heap = [(0, end)]
    # 反向前驱字典：记录每个结点的上游结点（反向路径）
    reverse_prev = {}

    # 已搜索的节点
    # 正向已访问节点集合
    visited_forward = set()
    # 反向已访问节点集合
    visited_reverse = set()

    # 双向搜索相遇节点及记录起终点间的最短距离
    meeting_node = None
    shortest_distance = float('inf')

    # 当两个队列都不为空时继续搜索
    while forward_heap and reverse_heap:
        # 选择正向或反向队列中距离更小的节点进行扩展
        # heapq模块实现优先队列
        # 正向队列中的最小距离
        # forward_heap[0]表示第零个元素,forward[0][0]表示距离长度
        forward_min = forward_heap[0][0]
        # 反向队列中最小距离
        reverse_min = reverse_heap[0][0]

        # 若正向距离更小，优先扩展正向
        # 判断哪一向最小距离更小可以让搜索更高效地收敛到相遇点，避免某一方向过度扩展浪费资源
        if forward_min <= reverse_min:
            # 正向扩展
            # 弹出正向距离最小的节点
            # 这是 Python 内置的heapq模块提供的函数
            # 用于从小根堆（优先队列的实现方式）中弹出并返回最小元素。
            # 这是正向搜索的优先队列，存储格式为(距离, 节点)的元组。
            # 例如(3, 1)表示 “从起点到节点 1 的当前最短距离是 3”。
            current_dist, u = heapq.heappop(forward_heap)
            # 标记该节点为正向已访问
            visited_forward.add(u)
            # 更新邻接节点
            # 遍历当前节点的所有邻接结点
            for i in range(len(adj)):
                # 若结点u和i之间有连接（权重 > 0）
                if adj[u][i] > 0:
                    # 标记邻接结点为已访问
                    visited_forward.add(i)
                    # 若通过u到i的距离更短，更新距离
                    if forward_dist[i] > forward_dist[u] + adj[u][i]:
                        # 更新距离
                        # forward_dist[u]是当前节点u到起点的距离（u=0时，forward_dist[0]=0）
                        # adj[u][i]是u到i的直接权重（如u=0,i=1时为 3）
                        # 右边结果是 “起点→u→i” 的总距离（0+3=3）
                        # 左边forward_dist[i]是之前记录的 i 到起点的距离（初始为inf）
                        forward_dist[i] = forward_dist[u] + adj[u][i]
                        # 加入队列
                        heapq.heappush(forward_heap, (forward_dist[i], i))
                        # 记录前驱节点
                        forward_prev[i] = u


                    # 检查是否与反向所搜相遇相遇，若相遇计算完整路径的长度
                    if i in visited_reverse:
                        # 计算总距离
                        total = forward_dist[i] + reverse_dist[i]
                        # 若找到更短路径
                        if total < shortest_distance:
                            # 更新最短距离
                            shortest_distance = total
                            # 记录相遇节点
                            meeting_node = i

        # 若反向距离更小，优先扩展反向
        else:
            # 弹出反向距离最小的节点
            current_dist, u = heapq.heappop(reverse_heap)
            # 标记该节点为反向已访问
            visited_reverse.add(u)
            # 遍历当前节点的所有邻接结点（反向搜索需检查adj[j][u]）
            for j in range(len(adj)):
                # 反向搜搜中j到u有连接
                if adj[j][u] > 0:
                    # 标记邻接结点为已访问
                    visited_reverse.add(j)
                    #通过u到j的反向距离更短，更新距离
                    if reverse_dist[j] > reverse_dist[u] + adj[j][u]:
                        # 更新距离
                        reverse_dist[j] = reverse_dist[u] + adj[j][u]
                        # 加入队列
                        heapq.heappush(reverse_heap, (reverse_dist[j], j))
                        # 记录反向前驱节点
                        reverse_prev[j] = u
                    # 检查是否与正向搜索相遇
                    if j in visited_forward:
                        # 计算总距离
                        total = forward_dist[j] + reverse_dist[j]
                        # 若找到更短路径
                        if total < shortest_distance:
                            # 更新最短距离
                            shortest_distance = total
                            # 记录相遇节点
                            meeting_node = j

        # 终止条件：两队列最小键值之和 >= 当前最短路径
        # 若两个队列都不为空
        if forward_heap and reverse_heap:
            # 正向队列当前最小距离
            current_forward_min = forward_heap[0][0]
            # 反向队列当前最小距离
            current_reverse_min = reverse_heap[0][0]
            # 若两队列最小距离之和 >= 已找到的最短距离，说明无法找到更短路径
            if current_forward_min + current_reverse_min >= shortest_distance:
                # 终止搜索
                break

    # 路径回溯：正向、反向
    # 若未找到相遇节点，说明无路径
    if meeting_node is None:
        return None, float('inf')

    # 正向路径回溯：从相遇节点回溯到起点
    path = []
    node = meeting_node
    # 循环至到回溯到起点
    while node != start:
        # 添加当前节点
        path.append(node)
        # 获取上游结点
        node = forward_prev.get(node)
    # 添加起点
    path.append(start)
    # 反转路径，得到从起点到相遇结点的正向路径
    path = path[::-1]

    # 反向路径回溯：从相遇节点回溯到终点
    node = meeting_node
    # 循环直到回溯到终点
    while node != end:
        # 获取反向上游节点
        node = reverse_prev.get(node)
        # 添加到路径
        path.append(node)

    # 返回完整路径和最短距离
    return path


def generate_large_adj_matrix(size, density=0.2, min_weight=1, max_weight=10):
    adj = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(i + 1, size):  # 只生成上三角，保证无向图对称性
            # 按密度随机生成边
            if random.random() < density:
                weight = random.randint(min_weight, max_weight)
                adj[i][j] = weight
                adj[j][i] = weight  # 无向图对称
    return adj

# if __name__ == '__main__':
#     # 论文中示例图的邻接矩阵
#     adj1 = generate_large_adj_matrix(10000, density=0.5, min_weight=1, max_weight=10)
#     t = time.thread_time()
#     path1, distance = bidirectional_dijkstra(adj1, 2602, 2601)
#     print(f"{time.thread_time() - t}s")
#     print(f"最短路径: {path1}")
#     print(f"最短距离: {distance}")