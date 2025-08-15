from Code.AIPart.graph.graph import Graph

def extract_and_format_data(graph):
    """
    从Graph类中提取数据并转换为指定格式
    :param graph: 已初始化的Graph实例
    :return: 包含edges、points_coords、node_names等的字典
    """
    # 1. 提取节点坐标和名称
    points_coords = []
    node_names = []
    for point in graph.points:
        # 提取坐标（保持原始单位，不额外缩放）
        points_coords.append((point.x, point.y))
        # 提取节点名称
        node_names.append(point.name)

    # 2. 提取边数据（start_id和end_id）
    edges = []
    for edge in graph.edges:
        edges.append({
            "start_id": edge.start_id,
            "end_id": edge.end_id
        })

    # 3. 生成道路参数（obstacles_params）
    obstacles_params = []
    for edge in edges:
        start_idx = edge["start_id"]
        end_idx = edge["end_id"]
        start_point = points_coords[start_idx]
        end_point = points_coords[end_idx]
        obstacles_params.append({
            "start_point": start_point,
            "end_point": end_point,
            "width": 12,  # 固定宽度，可根据实际需求调整
            "gap_offset_ratio": 1.0  # 固定比例
        })

    # 4. 计算地图边缘参数
    if points_coords:
        max_x = max(coord[0] for coord in points_coords) + 100
        max_y = max(coord[1] for coord in points_coords) + 100
        MAP_SIZE = max(max_x, max_y)
        # 添加地图边缘
        obstacles_params.extend([
            {"start_point": (0, 0), "end_point": (0, MAP_SIZE), "width": 3, "gap_offset_ratio": 0},
            {"start_point": (MAP_SIZE, 0), "end_point": (MAP_SIZE, MAP_SIZE), "width": 3, "gap_offset_ratio": 0},
            {"start_point": (0, 0), "end_point": (MAP_SIZE, 0), "width": 3, "gap_offset_ratio": 0},
            {"start_point": (0, MAP_SIZE), "end_point": (MAP_SIZE, MAP_SIZE), "width": 3, "gap_offset_ratio": 0}
        ])
    else:
        MAP_SIZE = 0  # 无节点时的默认值

    # 5. 提取节点热度（示例：使用度矩阵的行和作为热度，可根据实际逻辑调整）
    node_heat = []
    if hasattr(graph, 'degree'):
        degree_matrix = graph.degree
        for i in range(len(graph.points)):
            # 以节点的出度总和作为热度基础，归一化到0.01-0.1范围
            out_degree = degree_matrix[i].sum()
            normalized_heat = max(0.01, min(0.1, out_degree / (degree_matrix.sum() + 1e-6) * 0.1))
            node_heat.append(round(normalized_heat, 2))
    else:
        # 无度矩阵时的默认热度
        node_heat = [0.02 for _ in range(len(graph.points))]

    return {
        "edges": edges,
        "points_coords": points_coords,
        "node_names": node_names,
        "obstacles_params": obstacles_params,
        "MAP_SIZE": MAP_SIZE,
        "node_heat": node_heat
    }


# 使用示例
if __name__ == "__main__":
    # 初始化Graph并加载数据（假设从JSON加载了实际数据）


    graph = Graph()  # Graph内部会自动调用load_json加载数据

    # 提取并格式化数据
    result = extract_and_format_data(graph)

    # 打印结果示例
    print("提取的边数据（前5条）：")
    for edge in result["edges"][:5]:
        print(edge)
    print("\n提取的节点坐标（前5个）：")
    for coord in result["points_coords"][:5]:
        print(coord)
    print("\n生成的障碍物参数（前5条）：")
    for param in result["obstacles_params"][:5]:
        print(param)