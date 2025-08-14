import numpy as np
from Code.AIPart.graph.graph import Graph

import os
# 从提供的模块中导入Graph类



def generate_obstacles_params():
    """从Graph类提取数据并生成障碍物参数"""
    # 初始化图对象
    graph = Graph()

    # 加载数据（假设JSON文件路径正确，若有必要可调整）
    # 尝试不同的路径加载方式
    json_paths = [
        "%s\\graph\\data.json" % os.getcwd(),
        "data.json",
        "graph/data.json"
    ]

    load_success = False
    for path in json_paths:
        try:
            if graph.load_json(path):
                load_success = True
                break
        except Exception as e:
            continue

    if not load_success:
        raise FileNotFoundError("无法加载JSON数据文件，请检查路径是否正确")

    obstacles_params = []

    # 处理所有边（道路）
    for edge in graph.edges:
        start_id = edge.start_id
        end_id = edge.end_id

        # 从位置矩阵获取坐标
        start_point = tuple(graph.nodes_position[start_id])
        end_point = tuple(graph.nodes_position[end_id])

        # 添加道路参数，宽度设为12，gap_offset_ratio设为1.0
        obstacles_params.append({
            "start_point": start_point,
            "end_point": end_point,
            "width": 12,
            "gap_offset_ratio": 1.0
        })

    # 计算地图边缘（根据所有点的坐标范围）
    all_coords = graph.nodes_position
    max_x = np.max(all_coords[:, 0]) + 100  # 增加100作为边界缓冲
    max_y = np.max(all_coords[:, 1]) + 100
    min_x = np.min(all_coords[:, 0]) - 100
    min_y = np.min(all_coords[:, 1]) - 100

    # 计算地图大小（取最大维度）
    MAP_SIZE = max(max_x - min_x, max_y - min_y)

    # 添加地图边缘障碍物
    obstacles_params.extend([
        {"start_point": (min_x, min_y), "end_point": (min_x, max_y), "width": 3, "gap_offset_ratio": 0},
        {"start_point": (max_x, min_y), "end_point": (max_x, max_y), "width": 3, "gap_offset_ratio": 0},
        {"start_point": (min_x, min_y), "end_point": (max_x, min_y), "width": 3, "gap_offset_ratio": 0},
        {"start_point": (min_x, max_y), "end_point": (max_x, max_y), "width": 3, "gap_offset_ratio": 0}
    ])

    return obstacles_params, MAP_SIZE

# 目标点热度值（保留两位小数）
node_heat = [
    0.02, 0.04, 0.02, 0.01, 0.05, 0.09, 0.10, 0.03, 0.02, 0.04, 0.02, 0.04, 0.03,
    0.02, 0.01, 0.03, 0.02, 0.02, 0.03, 0.02, 0.05, 0.03, 0.03, 0.03, 0.01, 0.04,
    0.03, 0.03, 0.02, 0.02, 0.04
]

# 已生成的目标点（每个节点1-2个，确保不在边两侧12格范围内）
targets = np.array([
    # 松风涧 (0) - 2个目标点
    [4710.0, 310.0], [4650.0, 350.0],
    # 月栖滩 (1) - 1个目标点
    [5390.0, 790.0],
    # 云岫台 (2) - 2个目标点
    [3310.0, 1950.0], [3250.0, 1900.0],
    # 花溪渡 (3) - 1个目标点
    [3710.0, 2120.0],
    # 星垂浦 (4) - 2个目标点
    [4700.0, 2190.0], [4640.0, 2140.0],
    # 竹影潭 (5) - 1个目标点
    [5700.0, 2260.0],
    # 雾隐桥 (6) - 2个目标点
    [1850.0, 3340.0], [1790.0, 3290.0],
    # 棠香坞 (7) - 1个目标点
    [2820.0, 3360.0],
    # 枫径斜 (8) - 2个目标点
    [3240.0, 3410.0], [3180.0, 3360.0],
    # 荷风榭 (9) - 1个目标点
    [4220.0, 3140.0],
    # 砚池春 (10) - 2个目标点
    [4770.0, 3210.0], [4710.0, 3160.0],
    # 书声崖 (11) - 1个目标点
    [5410.0, 3230.0],
    # 画舫驿 (12) - 2个目标点
    [6050.0, 3340.0], [5990.0, 3290.0],
    # 棋趣坪 (13) - 1个目标点
    [1980.0, 3870.0],
    # 诗墙巷 (14) - 2个目标点
    [2870.0, 3960.0], [2810.0, 3910.0],
    # 灯影廊 (15) - 1个目标点
    [3310.0, 3910.0],
    # 拓片台 (16) - 2个目标点
    [3710.0, 4180.0], [3650.0, 4130.0],
    # 弦歌榭 (17) - 1个目标点
    [4190.0, 4140.0],
    # 忆旧轩 (18) - 2个目标点
    [4720.0, 4220.0], [4660.0, 4170.0],
    # 问津亭 (19) - 1个目标点
    [4720.0, 3850.0],
    # 雀跃坪 (20) - 2个目标点
    [5430.0, 3800.0], [5370.0, 3750.0],
    # 蝶踪径 (21) - 1个目标点
    [6030.0, 3780.0],
    # 萤火星 (22) - 2个目标点
    [3310.0, 4710.0], [3250.0, 4660.0],
    # 风铃渡 (23) - 1个目标点
    [3600.0, 5110.0],
    # 落英阶 (24) - 2个目标点
    [4330.0, 4910.0], [4270.0, 4860.0],
    # 镜心湖 (25) - 1个目标点
    [5300.0, 4780.0],
    # 踏浪矶 (26) - 2个目标点
    [2400.0, 5770.0], [2340.0, 5720.0],
    # 叠翠屏 (27) - 1个目标点
    [2510.0, 6280.0],
    # 听雪轩 (28) - 2个目标点
    [3270.0, 6120.0], [3210.0, 6070.0],
    # 陆小凤 (29) - 1个目标点
    [4550.0, 5930.0],
    # 小凤路 (30) - 2个目标点
    [5960.0, 5700.0], [5900.0, 5650.0]
])

# 对应的目标点热度值
targets_heat = [
    # 松风涧
    0.02, 0.02,
    # 月栖滩
    0.04,
    # 云岫台
    0.02, 0.02,
    # 花溪渡
    0.01,
    # 星垂浦
    0.05, 0.05,
    # 竹影潭
    0.09,
    # 雾隐桥
    0.05, 0.05,
    # 棠香坞
    0.03,
    # 枫径斜
    0.02, 0.02,
    # 荷风榭
    0.04,
    # 砚池春
    0.02, 0.02,
    # 书声崖
    0.04,
    # 画舫驿
    0.03, 0.03,
    # 棋趣坪
    0.02,
    # 诗墙巷
    0.01, 0.01,
    # 灯影廊
    0.03,
    # 拓片台
    0.02, 0.02,
    # 弦歌榭
    0.02,
    # 忆旧轩
    0.03, 0.03,
    # 问津亭
    0.02,
    # 雀跃坪
    0.05, 0.05,
    # 蝶踪径
    0.04,
    # 萤火星
    0.03, 0.03,
    # 风铃渡
    0.03,
    # 落英阶
    0.01, 0.01,
    # 镜心湖
    0.04,
    # 踏浪矶
    0.03, 0.03,
    # 叠翠屏
    0.03,
    # 听雪轩
    0.02, 0.02,
    # 陆小凤
    0.02,
    # 小凤路
    0.04, 0.04
]

# 生成并测试结果
if __name__ == "__main__":
    try:
        obstacles_params, map_size = generate_obstacles_params()
        print(f"成功生成障碍物参数，共{len(obstacles_params)}个障碍物")
        print(f"地图大小: {map_size}")

        # 打印前5个障碍物参数作为示例
        print("\n前5个障碍物参数示例:")
        for i, param in enumerate(obstacles_params[:5]):
            print(f"障碍物{i + 1}: {param}")
    except Exception as e:
        print(f"生成过程出错: {str(e)}")


# print(f"已生成固定目标点：共{len(points_coords)}个节点，生成了{len(targets)}个目标点")
