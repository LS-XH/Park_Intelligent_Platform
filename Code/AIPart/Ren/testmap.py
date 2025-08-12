import numpy as np

# 单向道路地图参数（所有坐标数字乘3）
obstacles_params = [
    # 一路：陆路 → 小路
    {"start_point": (4 * 3, 55 * 3), "end_point": (116 * 3, 55 * 3), "width": 12, "gap_offset_ratio": 1.0},
    # 二路：小路 → 凤路
    {"start_point": (116 * 3, 55 * 3), "end_point": (220 * 3, 72 * 3), "width": 12, "gap_offset_ratio": 1.0},
    # 四路：小路 → 文路
    {"start_point": (116 * 3, 55 * 3), "end_point": (116 * 3, 111 * 3), "width": 12, "gap_offset_ratio": 1.0},
    # 三路：文路 → 希路
    {"start_point": (116 * 3, 111 * 3), "end_point": (4 * 3, 111 * 3), "width": 12, "gap_offset_ratio": 1.0},
    # 五路：文路 → 锐路
    {"start_point": (116 * 3, 111 * 3), "end_point": (220 * 3, 111 * 3), "width": 12, "gap_offset_ratio": 1.0},
    # 七路：文路 → 争路
    {"start_point": (116 * 3, 111 * 3), "end_point": (116 * 3, 178 * 3), "width": 12, "gap_offset_ratio": 1.0},
    # 六路：凤路 → 锐路
    {"start_point": (220 * 3, 72 * 3), "end_point": (220 * 3, 111 * 3), "width": 12, "gap_offset_ratio": 1.0},
    # 八路：锐路 → 英路
    {"start_point": (220 * 3, 111 * 3), "end_point": (220 * 3, 148 * 3), "width": 12, "gap_offset_ratio": 1.0},
    # 地图边缘
    {"start_point": (0, 0), "end_point": (0, 700), "width": 3, "gap_offset_ratio": 0},
    {"start_point": (700, 0), "end_point": (700, 700), "width": 3, "gap_offset_ratio": 0},
    {"start_point": (0, 0), "end_point": (700, 0), "width": 3, "gap_offset_ratio": 0},
    {"start_point": (0, 700), "end_point": (700, 700), "width": 3, "gap_offset_ratio": 0},
]

# 目标点参数（所有坐标数字乘3，同步调整以适配新坐标范围，这里简单按原相对位置乘3 ）
targets = np.array([
    [91 * 3, 150 * 3],
    [210 * 3, 105 * 3],
    [30 * 3, 24 * 3],
    [110 * 3, 42 * 3],
    [200 * 3, 86 * 3],
    [110 * 3, 100 * 3],
    [195 * 3, 123 * 3]
])
targets_heat = [0.2, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2]

points = [
    (12, 165),   # 陆路：对应一路起点 (4*3, 55*3)
    (348, 165),  # 小路：对应一路终点/二路起点/四路起点 (116*3, 55*3)
    (660, 216),  # 凤路：对应二路终点/六路起点 (220*3, 72*3)
    (12, 333),   # 希路：对应三路终点 (4*3, 111*3)
    (348, 333),  # 文路：对应四路终点/三路起点/五路起点/七路起点 (116*3, 111*3)
    (660, 333),  # 锐路：对应五路终点/六路终点/八路起点 (220*3, 111*3)
    (348, 534),  # 争路：对应七路终点 (116*3, 178*3)
    (660, 444)   # 英路：对应八路终点 (220*3, 148*3)
]