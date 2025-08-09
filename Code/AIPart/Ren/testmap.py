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
]

# 目标点参数（所有坐标数字乘3，同步调整以适配新坐标范围，这里简单按原相对位置乘3 ）
targets = np.array([
    [91 * 3, 150 * 3],    # 小路
    [210 * 3, 105 * 3],   # 凤路
    [30 * 3, 24 * 3],    # 希路
    [110 * 3, 42 * 3],    # 文路
    [200 * 3, 86 * 3],   # 锐路
    [110 * 3, 100 * 3],   # 争路
    [195 * 3, 123 * 3]   # 英路
])
