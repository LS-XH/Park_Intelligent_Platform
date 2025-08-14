import re
def crowd_evacuation(crowd_density: float, emergency: str) -> tuple or None:
    """
    判断人群疏散等级

    :param crowd_density: 人群密度（0-10）
    :param emergency: 紧急情况描述字符串
    :return: (疏散等级, 灾害类型) 或 None
    """
    # 紧急情况等级映射（键：类型，值：基础等级）
    emergency_situation = {
        "minor_accident": 0,  # 轻微事故 0m
        "normal_accident": 1,  # 一般事故 5m
        "big_accident": 2,  # 较严重事故 25m
        "serious_accident": 3,  # 重大事故 125m
        "much_serious_accident": 4,  # 特大事故 625m
        "landslide": 4,  # 滑坡
        "earthquake": 5,  # 地震
        "tornado": 5,  # 龙卷风
        "flood": 5,  # 洪水
        "": None  # 无紧急情况
    }

    # 1. 匹配地质灾害（使用正则表达式，不区分描述顺序）
    geological_patterns = {
        "earthquake": re.compile(r'地震|地动|震感'),
        "tornado": re.compile(r'龙卷风|旋风|强风涡旋'),
        "flood": re.compile(r'洪水|内涝|暴雨淹没'),
        "landslide": re.compile(r'滑坡|塌方|山体滑动')
    }
    for disaster, pattern in geological_patterns.items():
        if pattern.search(emergency):
            base_level = emergency_situation[disaster]
            level = min(base_level + int(crowd_density // 2), 5)
            return level

    # 2. 匹配交通事故
    # 轻微事故
    if any(keyword in emergency for keyword in ["轻微刮擦", "无人员受伤", "车辆能正常移动", "不堵塞车道"]):
        emergency_type = "minor_accident"
    # 一般事故
    elif any(keyword in emergency for keyword in ["车辆中度损坏", "轻微人员受伤", "占用1-2条车道", "局部拥堵"]):
        emergency_type = "normal_accident"
    # 较严重事故
    elif any(keyword in emergency for keyword in ["严重损坏", "人员受伤", "送医治疗", "占用多条车道", "路段拥堵"]):
        emergency_type = "big_accident"
    # 重大事故
    elif any(keyword in emergency for keyword in
             ["多车连环碰撞", "车辆翻滚", "坠崖", "人员重伤", "被困车内", "完全阻断道路"]):
        emergency_type = "serious_accident"
    # 特大事故
    elif any(keyword in emergency for keyword in
             ["大规模连环事故", "危险品泄漏", "撞击建筑物", "撞击人群", "多人死亡", "批量重伤"]):
        emergency_type = "much_serious_accident"
    # 无匹配情况
    else:
        return emergency_situation.get(emergency, None)

    # 计算最终等级（结合人群密度）
    base_level = emergency_situation[emergency_type]
    level = min(base_level + int(crowd_density // 2), 5)  # 最高等级限制为5
    return level
