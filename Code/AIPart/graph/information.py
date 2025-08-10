# 拥挤程度float
# 事故str正则判断
# 车流量float
# 人流密度float
# 车辆统计分析
# 天气（温度，风向，空气质量）

import json

def read_json_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
        return data

def consgestion_degree()->float:
    """
    统计拥挤程度
    :return: float
    """



    pass


def accident_rate()->str:
    """
    统计事故率
    :return:
    """

    pass

def traffic_flow()->float:
    """
    统计车流量
    :return:
    """
    pass

def human_flow()->float:
    """
    统计人流密度
    :return:
    """
    pass

def vehicle_statistics()->dict:
    """
    统计车辆信息
    :return:
    """
    pass

def weather()->dict:
    """
    统计天气信息
    :return:
    """
    pass