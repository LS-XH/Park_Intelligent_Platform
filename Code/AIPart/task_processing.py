# ================================
# @File         : task_processing.py
# @Time         : 2025/08/14
# @Author       : Yingrui Chen
# @description  : AI端服务器数据处理模块
# ================================

import queue

from AIServer.common_util import common_logger
from graph import Graph
from Ren.mainpart import Crowd
from Ren.testmap import targets, targets_heat
from Algorithm.bidirectional_dijkstra import bidirectional_dijkstra
from load_frame import *

"""
# ====================================
#
#            全局调控变量
# 
# ====================================
"""

# 初始化图
graph = Graph()
message_queue = queue.Queue()

# 读取映射表
with open("AIServer/jsonData/reflect_table.json", "r", encoding='utf-8') as f:
    reflect_table = json.load(f)
f.close()

# 读取人群初始化所需的信息
with open("./graph/data.json", "r", encoding="utf-8") as f:
    people = json.load(f)
    edges = people['edges']
f.close()

# 初始化人群
crowd = Crowd(
    points=list(graph.nodes_position),
    edges=edges,
    targets=targets,
    targets_heat=targets_heat,
    num_agents=100
)

# 初始化智能模块选项
AGENT_CONFIG = {
    "bestRoute": False,
    "crowdEvacuation": False,
    "trafficLight": False,
    "CAV": False
}

ALL_EMERGENCY = []

common_logger.info("【AI】所有组件预加载完毕！")
"""
# ====================================
#            Web数据获取逻辑
# ====================================
# 0、        服务器通信测试✅
# 1、        获取全局天气✅
# 2、        修改全局天气✅
# 3、        开始接收全局数据✅
# 4、        停止接收全局数据✅
# 5、        推送节点数据✅
# 6、        推送路段数据✅
# 7、        推送节点风险预测❌
# 8、        推送路段风险预测❌
# 9、        智能模块开关✅
# 10、       寻路✅
# 11、       突发事件设置⭕️
# 12、       进程控制✅
# 13、       人口密度✅
# ====================================
#           Unity数据获取逻辑
# ====================================
# 0、        车辆数据推送⭕️
# 1、        车辆数据初始化✅
# 15、       聚焦节点❌对应7无法实现
# 16、       聚焦路段❌对应8无法实现
# ====================================
"""


def web_solve_status_0():
    """WEB端处理状态码0，用于通信测试"""
    try:
        response = {"status": 0, "response": {}}
        response["response"]["content"] = "success"

        return response
    except Exception as e:
        common_logger.error(f"WEB】状态码[0]处理异常：{e}")
        response = {"status": 0, "response": "error"}

        return response


def web_solve_status_1():
    """
    WEB端处理状态码1，用于获取天气数据（温度、风力等级、空气质量、降水量）
    * 温度：-20～40
    * 风力等级：0～100
    * 空气质量：0～100
    * 降水量：0～100
    """
    try:
        init_weather = graph.get_weather()
        response = {"status": 1, "response": {}}
        response["response"]["temperature"] = int(init_weather[0])
        response["response"]["wind"] = int(init_weather[1])
        response["response"]["airQuality"] = int(init_weather[2])
        response["response"]["rain"] = int(init_weather[3])

        return response
    except Exception as e:
        common_logger.error(f"WEB】状态码[1]处理异常：{e}")
        response = {"status": 1, "response": "error"}

        return response


def web_solve_status_2(message=None):
    """WEB端处理状态码2，用于修改天气数据"""
    try:
        weather_data = message.get("message")
        graph.set_weather(weather_data)

        response = {"status": 2, "response": {}}
        response["response"]["content"] = "success"

        return response
    except Exception as e:
        common_logger.error(f"WEB】状态码[2]处理异常：{e}")
        response = {"status": 2, "response": "error"}

        return response


def web_solve_status_3():
    """WEB端处理状态码3，向WEB端持续传输全局车流、人流、拥挤程度数据"""
    try:
        all_information = graph.get_all_information(crowd.density)
        response = {"status": 3, "response": {}}
        response["response"]["cars"] = all_information[0]
        response["response"]["people"] = all_information[1]
        response["response"]["crowding"] = all_information[2]

        return response
    except Exception as e:
        common_logger.error(f"WEB】状态码[3]处理异常：{e}")
        response = {"status": 3, "response": "error"}

        return response


def web_solve_status_4():
    """WEB端处理状态码4，停止接收全局车流、人流、拥挤程度数据"""

    # ===================================
    #     留空
    # ===================================

    pass


def web_solve_status_5(point_id=None):
    """WEB端处理状态码5，持续传输节点⻋流、⼈流、拥挤程度、突发事件、红绿灯数据"""
    try:
        point_information = graph.get_point_information(point_id, crowd.density)

        response = {"status": 5, "response": {}}

        response["response"]["name"] = graph.__getitem__(point_id)
        response["response"]["cars"] = point_information[0]
        response["response"]["people"] = point_information[1]
        response["response"]["crowding"] = point_information[2]
        response["response"]["emergency"] = random.choice(["", "车辆失灵", "车祸", "道路维修", "严重灾害"])
        response["response"]["trafficLight"] = [random.choice(['red', 'green']), random.randint(0, 50),
                                                random.randint(50, 60)]

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[5]处理异常：{e}")
        response = {"status": 5, "response": "error"}

        return response


def web_solve_status_6(road_name=None):
    """WEB端处理状态码6，持续传输路段统计信息"""
    try:
        road_name = reflect_table['edges'][str(road_name)]
        road_information = graph.get_road_information(road_name, crowd.density)

        response = {"status": 6, "response": {}}

        response["response"]["name"] = road_name
        response["response"]["cars"] = road_information[0]
        response["response"]["people"] = road_information[1]
        response["response"]["crowding"] = round(100 * road_information[2], 2)
        response["response"]["emergency"] = random.choice(["", "车辆失灵", "车祸", "道路维修", "严重灾害"])

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[6]处理异常：{e}")
        response = {"status": 6, "response": "error"}

        return response


def web_solve_status_7(point_id=None):
    """WEB端处理状态码7，持续传输节点风险预测数据"""
    try:
        # point_risk_data, point_predict_data = graph.get_point_risk_data(point_id, crowd.density)

        response = {"status": 7, "response": {}}

        response["response"]["name"] = point_id
        response["response"]["riskData"] = [round(random.uniform(0, 100), 1) for _ in range(12)]
        response["response"]["predict"] = random.uniform(0, 100)

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[7]处理异常：{e}")
        response = {"status": 7, "response": "error"}

        return response


def web_solve_status_8(road_name=None):
    """WEB端处理状态码8，持续传输路段风险预测数据"""
    try:
        road_name = reflect_table['edges'][str(road_name)]
        # road_risk_data, road_predict_data = graph.get_road_risk_data(road_name, crowd.density)

        response = {"status": 8, "response": {}}

        response["response"]["name"] = road_name
        response["response"]["riskData"] = [round(random.uniform(0, 100), 1) for _ in range(12)]
        response["response"]["predict"] = random.uniform(0, 100)

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[8]处理异常：{e}")
        response = {"status": 8, "response": "error"}

        return response


def web_solve_status_9(message=None):
    """WEB端处理状态码9，用于智能模块控制"""
    try:
        global AGENT_CONFIG
        AGENT_CONFIG = message.get("message")

        response = {"status": 9, "response": {}}
        response["response"]["content"] = "success"

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[9]处理异常：{e}")
        response = {"status": 9, "response": "error"}

        return response


def web_solve_status_10(message=None):
    """WEB端处理状态码10，用于寻路模块"""
    try:
        start_name = message['message']['startName']
        end_name = message['message']['endName']
        start_id = reflect_table['points'][start_name]
        end_id = reflect_table['points'][end_name]

        route = bidirectional_dijkstra(graph.length, start_id, end_id)
        name_route = []

        for point in route[0]:
            point = graph.__getitem__(point)
            name_route.append(point)

        response = {"status": 10, "response": {}}

        response["response"]["success"] = True
        response["response"]["route"] = name_route

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[10]处理异常：{e}")
        response = {"status": 10, "response": "error"}

        return response


def web_solve_status_11():
    """WEB端处理状态码11，用于突发事件设置"""
    try:

        response = {"status": 11, "response": {}}

        response["response"]["content"] = "success"

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[11]处理异常：{e}")
        response = {"status": 11, "response": "error"}

        return response


def web_solve_status_12():
    """WEB端处理状态码12，用于控制全局模拟进程"""
    try:
        # ===================================
        #         模拟进程控制逻辑
        # ===================================

        response = {"status": 12, "response": {}}

        response["response"]["content"] = "success"

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[12]处理异常：{e}")
        response = {"status": 12, "response": "error"}

        return response


def web_solve_status_13():
    """WEB端处理状态码13，用于持续传输人口密度"""
    try:
        global crowd
        density_data = crowd.density
        for _ in range(100):
            crowd.simulate(happened=None, trafficlight=None)
        min_val = density_data.min()
        max_val = density_data.max()
        normalize_data = (density_data - min_val) / (max_val - min_val)

        response = {"status": 13, "response": {}}

        response["response"]["people"] = normalize_data.tolist()

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[13]处理异常：{e}")
        response = {"status": 13, "response": "error"}

        return response


"""
# ====================================
#
#      WebSocket服务器代码处理逻辑
# 
# ====================================
"""


def web_process_data(message=None, sim_type=None):
    # 处理实时推送数据生成（由服务器内部调用）
    if sim_type is not None:
        return generate_realtime_data(message, sim_type)

    # 处理客户端请求（静态数据或实时启停指令）
    status = message.get("status")
    if status is None:
        return {
            "type": "static",
            "data": {"status": -1, "response": {"content": "Missing 'status' in message"}}
        }

    # 实时数据启动指令
    if status in [3, 5, 6, 7, 8, 13]:
        return {
            "type": "realtime_start",
            "sim_type": status
        }

    # 实时数据停止指令（4/6/8/10/12）
    stop_to_start_map = {4: 3}
    if status in stop_to_start_map:
        return {
            "type": "realtime_stop",
            "sim_type": stop_to_start_map[status],
            "original_status": status
        }

    # 静态数据处理
    return {
        "type": "static",
        "data": process_static_message(message, status)
    }


def process_static_message(message, status):
    """处理静态消息（独立函数）"""
    if status == 0:
        return web_solve_status_0()
    elif status == 1:
        return web_solve_status_1()
    elif status == 2:
        return web_solve_status_2(message=message)
    elif status == 9:
        return web_solve_status_9(message=message)
    elif status == 10:
        return web_solve_status_10(message=message)
    elif status == 11:
        return web_solve_status_11()
    elif status == 12:
        return web_solve_status_12()
    else:
        return {"status": -1, "response": {"content": f"Unknown status code: {status}"}}


def generate_realtime_data(sim_type):
    """生成实时推送数据（独立函数）"""
    if sim_type == 3:
        return web_solve_status_3()
    elif sim_type == 5:
        return web_solve_status_5()
    elif sim_type == 6:
        return web_solve_status_6()
    elif sim_type == 7:
        return web_solve_status_7()
    elif sim_type == 8:
        return web_solve_status_8()
    elif sim_type == 13:
        return web_solve_status_13()
    else:
        return {"status": -1, "response": {"content": "Unknown simulation type"}}


"""
# ====================================
#
#            Unity数据获取逻辑
# 
# ====================================
"""


def unity_solve_status_0():
    """TCP端处理状态码0，用于推送实时车辆模拟数据"""

    pass


def unity_solve_status_1():
    """TCP端处理状态码1，用于获取初始化车辆信息"""

    init_cars_data = initialize_cars(30)
    init_cars_data = cars_to_unity(init_cars_data)

    return json.dumps(init_cars_data)


def unity_solve_status_2(data):
    """TCP端处理状态码2，用于模拟用户点击节点"""

    node_id = data.get("message").get("id")
    if node_id:
        enqueue_information = {
            "status": 15,
            "message": {
                "id": node_id
            }
        }
        message_queue.put(enqueue_information)

    return


def unity_solve_status_3(data):
    """TCP端处理状态码3，用于模拟用户点击节点"""

    road_name = data.get("message").get("id")
    if road_name:
        enqueue_information = {
            "status": 16,
            "message": {
                "id": road_name
            }
        }
        message_queue.put(enqueue_information)

    return


"""
# ====================================
#
#         TCP服务器代码处理逻辑
# 
# ====================================
"""


def unity_process_message(data=None, tcpServer=None):
    req = json.loads(data)
    status = req.get("status")

    # 状态码0：车辆数据实时推送
    if status == 0:
        return unity_solve_status_1()

    # 状态码1：获取初始化信息
    elif status == 1:
        return unity_solve_status_1()

    # 状态码2：用户点击节点
    elif status == 15:
        return unity_solve_status_2(req)

    # 状态码3：用户点击路段
    elif status == 16:
        return unity_solve_status_3(req)

    return None
