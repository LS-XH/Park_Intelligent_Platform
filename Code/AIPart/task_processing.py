# ================================
# @File         : task_processing.py
# @Time         : 2025/08/14
# @Author       : Yingrui Chen
# @description  : AI端服务器数据处理模块
# ================================
import base64
import queue

import joblib

from AIServer.common_util import common_logger
from graph import Graph
from car.car import Cars
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
    graph_json = json.load(f)
    points = graph_json["points"]
    edges = graph_json['edges']
f.close()

with open("./graph/hot_data.json", "r", encoding="utf-8") as f:
    hot_data = json.load(f)
f.close()

# 初始化人群
crowd = joblib.load("./Ren/crowd.pkl")

# 初始化智能模块选项
AGENT_CONFIG = {
    "bestRoute": False,
    "crowdEvacuation": False,
    "trafficLight": False,
    "CAV": False
}

ALL_EMERGENCY = []

init_cars = initialize_cars(30)
init_cars_list = cars_to_calculate(init_cars)

init_cars_list = [

    [0, 1, 1, 0.1, 11],

    [0, 1, 1, 0.15, 11],

    [0, 1, 1, 0.25, 11],

    [0, 1, 0, 0.2, 11],
]

CARS = Cars(graph, init_cars_list)

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
# 7、        推送节点风险预测✅
# 8、        推送路段风险预测✅
# 9、        智能模块开关✅
# 10、       寻路✅
# 11、       突发事件设置✅
# 12、       进程控制✅
# 13、       人口密度✅
# ====================================
#           Unity数据获取逻辑
# ====================================
# 0、        车辆数据推送✅
# 1、        车辆数据初始化✅
# 15、       聚焦节点✅
# 16、       聚焦路段✅
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
        global crowd, CARS
        cal_cars = []
        for i in range(len(CARS.car_positions)):
            cal_cars.append((CARS.car_positions[i]['x'], CARS.car_positions[i]['y']))
        cal_people = crowd.position
        all_information = graph.get_all_information(cal_cars, cal_people, hot_data)

        response = {"status": 3, "response": {}}
        response["response"]["cars"] = all_information[0]
        response["response"]["people"] = all_information[1]
        response["response"]["crowding"] = all_information[2]

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[3]处理异常：{e}")
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
        global crowd, CARS
        cal_cars = []
        for i in range(len(CARS.car_positions)):
            cal_cars.append((CARS.car_positions[i]['x'], CARS.car_positions[i]['y']))
        cal_people = crowd.position

        point_information = graph.get_point_information(point_id, cal_cars, cal_people, hot_data)

        response = {"status": 5, "response": {}}

        response["response"]["name"] = graph.__getitem__(point_id)
        response["response"]["cars"] = point_information[0]
        response["response"]["people"] = point_information[1]
        response["response"]["crowding"] = point_information[2]
        response["response"]["emergency"] = point_information[3]
        response["response"]["trafficLight"] = point_information[4]

        response = json.dumps(response)
        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[5]处理异常：{e}")
        response = {"status": 5, "response": "error"}
        response = json.dumps(response)

        return response


def web_solve_status_6(road_id=None):
    """WEB端处理状态码6，持续传输路段统计信息"""
    try:
        global crowd, CARS
        cal_cars = []
        for i in range(len(CARS.car_positions)):
            cal_cars.append((CARS.car_positions[i]['x'], CARS.car_positions[i]['y']))
        cal_people = crowd.position

        road_name = reflect_table['edges'][str(road_id)]
        start_id, end_id = None, None
        for edge in edges:
            if edge['name'] == road_name:
                start_id = edge['start_id']
                end_id = edge['end_id']

        road_information = graph.get_road_information(road_id, start_id, end_id, cal_cars, cal_people, hot_data)

        response = {"status": 6, "response": {}}

        response["response"]["name"] = road_name
        response["response"]["cars"] = road_information[0]
        response["response"]["people"] = road_information[1]
        response["response"]["crowding"] = road_information[2]
        response["response"]["emergency"] = road_information[3]
        response = json.dumps(response)

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[6]处理异常：{e}")
        response = {"status": 6, "response": "error"}
        response = json.dumps(response)

        return response


def web_solve_status_7(point_id=None):
    """WEB端处理状态码7，持续传输节点风险预测数据"""
    try:
        global crowd, CARS
        cal_cars = []
        for i in range(len(CARS.car_positions)):
            cal_cars.append((CARS.car_positions[i]['x'], CARS.car_positions[i]['y']))
        cal_people = crowd.position

        point_risk_data, point_predict_data = graph.get_point_risk_data(point_id, cal_cars, cal_people, hot_data)
        if len(point_risk_data) >= 12:
            point_risk_data = point_risk_data[-12:]

        response = {"status": 7, "response": {}}

        response["response"]["name"] = point_id
        response["response"]["riskData"] = point_risk_data
        response["response"]["predict"] = point_predict_data
        response = json.dumps(response)

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[7]处理异常：{e}")
        response = {"status": 7, "response": "error"}
        response = json.dumps(response)

        return response


def web_solve_status_8(road_id=None):
    """WEB端处理状态码8，持续传输路段风险预测数据"""
    try:
        global crowd, CARS
        cal_cars = []
        for i in range(len(CARS.car_positions)):
            cal_cars.append((CARS.car_positions[i]['x'], CARS.car_positions[i]['y']))
        cal_people = crowd.position

        road_name = reflect_table['edges'][str(road_id)]
        road_risk_data, road_predict_data = graph.get_road_risk_data(road_id, cal_cars, cal_people, hot_data)
        if len(road_risk_data) >= 12:
            road_risk_data = road_risk_data[-12:]

        response = {"status": 8, "response": {}}

        response["response"]["name"] = road_name
        response["response"]["riskData"] = road_risk_data
        response["response"]["predict"] = road_predict_data
        response = json.dumps(response)

        return response
    except Exception as e:
        common_logger.error(f"【WEB】状态码[8]处理异常：{e}")
        response = {"status": 8, "response": "error"}
        response = json.dumps(response)

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

        for point in route:
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


def web_solve_status_11(message=None):
    """WEB端处理状态码11，用于突发事件设置"""
    try:
        global ALL_EMERGENCY
        place_name = message.get('message').get('name')
        emergency = message.get('message').get('emergency')
        all_points_name = [point.name for point in graph.points]

        if place_name in all_points_name:
            for point in graph.points:
                if place_name == point.name:
                    emergency_x, emergency_y = point.x, point.y
                    ALL_EMERGENCY.append(((emergency_x, emergency_y), emergency))

        for edge in edges:
            if place_name == edge['name']:
                edge_start_id, edge_end_id = edge['start_id'], edge['end_id']
                start_point, end_point = points[edge_start_id], points[edge_end_id]
                start_x, start_y, end_x, end_y = start_point['x'], start_point['y'], end_point['x'], end_point['y']
                random_percent = random.uniform(0, 1)
                emergency_x = start_x + (end_x - start_x) * random_percent
                emergency_y = start_y + (end_y - start_y) * random_percent
                ALL_EMERGENCY.append(((emergency_x, emergency_y), emergency))

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
        global crowd, ALL_EMERGENCY

        for _ in range(10):
            crowd.simulate(happened=ALL_EMERGENCY, trafficlight=None)

        crowd_img = crowd.generate_frame([])

        response = {"status": 13, "response": crowd_img}

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
        return generate_realtime_data(sim_type)

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
        return web_solve_status_11(message=message)
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
    try:
        global CARS

        for _ in range(1):
            CARS.simulate(dt=0.1)

        cars_info = CARS.car_positions
        result = {}
        for i in range(len(cars_info)):
            car_info = {"x": cars_info[i]['x'] / 10, "y": cars_info[i]['y'] / 10, "theta": cars_info[i]['theta']}
            result[str(i)] = car_info

        return json.dumps(result)
    except Exception as e:
        common_logger.error(f"【TCP】推送车辆模拟数据异常:{e}")


def unity_solve_status_1():
    """TCP端处理状态码1，用于获取初始化车辆信息"""
    try:
        global CARS

        cars_info = CARS.car_positions
        result = {str(index): value for index, value in enumerate(cars_info)}

        return json.dumps(result)
    except Exception as e:
        common_logger.error(f"【TCP】推送车辆模拟数据异常:{e}")


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
        return unity_solve_status_0()

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
