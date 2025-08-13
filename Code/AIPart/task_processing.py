import json
import random
import threading

from graph import Graph
from Algorithm.Optimal_pathfinding.bidirectional_dijkstra import bidirectional_dijkstra

graph = Graph()
graph.load_json("./graph/data.json")
with open("./graph/reflect_table.json", "r", encoding='utf-8') as file:
    reflect_table = json.load(file)
length_matrix = graph.length


"""
# ====================================
#
#            Web数据获取逻辑
# 
# ====================================
"""


def web_solve_status_0():
    """WEB端处理状态码0，用于通信测试"""
    response = {"status": 0, "response": {}}

    response["response"]["content"] = "success"

    return response


def web_solve_status_1():
    """
    WEB端处理状态码1，用于获取天气数据（温度、风力等级、空气质量、降水量）
    * 温度：-20～40
    * 风力等级：0～100
    * 空气质量：0～100
    * 降水量：0～100
    """

    # ===================================
    #          图接收天气逻辑函数
    # ===================================

    response = {"status": 1, "response": {}}

    response["response"]["temperature"] = random.uniform(-20, 40)
    response["response"]["wind"] = random.uniform(0, 100)
    response["response"]["airQuality"] = random.uniform(0, 100)
    response["response"]["rain"] = random.uniform(0, 100)

    return response


def web_solve_status_2(message=None):
    """WEB端处理状态码2，用于修改天气数据"""

    # ===================================
    #          图修改天气逻辑函数
    # ===================================

    response = {"status": 2, "response": {}}
    response["response"]["content"] = "success"

    return response


def web_solve_status_3():
    """WEB端处理状态码3，向WEB端持续传输全局车流、人流、拥挤程度数据"""

    # ===================================
    #     图获取全局车流、人流、拥挤程度
    # ===================================

    response = {"status": 3, "response": {}}

    response["response"]["cars"] = random.uniform(0, 100)
    response["response"]["people"] = random.uniform(0, 100)
    response["response"]["crowding"] = random.uniform(0, 100)

    return response


def web_solve_status_4():
    """WEB端处理状态码4，停止接收全局车流、人流、拥挤程度数据"""

    # ===================================
    #     留空
    # ===================================

    pass


def web_solve_status_5(point_name=None):
    """WEB端处理状态码5，持续传输节点⻋流、⼈流、拥挤程度、突发事件、红绿灯数据"""

    # ===================================
    #        获取节点统计信息逻辑
    # ===================================

    response = {"status": 5, "response": {}}

    response["response"]["name"] = point_name
    response["response"]["cars"] = random.uniform(0, 30)
    response["response"]["people"] = random.uniform(0, 3000)
    response["response"]["crowding"] = random.uniform(0, 100)
    response["response"]["emergency"] = random.choice(["", "车辆失灵", "车祸", "道路维修", "严重灾害"])
    response["response"]["trafficLight"] = [random.choice(['red', 'green']), random.randint(0, 50),
                                            random.randint(50, 60)]

    return response


def web_solve_status_6(road_name=None):
    """WEB端处理状态码6，持续传输路段统计信息"""

    # ===================================
    #         获取路段统计信息逻辑
    # ===================================

    response = {"status": 6, "response": {}}

    response["response"]["name"] = road_name
    response["response"]["cars"] = random.uniform(0, 30)
    response["response"]["people"] = random.uniform(0, 3000)
    response["response"]["crowding"] = random.uniform(0, 100)
    response["response"]["emergency"] = random.choice(["", "车辆失灵", "车祸", "道路维修", "严重灾害"])

    return response


def web_solve_status_7(point_name=None):
    """WEB端处理状态码7，持续传输节点风险预测数据"""

    # ===================================
    #         获取节点风险数据逻辑
    # ===================================

    response = {"status": 7, "response": {}}

    response["response"]["name"] = point_name
    response["response"]["riskData"] = [round(random.uniform(0, 100), 1) for _ in range(12)]
    response["response"]["predict"] = random.uniform(0, 100)

    return response


def web_solve_status_8(road_name=None):
    """WEB端处理状态码8，持续传输路段风险预测数据"""

    # ===================================
    #         获取路段风险数据逻辑
    # ===================================

    response = {"status": 8, "response": {}}

    response["response"]["name"] = road_name
    response["response"]["riskData"] = [round(random.uniform(0, 100), 1) for _ in range(12)]
    response["response"]["predict"] = random.uniform(0, 100)

    return response


def web_solve_status_9(message=None):
    """WEB端处理状态码9，用于智能模块控制"""

    # ===================================
    #         智能模块启停逻辑
    # ===================================

    response = {"status": 9, "response": {}}

    response["response"]["content"] = "success"

    return response


def web_solve_status_10(message=None):
    """WEB端处理状态码10，用于寻路模块"""

    # ===================================
    #           寻路模块逻辑
    start_name = message['data']['start']
    end_name = message['data']['end']
    start_id = reflect_table[start_name]
    end_id = reflect_table[end_name]

    route = bidirectional_dijkstra(length_matrix, start_id, end_id)
    name_route = []

    for point in route:
        point = graph.__getitem__(point)
        name_route.append(point)
    # ===================================

    response = {"status": 10, "response": {}}

    response["response"]["success"] = True
    response["response"]["route"] = name_route

    return response


def web_solve_status_11(message=None):
    """WEB端处理状态码11，用于突发事件设置"""

    # ===================================
    #           突发事件设置逻辑
    # ===================================

    response = {"status": 11, "response": {}}

    response["response"]["content"] = "success"

    return response


def web_solve_status_12(message=None):
    """WEB端处理状态码12，用于控制全局模拟进程"""

    # ===================================
    #         模拟进程控制逻辑
    # ===================================

    response = {"status": 12, "response": {}}

    response["response"]["content"] = "success"

    return response


def web_solve_status_13(message=None):
    """WEB端处理状态码13，用于持续传输人口密度"""

    # ===================================
    #          人口密度获取逻辑
    # ===================================

    response = {"status": 13, "response": {}}

    with open("./AIServer/jsonData/testdensitymatrix.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    response["response"]["people"] = data

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
        return web_solve_status_11(message=message)
    elif status == 12:
        return web_solve_status_12(message=message)
    else:
        return {"status": -1, "response": {"content": f"Unknown status code: {status}"}}


def generate_realtime_data(message, sim_type):
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
#            Web数据获取逻辑
# 
# ====================================
"""


def tcp_solve_status_1():
    """TCP端处理状态码1，用于获取初始化车辆信息"""

    # ===================================
    #          获取初始化车辆位置逻辑
    # ===================================

    response = {"status": 0, "response": {}}
    with open("./AIServer/jsonData/cars_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    response["response"]["cars"] = data

    return response


def tcp_solve_status_2():
    """TCP端处理状态码2，用于模拟用户点击节点"""

    # ===================================
    #          用户点击节点逻辑
    # ===================================

    response = {"status": 2, "response": {}}
    response["response"]["content"] = "success"

    return response


def tcp_solve_status_3():
    """TCP端处理状态码3，用于模拟用户点击节点"""

    # ===================================
    #          用户点击路段逻辑
    # ===================================

    response = {"status": 3, "response": {}}
    response["response"]["content"] = "success"

    return response


"""
# ====================================
#
#         TCP服务器代码处理逻辑
# 
# ====================================
"""


def tcp_process_message(data=None, tcpServer=None):
    req = json.loads(data)
    status = req.get("status")
    response = {"status": status}

    # 状态码0：通信测试
    if status == 0:
        print("留空处理")
        return {"status": 0}

    # 状态码1：获取初始化信息
    elif status == 1:
        return tcp_solve_status_1()

    # 状态码2：用户点击节点
    elif status == 2:
        return tcp_solve_status_2()

    # 状态码3：用户点击路段
    elif status == 3:
        return tcp_solve_status_3()

    # 状态码8：模拟进程控制
    elif status == 8:
        process = req["message"]["process"]
        tcpServer.simulation_speed = req["message"].get("time", 30)

        if process == "start":
            # 启动模拟推送
            with tcpServer.client_lock:
                for sock in tcpServer.clients:
                    addr, _ = tcpServer.clients[sock]
                    tcpServer.clients[sock] = (addr, True)
                    threading.Thread(
                        target=tcpServer.simulation_handler,
                        args=(sock,),
                        daemon=True
                    ).start()
        else:
            # 停止模拟推送
            with tcpServer.client_lock:
                for sock in tcpServer.clients:
                    addr, _ = tcpServer.clients[sock]
                    tcpServer.clients[sock] = (addr, False)
            response["response"] = {"process": "stop"}

    return response
