# ================================
# @File         : server.py
# @Time         : 2025/08/07
# @Author       : Yingrui Chen
# @description  : AI端服务器主程序
# ================================

import asyncio
import threading
import time

from task_processing import *
from AIServer.WebSocketServer import WebSocketServer
from AIServer.UnityServer import UnityServer

vehicle_simulation_running = False
node_simulation_running = False
road_simulation_running = False
node_risk_data_running = False
road_risk_data_running = False
crowd_density_running = False
current_node_name = None
current_road_name = None

websocketApp = WebSocketServer(__name__)
unityApp = UnityServer(__name__)


with open("./AIServer/jsonData/cars_test.json", "r") as jsonFile:
    jsonData = json.load(jsonFile)


# 消息转发线程
def message_forwarder():
    global vehicle_simulation_running, node_simulation_running, road_simulation_running, node_risk_data_running, road_risk_data_running, crowd_density_running
    global current_node_name, current_road_name

    while True:
        if not message_queue.empty():
            data = message_queue.get()
            # 状态码12启停指令
            if data.get("status") == 12:
                process = data.get("message").get("process")
                if process == "start":
                    vehicle_simulation_running = True
                    crowd_density_running = True
                    simulation_speed = data.get("message").get("time")
                    unityApp.simulation_speed = simulation_speed
                    unityApp.logger.info("开始向Unity客户端推送车辆数据")
                elif process == "stop":
                    vehicle_simulation_running = False
                    node_simulation_running = False
                    road_simulation_running = False
                    node_risk_data_running = False
                    road_risk_data_running = False
                    crowd_density_running = False
                    unityApp.logger.info("停止向Unity客户端推送车辆数据")

            # 状态码15节点数据推送指令
            elif data.get("status") == 15:
                node_name = data.get("message").get("id")
                if node_name:
                    node_simulation_running = True
                    node_risk_data_running = True
                    road_simulation_running = False
                    road_risk_data_running = False
                    current_node_name = node_name
                    websocketApp.logger.info(f"开始向Web客户端推送节点[{node_name}]统计数据")
            elif data.get("status") == 16:
                road_name = data.get("message").get("id")
                if road_name:
                    road_simulation_running = True
                    road_risk_data_running = True
                    node_simulation_running = False
                    node_risk_data_running = False
                    current_road_name = road_name
                    websocketApp.logger.info(f"开始向Web客户端推送节点[{road_name}]统计数据")

        # 持续推送车辆数据（当模拟运行时）
        if vehicle_simulation_running and unityApp.running:
            vehicle_data = unity_solve_status_1()

            if unityApp.running and unityApp.event_loop is not None:
                loop = unityApp.event_loop
                asyncio.run_coroutine_threadsafe(
                    unityApp.broadcast(vehicle_data), loop
                )
            time.sleep(1 / unityApp.simulation_speed)

        # 持续推送节点统计数据到Web客户端
        if node_simulation_running and websocketApp.running:
            if current_node_name:
                node_data = web_solve_status_5(point_id=current_node_name)
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(json.dumps(node_data)), loop
                    )
            time.sleep(1)  # 1秒推送间隔

        # 持续推送路段统计数据到Web客户端
        if road_simulation_running and websocketApp.running:
            if current_road_name:
                road_data = web_solve_status_6(road_name=current_road_name)
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(json.dumps(road_data)), loop
                    )
            time.sleep(1)  # 1秒推送间隔

        if node_risk_data_running and websocketApp.running:
            if current_node_name:
                node_risk_data = web_solve_status_7(point_id=current_node_name)
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(json.dumps(node_risk_data)), loop
                    )
            time.sleep(1)

        if road_risk_data_running and websocketApp.running:
            if current_road_name:
                road_risk_data = web_solve_status_8(road_name=current_road_name)
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(json.dumps(road_risk_data)), loop
                    )
            time.sleep(1)

        if crowd_density_running and websocketApp.running:
            density_data = web_solve_status_13()
            if websocketApp.running and websocketApp.event_loop is not None:
                loop = websocketApp.event_loop
                asyncio.run_coroutine_threadsafe(
                    websocketApp.broadcast(json.dumps(density_data)), loop
                )

        time.sleep(0.01)


"""
# ====================================================================
#
#                           服务器装饰器代码
#
# ====================================================================
"""


@websocketApp.on_data
async def data_processor(message=None, websocket=None, sim_type=None):
    """
    统一数据处理函数
    - 当message不为None时：处理客户端请求（静态数据/实时启停）
    - 当sim_type不为None时：生成实时推送数据
    """
    data = web_process_data(message, sim_type)

    # 收到控制指令时将信息加入消息转发队列
    if message and message.get("status") == 12:
        message_queue.put(message)

    return data


@unityApp.on_message
def handle_message(data):
    try:
        response = unity_process_message(data=data, tcpServer=unityApp)
        return response
    except Exception as e:
        unityApp.logger.error(f"消息处理错误: {str(e)}")
        return {"status": -1, "response": {"content": "error"}}


# 注册模拟数据生成函数
@unityApp.on_simulation
def generate_simulation_data():
    data = {
        "status": 8,
        "response": {"process": "start"}
    }
    return data


if __name__ == '__main__':
    def start_websocket():
        websocketApp.run(port=5566, debug=True)

    def start_tcp():
        unityApp.run(port=5567, debug=False)

    # 启动信息转发线程
    forwarder_thread = threading.Thread(target=message_forwarder, daemon=True)
    forwarder_thread.start()

    # 分别在两个线程中启动服务
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    unity_thread = threading.Thread(target=start_tcp, daemon=True)

    # 启动线程
    websocket_thread.start()
    unity_thread.start()

    # 主线程等待子线程, 防止程序退出
    websocket_thread.join()
    unity_thread.join()
    forwarder_thread.join()
