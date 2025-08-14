# ================================
# @File         : server.py
# @Time         : 2025/08/07
# @Author       : Yingrui Chen
# @description  : AI端服务器主程序
# ================================

import threading
import json
import time
import queue

from task_processing import web_process_data, tcp_process_message, tcp_solve_status_1
from AIServer.WebSocketServer import WebSocketServer
from AIServer.SocketServer import TCPServer

message_queue = queue.Queue()
vehicle_simulation_running = False

websocketApp = WebSocketServer(__name__)
tcpServer = TCPServer(__name__)


with open("./AIServer/jsonData/cars_test.json", "r") as jsonFile:
    jsonData = json.load(jsonFile)


# 消息转发线程
def message_forwarder():
    global vehicle_simulation_running
    while True:
        if not message_queue.empty():
            data = message_queue.get()
            # 处理前端发来的状态码12控制指令
            if data.get("status") == 12:
                process = data.get("message").get("process")
                if process == "start":
                    vehicle_simulation_running = True
                    simulation_time = data.get("message").get("time")
                    tcpServer.simulation_time = simulation_time
                    tcpServer.logger.info("开始向TCP客户端推送车辆数据")
                elif process == "stop":
                    vehicle_simulation_running = False
                    tcpServer.logger.info("停止向TCP客户端推送车辆数据")

        # 持续推送车辆数据（当模拟运行时）
        if vehicle_simulation_running:
            vehicle_data = tcp_solve_status_1()
            tcpServer.broadcast(json.dumps(vehicle_data))
            time.sleep(1 / tcpServer.simulation_speed)  # 考虑模拟速度

        time.sleep(1 / tcpServer.simulation_speed)


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


@tcpServer.on_message
def handle_message(data):
    try:
        response = tcp_process_message(data=data, tcpServer=tcpServer)
        return response
    except Exception as e:
        tcpServer.logger.error(f"消息处理错误: {str(e)}")
        return {"status": -1, "response": {"content": "error"}}


# 注册模拟数据生成函数
@tcpServer.on_simulation
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
        tcpServer.run(port=5567, debug=False)

    # 启动信息转发线程
    forwarder_thread = threading.Thread(target=message_forwarder, daemon=True)
    forwarder_thread.start()

    # 分别在两个线程中启动服务
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    tcp_thread = threading.Thread(target=start_tcp, daemon=True)

    # 启动线程
    websocket_thread.start()
    tcp_thread.start()

    # 主线程等待子线程, 防止程序退出
    websocket_thread.join()
    tcp_thread.join()
    forwarder_thread.join()
