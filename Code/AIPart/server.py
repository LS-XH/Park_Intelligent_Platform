# ================================
# @File         : server.py
# @Time         : 2025/08/07
# @Author       : Yingrui Chen
# @description  : AI端服务器主程序
# ================================

import asyncio
import threading
import time
import argparse
import webbrowser

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
web_vue_path = "./Web"


with open("./AIServer/jsonData/cars_test.json", "r") as jsonFile:
    jsonData = json.load(jsonFile)


def crowd_density_pusher():
    global crowd_density_running, websocketApp
    while True:
        # 持续推送人口密度数据（根据crowd_density_running状态控制）
        if crowd_density_running and websocketApp.running:
            try:
                density_data = web_solve_status_13()
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(json.dumps(density_data)), loop
                    )
            except Exception as e:
                websocketApp.logger.error(f"人口密度数据推送错误: {str(e)}")

        # 控制推送频率（可根据需要调整）
        time.sleep(2)


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
            vehicle_data = unity_solve_status_0()

            if unityApp.running and unityApp.event_loop is not None:
                loop = unityApp.event_loop
                asyncio.run_coroutine_threadsafe(
                    unityApp.broadcast(vehicle_data), loop
                )
            time.sleep(1 / 30)

        # 持续推送节点统计数据到Web客户端
        if node_simulation_running and websocketApp.running:
            if current_node_name:
                node_data = web_solve_status_5(point_id=current_node_name)
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(node_data), loop
                    )
            time.sleep(1)  # 1秒推送间隔

        # 持续推送路段统计数据到Web客户端
        if road_simulation_running and websocketApp.running:
            if current_road_name:
                road_data = web_solve_status_6(road_id=current_road_name)
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(road_data), loop
                    )
            time.sleep(1)  # 1秒推送间隔

        if node_risk_data_running and websocketApp.running:
            if current_node_name:
                node_risk_data = web_solve_status_7(point_id=current_node_name)
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(node_risk_data), loop
                    )
            time.sleep(1)

        if road_risk_data_running and websocketApp.running:
            if current_road_name:
                road_risk_data = web_solve_status_8(road_id=current_road_name)
                if websocketApp.running and websocketApp.event_loop is not None:
                    loop = websocketApp.event_loop
                    asyncio.run_coroutine_threadsafe(
                        websocketApp.broadcast(road_risk_data), loop
                    )
            time.sleep(1)

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


def get_parser():
    parser = argparse.ArgumentParser(description="Welcome to AI Server!")
    parser.add_argument('--debug', action='store_true', help="开启调试模式")

    return parser


def start_vue_application(vue_path, debug=False):
    """在新线程中启动Vue应用"""
    try:
        # 切换到Vue项目目录并执行npm run dev命令
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=vue_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 输出Vue启动日志
        if debug:
            print(f"Starting Vue application from {vue_path}")

            # 实时打印stdout
            def print_output():
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(f"Vue: {output.strip()}")

            output_thread = threading.Thread(target=print_output, daemon=True)
            output_thread.start()

        time.sleep(2)
        webbrowser.open("http://localhost:5173")
        # 等待进程完成
        process.wait()

        # 如果进程异常退出，打印错误信息
        if process.returncode != 0:
            error = process.stderr.read()
            print(f"Vue application exited with error: {error}")

    except Exception as e:
        print(f"Failed to start Vue application: {str(e)}")


def main():
    server_parser = get_parser()
    args = server_parser.parse_args()

    # 启动信息转发线程
    forwarder_thread = threading.Thread(target=message_forwarder, daemon=True)
    forwarder_thread.start()

    # 人口密度推送进程
    crowd_density_thread = threading.Thread(target=crowd_density_pusher, daemon=True)
    crowd_density_thread.start()

    # 新增：Vue应用启动线程
    vue_thread = threading.Thread(
        target=start_vue_application,
        args=(web_vue_path, args.debug),
        daemon=True
    )
    vue_thread.start()

    # WebSocket服务启动线程
    def start_websocket():
        websocketApp.run(port=5566, debug=args.debug)

    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    websocket_thread.start()

    # TCP服务启动线程
    def start_tcp():
        unityApp.run(port=5567, debug=False)

    unity_thread = threading.Thread(target=start_tcp, daemon=True)
    unity_thread.start()

    # 主线程等待子线程, 防止程序退出
    websocket_thread.join()
    unity_thread.join()
    forwarder_thread.join()
    crowd_density_thread.join()
    vue_thread.join()  # 等待Vue线程结束


main()
