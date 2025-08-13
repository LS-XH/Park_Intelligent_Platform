# ================================
# @File         : WebSocketServer.py
# @Time         : 2025/08/07
# @Author       : Yingrui Chen
# @description  : 基于基础WebSocket搭建的服务端类
#                 支持程序热启动、动态实时数据推送、静态数据收发处理
# ================================

import asyncio
import json
import os
import sys
from functools import wraps

import websockets
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from .common_util import *


class WebSocketServer:
    def __init__(self, import_name=None):
        """初始化WebSocket服务器"""
        self.host = "0.0.0.0"
        self.port = 5566
        self.debug = True  # 默认开启调试模式
        self.local_ip = get_local_ip()
        self.current_file = os.path.abspath(sys.modules[import_name].__file__) \
            if import_name \
            else os.path.abspath(__file__)
        self.observer = None
        self.logger = common_logger
        self.stop_to_start_map = {4: 3, 6: 5, 8: 7, 10: 9, 12: 11}

        # 统一数据处理函数
        self.data_processor = None

    def on_data(self, func):
        """注册统一数据处理函数的装饰器（处理静态和实时数据）"""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        self.data_processor = wrapper
        return wrapper

    async def _realtime_pusher(self, websocket, stop_event, sim_type):
        """实时数据推送循环"""
        while not stop_event.is_set():
            # 调用数据处理器生成实时数据
            result = await self.data_processor(sim_type=sim_type)
            await websocket.send(json.dumps(result))
            await asyncio.sleep(1)  # 1秒间隔推送

    async def handle_client(self, websocket):
        """处理客户端连接的主逻辑"""
        client_address = websocket.remote_address
        client_ip = f"{client_address[0]}:{client_address[1]}"
        self.logger.info(f"【WEB】客户端 {client_ip}连接")

        # 管理实时任务：{sim_type: (stop_event, task)}
        realtime_tasks = {}

        try:
            async for message in websocket:
                self.logger.info(f"【WEB】收到来自 {client_ip} 的消息: {message}")
                message = json.loads(message)

                # 调用统一数据处理器处理消息
                if not self.data_processor:
                    response = json.dumps({
                        "status": -1,
                        "response": {"content": "No data processor registered"}
                    })
                    await websocket.send(response)
                    continue

                # 处理消息并获取结果
                result = await self.data_processor(message=message, websocket=websocket)

                # 根据处理结果类型进行分发
                if result["type"] == "static":
                    # 静态数据直接返回
                    await websocket.send(json.dumps(result["data"]))

                elif result["type"] == "realtime_start":
                    # 启动实时数据推送
                    sim_type = result["sim_type"]
                    # 停止同类型已有任务
                    if sim_type in realtime_tasks:
                        existing_stop, existing_task = realtime_tasks[sim_type]
                        existing_stop.set()
                        await existing_task

                    # 创建新任务
                    stop_event = asyncio.Event()
                    task = asyncio.create_task(
                        self._realtime_pusher(websocket, stop_event, sim_type)
                    )
                    realtime_tasks[sim_type] = (stop_event, task)
                    self.logger.info(f"【WEB】开始向 {client_ip} 推送类型 {sim_type} 的实时数据")

                elif result["type"] == "realtime_stop":
                    # 停止实时数据推送
                    sim_type = result["sim_type"]
                    if sim_type in realtime_tasks:
                        stop_event, task = realtime_tasks[sim_type]
                        stop_event.set()
                        await task
                        del realtime_tasks[sim_type]
                        self.logger.info(f"【WEB】已停止 {client_ip} 的类型 {sim_type} 实时推送")

                    await websocket.send(json.dumps({
                        "status": result["original_status"],
                        "response": {"content": "success"}
                    }))

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info(f"【WEB】客户端 {client_ip} 关闭连接: {str(e)}")
        except Exception as e:
            self.logger.error(f"【WEB】处理客户端 {client_ip} 时发生错误: {str(e)}", exc_info=True)
        finally:
            # 清理所有实时任务
            for sim_type, (stop_event, task) in realtime_tasks.items():
                stop_event.set()
                await task
            self.logger.info(f"【WEB】客户端 {client_ip} 断开连接")

    async def start_server(self):
        """启动WebSocket服务器"""
        if self.debug:
            self.logger.info(f"【WEB】 * WebSocket服务器在开发者模式下运行！")
        else:
            self.logger.info(f"【WEB】 * WebSocket服务器在生产环境下运行！")
        self.logger.info(f"【WEB】 * WebSocket服务器运行在 ws://{self.local_ip}:{self.port}")

        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # 保持服务器运行

    def restart_server(self):
        """重启服务器进程"""
        self.logger.debug("【WEB】服务重启中... ...")
        python = sys.executable
        script = os.path.abspath(sys.argv[0])
        os.execl(python, python, script)

    def run(self, host=None, port=None, debug=None):
        """运行服务器主循环"""
        if host:
            self.host = host
        if port:
            self.port = port
        if debug is not None:
            self.debug = debug

        # 屏蔽websockets库的INFO级日志
        websockets_logger = logging.getLogger("websockets")
        websockets_logger.setLevel(logging.WARNING)

        if self.debug and not self.observer:
            event_handler = self.CodeChangeHandler(self.restart_server, self.current_file, self.logger)
            self.observer = Observer()
            self.observer.schedule(event_handler, os.path.dirname(self.current_file), recursive=False)
            self.observer.start()
            self.logger.level = logging.DEBUG

        try:
            asyncio.run(self.start_server())
        except KeyboardInterrupt:
            self.logger.debug("【WEB】关闭服务器中... ...")
        except Exception as e:
            self.logger.error(f"【WEB】服务器运行出错: {str(e)}", exc_info=True)
        finally:
            if self.observer and self.debug:
                self.observer.stop()
                self.observer.join()
            self.logger.debug("【WEB】服务器成功关闭！")

    class CodeChangeHandler(FileSystemEventHandler):
        """监控当前文件变化的处理器"""

        def __init__(self, restart_callback, target_file, logger):
            self.restart_callback = restart_callback
            self.target_file = target_file
            self.logger = logger

        def on_modified(self, event):
            if not event.is_directory and event.src_path == self.target_file:
                self.logger.info(f"【WEB】\nFile {self.target_file} Changed！Restarting AI Server...")
                self.restart_callback()


if __name__ == "__main__":
    server = WebSocketServer()
    server.run(port=5569, debug=True)
