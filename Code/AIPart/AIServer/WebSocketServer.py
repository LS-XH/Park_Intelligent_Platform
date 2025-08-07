# ================================
# @File         : WebSocketServer.py
# @Time         : 2025/08/07
# @Author       : Yingrui Chen
# @description  : 基于基础WebSocket搭建的服务端类
#                 支持程序热启动、动态实时数据推送、静态数据收发处理
# ================================

import asyncio
import json
import logging
import os
import socket
import sys
from functools import wraps

import websockets
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class WebSocketServer:
    def __init__(self, import_name=None):
        """初始化WebSocket服务器"""
        self.host = "0.0.0.0"
        self.port = 5566
        self.debug = True  # 默认开启调试模式
        self.local_ip = self.get_local_ip()
        self.current_file = os.path.abspath(sys.modules[import_name].__file__) \
            if import_name \
            else os.path.abspath(__file__)
        self.observer = None
        self.logger = self.setup_logging()

        # 装饰器
        self.message_handler = None
        self.simulation_handler = None

    @staticmethod
    def setup_logging():
        """配置日志系统"""
        log_format = ' [%(levelname)s] %(asctime)s %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format
        )

        return logging.getLogger(__name__)

    def on_message(self, func):
        """注册消息处理的函数装饰器"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        self.message_handler = wrapper

        return wrapper

    def on_simulation(self, func):
        """注册模拟数据生成函数的装饰器"""
        @wraps(func)
        async def wrapper(websocket, stop_event):
            while not stop_event.is_set():
                data = await func()  # 调用被装饰的生成函数
                await websocket.send(json.dumps(data))
                await asyncio.sleep(1)  # 设置间隔1秒推送数据

        self.simulation_handler = wrapper
        return func

    def get_local_ip(self):
        """
        获取本机局域网IP地址
        :return:    本机局域网IP
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            self.logger.error(f"获取本地IP失败: {str(e)}")
            return "127.0.0.1"

    async def handle_client(self, websocket):
        """处理客户端连接"""
        # 获取连接的客户端信息
        client_address = websocket.remote_address
        client_ip = f"{client_address[0]}:{client_address[1]}"
        self.logger.info(f"A New Client Connection: {client_ip}")

        # 用于控制模拟任务的事件和任务对象
        simulation_stop_event = asyncio.Event()
        simulation_task = None

        try:
            # 首次建立连接
            await websocket.send(json.dumps("AI Server Connected!"))
            self.logger.debug(f"已向客户端 {client_ip} 发送连接确认")

            # 持续接收客户端消息
            async for message in websocket:
                self.logger.info(f"收到来自 {client_ip} 的消息: {message}")
                message = json.loads(message)
                message_type = message["type"]

                if message_type == "message":
                    if self.message_handler:
                        await self.message_handler(message, websocket)
                    else:
                        await websocket.send(json.dumps("AI Server Error!No message handler!"))

                elif message_type == "control":
                    action = message['content']['action']
                    if action == "start_simulation":
                        if simulation_task and not simulation_task.done():
                            simulation_stop_event.set()
                            await simulation_task

                        simulation_stop_event.clear()
                        # 使用装饰器注册的处理函数
                        simulation_task = asyncio.create_task(
                            self.simulation_handler(websocket, simulation_stop_event)
                        )
                        self.logger.info(f"开始向 {client_ip} 发送模拟数据")

                    elif action == "stop_simulation":
                        # 触发停止事件
                        if simulation_task and not simulation_task.done():
                            simulation_stop_event.set()
                            await simulation_task  # 等待任务真正停止
                            simulation_task = None
                            await websocket.send(json.dumps("simulation_stopped"))
                            self.logger.info(f"已响应 {client_ip} 的停止请求")

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info(f"客户端 {client_ip} 关闭连接: {str(e)}")
        except Exception as e:
            self.logger.error(f"处理客户端 {client_ip} 时发生错误: {str(e)}", exc_info=True)
        finally:
            self.logger.info(f"客户端 {client_ip} 断开连接")

    async def start_server(self):
        """启动WebSocket服务器"""
        self.logger.info(f" * Starting AI Server on {self.host}:{self.port}...")
        if self.debug:
            self.logger.info(f" * Debugger is Activated!")
        else:
            self.logger.info(f" * Running in the production environment!")
        self.logger.info(f" * The AI server is running on the ws://{self.local_ip}:{self.port}")
        self.logger.info(f" * Press Ctrl+C to exit")

        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # 保持服务器运行

    def restart_server(self):
        """重启服务器进程"""
        self.logger.debug("Restarting AI Server...")
        python = sys.executable
        script = os.path.abspath(sys.argv[0])
        os.execl(python, python, script)

    def run(self, host=None, port=None, debug=None):
        """
        运行服务器主循环

        参数:
            host (str): 服务器绑定的主机地址，默认为"0.0.0.0"
            port (int): 服务器监听的端口，默认为5566
            debug (bool): 是否开启调试模式（自动重载），默认为True
        """
        # 更新配置（如果提供了参数）
        if host:
            self.host = host
        if port:
            self.port = port
        if debug is not None:
            self.debug = debug

        # 如果开启调试模式，启动文件监控
        if self.debug:
            event_handler = self.CodeChangeHandler(self.restart_server, self.current_file, self.logger)
            self.observer = Observer()
            self.observer.schedule(event_handler, os.path.dirname(self.current_file), recursive=False)
            self.observer.start()
            self.logger.debug("文件监控已启动，仅监控当前文件变化")

        try:
            # 启动WebSocket服务器
            asyncio.run(self.start_server())
        except KeyboardInterrupt:
            self.logger.debug("Closing AI Server...")
        except Exception as e:
            self.logger.error(f"服务器运行出错: {str(e)}", exc_info=True)
        finally:
            if self.observer and self.debug:
                self.observer.stop()
                self.observer.join()
            self.logger.debug("AI Server Closed.")

    class CodeChangeHandler(FileSystemEventHandler):
        """监控当前文件变化的处理器"""

        def __init__(self, restart_callback, target_file, logger):
            self.restart_callback = restart_callback
            self.target_file = target_file  # 只监控指定文件
            self.logger = logger  # 接收日志实例

        def on_modified(self, event):
            # 只关注目标文件的变化
            if not event.is_directory and event.src_path == self.target_file:
                self.logger.info(f"\nFile {self.target_file} Changed！Restarting AI Server...")
                self.restart_callback()


if __name__ == "__main__":
    server = WebSocketServer()
    server.run(port=5566, debug=True)
