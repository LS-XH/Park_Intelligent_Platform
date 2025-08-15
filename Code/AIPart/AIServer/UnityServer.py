# ================================
# @File         : WebSocketServer.py
# @Time         : 2025/08/08
# @Author       : Yingrui Chen
# @description  : 基于websocket的服务端类（兼容模式）
#                 支持程序热启动、动态实时数据推送、静态数据收发处理
# ================================

import json
import os
import sys
import threading
import time
from functools import wraps
import asyncio
from websockets import serve, exceptions

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .common_util import *


class UnityServer:
    def __init__(self, import_name=None):
        """初始化WebSocket服务器"""
        self.host = "0.0.0.0"
        self.port = 5567
        self.debug = True  # 默认开启调试模式
        self.local_ip = get_local_ip()
        self.current_file = os.path.abspath(sys.modules[import_name].__file__) \
            if import_name \
            else os.path.abspath(__file__)
        self.observer = None
        self.logger = common_logger

        # 跨域配置 - 允许的源列表
        self.allowed_origins = ["*"]  # 开发环境允许所有源，生产环境指定具体域名

        # 服务器实例
        self.server = None

        # 客户端管理（使用连接对象作为键，不显式指定类型）
        self.clients = {}  # {client: (address, is_simulation_running)}
        self.client_lock = threading.Lock()

        # 装饰器
        self.message_handler = None
        self.simulation_handler = None

        # 控制标志
        self.running = False
        self.event_loop = None

        # 模拟数据存储
        self.emergency_active = False
        self.simulation_speed = 30

    def _is_origin_allowed(self, origin):
        """检查来源是否允许跨域连接"""
        if not origin:
            return True  # 允许没有指定origin的请求

        # 处理通配符情况
        if "*" in self.allowed_origins:
            return True

        # 检查是否在允许的源列表中
        return origin in self.allowed_origins

    async def _handle_cors_handshake(self, client):
        """处理跨域握手，添加必要的响应头"""
        # 获取客户端请求的origin
        print("RUNNING POINT")
        origin = client.request_headers.get("Origin", "")

        # 验证跨域权限
        if not self._is_origin_allowed(origin):
            self.logger.warning(f"【Unity】拒绝跨域连接: {origin}")

        # 添加跨域响应头
        client.response_headers["Access-Control-Allow-Origin"] = origin if origin else "*"
        client.response_headers["Access-Control-Allow-Credentials"] = "true"
        client.response_headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        client.response_headers["Access-Control-Allow-Headers"] = "Content-Type"

    def on_message(self, func):
        """注册消息处理的函数装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        self.message_handler = wrapper
        return wrapper

    def on_simulation(self, func):
        """注册模拟数据生成函数的装饰器"""
        @wraps(func)
        async def wrapper(client):
            """模拟数据推送循环"""
            while True:
                # 检查客户端是否仍在连接且需要模拟数据
                with self.client_lock:
                    if client not in self.clients or not self.clients[client][1]:
                        break

                try:
                    data = func()  # 调用被装饰的生成函数
                    await self.send_data(client, json.dumps(data))
                    # 根据模拟速度调整间隔
                    await asyncio.sleep(1 / self.simulation_speed)
                except Exception as e:
                    self.logger.error(f"【Unity】模拟数据推送错误: {str(e)}")
                    break

            # 更新客户端状态
            with self.client_lock:
                if client in self.clients:
                    addr, _ = self.clients[client]
                    self.clients[client] = (addr, False)

        self.simulation_handler = wrapper
        return func

    async def send_data(self, client, data: str):
        """向客户端发送数据"""
        try:
            await client.send(data)
            return True
        except exceptions.ConnectionClosed:
            self.logger.error(f"【Unity】客户端已断开连接")
            await self.disconnect_client(client)
            return False
        except Exception as e:
            self.logger.error(f"【Unity】发送数据失败: {str(e)}")
            await self.disconnect_client(client)
            return False

    async def broadcast(self, data: str):
        """向所有连接的客户端广播数据"""
        with self.client_lock:
            for client in list(self.clients.keys()):
                await self.send_data(client, data)

    async def receive_data(self, client):
        """从客户端接收数据"""
        try:
            data = await client.recv()
            return data
        except exceptions.ConnectionClosed:
            self.logger.error(f"【Unity】客户端断开连接")
            await self.disconnect_client(client)
            return None
        except Exception as e:
            self.logger.error(f"【Unity】接收数据失败: {str(e)}")
            await self.disconnect_client(client)
            return None

    async def disconnect_client(self, client):
        """断开客户端连接"""
        with self.client_lock:
            if client in self.clients:
                addr, _ = self.clients[client]
                self.logger.info(f"【Unity】客户端 {addr} 断开连接")
                del self.clients[client]

        try:
            await client.close()
        except Exception as e:
            self.logger.error(f"【Unity】关闭客户端连接错误: {str(e)}")

    async def handle_client(self, client):
        """处理客户端连接"""
        # await self._handle_cors_handshake(client)
        # 获取客户端地址
        client_addr = f"{client.remote_address[0]}:{client.remote_address[1]}"
        self.logger.info(f"【Unity】新客户端连接: {client_addr}")

        # 记录客户端信息
        with self.client_lock:
            self.clients[client] = (client_addr, False)

        try:
            # 持续接收客户端消息
            while self.running:
                data = await self.receive_data(client)
                if not data:
                    break

                self.logger.info(f"【Unity】收到 {client_addr} 消息: {data}")

                # 处理消息
                if self.message_handler:
                    response = self.message_handler(data)
                    if response:
                        await self.send_data(client, json.dumps(response))

        except Exception as e:
            self.logger.error(f"【Unity】处理客户端 {client_addr} 时发生错误: {str(e)}", exc_info=True)
        finally:
            await self.disconnect_client(client)

    async def start_server(self):
        """启动WebSocket服务器"""
        try:
            self.event_loop = asyncio.get_event_loop()
            self.server = await serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=60
            )
            self.running = True

            if self.debug:
                self.logger.info(f"【Unity】 * 服务器在开发者模式下运行!")
            else:
                self.logger.info(f"【Unity】 * 服务器在生产环境下运行!")
            self.logger.info(f"【Unity】 * 服务器地址: ws://{self.local_ip}:{self.port}")

            # 保持服务器运行
            await self.server.wait_closed()

        except Exception as e:
            self.logger.error(f"服务器启动失败: {str(e)}", exc_info=True)
        finally:
            await self.stop_server()

    async def stop_server(self):
        """停止服务器"""
        self.running = False
        self.logger.debug("正在关闭服务器...")

        # 关闭所有客户端连接
        with self.client_lock:
            for client in list(self.clients.keys()):
                await self.disconnect_client(client)

        # 关闭服务器
        if self.server:
            try:
                self.server.close()
                await self.server.wait_closed()
            except Exception as e:
                self.logger.error(f"关闭服务器错误: {str(e)}")

        self.logger.debug("服务器已关闭")

    def restart_server(self):
        """重启服务器进程"""
        self.logger.debug("重启服务器...")
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

        if self.debug:
            event_handler = self.CodeChangeHandler(self.restart_server, self.current_file, self.logger)
            self.observer = Observer()
            self.observer.schedule(event_handler, os.path.dirname(self.current_file), recursive=False)
            self.observer.start()
            self.logger.debug("文件监控已启动，仅监控当前文件变化")

        # 屏蔽websockets库的INFO级日志
        websockets_logger = logging.getLogger("websockets")
        websockets_logger.setLevel(logging.WARNING)

        try:
            # 运行异步服务器
            asyncio.run(self.start_server())
        except KeyboardInterrupt:
            self.logger.debug("接收到退出信号")
        except Exception as e:
            self.logger.error(f"服务器运行出错: {str(e)}", exc_info=True)
        finally:
            if self.observer and self.debug:
                self.observer.stop()
                self.observer.join()
            # 停止服务器
            asyncio.run(self.stop_server())

    class CodeChangeHandler(FileSystemEventHandler):
        """监控当前文件变化的处理器"""
        def __init__(self, restart_callback, target_file, logger):
            self.restart_callback = restart_callback
            self.target_file = target_file
            self.logger = logger
            self.last_modified = 0

        def on_modified(self, event):
            now = time.time()
            if now - self.last_modified < 1:
                return
            self.last_modified = now

            if not event.is_directory and event.src_path == self.target_file:
                self.logger.info(f"\n文件 {self.target_file} 已更改！重启服务器...")
                self.restart_callback()


if __name__ == "__main__":
    server = UnityServer(__name__)
    server.run()
