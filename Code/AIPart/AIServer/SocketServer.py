# ================================
# @File         : TCPServer.py
# @Time         : 2025/08/08
# @Author       : Yingrui Chen
# @description  : 基于基础socket搭建的TCP服务端类
#                 支持程序热启动、动态实时数据推送、静态数据收发处理
# ================================

import json
import os
import select
import sys
import threading
import time
from functools import wraps

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from common_util import *


class TCPServer:
    def __init__(self, import_name=None):
        """初始化TCP服务器"""
        self.host = "0.0.0.0"
        self.port = 5566
        self.debug = True  # 默认开启调试模式
        self.local_ip = get_local_ip()
        self.current_file = os.path.abspath(sys.modules[import_name].__file__) \
            if import_name \
            else os.path.abspath(__file__)
        self.observer = None
        self.logger = common_logger

        # 服务器socket
        self.server_socket = None

        # 客户端管理
        self.clients = {}  # {client_socket: (address, is_simulation_running)}
        self.client_lock = threading.Lock()

        # 装饰器
        self.message_handler = None
        self.simulation_handler = None

        # 控制标志
        self.running = False

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
        def wrapper(client_socket):
            """模拟数据推送循环"""
            while True:
                # 检查客户端是否仍在连接且需要模拟数据
                with self.client_lock:
                    if client_socket not in self.clients or not self.clients[client_socket][1]:
                        break

                try:
                    data = func()  # 调用被装饰的生成函数
                    self.send_data(client_socket, json.dumps(data))
                    time.sleep(1)  # 设置间隔1秒推送数据
                except Exception as e:
                    self.logger.error(f"模拟数据推送错误: {str(e)}")
                    break

            # 更新客户端状态
            with self.client_lock:
                if client_socket in self.clients:
                    addr, _ = self.clients[client_socket]
                    self.clients[client_socket] = (addr, False)

        self.simulation_handler = wrapper
        return func

    def send_data(self, client_socket, data):
        """向客户端发送数据"""
        try:
            # 发送数据长度前缀 + 数据，解决粘包问题
            data_len = len(data.encode('utf-8'))
            client_socket.sendall(f"{data_len:010d}".encode('utf-8') + data.encode('utf-8'))
            return True
        except Exception as e:
            self.logger.error(f"发送数据失败: {str(e)}")
            self.disconnect_client(client_socket)
            return False

    def receive_data(self, client_socket):
        """从客户端接收数据"""
        try:
            # 先接收10字节的长度前缀
            len_prefix = client_socket.recv(10)
            if not len_prefix:
                return None

            data_len = int(len_prefix.decode('utf-8'))

            # 接收指定长度的数据
            data = b''
            while len(data) < data_len:
                chunk = client_socket.recv(min(data_len - len(data), 4096))
                if not chunk:
                    return None
                data += chunk

            return data.decode('utf-8')
        except Exception as e:
            self.logger.error(f"接收数据失败: {str(e)}")
            self.disconnect_client(client_socket)
            return None

    def disconnect_client(self, client_socket):
        """断开客户端连接"""
        with self.client_lock:
            if client_socket in self.clients:
                addr, _ = self.clients[client_socket]
                self.logger.info(f"客户端 {addr[0]}:{addr[1]} 断开连接")
                del self.clients[client_socket]

        try:
            client_socket.close()
        except Exception as e:
            self.logger.error(f"关闭客户端连接错误: {str(e)}")

    def handle_client(self, client_socket, client_address):
        """处理客户端连接"""
        client_ip = f"{client_address[0]}:{client_address[1]}"
        self.logger.info(f"新客户端连接: {client_ip}")

        # 记录客户端信息
        with self.client_lock:
            self.clients[client_socket] = (client_address, False)

        try:
            # 首次建立连接
            self.send_data(client_socket, json.dumps("Server Connected!"))
            self.logger.debug(f"已向客户端 {client_ip} 发送连接确认")

            # 持续接收客户端消息
            while self.running:
                data = self.receive_data(client_socket)
                if not data:
                    break

                self.logger.info(f"收到来自 {client_ip} 的消息: {data}")
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "message":
                    if self.message_handler:
                        self.message_handler(message, client_socket)
                    else:
                        self.send_data(client_socket, json.dumps("Error! No message handler!"))

                elif message_type == "control":
                    action = message.get('content', {}).get('action')
                    if action == "start_simulation":
                        with self.client_lock:
                            if client_socket in self.clients and not self.clients[client_socket][1]:
                                # 更新状态并启动模拟线程
                                self.clients[client_socket] = (client_address, True)
                                threading.Thread(
                                    target=self.simulation_handler,
                                    args=(client_socket,),
                                    daemon=True
                                ).start()
                                self.logger.info(f"开始向 {client_ip} 发送模拟数据")
                                self.send_data(client_socket, json.dumps("simulation_started"))

                    elif action == "stop_simulation":
                        with self.client_lock:
                            if client_socket in self.clients and self.clients[client_socket][1]:
                                # 更新状态，模拟线程会自动退出
                                self.clients[client_socket] = (client_address, False)
                                self.logger.info(f"已响应 {client_ip} 的停止请求")
                                self.send_data(client_socket, json.dumps("simulation_stopped"))

        except Exception as e:
            self.logger.error(f"处理客户端 {client_ip} 时发生错误: {str(e)}", exc_info=True)
        finally:
            self.disconnect_client(client_socket)

    def start_server(self):
        """启动TCP服务器"""
        try:
            # 创建服务器socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.setblocking(False)  # 非阻塞模式
            self.running = True

            self.logger.info(f" * 服务器启动在 {self.host}:{self.port}...")
            if self.debug:
                self.logger.info(f" * 调试模式已激活!")
            else:
                self.logger.info(f" * 运行在生产环境!")
            self.logger.info(f" * 服务器地址: tcp://{self.local_ip}:{self.port}")
            self.logger.info(f" * 按Ctrl+C退出")

            # 监听客户端连接的主循环
            while self.running:
                try:
                    # 使用select处理非阻塞IO
                    read_sockets, _, exception_sockets = select.select(
                        [self.server_socket] + list(self.clients.keys()),
                        [],
                        [],
                        1.0  # 超时时间，允许定期检查running标志
                    )

                    for sock in read_sockets:
                        if sock == self.server_socket:
                            # 新客户端连接
                            client_socket, client_address = self.server_socket.accept()
                            client_socket.setblocking(True)
                            # 启动新线程处理客户端
                            client_thread = threading.Thread(
                                target=self.handle_client,
                                args=(client_socket, client_address),
                                daemon=True
                            )
                            client_thread.start()
                        else:
                            # 已有客户端数据，在各自的线程中处理
                            pass

                    # 处理异常的socket
                    for sock in exception_sockets:
                        self.disconnect_client(sock)

                except Exception as e:
                    self.logger.error(f"服务器主循环错误: {str(e)}", exc_info=True)
                    time.sleep(1)

        except Exception as e:
            self.logger.error(f"服务器启动失败: {str(e)}", exc_info=True)
        finally:
            self.stop_server()

    def stop_server(self):
        """停止服务器"""
        self.running = False
        self.logger.debug("正在关闭服务器...")

        # 关闭所有客户端连接
        with self.client_lock:
            for client_socket in list(self.clients.keys()):
                self.disconnect_client(client_socket)

        # 关闭服务器socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                self.logger.error(f"关闭服务器socket错误: {str(e)}")

        self.logger.debug("服务器已关闭")

    def restart_server(self):
        """重启服务器进程"""
        self.logger.debug("重启服务器...")
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
            # 启动TCP服务器
            self.start_server()
        except KeyboardInterrupt:
            self.logger.debug("接收到退出信号")
        except Exception as e:
            self.logger.error(f"服务器运行出错: {str(e)}", exc_info=True)
        finally:
            if self.observer and self.debug:
                self.observer.stop()
                self.observer.join()
            self.stop_server()

    class CodeChangeHandler(FileSystemEventHandler):
        """监控当前文件变化的处理器"""

        def __init__(self, restart_callback, target_file, logger):
            self.restart_callback = restart_callback
            self.target_file = target_file  # 只监控指定文件
            self.logger = logger  # 接收日志实例
            self.last_modified = 0

        def on_modified(self, event):
            # 防止重复触发
            now = time.time()
            if now - self.last_modified < 1:
                return
            self.last_modified = now

            # 只关注目标文件的变化
            if not event.is_directory and event.src_path == self.target_file:
                self.logger.info(f"\n文件 {self.target_file} 已更改！重启服务器...")
                self.restart_callback()


if __name__ == "__main__":
    server = TCPServer()

    # 示例消息处理
    @server.on_message
    def handle_message(message, client_socket):
        response = {
            "type": "response",
            "content": f"服务器已收到: {message.get('content')}"
        }
        server.send_data(client_socket, json.dumps(response))

    # 示例模拟数据生成
    @server.on_simulation
    def generate_simulation_data():
        import random
        return {
            "type": "simulation_data",
            "value": random.uniform(0, 100),
            "timestamp": time.time()
        }


    server.run(port=5566, debug=True)
