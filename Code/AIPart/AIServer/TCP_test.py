# ================================
# @File         : tcp_client_test.py
# @Time         : 2025/08/08
# @description  : 测试TCP服务器的客户端程序
# ================================

import socket
import json
import time
import threading
import sys


class TCPClient:
    def __init__(self, host='127.0.0.1', port=5566):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.receive_thread = None

    def connect(self):
        """连接到TCP服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"成功连接到服务器 {self.host}:{self.port}")

            # 启动接收消息线程
            self.receive_thread = threading.Thread(target=self.receive_messages, daemon=True)
            self.receive_thread.start()
            return True
        except Exception as e:
            print(f"连接服务器失败: {str(e)}")
            return False

    def send_data(self, data):
        """发送数据到服务器"""
        if not self.connected:
            print("未连接到服务器")
            return False

        try:
            # 按照服务器协议，先发送10字节的长度前缀，再发送数据
            json_data = json.dumps(data)
            data_len = len(json_data.encode('utf-8'))
            self.socket.sendall(f"{data_len:010d}".encode('utf-8') + json_data.encode('utf-8'))
            print(f"发送数据: {json_data}")
            return True
        except Exception as e:
            print(f"发送数据失败: {str(e)}")
            self.disconnect()
            return False

    def receive_messages(self):
        """接收服务器消息的线程函数"""
        while self.connected:
            try:
                # 先接收10字节的长度前缀
                len_prefix = self.socket.recv(10)
                if not len_prefix:
                    print("服务器已断开连接")
                    self.disconnect()
                    break

                data_len = int(len_prefix.decode('utf-8'))

                # 接收指定长度的数据
                data = b''
                while len(data) < data_len:
                    chunk = self.socket.recv(min(data_len - len(data), 4096))
                    if not chunk:
                        print("服务器已断开连接")
                        self.disconnect()
                        return
                    data += chunk

                message = data.decode('utf-8')
                try:
                    # 尝试解析为JSON
                    json_message = json.loads(message)
                    print(f"收到数据: {json.dumps(json_message, indent=2)}")
                except:
                    print(f"收到数据: {message}")

            except Exception as e:
                print(f"接收数据错误: {str(e)}")
                self.disconnect()
                break

    def disconnect(self):
        """断开与服务器的连接"""
        if self.connected:
            self.connected = False
            try:
                self.socket.close()
                print("已断开与服务器的连接")
            except Exception as e:
                print(f"关闭连接错误: {str(e)}")


if __name__ == "__main__":
    # 创建客户端实例
    client = TCPClient(host='127.0.0.1', port=5566)

    # 连接服务器
    if not client.connect():
        sys.exit(1)

    try:
        # 发送一条测试消息
        client.send_data({
            "type": "message",
            "content": "Hello from TCP client!"
        })

        # 等待服务器响应
        time.sleep(1)

        # 发送开始模拟数据命令
        client.send_data({
            "type": "control",
            "content": {
                "action": "start_simulation"
            }
        })

        # 保持连接接收模拟数据
        print("\n正在接收模拟数据（按Ctrl+C停止）...")
        while client.connected:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n用户中断")
        # 发送停止模拟数据命令
        client.send_data({
            "type": "control",
            "content": {
                "action": "stop_simulation"
            }
        })
        time.sleep(1)
    finally:
        client.disconnect()
