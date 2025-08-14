import json
import socket
import time


class TCPClientTester:
    def __init__(self, host='localhost', port=5567):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        """建立与服务器的连接"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"成功连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接服务器失败: {str(e)}")
            return False

    def send_request(self, status, message=None):
        """发送请求到服务器并返回响应"""
        if not self.connected:
            print("未连接到服务器，请先调用connect()")
            return None

        try:
            # 构建请求数据
            request = {"status": status}
            if message is not None:
                request["message"] = message

            # 序列化为JSON并发送
            request_json = json.dumps(request)
            data_bytes = request_json.encode('utf-8')
            data_len = len(data_bytes)
            len_prefix = f"{data_len:010d}".encode('utf-8')
            self.socket.sendall(data_bytes)
            print(f"\n发送请求: status={status}")
            print(f"请求内容: {request_json}")

            # 接收响应
            len_prefix = self.socket.recv(10)
            if not len_prefix:
                print("未收到响应长度前缀")
                return None

            data_len = int(len_prefix.decode('utf-8'))
            data = b''
            while len(data) < data_len:
                chunk = self.socket.recv(min(data_len - len(data), 4096))
                if not chunk:
                    print("接收响应时连接中断")
                    return None
                data += chunk

            response = json.loads(data.decode('utf-8'))
            print("收到响应:")
            print(json.dumps(response, indent=2))
            return response

        except Exception as e:
            print(f"发送/接收数据出错: {str(e)}")
            return None

    def test_status_0(self):
        """测试状态码0: 通信测试"""
        print("\n=== 测试状态码0: 通信测试 ===")
        return self.send_request(0)

    def test_status_1(self):
        """测试状态码1: 获取初始化信息"""
        print("\n=== 测试状态码1: 获取初始化信息 ===")
        return self.send_request(1)

    def test_status_2(self):
        """测试状态码2: 用户点击节点"""
        print("\n=== 测试状态码2: 用户点击节点 ===")
        message = {"id": 6}  # 示例节点ID
        return self.send_request(2, message)

    def test_status_3(self):
        """测试状态码3: 用户点击路段"""
        print("\n=== 测试状态码3: 用户点击路段 ===")
        message = {"id": 6}  # 示例路段ID
        return self.send_request(3, message)

    def test_status_4(self):
        """测试状态码4: 智能模块开关"""
        print("\n=== 测试状态码4: 智能模块开关 ===")
        message = {
            "bestRoute": True,
            "crowdEvacuation": False,
            "trafficLight": True,
            "CAV": False
        }
        return self.send_request(4, message)

    def test_status_5(self):
        """测试状态码5: 寻路界面"""
        print("\n=== 测试状态码5: 寻路界面 ===")
        message = {
            "startName": "陆小凤",
            "endName": "雀跃坪"
        }
        return self.send_request(5, message)

    def test_status_6(self):
        """测试状态码6: 突发事件选择"""
        print("\n=== 测试状态码6: 突发事件选择 ===")
        message = {
            "id": 8,
            "emergency": "车祸"
        }
        return self.send_request(6, message)

    def test_status_7(self):
        """测试状态码7: 突发事件停止"""
        print("\n=== 测试状态码7: 突发事件停止 ===")
        return self.send_request(7)

    def test_status_8(self):
        """测试状态码8: 模拟进程控制区"""
        print("\n=== 测试状态码8: 模拟进程控制区 ===")

        # 测试启动模拟
        print("\n--- 测试启动模拟 ---")
        start_message = {
            "process": "start",
            "time": 30
        }
        start_response = self.send_request(8, start_message)

        # 运行5秒模拟并接收推送数据
        if start_response:
            print("\n等待5秒并接收模拟数据推送...")
            end_time = time.time() + 5  # 持续5秒
            self.socket.settimeout(0.1)  # 设置超时，避免阻塞

            while time.time() < end_time:
                try:
                    # 接收服务器推送的实时数据
                    len_prefix = self.socket.recv(10)
                    if not len_prefix:
                        print("未收到推送数据长度前缀，连接可能已断开")
                        break

                    data_len = int(len_prefix.decode('utf-8'))
                    data = b''
                    while len(data) < data_len:
                        chunk = self.socket.recv(min(data_len - len(data), 4096))
                        if not chunk:
                            print("接收推送数据时连接中断")
                            break
                        data += chunk

                    if data:
                        push_data = json.loads(data.decode('utf-8'))
                        print("收到实时推送数据:")
                        print(json.dumps(push_data, indent=2))

                except socket.timeout:
                    continue  # 超时不处理，继续等待下一次推送
                except Exception as e:
                    print(f"接收推送数据出错: {str(e)}")
                    break

            # 测试停止模拟
            print("\n--- 测试停止模拟 ---")
            self.socket.settimeout(None)  # 恢复默认超时
            stop_message = {
                "process": "stop"
            }
            stop_response = self.send_request(8, stop_message)
            return stop_response

        return None

    def run_all_tests(self):
        """运行所有状态码测试"""
        if not self.connect():
            return

        try:
            # 按顺序执行所有测试
            self.test_status_0()
            time.sleep(1)

            self.test_status_1()
            time.sleep(1)

            self.test_status_2()
            time.sleep(1)

            self.test_status_3()
            time.sleep(1)

            print("\n所有测试完成")

        finally:
            self.disconnect()

    def disconnect(self):
        """断开与服务器的连接"""
        if self.connected and self.socket:
            self.socket.close()
            self.connected = False
            print("\n已断开与服务器的连接")


if __name__ == "__main__":
    # 创建测试客户端并运行所有测试
    tester = TCPClientTester(host='127.0.0.1', port=5567)
    tester.run_all_tests()
