# ================================
# @File         : server.py
# @Time         : 2025/08/07
# @Author       : Yingrui Chen
# @description  : AI端服务器主程序
# ================================

import random
from task_processing import process
from AIServer.WebSocketServer import WebSocketServer

app = WebSocketServer(__name__)


@app.on_simulation
async def generate_simulation_data():
    """模拟数据实时推送"""
    return random.randint(1, 10086)


@app.on_message
async def process_message(message, websocket):
    """静态数据处理、收发"""
    response = await process(message)
    await websocket.send(response)

if __name__ == '__main__':
    port = 5566
    debug = True

    app.run(port=port, debug=debug)
