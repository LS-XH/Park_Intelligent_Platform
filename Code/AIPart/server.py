# ================================
# @File         : server.py
# @Time         : 2025/08/07
# @Author       : Yingrui Chen
# @description  : AI端服务器主程序
# ================================
import json
import random
from AIServer.WebSocketServer import WebSocketServer

app = WebSocketServer(__name__)


@app.on_simulation
async def generate_simulation_data(sim_type):
    """根据模拟类型生成对应格式的随机数据"""
    if sim_type == 3:
        # 全局车流、人流、拥挤程度（状态码3）
        return {
            "status": 3,
            "response": {
                "cars": round(random.uniform(500, 20000), 1),  # 车流量
                "people": round(random.uniform(1000, 100000), 1),  # 人流量
                "crowding": random.choice(["畅通", "轻度拥挤", "中度拥挤", "严重拥挤"])
            }
        }
    elif sim_type == 5:
        # 节点数据（状态码5）
        return {
            "status": 5,
            "response": {
                "cars": round(random.uniform(50, 5000), 1),
                "people": round(random.uniform(100, 20000), 1),
                "crowding": random.choice(["畅通", "轻度拥挤", "中度拥挤", "严重拥挤"]),
                "emergency": random.choice(["无", "交通事故", "设备故障", "人群聚集", "其他"]),
                "trafficLight": random.choice(["红灯", "绿灯", "黄灯"])
            }
        }
    elif sim_type == 7:
        # 路段数据（状态码7）
        return {
            "status": 7,
            "response": {
                "cars": random.randint(0, 3000),  # 车流（整数）
                "people": random.randint(0, 15000),  # 人流（整数）
                "crowding": round(random.uniform(0, 100), 1),  # 拥挤度（百分比）
                "emergency": random.choice(["无", "道路施工", "交通事故", "积水", "其他"])
            }
        }
    elif sim_type == 9:
        # 节点风险预测（状态码9）
        return {
            "status": 9,
            "response": {
                "riskData": [round(random.uniform(0, 100), 1) for _ in range(12)],  # 12个历史数据
                "predict": round(random.uniform(0, 100), 1)  # 预测数据
            }
        }
    elif sim_type == 11:
        # 路段风险预测（状态码11）
        return {
            "status": 11,
            "response": {
                "riskData": [round(random.uniform(0, 100), 1) for _ in range(12)],  # 12个历史数据
                "predict": round(random.uniform(0, 100), 1)  # 预测数据
            }
        }
    else:
        return {"status": -1, "response": {"content": "Unknown simulation type"}}


@app.on_message
async def process_message(message, websocket):
    """处理静态消息（调用task_processing中的处理逻辑）"""
    """根据消息中的status处理静态请求"""
    status = message.get("status")

    if status == 0:
        # 通信测试
        return json.dumps({
            "status": 0,
            "response": {"content": "success"}
        })

    elif status == 1:
        # 获取全局天气数据
        return json.dumps({
            "status": 1,
            "response": {
                "temperature": round(random.uniform(-10, 40), 1),  # 温度（-10~40℃）
                "wind": random.choice(["东风", "南风", "西风", "北风", "东北风", "东南风", "西北风", "西南风"]),
                "airQuality": round(random.uniform(0, 500), 1)  # 空气质量（0~500）
            }
        })

    elif status == 2:
        # 修改全局天气数据（仅返回成功，实际应用中可添加保存逻辑）
        return json.dumps({
            "status": 2,
            "response": {"content": "success"}
        })

    else:
        # 未知状态码
        return json.dumps({
            "status": -1,
            "response": {"content": f"Unknown status code: {status}"}
        })


if __name__ == '__main__':
    port = 5566
    debug = True
    app.run(port=port, debug=debug)
