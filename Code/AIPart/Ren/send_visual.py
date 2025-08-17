# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import torch
# from io import BytesIO, StringIO
# from PIL import Image
# import base64
# from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
# import threading
# import time
#
# # 导入项目相关模块
# from Code.AIPart.Ren.agents import AgentGroup
# from Code.AIPart.Ren.obstacles import ObstacleGenerator
# from testmap import obstacles_params, targets, targets_heat, points, MAP_SIZE
# from Code.AIPart.Ren.config import *
#
# # 配置非交互式后端，适合服务器环境
# matplotlib.use('Agg')
# # 设置中文显示
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # 初始化Flask和WebSocket
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'agent_visualization_secret_key'
# socketio = SocketIO(app, cors_allowed_origins="*")  # 允许跨域访问
#
# emergency = [
#     ((MAP_SIZE * 0.3, MAP_SIZE * 0.4), 1),
#     ((MAP_SIZE * 0.5, MAP_SIZE * 0.7), 5),
#     ((MAP_SIZE * 0.2, MAP_SIZE * 0.9), 2),
#     ((MAP_SIZE * 0.8, MAP_SIZE * 0.6), 3),
#     ((MAP_SIZE * 0.6, MAP_SIZE * 0.3), 4),
#     ((MAP_SIZE * 0.4, MAP_SIZE * 0.8), 1),
#     ((MAP_SIZE * 0.9, MAP_SIZE * 0.3), 0),
# ]
#
#
# # 创建智能体群体
# def create_agents(num_agents,
#                   obstacle_gen,
#                   targets,
#                   targets_heat):
#     return AgentGroup(num_agents,
#                       obstacle_gen,
#                       targets,
#                       targets_heat)
#
#
# # 生成地图障碍物
# obstacle_gen = ObstacleGenerator(MAP_SIZE, points)
# obstacle_gen.add_obstacles(obstacles_params)
#
# # 初始化智能体群体
# agents = create_agents(num_agents,
#                        obstacle_gen,
#                        targets,
#                        targets_heat)
#
# # 绘制画布
# fig, ax = plt.subplots(figsize=(12, 10))
# ax.set_xlim(0.0, MAP_SIZE)
# ax.set_ylim(0.0, MAP_SIZE)
# ax.set_xlabel('X坐标')
# ax.set_ylabel('Y坐标')
#
# # 1. 绘制密度热图（作为底层）
# density_data = agents.get_density().cpu().numpy()
# density_plot = ax.imshow(
#     density_data.T,  # 转置以正确显示坐标
#     extent=[0, MAP_SIZE, 0, MAP_SIZE],  # 保持原始地图坐标范围
#     cmap=DENSITY_CMAP,
#     alpha=DENSITY_ALPHA,
#     origin='lower',
#     vmin=0,
#     vmax=min(10, num_agents // 100)  # 限制最大显示值，使颜色更明显
# )
# # 添加密度颜色条
# cbar = fig.colorbar(density_plot, ax=ax)
# cbar.set_label('智能体密度')
#
# # 绘制障碍物
# obstacle_x, obstacle_y = np.where(obstacle_gen.map_matrix == 1.0)
# ax.scatter(obstacle_x, obstacle_y, c='black', s=10, label='道路/障碍物')
#
# # 标记交点
# for idx, (x, y) in enumerate(obstacle_gen.intersection_points):
#     ax.scatter(x, y, c='yellow', s=80, marker='+', label=f'交点 {idx}' if idx == 0 else "")
#     ax.text(x + 5.0, y + 5.0, f'ID:{idx}', fontsize=9, color='yellow')
#
# # 绘制所有缺口中心和斥力线段
# for i, obs in enumerate(obstacle_gen.obstacles):
#     for idx, gap_center in obs.gap_center_map.items():
#         ax.scatter(gap_center[0], gap_center[1], c='blue', s=5, marker='o',
#                    alpha=0.5, label='缺口中心' if (i == 0 and idx == 0) else "")
#         # 计算并绘制斥力线段
#         seg_length = obs.gap_offset_ratio + (obstacle_gen.intersection_gap_size / 2.0) + 2.0
#         dir_vec = np.array([obs.dir_x, obs.dir_y], dtype=np.float32)
#         seg_start = gap_center - 0.5 * seg_length * dir_vec
#         seg_end = gap_center + 0.5 * seg_length * dir_vec
#         ax.plot([seg_start[0], seg_end[0]], [seg_start[1], seg_end[1]], 'g--', linewidth=1,
#                 alpha=0.5, label='斥力线段' if (i == 0 and idx == 0) else "")
#
# # 绘制目标点
# targets_np = np.array(targets, dtype=np.float32)
# targets_plot = ax.scatter(
#     targets_np[:, 0], targets_np[:, 1],
#     c='red', s=150, marker='*',
#     edgecolors='black', linewidths=2,
#     label='目标点'
# )
#
# # 初始化智能体绘制
# agent_positions = agents.positions.cpu().numpy()
# agent_sizes = agents.sizes.cpu().numpy()
# agent_colors = agents.colors.cpu().numpy()
#
# agents_plot = ax.scatter(
#     agent_positions[:, 0], agent_positions[:, 1],
#     s=agent_sizes, c=agent_colors,
#     alpha=0.7, edgecolors='white', linewidths=0.5,
#     label='智能体'
# )
#
# # 解析紧急情况列表
# eme_pos = []
# eme_lev = []
# for pos, lev in emergency:
#     eme_pos.append(pos)
#     eme_lev.append(lev)
# eme_pos_np = np.array(eme_pos, dtype=np.float32)
#
# # 紧急情况标记
# eme_sizes = [50.0 + lev * 30.0 for lev in eme_lev]
# emergency_plot = ax.scatter(
#     eme_pos_np[:, 0], eme_pos_np[:, 1],
#     s=eme_sizes,
#     marker='X',
#     c='red',
#     edgecolors='black',
#     linewidths=1.5,
#     label='紧急情况',
#     visible=False
# )
#
# ax.legend()
#
# # 全局变量，用于控制动画帧
# current_frame = 0
# is_running = True
#
#
# # 动画更新函数
# def update(frame):
#     global current_frame
#     current_frame = frame
#
#     # 模拟交通灯状态
#     if frame < 200:
#         traffic_light = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                          1]  # ID=0红灯
#     elif frame < 400:
#         traffic_light = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                          0]  # ID=1红灯
#     elif frame < 600:
#         traffic_light = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                          1]  # ID=2红灯
#     else:
#         traffic_light = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                          0]  # ID=4红灯
#
#     # 每隔150帧添加和删除智能体
#     if frame % 150 == 0 and frame > 0:
#         # 大门附近添加智能体
#         if len(targets) > 1:
#             target_x, target_y = targets[0]
#             target_x1, target_y1 = targets[1]
#             agents.birth(target_x, target_y, 80.0, 25)
#             agents.birth(target_x1, target_y1, 80.0, 25)
#             agents.kill(target_x, target_y, 80.0, 10)
#             agents.kill(target_x1, target_y1, 80.0, 10)
#
#         # 在紧急情况点附近删除智能体
#         if eme_pos:
#             for e_pos, e_lev in zip(eme_pos, eme_lev):
#                 eme_x, eme_y = e_pos
#                 agents.kill(eme_x, eme_y, 5 ** (e_lev - 1), 3 ** (e_lev - 1))
#
#     # 更新智能体位置
#     if 300 <= frame <= 500:
#         agents.move_towards_target((eme_pos, eme_lev), traffic_light)
#     else:
#         agents.move_towards_target(traffic_light=traffic_light)
#     agents.move_towards_target(traffic_light=traffic_light)
#
#     # 更新智能体显示
#     positions = agents.positions.cpu().numpy()
#     agents_plot.set_offsets(positions)
#
#     # 更新智能体大小显示（处理数量变化）
#     agent_sizes = agents.sizes.cpu().numpy()
#     agents_plot.set_sizes(agent_sizes)
#
#     # 更新智能体颜色（处理数量变化）
#     agent_colors = agents.colors.cpu().numpy()
#     agents_plot.set_facecolors(agent_colors)
#
#     if frame % 50 == 0:
#         # 更新密度热图
#         density_data = agents.get_density().cpu().numpy()
#         # print(np.max(density_data), np.min(density_data))
#         density_plot.set_data(density_data.T)  # 转置以正确显示
#
#         # 动态调整颜色范围以适应密度变化
#         current_max = min(density_data.max(), agents.num_agents // 10)
#         if current_max > 0:
#             density_plot.set_clim(vmin=0, vmax=current_max)
#
#     # 控制紧急情况显示
#     emergency_plot.set_visible(300 <= frame <= 500)
#
#     return agents_plot, emergency_plot
#
#
# # 生成图像并转换为Base64
# def generate_frame(frame):
#     # 更新当前帧
#     update(frame)
#
#     # 保存图像到内存缓冲区
#     buf = BytesIO()
#     fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
#     buf.seek(0)
#
#     # 转换为Base64编码
#     img = Image.open(buf)
#     img_byte_arr = BytesIO()
#     img.save(img_byte_arr, format='PNG')
#     img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
#
#     return img_base64
#
#
# # 定时发送帧数据的线程函数
# def send_frames():
#     global current_frame, is_running
#     frame = 0
#     while is_running:
#         try:
#             # 生成当前帧的图像
#             img_base64 = generate_frame(frame)
#
#             # 发送图像数据到前端
#             socketio.emit('frame_update', {'image': img_base64, 'frame': frame})
#
#             # 控制帧率，每50ms发送一次（约20fps）
#             time.sleep(0.05)
#
#             # 循环播放动画（800帧后重置）
#             frame = (frame + 1) % 800
#         except Exception as e:
#             print(f"发送帧时出错: {e}")
#             # 出错时稍等再重试
#             time.sleep(1)
#
#
# # WebSocket事件处理
# @socketio.on('connect')
# def handle_connect():
#     print('客户端已连接')
#     emit('connection_response', {'message': '已成功连接到服务器'})
#
#
# @socketio.on('disconnect')
# def handle_disconnect():
#     print('客户端已断开连接')
#
#
# # Flask路由
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# # 主函数
# if __name__ == '__main__':
#     try:
#         # 启动发送帧的线程
#         frame_thread = threading.Thread(target=send_frames, daemon=True)
#         frame_thread.start()
#
#         # 启动Flask-SocketIO服务器，添加允许不安全Werkzeug服务器的参数
#         print("启动服务器，访问 http://localhost:5000 查看可视化结果")
#         socketio.run(
#             app,
#             host='0.0.0.0',
#             port=5000,
#             debug=False,
#             allow_unsafe_werkzeug=True  # 添加此参数解决错误
#         )
#     except KeyboardInterrupt:
#         print("服务器正在关闭...")
#         is_running = False
#         frame_thread.join()
#     finally:
#         plt.close('all')
#
