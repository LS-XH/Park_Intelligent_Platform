# ================================
# @File         : common_util.py
# @Time         : 2025/08/08
# @Author       : Yingrui Chen
# @description  : AI Server公共方法
# ================================

"""
* 配置系统日志，用于服务器调试            setup_logging()
* 获取本机局域网IP，方便开发者操作         get_local_ip()
"""

import logging
import socket
from logging.handlers import RotatingFileHandler


def setup_logging():
    """配置日志系统"""
    log_format = ' 【%(levelname)s】%(asctime)s %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format
    )

    file_handler = RotatingFileHandler(
        './AIServer/ServerLog/app.log',
        maxBytes=1024 * 1024 * 5,  # 5MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter(log_format))

    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    return logger


def get_local_ip():
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
        common_logger.error(f"获取本地IP失败: {str(e)}")
        return "127.0.0.1"


common_logger = setup_logging()

if __name__ == "__main__":
    print(get_local_ip())
