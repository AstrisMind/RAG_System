import os
import logging
from loguru import logger
from typing import Any

# 配置日志
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 删除默认的logger
logger.remove()

# 添加自定义logger，输出到控制台
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# 添加文件输出
logger.add(
    "rag_system.log",
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="7 days"
)

# 创建兼容标准库logging的适配器
class LoguruHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

# 配置标准库logging使用loguru
def configure_standard_logging():
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(LoguruHandler())
    root_logger.setLevel(getattr(logging, LOG_LEVEL))

# 确保目录存在
def ensure_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")

# 验证文件存在
def validate_file(file_path: str) -> bool:
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False
    if not os.path.isfile(file_path):
        logger.error(f"路径不是文件: {file_path}")
        return False
    return True

# 获取文件扩展名
def get_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[1].lower()

__all__ = ["logger", "configure_standard_logging", "ensure_directory", "validate_file", "get_file_extension"]