import logging
import sys

from src.core.settings import settings

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

_configured = False


def setup_logging() -> None:
    """幂等初始化全局日志配置。"""
    global _configured
    if _configured:
        return

    level_name = settings.LOG_LEVEL.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format=settings.LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取模块 logger，首次调用时自动完成日志初始化。"""
    setup_logging()
    return logging.getLogger(name)
