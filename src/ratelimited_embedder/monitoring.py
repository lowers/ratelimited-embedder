"""硬件监控与速率建议"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_hardware_suggestion() -> Dict[str, Any]:
    """
    根据本机硬件资源推荐向量化参数

    Returns:
        dict: batch_size, delay, mem_gb, mem_percent, cpu_count
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        mem_gb = round(mem.total / (1024**3), 1)
        mem_percent = mem.percent
        cpu_count = psutil.cpu_count(logical=True) or 4

        # 根据可用内存推荐 batch_size
        available_gb = mem.available / (1024**3)
        if available_gb >= 16:
            batch_size = 64
            delay = 0.2
        elif available_gb >= 8:
            batch_size = 32
            delay = 0.3
        elif available_gb >= 4:
            batch_size = 16
            delay = 0.5
        else:
            batch_size = 8
            delay = 1.0

        return {
            "batch_size": batch_size,
            "delay": delay,
            "mem_gb": mem_gb,
            "mem_percent": mem_percent,
            "cpu_count": cpu_count,
        }

    except ImportError:
        logger.warning("psutil 未安装，返回默认建议")
        return {
            "batch_size": 16,
            "delay": 0.5,
            "mem_gb": 0,
            "mem_percent": 0,
            "cpu_count": 4,
        }
