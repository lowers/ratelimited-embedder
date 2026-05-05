"""
ratelimited-embedder — LangChain 速率控制向量化工具

支持批处理、自动降速、SQLite 向量缓存、硬件建议。
"""

__version__ = "0.1.0"

from .core import RateControlledEmbedder
from .cache import VectorCache
from .langchain_adapter import wrap_embeddings
from .monitoring import get_hardware_suggestion

__all__ = [
    "RateControlledEmbedder",
    "VectorCache",
    "wrap_embeddings",
    "get_hardware_suggestion",
]
