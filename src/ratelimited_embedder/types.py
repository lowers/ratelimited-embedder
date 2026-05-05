"""类型定义"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Protocol


class EmbeddingsProtocol(Protocol):
    """LangChain Embeddings 接口协议"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


ProgressCallback = Optional[Callable[[str], None]]
