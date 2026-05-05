"""
LangChain Embeddings 适配器

wrap_embeddings() 包装任意 LangChain Embeddings 实例，
自动启用 SQLite 向量缓存，对外接口不变。
"""

import logging
from typing import Any, Dict, List, Optional, Type

from .cache import VectorCache
from .types import EmbeddingsProtocol

logger = logging.getLogger(__name__)


def wrap_embeddings(
    embeddings: EmbeddingsProtocol,
    cache_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_class: Optional[Type[VectorCache]] = None,
) -> "_CachedEmbeddingsProxy":
    """
    包装 LangChain Embeddings 实例，自动启用向量缓存

    Args:
        embeddings: 任意 LangChain Embeddings 实例
        cache_path: SQLite 缓存文件完整路径（优先级高于 cache_dir）
        cache_dir: 缓存目录，文件名自动生成为 vector_cache.db
        cache_class: 自定义缓存类（需实现 get/put/get_batch/put_batch 接口）

    Returns:
        包装后的 Embeddings 代理对象，接口与原实例一致
    """
    cache_cls = cache_class or VectorCache
    if cache_path:
        cache = cache_cls(db_path=cache_path)
    elif cache_dir:
        cache = cache_cls(cache_dir=cache_dir)
    else:
        cache = cache_cls()

    return _CachedEmbeddingsProxy(embeddings, cache)


class _CachedEmbeddingsProxy:
    """Embeddings 代理，自动查询/写入缓存"""

    def __init__(self, embeddings: EmbeddingsProtocol, cache: VectorCache):
        self._embeddings = embeddings
        self._cache = cache
        self._cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """带缓存的批量 embedding"""
        results, miss_indices = self._cache.get_batch(texts)

        if not miss_indices:
            self._cache_stats["hits"] += len(texts)
            return results  # type: ignore[return-value]

        # 只对未命中的文本调用原始 API
        miss_texts = [texts[i] for i in miss_indices]
        self._cache_stats["misses"] += len(miss_texts)

        miss_vectors = self._embeddings.embed_documents(miss_texts)
        self._cache.put_batch(miss_texts, miss_vectors)

        for idx, vec in zip(miss_indices, miss_vectors):
            results[idx] = vec

        logger.debug(
            "缓存命中: %d, 未命中: %d",
            len(texts) - len(miss_indices),
            len(miss_indices),
        )
        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        """带缓存的单条 embedding"""
        cached = self._cache.get(text)
        if cached is not None:
            self._cache_stats["hits"] += 1
            return cached

        self._cache_stats["misses"] += 1
        vector = self._embeddings.embed_query(text)
        self._cache.put(text, vector)
        return vector

    def get_cache_stats(self) -> Dict[str, int]:
        """返回缓存命中统计"""
        return dict(self._cache_stats)

    def __getattr__(self, name: str) -> Any:
        """透传未定义的属性到原始 embeddings"""
        return getattr(self._embeddings, name)
