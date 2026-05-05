"""
速率控制向量化核心模块

RateControlledEmbedder: 分批向量化 + 自动降速 + 进度条 + 流式预览
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm
from langchain_community.vectorstores import FAISS

from .cache import VectorCache
from .monitoring import get_hardware_suggestion
from .types import ProgressCallback

logger = logging.getLogger(__name__)


class RateControlledEmbedder:
    """
    速率控制向量化器

    分批处理文档块的 embedding，支持：
    - 可配置 batch_size / delay
    - 自动降速（单批耗时超过阈值时减半 batch_size、增大 delay）
    - SQLite 向量缓存
    - tqdm 进度条
    - 流式预览（每批完成后输出统计）
    """

    def __init__(
        self,
        embeddings,
        batch_size: int = 16,
        delay: float = 0.5,
        slow_threshold: float = 2.0,
        cache: Optional[VectorCache] = None,
    ):
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.delay = delay
        self.slow_threshold = slow_threshold
        self.cache = cache

        self._stats: Dict[str, Any] = {
            "total_chunks": 0,
            "total_batches": 0,
            "degrade_count": 0,
            "avg_batch_time": 0.0,
            "final_batch_size": batch_size,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    # ------------------------------------------------------------------
    # 速率控制
    # ------------------------------------------------------------------

    def set_rate(self, batch_size: int, delay: float):
        """动态调整速率"""
        self.batch_size = max(1, batch_size)
        self.delay = max(0.0, delay)

    @staticmethod
    def get_rate_suggestion() -> Dict[str, Any]:
        """获取硬件建议的速率参数"""
        return get_hardware_suggestion()

    def get_stats(self) -> Dict[str, Any]:
        """返回本轮向量化的统计信息"""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # 向量化核心
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """对一批文本做 embedding，处理缓存"""
        if self.cache:
            results, miss_indices = self.cache.get_batch(texts)
            if not miss_indices:
                self._stats["cache_hits"] += len(texts)
                return results

            # 只对未命中的文本调用 API
            miss_texts = [texts[i] for i in miss_indices]
            self._stats["cache_misses"] += len(miss_texts)
            miss_vectors = self.embeddings.embed_documents(miss_texts)

            # 写回缓存
            self.cache.put_batch(miss_texts, miss_vectors)

            # 合并结果
            for idx, vec in zip(miss_indices, miss_vectors):
                results[idx] = vec
            return results
        else:
            return self.embeddings.embed_documents(texts)

    def build_vectorstore(
        self,
        chunks,
        save_path: str = "faiss_index",
        progress_callback: ProgressCallback = None,
    ) -> FAISS:
        """
        分批构建 FAISS 向量库

        Args:
            chunks: Document 列表
            save_path: FAISS 索引保存路径
            progress_callback: 可选进度回调

        Returns:
            构建好的 FAISS 向量库
        """
        from langchain_core.documents import Document

        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        total = len(texts)

        self._stats["total_chunks"] = total
        self._stats["final_batch_size"] = self.batch_size

        all_vectors: List[List[float]] = []
        batch_times: List[float] = []
        current_batch_size = self.batch_size
        current_delay = self.delay

        pbar = tqdm(total=total, desc="向量化", unit="chunk")
        idx = 0

        while idx < total:
            batch_texts = texts[idx: idx + current_batch_size]
            batch_start = time.time()

            try:
                batch_vectors = self._embed_batch(batch_texts)
            except Exception as e:
                logger.error("批次 %d 向量化失败: %s", idx // current_batch_size + 1, e)
                raise

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            all_vectors.extend(batch_vectors)

            processed = min(idx + current_batch_size, total)
            pbar.update(len(batch_texts))

            # 流式预览
            if progress_callback:
                msg = (
                    f"批次 {len(batch_times)} | "
                    f"{processed}/{total} | "
                    f"耗时 {batch_time:.2f}s | "
                    f"batch_size={current_batch_size}"
                )
                progress_callback(msg)

            # 自动降速
            if batch_time > self.slow_threshold and current_batch_size > 1:
                old_size = current_batch_size
                current_batch_size = max(1, current_batch_size // 2)
                current_delay = min(current_delay * 1.5, 30.0)
                self._stats["degrade_count"] += 1
                logger.warning(
                    "自动降速: batch_size %d → %d, delay %.1f → %.1f",
                    old_size, current_batch_size, self.delay, current_delay,
                )

            idx += current_batch_size
            if idx < total:
                time.sleep(current_delay)

        pbar.close()

        # 构建 FAISS
        vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        vectorstore.save_local(save_path)

        # 更新统计
        self._stats["total_batches"] = len(batch_times)
        self._stats["avg_batch_time"] = sum(batch_times) / len(batch_times) if batch_times else 0
        self._stats["final_batch_size"] = current_batch_size

        return vectorstore
