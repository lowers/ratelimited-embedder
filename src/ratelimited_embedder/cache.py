"""SQLite 向量缓存，MD5 哈希去重"""

import hashlib
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorCache:
    """
    基于 SQLite 的向量缓存

    以文本的 MD5 哈希作为 key，存储 embedding 向量和可选元数据。
    避免对相同文本重复调用 embedding API。
    """

    def __init__(self, db_path: str = "vector_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_cache (
                    hash TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """查询缓存，命中返回向量，未命中返回 None"""
        h = self._hash(text)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT vector FROM vector_cache WHERE hash = ?", (h,)
            ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, text: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """写入缓存"""
        h = self._hash(text)
        vec_str = json.dumps(vector)
        meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vector_cache (hash, text, vector, metadata) VALUES (?, ?, ?, ?)",
                (h, text, vec_str, meta_str),
            )
            conn.commit()

    def get_batch(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        批量查询缓存

        Returns:
            (results, miss_indices)
            results[i] 为向量或 None
            miss_indices 为未命中的索引列表
        """
        results = []
        miss_indices = []
        for i, text in enumerate(texts):
            vec = self.get(text)
            results.append(vec)
            if vec is None:
                miss_indices.append(i)
        return results, miss_indices

    def put_batch(self, texts: List[str], vectors: List[List[float]], metadata: Optional[Dict[str, Any]] = None):
        """批量写入缓存"""
        for text, vec in zip(texts, vectors):
            self.put(text, vec, metadata)

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM vector_cache").fetchone()
            count = row[0] if row else 0
        return {"total_entries": count, "db_path": self.db_path}

    def clear(self):
        """清空缓存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM vector_cache")
            conn.commit()
