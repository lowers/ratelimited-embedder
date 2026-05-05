"""VectorCache 单元测试"""

import os
import tempfile

import pytest

from ratelimited_embedder.cache import VectorCache


@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test_cache.db")
    return VectorCache(db_path=db_path)


class TestVectorCache:
    def test_put_and_get(self, cache):
        vector = [0.1, 0.2, 0.3]
        cache.put("hello", vector)
        result = cache.get("hello")
        assert result == vector

    def test_get_miss(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_overwrite(self, cache):
        cache.put("hello", [1.0])
        cache.put("hello", [2.0])
        assert cache.get("hello") == [2.0]

    def test_get_batch(self, cache):
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        texts = ["a", "b", "c"]
        results, miss_indices = cache.get_batch(texts)
        assert len(results) == 3
        assert results[0] == [1.0]
        assert results[1] == [2.0]
        assert results[2] is None
        assert miss_indices == [2]

    def test_put_batch(self, cache):
        texts = ["x", "y"]
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        cache.put_batch(texts, vectors)
        assert cache.get("x") == [1.0, 2.0]
        assert cache.get("y") == [3.0, 4.0]

    def test_stats(self, cache):
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        stats = cache.stats()
        assert stats["total_entries"] == 2

    def test_clear(self, cache):
        cache.put("a", [1.0])
        cache.clear()
        assert cache.get("a") is None
        assert cache.stats()["total_entries"] == 0

    def test_metadata(self, cache):
        cache.put("doc1", [0.5], metadata={"source": "test.pdf"})
        # metadata 存储不影响向量读取
        assert cache.get("doc1") == [0.5]

    def test_cache_dir(self, tmp_path):
        cache_dir = str(tmp_path / "my_cache")
        cache = VectorCache(cache_dir=cache_dir)
        cache.put("hello", [1.0, 2.0])
        assert cache.get("hello") == [1.0, 2.0]
        assert "vector_cache.db" in cache.db_path
        assert cache_dir in cache.db_path
