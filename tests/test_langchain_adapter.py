"""wrap_embeddings 适配器测试"""

import pytest

from ratelimited_embedder.langchain_adapter import wrap_embeddings


class MockEmbeddings:
    def __init__(self):
        self.call_log = []

    def embed_documents(self, texts):
        self.call_log.append(("docs", texts))
        return [[float(i)] * 3 for i in range(len(texts))]

    def embed_query(self, text):
        self.call_log.append(("query", text))
        return [0.1, 0.2, 0.3]


@pytest.fixture
def wrapped(tmp_path):
    emb = MockEmbeddings()
    cache_path = str(tmp_path / "adapter_cache.db")
    return wrap_embeddings(emb, cache_path=cache_path), emb


class TestWrapEmbeddings:
    def test_embed_documents_caches(self, wrapped):
        proxy, original = wrapped
        texts = ["hello", "world"]

        result1 = proxy.embed_documents(texts)
        assert len(result1) == 2
        assert len(original.call_log) == 1

        # 第二次调用应命中缓存，不增加 call_log
        result2 = proxy.embed_documents(texts)
        assert result2 == result1
        assert len(original.call_log) == 1

    def test_embed_query_caches(self, wrapped):
        proxy, original = wrapped

        result1 = proxy.embed_query("test")
        assert len(result1) == 3
        assert len(original.call_log) == 1

        result2 = proxy.embed_query("test")
        assert result2 == result1
        assert len(original.call_log) == 1

    def test_cache_stats(self, wrapped):
        proxy, original = wrapped
        proxy.embed_query("a")
        proxy.embed_query("a")  # cache hit
        stats = proxy.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1

    def test_attribute_proxy(self, wrapped):
        proxy, original = wrapped
        # 自定义属性透传
        original.custom_attr = "hello"
        assert proxy.custom_attr == "hello"

    def test_cache_dir(self, tmp_path):
        emb = MockEmbeddings()
        cache_dir = str(tmp_path / "embed_cache")
        proxy = wrap_embeddings(emb, cache_dir=cache_dir)
        result = proxy.embed_query("test")
        assert len(result) == 3
        # 第二次应命中缓存
        result2 = proxy.embed_query("test")
        assert result2 == result
