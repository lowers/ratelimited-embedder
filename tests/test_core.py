"""RateControlledEmbedder 单元测试"""

from unittest.mock import MagicMock, patch

import pytest

from ratelimited_embedder.core import RateControlledEmbedder


class MockEmbeddings:
    """模拟 LangChain Embeddings"""

    def __init__(self, dim=4):
        self.dim = dim
        self.call_count = 0

    def embed_documents(self, texts):
        self.call_count += 1
        return [[float(i) / len(texts)] * self.dim for i in range(len(texts))]

    def embed_query(self, text):
        return [0.5] * self.dim


@pytest.fixture
def mock_embeddings():
    return MockEmbeddings()


@pytest.fixture
def embedder(mock_embeddings):
    return RateControlledEmbedder(
        embeddings=mock_embeddings,
        batch_size=4,
        delay=0.0,
        slow_threshold=10.0,
    )


class TestRateControlledEmbedder:
    def test_set_rate(self, embedder):
        embedder.set_rate(batch_size=32, delay=1.0)
        assert embedder.batch_size == 32
        assert embedder.delay == 1.0

    def test_set_rate_clamp(self, embedder):
        embedder.set_rate(batch_size=0, delay=-1.0)
        assert embedder.batch_size == 1
        assert embedder.delay == 0.0

    def test_get_stats_initial(self, embedder):
        stats = embedder.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["degrade_count"] == 0

    @patch("langchain_community.vectorstores.FAISS")
    def test_build_vectorstore(self, mock_faiss_cls, embedder, tmp_path):
        from langchain_core.documents import Document

        chunks = [Document(page_content=f"doc {i}") for i in range(10)]
        mock_vs = MagicMock()
        mock_faiss_cls.from_embeddings.return_value = mock_vs

        result = embedder.build_vectorstore(chunks, save_path=str(tmp_path / "idx"))

        assert result is mock_vs
        stats = embedder.get_stats()
        assert stats["total_chunks"] == 10
        assert stats["total_batches"] > 0

    def test_rate_suggestion(self):
        suggestion = RateControlledEmbedder.get_rate_suggestion()
        assert "batch_size" in suggestion
        assert "delay" in suggestion
        assert suggestion["batch_size"] >= 1

    def test_cache_path(self, mock_embeddings, tmp_path):
        embedder = RateControlledEmbedder(
            embeddings=mock_embeddings,
            cache_path=str(tmp_path / "cache.db"),
        )
        assert embedder.cache is not None
        assert "cache.db" in embedder.cache.db_path

    def test_cache_dir(self, mock_embeddings, tmp_path):
        embedder = RateControlledEmbedder(
            embeddings=mock_embeddings,
            cache_dir=str(tmp_path / "my_cache"),
        )
        assert embedder.cache is not None
        assert "vector_cache.db" in embedder.cache.db_path

    def test_cache_priority(self, mock_embeddings, tmp_path):
        """cache 参数优先级高于 cache_path 和 cache_dir"""
        from ratelimited_embedder.cache import VectorCache
        existing = VectorCache(db_path=str(tmp_path / "existing.db"))
        embedder = RateControlledEmbedder(
            embeddings=mock_embeddings,
            cache=existing,
            cache_path=str(tmp_path / "other.db"),
            cache_dir=str(tmp_path / "dir"),
        )
        assert embedder.cache is existing


class TestTopLevelImports:
    """验证 __init__.py 导出正确"""

    def test_import_all_exports(self):
        import ratelimited_embedder
        for name in ratelimited_embedder.__all__:
            assert hasattr(ratelimited_embedder, name), f"Missing export: {name}"

    def test_version_defined(self):
        from ratelimited_embedder import __version__
        assert isinstance(__version__, str)
        assert len(__version__.split(".")) >= 2
