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

    @patch("ratelimited_embedder.core.FAISS")
    def test_build_vectorstore(self, mock_faiss_cls, embedder, tmp_path):
        from langchain_core.documents import Document

        chunks = [Document(page_content=f"doc {i}") for i in range(10)]
        mock_vs = MagicMock()
        mock_faiss_cls.from_texts.return_value = mock_vs

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
