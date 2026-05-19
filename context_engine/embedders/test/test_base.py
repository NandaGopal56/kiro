import pytest

from context_engine.embedders.base import Embedder


class MockEmbedder(Embedder):
    """Mock embedder for testing protocol."""
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]
    
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


@pytest.mark.asyncio
async def test_mock_embedder_embed_documents():
    embedder = MockEmbedder()
    result = await embedder.embed_documents(["text1", "text2"])
    assert len(result) == 2
    assert all(len(emb) == 2 for emb in result)


@pytest.mark.asyncio
async def test_mock_embedder_embed_query():
    embedder = MockEmbedder()
    result = await embedder.embed_query("test")
    assert len(result) == 2


@pytest.mark.asyncio
async def test_mock_embedder_empty_list():
    embedder = MockEmbedder()
    result = await embedder.embed_documents([])
    assert result == []


@pytest.mark.asyncio
async def test_mock_embedder_single_document():
    embedder = MockEmbedder()
    result = await embedder.embed_documents(["single text"])
    assert len(result) == 1
