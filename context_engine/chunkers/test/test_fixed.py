import pytest

from context_engine.chunkers.fixed import FixedChunker


@pytest.mark.asyncio
async def test_fixed_chunker_initialization():
    chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.name == "fixed"


@pytest.mark.asyncio
async def test_fixed_chunker_split_empty_list():
    chunker = FixedChunker(chunk_size=100)
    result = await chunker.split([])
    assert result == []


@pytest.mark.asyncio
async def test_fixed_chunker_split_single_text():
    chunker = FixedChunker(chunk_size=50)
    texts = ["This is a test text that should be chunked"]
    result = await chunker.split(texts)
    assert len(result) > 0
    assert all(doc.page_content for doc in result)


@pytest.mark.asyncio
async def test_fixed_chunker_split_multiple_texts():
    chunker = FixedChunker(chunk_size=50)
    texts = ["First text", "Second text"]
    result = await chunker.split(texts)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_fixed_chunker_metadata():
    chunker = FixedChunker(chunk_size=50, source="test_source")
    texts = ["Test text"]
    result = await chunker.split(texts)
    assert result[0].metadata["source"] == "test_source"
    assert result[0].metadata["chunker"] == "fixed"
