import pytest
from pathlib import Path

from context_engine.loaders.base import ExtractionResult, Loader


class MockLoader(Loader):
    """Mock loader for testing base class."""
    async def load(self, source: str):
        from context_engine.loaders.schema import ParsedDoc
        return ParsedDoc(texts=["mock content"])


def test_extraction_result_creation():
    result = ExtractionResult(
        source_path=Path("/test/source.pdf"),
        text_path=Path("/test/source.txt"),
        text="extracted text"
    )
    assert result.source_path == Path("/test/source.pdf")
    assert result.text_path == Path("/test/source.txt")
    assert result.text == "extracted text"
    assert result.artifacts_dir is None
    assert result.metadata is None


def test_extraction_result_with_metadata():
    result = ExtractionResult(
        source_path=Path("/test/source.pdf"),
        text_path=Path("/test/source.txt"),
        text="extracted text",
        artifacts_dir=Path("/test/artifacts"),
        metadata={"key": "value"}
    )
    assert result.artifacts_dir == Path("/test/artifacts")
    assert result.metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_mock_loader_load():
    loader = MockLoader()
    result = await loader.load("test.pdf")
    assert result.texts == ["mock content"]


def test_loader_is_abstract():
    from context_engine.loaders.base import Loader
    with pytest.raises(TypeError):
        Loader()
