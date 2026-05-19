import pytest
from pathlib import Path

from context_engine.loaders.auto import AutoLoader


def test_autoloader_initialization_default():
    loader = AutoLoader()
    assert ".pdf" in loader.loaders_by_suffix


def test_autoloader_initialization_with_output_dir():
    loader = AutoLoader(output_dir="/test/output")
    assert ".pdf" in loader.loaders_by_suffix


def test_autoloader_custom_loaders():
    from context_engine.loaders.base import Loader
    
    class CustomLoader(Loader):
        async def load(self, source: str):
            from context_engine.loaders.schema import ParsedDoc
            return ParsedDoc(texts=["custom"])
    
    loader = AutoLoader(loaders_by_suffix={".custom": CustomLoader()})
    assert ".custom" in loader.loaders_by_suffix


def test_autoloader_loader_for_pdf():
    loader = AutoLoader()
    result = loader._loader_for("test.pdf")
    assert result is not None


def test_autoloader_loader_for_unsupported():
    loader = AutoLoader()
    with pytest.raises(ValueError, match="No loader registered"):
        loader._loader_for("test.unknown")
