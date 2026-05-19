import pytest
from unittest.mock import patch, MagicMock

from context_engine.embedders.openai import OpenAIEmbedder


@patch("context_engine.embedders.openai.OpenAIEmbeddings")
def test_init_defaults(mock_cls):
    mock_cls.return_value = MagicMock()
    e = OpenAIEmbedder()
    mock_cls.assert_called_once()
    assert e.embeddings is mock_cls.return_value


@patch("context_engine.embedders.openai.OpenAIEmbeddings")
def test_init_with_params(mock_cls):
    mock_cls.return_value = MagicMock()
    e = OpenAIEmbedder(model="x", api_key="y", timeout=10)

    mock_cls.assert_called_once_with(
        model="x",
        api_key="y",
        timeout=10,
    )


def test_has_methods():
    e = OpenAIEmbedder.__new__(OpenAIEmbedder)  # no init
    assert hasattr(e, "embed_documents")
    assert hasattr(e, "embed_query")