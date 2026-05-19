import pytest

from context_engine.chunkers.schema import Doc


def test_doc_creation():
    doc = Doc(page_content="Test content")
    assert doc.page_content == "Test content"
    assert doc.metadata == {}


def test_doc_with_metadata():
    doc = Doc(page_content="Test", metadata={"key": "value"})
    assert doc.page_content == "Test"
    assert doc.metadata == {"key": "value"}


def test_doc_as_dict():
    doc = Doc(page_content="Test", metadata={"key": "value"})
    result = doc.as_dict()
    assert result == {"page_content": "Test", "metadata": {"key": "value"}}


def test_doc_default_metadata_factory():
    doc1 = Doc(page_content="Test1")
    doc2 = Doc(page_content="Test2")
    assert doc1.metadata is not doc2.metadata


def test_doc_metadata_mutation():
    doc = Doc(page_content="Test")
    doc.metadata["new_key"] = "new_value"
    assert doc.metadata == {"new_key": "new_value"}
