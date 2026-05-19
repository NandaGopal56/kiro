import pytest

from context_engine.loaders.schema import ParsedDoc


def test_parsed_doc_creation():
    doc = ParsedDoc()
    assert doc.texts == []
    assert doc.images == []
    assert doc.tables == []


def test_parsed_doc_with_texts():
    doc = ParsedDoc(texts=["text1", "text2"])
    assert doc.texts == ["text1", "text2"]
    assert doc.images == []
    assert doc.tables == []


def test_parsed_doc_as_dict():
    doc = ParsedDoc(texts=["text1"], images=["img1"], tables=["table1"])
    result = doc.as_dict()
    assert result == {"texts": ["text1"], "images": ["img1"], "tables": ["table1"]}


def test_parsed_doc_all_texts_default():
    doc = ParsedDoc(texts=["text1"], images=["img1"], tables=["table1"])
    result = doc.all_texts()
    assert result == ["text1", "img1", "table1"]


def test_parsed_doc_all_texts_exclude_images():
    doc = ParsedDoc(texts=["text1"], images=["img1"], tables=["table1"])
    result = doc.all_texts(include_images=False)
    assert result == ["text1", "table1"]


def test_parsed_doc_all_texts_exclude_tables():
    doc = ParsedDoc(texts=["text1"], images=["img1"], tables=["table1"])
    result = doc.all_texts(include_tables=False)
    assert result == ["text1", "img1"]


def test_parsed_doc_default_factory_independence():
    doc1 = ParsedDoc()
    doc2 = ParsedDoc()
    doc1.texts.append("text")
    assert doc1.texts == ["text"]
    assert doc2.texts == []
