import pytest

from context_engine import ContextEngine
from context_engine.chunkers.schema import Doc
from context_engine.loaders.schema import ParsedDoc


class FakeLoader:
    async def load(self, source: str):
        return ParsedDoc(texts=[source])


class FakeChunker:
    async def split(self, texts: list[str]):
        return [Doc(page_content=text) for text in texts]


class FakeStore:
    def __init__(self):
        self.docs = []

    async def add_documents(self, docs, ids=None):
        self.docs.extend(docs)
        return ["added"]

    async def upsert_documents(self, docs, ids=None):
        self.docs.extend(docs)
        return ["upserted"]

    async def similarity_search(self, query, k=4, where=None):
        return [Doc(page_content=query, metadata={"k": k})]


@pytest.mark.asyncio
async def test_context_engine_uses_injected_components():
    store = FakeStore()
    engine = ContextEngine(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        store=store,
    )

    ids = await engine.ingest("hello")
    docs = await engine.retrieve("query", k=2)

    assert ids == ["upserted"]
    assert store.docs == [Doc(page_content="hello")]
    assert docs == [Doc(page_content="query", metadata={"k": 2})]
