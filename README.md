# RAG

A small, composable toolkit for building a production RAG pipeline.
Start with a single PDF; extend to more sources, providers, and stores by
dropping a new class into the matching subpackage.

## Layout

```
context_engine/
  types.py          # Doc, ParsedDoc
  text.py           # clean_text, clean_chunk
  config.py         # RAGConfig
  rag.py            # Ingestor (load -> chunk -> store)

  loaders/          # AutoLoader, PdfLoader, PdfMarkdownLoader (+ future: web, docx, ...)
  summarizers/      # OpenAISummarizer for tables/images
  chunkers/         # FixedChunker, RecursiveChunker, MarkdownChunker, SemanticChunker
  embedders/        # HashEmbedder (test) (+ future: openai, cohere, ...)
  stores/           # ChromaStore (+ future: pinecone, qdrant, ...)
  evals/            # Sample/Result/Answer + simple metrics + runner
```

Each subpackage exposes:

- `base.py` — the contract every implementation must satisfy
- one file per implementation
- `REGISTRY` dict + `get(name, **kwargs)` for name-based instantiation

Adding a new source/provider = add a class, add one line to `REGISTRY`.

## End-to-end ingestion

```python
from context_engine import Ingestor, RAGConfig

config = RAGConfig(
    loader="pdf",
    loader_kwargs={"context_window": 3},
    chunker="recursive",
    chunker_kwargs={"chunk_size": 500, "chunk_overlap": 80, "source": "coffee_guide"},
    store="chroma",
    store_kwargs={"collection_name": "coffee", "persist_directory": ".rag_chroma"},
    embedder="hash",          # swap to a real provider for production
    summarizer=None,          # set to "openai" to summarize tables/images
)

ingestor = Ingestor.from_config(config)
ids = ingestor.ingest("data/coffee_processing.pdf")
matches = ingestor.search("How is washed coffee processed?", k=3)
```

Compose by hand if you want full control:

```python
from context_engine import Ingestor
from context_engine.chunkers import RecursiveChunker
from context_engine.embedders import HashEmbedder
from context_engine.loaders import PdfLoader
from context_engine.stores import ChromaStore
from context_engine.summarizers import OpenAISummarizer

ingestor = Ingestor(
    loader=PdfLoader(summarizer=OpenAISummarizer()),
    chunker=RecursiveChunker(chunk_size=500, chunk_overlap=80, source="coffee_guide"),
    store=ChromaStore(collection_name="coffee", embedder=HashEmbedder()),
)
chunks = ingestor.run("data/coffee_processing.pdf")
```

## PDF Markdown extraction

For structure-aware PDF chunking, use the Markdown PDF loader and then chunk
the Markdown output:

```python
from context_engine import Ingestor, RAGConfig

config = RAGConfig(
    loader="auto",
    loader_kwargs={"output_dir": "data/extracted"},
    chunker="markdown",
    chunker_kwargs={"chunk_size": 1000, "chunk_overlap": 150, "source": "contract"},
)

ingestor = Ingestor.from_config(config)
chunks = ingestor.run("data/SampleContract.pdf")
```

`AutoLoader` chooses the loader from the file extension. Currently `.pdf` uses
`PdfMarkdownLoader`, which writes `data/extracted/<pdf-name>/<pdf-name>.md`
plus an image artifact folder. Use the loader directly when you only need
conversion:

```python
from context_engine.loaders import PdfMarkdownLoader

result = PdfMarkdownLoader(output_dir="data/extracted").extract(
    "data/SampleContract.pdf"
)
print(result.text_path)
```

## Chunkers

`fixed` / `character`, `recursive`, `markdown` / `md`, `semantic`. Each lives in
`context_engine/chunkers/`. Pass strategy-specific kwargs via
`RAGConfig.chunker_kwargs` or directly to the class.

## Vector store

`ChromaStore` exposes `add_documents`, `upsert_documents`, `get_documents`,
`update_documents`, `delete_documents`, `similarity_search`, `count`, `clear`.

Smoke-test the store without any OpenAI key:

```bash
uv run python -m context_engine.stores.chroma --reset
```

The smoke test uses `HashEmbedder` — deterministic local embeddings intended
only for verifying database behavior. Swap in a real provider for retrieval
quality.

## Independent evaluation

The eval layer is decoupled from any specific RAG implementation. Pass it any
callable that turns a question into an `Answer`.

```python
from context_engine.evals import (
    Answer,
    AnswerContains,
    ContextContains,
    Sample,
    evaluate,
)


def my_rag(question: str) -> Answer:
    return Answer(answer="washed coffee uses water", contexts=["washed coffee uses water"])


results = evaluate(
    rag=my_rag,
    samples=[Sample(question="How is washed coffee processed?", expected_answer="water")],
    metrics=[AnswerContains(), ContextContains()],
)
```

## Adding a new source or provider

1. Drop a new class into the matching subpackage (e.g. `loaders/web.py`).
2. Make it inherit from the `base.py` contract.
3. Register it in that subpackage's `__init__.py` `REGISTRY` under a short name.
4. Reference the new name from `RAGConfig`.

That's the whole extension story — no factories, no orchestrator edits.

## Topics covered

- Chunking: fixed, recursive, semantic, document-structure, agentic
- Retrieval: vector search, reranking, evaluation
- RAG variants: simple vector, hybrid, graph

## Articles

- Chunking with clustering — https://towardsdatascience.com/improving-rag-chunking-with-clustering-03c1cf41f1cd/
- Query translation — https://raghunaathan.medium.com/query-translation-for-rag-retrieval-augmented-generation-applications-46d74bff8f07
