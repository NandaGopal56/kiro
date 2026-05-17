# RAG

A small, composable toolkit for building a context engine that can be used
from an agent tool, a web API, a script, or a notebook.

Use `ContextEngine` as the main entry point. The lower-level loader, chunker,
embedder, and store packages stay independently replaceable.

## Layout

```
context_engine/
  engine.py         # ContextEngine, ContextEngineConfig, Doc, ParsedDoc

  loaders/          # AutoLoader, PdfMarkdownLoader (+ future: web, docx, ...)
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

## Main Entry Point

```python
from context_engine import ContextEngine, ContextEngineConfig

engine = ContextEngine(
    ContextEngineConfig(
        loader="auto",
        chunker="recursive",
        store="chroma",
        store_kwargs={
            "collection_name": "coffee",
            "persist_directory": ".rag_chroma",
        },
        embedder="hash",  # local deterministic embeddings for testing
    )
)

ids = engine.ingest("data/coffee_processing.pdf")
matches = engine.retrieve("How is washed coffee processed?", k=3)
```

Override a process per request when an agent or API caller needs a different
strategy:

```python
ids = engine.ingest(
    "data/contract.pdf",
    loader="pdf",
    chunker="markdown",
    chunker_kwargs={"chunk_size": 1000, "chunk_overlap": 150, "source": "contract"},
)

matches = engine.retrieve(
    "What are the termination terms?",
    k=5,
    where={"source": "contract"},
)
```

For a web API, keep one engine instance at app startup and serialize returned
docs:

```python
from context_engine import ContextEngine

engine = ContextEngine()


def retrieve_handler(query: str) -> list[dict]:
    return [doc.as_dict() for doc in engine.retrieve(query, k=4)]
```

For an agent framework, adapt the framework-neutral callables:

```python
engine = ContextEngine()
tools = engine.tools()

tools["context_ingest"](source="data/handbook.pdf")
tools["context_retrieve"](query="What is the refund policy?", k=3)
```

## Configuration

```python
from context_engine import ContextEngineConfig

config = ContextEngineConfig(
    loader="auto",
    loader_kwargs={"output_dir": "data/extracted"},
    chunker="recursive",
    chunker_kwargs={"chunk_size": 500, "chunk_overlap": 80, "source": "coffee_guide"},
    store="chroma",
    store_kwargs={"collection_name": "coffee", "persist_directory": ".rag_chroma"},
    embedder="hash",
    summarizer=None,  # set to "openai" to summarize tables/images
)
```

Compose by hand if you want full control:

```python
from context_engine import ContextEngine
from context_engine.chunkers import RecursiveChunker
from context_engine.embedders import HashEmbedder
from context_engine.loaders import PdfMarkdownLoader
from context_engine.stores import ChromaStore

engine = ContextEngine(
    loader=PdfMarkdownLoader(output_dir="data/extracted"),
    chunker=RecursiveChunker(chunk_size=500, chunk_overlap=80, source="coffee_guide"),
    store=ChromaStore(collection_name="coffee", embedder=HashEmbedder()),
)
chunks = engine.run("data/coffee_processing.pdf")
```

## PDF Markdown extraction

For structure-aware PDF chunking, use the Markdown PDF loader and then chunk
the Markdown output:

```python
from context_engine import ContextEngine, ContextEngineConfig

config = ContextEngineConfig(
    loader="auto",
    loader_kwargs={"output_dir": "data/extracted"},
    chunker="markdown",
    chunker_kwargs={"chunk_size": 1000, "chunk_overlap": 150, "source": "contract"},
)

engine = ContextEngine(config)
chunks = engine.run("data/SampleContract.pdf")
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
`ContextEngineConfig.chunker_kwargs`, per-call `chunker_kwargs`, or directly to
the class.

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
4. Reference the new name from `ContextEngineConfig` or pass it to a method.

That's the whole extension story — no factories, no orchestrator edits.

## Topics covered

- Chunking: fixed, recursive, semantic, document-structure, agentic
- Retrieval: vector search, reranking, evaluation
- RAG variants: simple vector, hybrid, graph

## Articles

- Chunking with clustering — https://towardsdatascience.com/improving-rag-chunking-with-clustering-03c1cf41f1cd/
- Query translation — https://raghunaathan.medium.com/query-translation-for-rag-retrieval-augmented-generation-applications-46d74bff8f07
