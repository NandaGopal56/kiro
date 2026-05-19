"""Entry point for testing stores individually."""

from __future__ import annotations

import asyncio
import os

from ..chunkers import get as get_chunker
from ..embedders import get as get_embedder
from ..chunkers.schema import Doc
from . import get


async def main() -> None:
    # Test ChromaStore
    print("Testing stores...")
    print("=" * 50)
    
    # Check if API key is available for embedder
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        print("Please set it to test the store with embeddings.")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        print("\nSkipping store testing that requires embeddings.")
        exit(0)
    
    print("\n1. Testing ChromaStore:")
    store = get("chroma", collection_name="test_collection", persist_directory="./context_engine/data/chroma_test")
    
    # Clear any existing data
    await store.clear()
    print(f"Cleared existing data")
    
    # Create sample documents
    sample_docs = [
        Doc(
            page_content="Paris is the capital of France and known for the Eiffel Tower.",
            metadata={"source": "test", "chunk_id": 0},
        ),
        Doc(
            page_content="London is the capital of the United Kingdom and home to Big Ben.",
            metadata={"source": "test", "chunk_id": 1},
        ),
        Doc(
            page_content="Berlin is the capital of Germany and famous for the Brandenburg Gate.",
            metadata={"source": "test", "chunk_id": 2},
        ),
    ]
    
    # Add documents to the store
    print(f"\nAdding {len(sample_docs)} documents to the store...")
    ids = await store.add_documents(sample_docs)
    print(f"Added documents with IDs: {ids}")
    
    # Count documents
    count = await store.count()
    print(f"Total documents in store: {count}")
    
    # Similarity search
    print("\nPerforming similarity search...")
    query = "What is the capital of France?"
    results = await store.similarity_search(query, k=2)
    print(f"Found {len(results)} results for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"Result {i}: {doc.page_content[:80]}...")
    
    # Get documents by metadata
    print("\nGetting documents by metadata filter...")
    filtered = await store.get_documents(where={"source": "test"})
    print(f"Found {len(filtered)} documents with source='test'")
    
    # Clean up
    print("\nCleaning up test data...")
    await store.clear()
    print("Cleared test collection")
    
    print("\n" + "=" * 50)
    print("Store testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
