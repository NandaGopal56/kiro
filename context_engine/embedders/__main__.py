"""Entry point for testing embedders individually."""

from __future__ import annotations

import asyncio
import os

from . import get


async def main() -> None:
    # Test OpenAI embedder
    print("Testing embedders...")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        print("Please set it to test the OpenAI embedder.")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    print("\n1. Testing OpenAIEmbedder:")
    embedder = get("openai", model="text-embedding-3-small", api_key=api_key)
    
    # Test single query embedding
    query = "What is the capital of France?"
    query_embedding = await embedder.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    print(f"First 5 values: {query_embedding[:5]}")
    
    # Test document embeddings
    documents = [
        "Paris is the capital of France.",
        "London is the capital of the United Kingdom.",
        "Berlin is the capital of Germany.",
    ]
    doc_embeddings = await embedder.embed_documents(documents)
    print(f"\nNumber of document embeddings: {len(doc_embeddings)}")
    print(f"Each embedding dimension: {len(doc_embeddings[0])}")
    print(f"First document embedding first 5 values: {doc_embeddings[0][:5]}")
    
    print("\n" + "=" * 50)
    print("Embedder testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
