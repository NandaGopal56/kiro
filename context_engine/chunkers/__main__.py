"""Entry point for testing chunkers individually."""

from __future__ import annotations

import asyncio
from pathlib import Path

from . import get


async def main() -> None:
    # Test different chunkers
    # sample_text = """
    # This is a sample document for testing chunkers.
    # It contains multiple paragraphs that should be split into chunks.
    # The chunking strategy determines how the text is divided.
    # """
    
    # print("Testing chunkers...")
    # print("=" * 50)
    
    # # Test FixedChunker
    # print("\n1. Testing FixedChunker:")
    # fixed_chunker = get("fixed", chunk_size=100, chunk_overlap=10)
    # chunks = await fixed_chunker.split([sample_text])
    # print(f"Number of chunks: {len(chunks)}")
    # for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
    #     print(f"Chunk {i}: {chunk.page_content[:100]}...")
    
    # # Test RecursiveChunker
    # print("\n2. Testing RecursiveChunker:")
    # recursive_chunker = get("recursive")
    # chunks = await recursive_chunker.split([sample_text])
    # print(f"Number of chunks: {len(chunks)}")
    # for i, chunk in enumerate(chunks[:2]):
    #     print(f"Chunk {i}: {chunk.page_content[:100]}...")
    
    # Test MarkdownChunker
    print("\n3. Testing MarkdownChunker:")
    markdown_path = Path("/Users/nnandagopal/Desktop/personal_projects/RAG/context_engine/data/extracted/example1/example1.md")  # TODO: Update this path
    markdown_text = markdown_path.read_text()
    md_chunker = get("markdown")
    chunks = await md_chunker.split([markdown_text])
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):
        print(f"Chunk {i}: {chunk.page_content[:100]}...")
    
    print("\n" + "=" * 50)
    print("Chunker testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
