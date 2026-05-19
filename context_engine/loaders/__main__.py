"""Entry point for testing loaders individually."""

from __future__ import annotations

import asyncio

from pathlib import Path

from . import get


async def main() -> None:
    # Test different loaders
    print("Testing loaders...")
    print("=" * 50)
    
    # Test AutoLoader
    print("\n1. Testing AutoLoader:")
    auto_loader = get("auto")
    print(f"AutoLoader initialized with suffixes: {list(auto_loader.loaders_by_suffix.keys())}")
    
    # Test PdfMarkdownLoader
    print("\n2. Testing PdfMarkdownLoader:")
    pdf_loader = get("pdf", output_dir="./context_engine/data/extracted")
    print(f"PdfMarkdownLoader initialized")
    
    # Example: Load a PDF file if available
    # Uncomment the following lines to test with an actual PDF file
    sample_pdf = "././data/example1.pdf"
    if Path(sample_pdf).exists():
        print(f"\nLoading PDF: {sample_pdf}")
        parsed_doc = await pdf_loader.load(sample_pdf)
        print(f"Loaded document with {len(parsed_doc.texts)} characters")
        print(f"First 200 characters: {parsed_doc.texts[0][:200]}...")
    else:
        raise FileNotFoundError(f"Sample PDF not found at: {sample_pdf}")
    
    print("\n" + "=" * 50)
    print("Loader testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
