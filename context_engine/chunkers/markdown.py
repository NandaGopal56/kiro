from __future__ import annotations

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .schema import Doc
from .base import Chunker, clean_chunk


class MarkdownChunker(Chunker):
    """Split Markdown by headings, then recursively split large sections."""

    name = "markdown"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        strip_headers: bool = False,
        separators: list[str] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("clean_output", False)
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
            or [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=strip_headers,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=False,
            separators=separators
            or [
                "\n\n",
                "\n",
                ". ",
                ", ",
                " ",
                "",
            ],
        )

    def split(self, texts: list[str]) -> list[Doc]:
        chunks = []
        for text in texts:
            header_splits = self.header_splitter.split_text(text)
            for doc in header_splits:
                if len(doc.page_content) <= self.chunk_size:
                    chunks.append(doc)
                else:
                    chunks.extend(self.text_splitter.split_documents([doc]))

        docs: list[Doc] = []
        for chunk in chunks:
            text = (
                clean_chunk(chunk.page_content)
                if self.clean_output
                else chunk.page_content.strip()
            )
            if not text and not self.include_empty:
                continue
            docs.append(
                Doc(
                    page_content=text,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": len(docs),
                        "source": self.source,
                        "chunker": self.name,
                    },
                )
            )
        return docs
