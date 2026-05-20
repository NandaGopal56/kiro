from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import ContentLayer

from .schema import ParsedDoc, ExtractionResult
from .base import Loader
from .utils import clean_rag_markdown, clean_repeated_special_chars


class PdfMarkdownLoader(Loader):
    """Extract a PDF to cleaned Markdown and load it as ParsedDoc text."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        clean: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.clean = clean

    async def load(self, source: str) -> ParsedDoc:
        result = await self.extract(source)
        return ParsedDoc(texts=[result.text])

    async def extract(self, source: str | Path) -> ExtractionResult:
        pdf_path = Path(source).resolve()
        document_dir = self._document_dir(pdf_path)
        document_dir.mkdir(parents=True, exist_ok=True)

        markdown_path = document_dir / f"{pdf_path.stem}.md"
        artifacts_dir = document_dir / f"{pdf_path.stem}_artifacts"

        result = self._build_converter().convert(str(pdf_path), raises_on_error=True)

        result.document.save_as_markdown(
            markdown_path,
            page_break_placeholder="\n\n---\n\n",
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=artifacts_dir
        )

        markdown_text = markdown_path.read_text(encoding="utf-8")
        # if self.clean:
        #     markdown_text = clean_repeated_special_chars(clean_rag_markdown(markdown_text))
        #     markdown_path.write_text(markdown_text, encoding="utf-8")

        return ExtractionResult(
            source_path=pdf_path,
            text_path=markdown_path,
            text=markdown_text,
            metadata={"loader": "pdf_markdown"},
        )

    def extract_batch(self, input_dir: str | Path) -> list[ExtractionResult]:
        """Extract every PDF in a folder, skipping files that fail."""

        results: list[ExtractionResult] = []
        for pdf_path in sorted(Path(input_dir).resolve().glob("*.pdf")):
            try:
                results.append(asyncio.run(self.extract(pdf_path)))
            except Exception as exc:  # noqa: BLE001
                print(f"Skipped {pdf_path.name}: {exc}", file=sys.stderr)
        return results

    def _document_dir(self, pdf_path: Path) -> Path:
        if self.output_dir is None:
            return pdf_path.parent / "extracted" / pdf_path.stem
        return self.output_dir / pdf_path.stem

    @staticmethod
    def _pipeline_options() -> PdfPipelineOptions:
        return PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE,
                do_cell_matching=True,
            ),
            do_picture_extraction=True,
            do_picture_classification=True,
            images_scale=2.0,
            generate_page_images=True,
            generate_picture_images=True,
        )

    @classmethod
    def _build_converter(cls) -> DocumentConverter:
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=cls._pipeline_options())
            },
        )
