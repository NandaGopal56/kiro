from __future__ import annotations

import re
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
from markdowncleaner import CleanerOptions, MarkdownCleaner

from ..types import ParsedDoc
from .base import ExtractionResult, Loader


def clean_rag_markdown(text: str) -> str:
    """Clean PDF-converted Markdown without removing meaningful sections."""

    options = CleanerOptions()
    options.fix_encoding_mojibake = True
    options.normalize_quotation_symbols = True
    options.contract_empty_lines = True
    options.crimp_linebreaks = True
    options.remove_duplicate_headlines = True

    options.remove_short_lines = False
    options.remove_sections = False
    options.remove_references_heuristically = False
    options.remove_footnotes_in_text = False

    cleaner = MarkdownCleaner(options=options)
    return cleaner.clean_markdown_string(text)


def clean_repeated_special_chars(text: str) -> str:
    """Normalize repeated punctuation noise commonly introduced by PDFs."""

    text = re.sub(r"(\\_){2,}", "______", text)
    text = re.sub(r"_{3,}", "______", text)
    text = re.sub(r"([ \t])\1{2,}", r"\1", text)

    replacements = {
        r"!{2,}": "!",
        r"@{2,}": "@",
        r"\${2,}": "$",
        r"%{2,}": "%",
        r"\^{2,}": "^",
        r"&{2,}": "&",
        r"={2,}": "=",
        r"\+{2,}": "+",
        r";{2,}": ";",
        r":{2,}": ":",
        r",{2,}": ",",
        r"\?{2,}": "?",
        r"/{2,}": "/",
        r"[\\]{2,}": "\\\\",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    return text


class PdfMarkdownLoader(Loader):
    """Extract a PDF to cleaned Markdown and load it as ParsedDoc text."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        clean: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.clean = clean

    def load(self, source: str) -> ParsedDoc:
        result = self.extract(source)
        return ParsedDoc(texts=[result.text])

    def extract(self, source: str | Path) -> ExtractionResult:
        pdf_path = Path(source).resolve()
        document_dir = self._document_dir(pdf_path)
        document_dir.mkdir(parents=True, exist_ok=True)

        markdown_path = document_dir / f"{pdf_path.stem}.md"
        artifacts_dir = document_dir / f"{pdf_path.stem}_images"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        result = self._build_converter().convert(str(pdf_path), raises_on_error=True)
        result.document.save_as_markdown(
            markdown_path,
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
            page_break_placeholder="\n\n---\n\n",
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=artifacts_dir,
        )

        markdown_text = markdown_path.read_text(encoding="utf-8")
        if self.clean:
            markdown_text = clean_repeated_special_chars(clean_rag_markdown(markdown_text))
            markdown_path.write_text(markdown_text, encoding="utf-8")

        return ExtractionResult(
            source_path=pdf_path,
            text_path=markdown_path,
            text=markdown_text,
            artifacts_dir=artifacts_dir,
            metadata={"loader": "pdf_markdown"},
        )

    def extract_batch(self, input_dir: str | Path) -> list[ExtractionResult]:
        """Extract every PDF in a folder, skipping files that fail."""

        results: list[ExtractionResult] = []
        for pdf_path in sorted(Path(input_dir).resolve().glob("*.pdf")):
            try:
                results.append(self.extract(pdf_path))
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
            do_picture_classification=True,
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
