from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(slots=True)
class ParsedDoc:
    """Normalized loader output before chunking."""

    texts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {"texts": self.texts, "images": self.images, "tables": self.tables}

    def all_texts(
        self,
        include_images: bool = True,
        include_tables: bool = True,
    ) -> list[str]:
        texts = list(self.texts)
        if include_images:
            texts.extend(self.images)
        if include_tables:
            texts.extend(self.tables)
        return texts
