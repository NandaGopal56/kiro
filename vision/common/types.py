from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class Identity:
    """Recognition result for a tracked object (which instance / description)."""

    track_id: int
    label: str
    confidence: float = 1.0
    description: str = ""


@dataclass(slots=True)
class FrameResult:
    """Unified output of the vision pipeline for a single frame."""

    frame: object = None
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    identities: list[Identity] = field(default_factory=list)
    caption: str = ""
