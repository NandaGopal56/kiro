"""
Unified input schema.

Every input surface (text, audio, future modalities) produces an ``InputEvent``.
The agent core never sees modality-specific types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class Modality(str, Enum):
    TEXT  = "text"
    AUDIO = "audio"


@dataclass
class InputEvent:
    """A finalized, agent-ready input regardless of how it was captured."""

    text:      str
    modality:  Modality           = Modality.TEXT
    timestamp: float              = field(default_factory=time.monotonic)
    language:  Optional[str]      = None   # e.g. "en-IN" from STT
    metadata:  dict               = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.modality.value}] {self.text!r}"