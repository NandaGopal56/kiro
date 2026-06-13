from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class ToolEnum(str, Enum):
    INTERNET_SEARCH = "internet_search"
    VIDEO_CAPTURE = "video_capture"
    DOCUMENT_RAG = "document_rag"


class ToolClassifierOutput(BaseModel):
    tools: List[ToolEnum] = Field(
        default_factory=list,
        description="List of required tools. Empty if none."
    )