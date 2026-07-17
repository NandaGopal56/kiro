from dataclasses import dataclass


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
    bbox: tuple[int, int,int, int]