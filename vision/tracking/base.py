from abc import ABC, abstractmethod

from vision.common.types import Detection, Track


class ObjectTracker(ABC):

    @abstractmethod
    def update(
        self,
        frame,
        detections: list[Detection],
    ) -> list[Track]:
        pass