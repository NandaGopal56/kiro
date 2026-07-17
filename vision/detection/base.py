from abc import ABC, abstractmethod

from vision.common.types import Detection


class ObjectDetector(ABC):

    @abstractmethod
    def detect(self, frame) -> list[Detection]:
        pass