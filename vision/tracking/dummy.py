from vision.common.types import Detection, Track
from vision.tracking.base import ObjectTracker


class DummyTracker(ObjectTracker):

    def __init__(self):
        self.next_track_id = 1

    def update(
        self,
        frame,
        detections: list[Detection],
    ) -> list[Track]:

        tracks = []

        for det in detections:

            tracks.append(
                Track(
                    track_id=self.next_track_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )

            self.next_track_id += 1

        return tracks