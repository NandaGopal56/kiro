from vision.tracking.dummy import DummyTracker


def create_tracker(name: str):

    name = name.lower()

    if name == "dummy":
        return DummyTracker()

    raise ValueError(f"Unsupported tracker: {name}")