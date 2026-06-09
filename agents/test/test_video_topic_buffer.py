import agents.video_topic_buffer as video_topic_buffer
from agents.video_topic_buffer import VideoTopicBuffer


def test_latest_returns_most_recent_payload():
    buffer = VideoTopicBuffer(window_seconds=60)

    buffer.on_frame("camera/front", {"frame": 1})
    buffer.on_frame("camera/front", {"frame": 2})

    assert buffer.latest() == {"frame": 2}


def test_window_prunes_old_frames(monkeypatch):
    current_time = 1000.0
    monkeypatch.setattr(video_topic_buffer.time, "time", lambda: current_time)
    buffer = VideoTopicBuffer(window_seconds=10)

    buffer.on_frame("camera/front", "old")
    current_time = 1012.0
    buffer.on_frame("camera/front", "new")

    assert list(buffer.buffer) == [(1012.0, "new")]
    assert buffer.latest() == "new"


def test_clip_returns_frames_inside_requested_window(monkeypatch):
    current_time = 1000.0
    monkeypatch.setattr(video_topic_buffer.time, "time", lambda: current_time)
    buffer = VideoTopicBuffer(window_seconds=60)

    buffer.on_frame("camera/front", "first")
    current_time = 1020.0
    buffer.on_frame("camera/front", "second")

    assert buffer.clip(seconds=10) == ["second"]
    assert buffer.clip(seconds=30) == ["first", "second"]
