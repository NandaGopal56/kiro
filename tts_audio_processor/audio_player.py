import threading
import queue
import simpleaudio as sa
from pydub import AudioSegment
from stt_audio_processor.audio_handler import audio_handler
from .logger import logger

# Shared queue
audio_queue: "queue.Queue[AudioSegment]" = queue.Queue()

# Flag to stop playback
stop_playback = threading.Event()


def playback_worker():
    """Continuously play audio from queue."""

    logger.info('Starting playback worker')

    while not stop_playback.is_set():
        logger.debug('Waiting for audio to play')
        try:
            segment = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        
        # Mute the audio handler while playing TTS audio
        audio_handler.mute()
        logger.info('Muted audio handler')

        # Play the TTS audio
        play_obj = sa.play_buffer(
            segment.raw_data,
            num_channels=segment.channels,
            bytes_per_sample=segment.sample_width,
            sample_rate=segment.frame_rate,
        )
        play_obj.wait_done()
        logger.info('Played TTS audio')

        # Unmute the audio handler after playing TTS audio
        audio_handler.unmute()
        logger.info('Unmuted audio handler')
        audio_queue.task_done()

    logger.info('Playback worker stopped')

# Start playback thread immediately when module is imported
threading.Thread(target=playback_worker, daemon=True).start()
