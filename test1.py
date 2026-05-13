import asyncio
import base64
import os
import time
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeEvents,
    RealtimeAudioOptions,
)

load_dotenv()

SAMPLE_RATE = 16000
FRAME_MS = 200
FRAMES = int(SAMPLE_RATE * FRAME_MS / 1000)



async def main():
    loop = asyncio.get_running_loop()

    elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    connection = await elevenlabs.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id="scribe_v2_realtime",
            audio_format=AudioFormat.PCM_16000,
            sample_rate=SAMPLE_RATE,
            commit_strategy=CommitStrategy.MANUAL,
            include_timestamps=True,
        )
    )

    audio_queue = asyncio.Queue()
    ready = asyncio.Event()

    # ---------- EVENTS ----------

    def on_session_started(_):
        print("Session started. Speak now.")
        ready.set()

    # ---------- CLEAN PRINT STATE ----------
    current_line = ""

    def on_partial_transcript(data):
        global current_line
        text = data.get("text", "")
        if not text:
            return
        current_line = text
        print("\r" + current_line, end="", flush=True)

    def on_committed_transcript(data):
        global current_line
        final_text = data.get("text", "")
        if not final_text:
            return
        print("\r" + " " * len(current_line), end="\r")
        print(final_text)
        current_line = ""


    def on_timestamps(_):
        # ignore completely
        pass

    connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial_transcript)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed_transcript)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS, on_timestamps)

    # ---------- MIC CALLBACK (thread) ----------

    def mic_callback(indata, frames, time_info, status):
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAMES,
        callback=mic_callback,
    )

    # ---------- SENDER (async loop) ----------

    async def sender():
        await ready.wait()
        mic_stream.start()

        last_voice = time.time()

        def is_silence(chunk):
            return np.abs(chunk).mean() < 500

        while True:
            chunk = await audio_queue.get()

            b64 = base64.b64encode(chunk.tobytes()).decode()

            await connection.send({
                "audio_base_64": b64,
                "sample_rate": SAMPLE_RATE,
            })

            if not is_silence(chunk):
                last_voice = time.time()
            else:
                if (time.time() - last_voice) * 1000 > 900:
                    await connection.commit()
                    last_voice = time.time()

    asyncio.create_task(sender())

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())