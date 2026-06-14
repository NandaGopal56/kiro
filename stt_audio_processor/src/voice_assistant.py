"""
voice_processor.py
-------------------

The VoiceProcessor is the central coordinator for the voice pipeline. It
manages two operating modes — sleeping and active — and handles the
transitions between them.

Sleeping mode
-------------
In this mode the processor listens continuously for a wake word. Rather than
collecting fixed-length bursts and transcribing them in batches, it feeds
audio into a rolling overlap buffer. Each iteration sends a fresh chunk plus
some audio from just before it, so a wake word spoken across a chunk boundary
is never missed. Once a wake word is detected the processor switches to active
mode.

Active mode
-----------
In active mode the processor opens a single long-lived WebSocket session with
the transcription provider and keeps it open for the duration of the session.
This avoids the reconnection overhead that would otherwise create a gap in
coverage between utterances.

Within active mode the processor runs a sender and a receiver concurrently:

  Sender  — reads 100ms PCM chunks from the microphone, computes their RMS
            energy, and sends only the chunks that exceed the silence
            threshold. When the audio has been below the threshold for longer
            than EOU_SILENCE_TIMEOUT seconds, the sender raises an end-of-
            utterance (EOU) signal.

  Receiver — reads transcript events from the WebSocket queue and accumulates
             them into the current utterance. Events are printed as they
             arrive, so the output feels responsive even on longer recordings.

When the EOU fires, any audio that arrived during the handoff is captured
from the queue and replayed into the same open session so nothing is dropped.
The assembled utterance text is then emitted as a structured JSON event.

If no speech is detected for SESSION_IDLE_TIMEOUT seconds, the processor
closes the WebSocket session and returns to sleeping mode.

Silence detection
-----------------
PyAudio delivers audio continuously — including silence — so the processor
cannot rely on queue emptiness as a silence signal. Instead it measures the
RMS energy of every chunk. A chunk is considered silent when its RMS falls
below SILENCE_RMS_THRESHOLD. The EOU timer only counts down once at least
one speech chunk has been seen in the current utterance, so the system does
not fire spuriously before the user has started speaking.
"""

import asyncio
from collections.abc import Callable
import json
from datetime import datetime, timedelta

import numpy as np

from stt_audio_processor.utils.config import AUDIO_CONFIG
from stt_audio_processor.src.wake_word_detector import WakeWordDetector
from stt_audio_processor.src.transcription_providers import (
    SarvamTranscriptionProvider,
    BaseTranscriptionSession,
    TranscriptEvent,
)
from stt_audio_processor.src.audio_handler import audio_handler


EOU_SILENCE_TIMEOUT  = AUDIO_CONFIG.eou_silence_timeout
SESSION_IDLE_TIMEOUT = AUDIO_CONFIG.session_idle_timeout
WAKE_BURST_SECONDS   = AUDIO_CONFIG.wake_burst_seconds
SILENCE_RMS_THRESHOLD = AUDIO_CONFIG.silence_rms_threshold

# How much audio (in seconds) to carry over from the previous chunk into each
# wake-word transcription window so a word spoken across a boundary is caught.
WAKE_OVERLAP_SECONDS = 0.5


def _rms(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0


class VoiceProcessor:

    def __init__(
        self,
        provider=None,
        audio_handler=None,
        wake_word=None,
        on_utterance: Callable[[str], None] | None = None,
    ):
        self.provider      = provider or SarvamTranscriptionProvider()
        self.audio_handler = audio_handler or globals()["audio_handler"]
        self.wake_word     = wake_word or WakeWordDetector()
        self.on_utterance  = on_utterance
        self._is_running   = False

    async def run(self):
        await self.audio_handler.start()
        self._is_running = True
        print(f"[VP] Started | wake words: {self.wake_word.wake_words}")
        print(
            f"[VP] EOU silence: {EOU_SILENCE_TIMEOUT}s | "
            f"idle timeout: {SESSION_IDLE_TIMEOUT}s | "
            f"silence RMS threshold: {SILENCE_RMS_THRESHOLD}"
        )

        try:
            while self._is_running:
                await self._sleeping_phase()
                if self._is_running:
                    await self._active_phase()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        self._is_running = False
        await self.audio_handler.stop()
        print("[VP] Stopped")

    # ------------------------------------------------------------------
    # SLEEPING
    # ------------------------------------------------------------------

    async def _sleeping_phase(self):
        """
        Listen continuously for a wake word.

        Audio is collected into a rolling window that overlaps with the
        previous window by WAKE_OVERLAP_SECONDS. This means a wake word
        spoken right at the edge of two windows is still captured in full
        and recognised correctly.

        Once the transcription of any window contains a wake word, this
        method returns and the caller switches to active mode.
        """
        print(f"\n[Sleeping] Listening for wake words: {self.wake_word.wake_words}")

        overlap_bytes = b""
        overlap_size  = int(AUDIO_CONFIG.sample_rate * WAKE_OVERLAP_SECONDS) * 2  # int16 = 2 bytes/sample

        while self._is_running:
            # Collect one burst window
            window = await self._collect_burst(WAKE_BURST_SECONDS)
            if not window:
                await asyncio.sleep(0.05)
                continue

            # Prepend overlap from the previous window to catch boundary words
            audio_to_transcribe = overlap_bytes + window

            print(f"[Sleeping] Transcribing {len(audio_to_transcribe) // 2} samples...", end=" ", flush=True)
            text = await self._transcribe_burst(audio_to_transcribe)

            # Keep the tail of this window as the overlap for the next iteration
            overlap_bytes = window[-overlap_size:] if len(window) > overlap_size else window

            if not text:
                print("(no speech)")
                continue

            print(f'heard: "{text}"')
            if self.wake_word.check_for_wake_word(text):
                return  # switch to active mode

    # ------------------------------------------------------------------
    # ACTIVE
    # ------------------------------------------------------------------

    async def _active_phase(self):
        """
        Keep a single WebSocket session open and transcribe utterances one
        after another until the idle timeout expires.

        The session is opened once here and reused for every utterance.
        Between utterances, any audio that accumulated in the microphone
        queue while the previous turn was wrapping up is drained and
        replayed into the still-open session, so no speech is lost.
        """
        print(f"\n[Active] Session open (idle timeout: {SESSION_IDLE_TIMEOUT}s)")
        last_speech_time = datetime.now()

        async with self.provider.streaming_session() as session:
            while self._is_running:
                idle_seconds = (datetime.now() - last_speech_time).total_seconds()
                if idle_seconds >= SESSION_IDLE_TIMEOUT:
                    print(f"\n[Active] {SESSION_IDLE_TIMEOUT}s of silence — returning to sleep")
                    self.wake_word.deactivate()
                    return

                # Drain anything that arrived between utterances and replay it
                # so speech that started during the previous turn's teardown
                # is not lost.
                carried_over = self.audio_handler.drain()

                utterance, had_speech = await self._transcribe_until_eou(
                    session, replay_chunks=carried_over
                )

                if had_speech:
                    last_speech_time = datetime.now()

                if utterance.strip():
                    self._emit_event(utterance.strip())
                else:
                    await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Core — one utterance per call, reuses an existing WS session
    # ------------------------------------------------------------------

    async def _transcribe_until_eou(
        self,
        session: BaseTranscriptionSession,
        replay_chunks: list[bytes] | None = None,
    ) -> tuple[str, bool]:
        """
        Collect one complete utterance from the microphone and return it.

        This method runs a sender and a receiver concurrently on the
        provided session. The sender reads microphone chunks, checks their
        RMS energy, and forwards only the speech portions to the provider.
        The receiver accumulates transcript events from the provider queue.

        If replay_chunks is provided, those bytes are sent to the provider
        before any new microphone audio. This replays audio that was
        captured in the microphone queue between the end of the previous
        utterance and the start of this call, preventing any gap in coverage.

        Returns a tuple of (full_utterance_text, had_speech_flag).
        """
        sentence_parts: list[str] = []
        had_speech = False
        eou_event  = asyncio.Event()

        # ── Sender ────────────────────────────────────────────────────
        async def _sender():
            nonlocal had_speech
            last_speech_time = datetime.now()

            # Replay any audio that was buffered during the previous handoff
            if replay_chunks:
                for chunk in replay_chunks:
                    if _rms(chunk) > SILENCE_RMS_THRESHOLD:
                        await session.send_audio(chunk)
                        had_speech = True
                        last_speech_time = datetime.now()

            while not eou_event.is_set():
                chunk = await self.audio_handler.get_chunk()

                if chunk:
                    rms = _rms(chunk)

                    if rms > SILENCE_RMS_THRESHOLD:
                        await session.send_audio(chunk)
                        last_speech_time = datetime.now()
                        had_speech = True
                        # print(f"[Sender] speech  RMS={rms:.0f}", flush=True)
                    else:
                        if had_speech:
                            silent_for = (datetime.now() - last_speech_time).total_seconds()
                            # print(f"[Sender] silence RMS={rms:.0f} ({silent_for:.1f}s)", flush=True)
                            if silent_for >= EOU_SILENCE_TIMEOUT:
                                print(f"\n[Sender] {EOU_SILENCE_TIMEOUT}s silence → EOU")
                                eou_event.set()
                                return

                await asyncio.sleep(0.02)  # tighter loop for lower latency

        # ── Receiver ──────────────────────────────────────────────────
        async def _receiver():
            async for event in session.events():
                if event.text:
                    sentence_parts.append(event.text)

                if event.is_endpoint:
                    eou_event.set()

                if eou_event.is_set():
                    break

        sender_task   = asyncio.create_task(_sender())
        receiver_task = asyncio.create_task(_receiver())

        await sender_task
        eou_event.set()

        # Give the receiver a moment to flush any in-flight transcripts
        try:
            async with asyncio.timeout(1.0):
                await receiver_task
        except asyncio.TimeoutError:
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

        full = " ".join(sentence_parts)
        if full.strip():
            print(f"\n[EOU] Utterance: \"{full}\"")
        return full, had_speech

    # ------------------------------------------------------------------
    # Sleeping helpers
    # ------------------------------------------------------------------

    async def _collect_burst(self, duration: float) -> bytes:
        """
        Read from the microphone queue for the given number of seconds and
        return the concatenated PCM bytes. Returns an empty bytes object if
        nothing arrived.
        """
        buf      = bytearray()
        deadline = datetime.now() + timedelta(seconds=duration)
        while datetime.now() < deadline:
            chunk = await self.audio_handler.get_chunk()
            if chunk:
                buf.extend(chunk)
            await asyncio.sleep(0.02)
        return bytes(buf)

    async def _transcribe_burst(self, audio_bytes: bytes) -> str:
        """
        Open a short-lived WebSocket session, send the given PCM bytes all
        at once, and return the transcript. Used only during sleeping mode
        for wake-word detection; active mode uses a persistent session instead.
        """
        parts: list[str] = []
        async with self.provider.streaming_session() as session:
            await session.send_audio(audio_bytes)
            try:
                async with asyncio.timeout(WAKE_BURST_SECONDS + 1.0):
                    async for event in session.events():
                        if event.text:
                            parts.append(event.text)
                        if event.is_endpoint:
                            break
            except asyncio.TimeoutError:
                pass
        return " ".join(parts).strip()

    def _emit_event(self, utterance: str):
        payload = {
            "event": "utterance_complete",
            "timestamp": datetime.now().isoformat(),
            "utterance": utterance,
        }
        print("\n" + "=" * 55)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("=" * 55 + "\n")
        if self.on_utterance is not None:
            self.on_utterance(utterance)
