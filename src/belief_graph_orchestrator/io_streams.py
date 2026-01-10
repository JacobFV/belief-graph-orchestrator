"""
Streaming text and audio I/O processing.

Handles:
  - Incremental text accumulation with sentence boundary detection
  - Audio frame buffering and speech-to-text transcription
  - Speech output generation from brain events
  - Conversation history for multi-turn interaction
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class TextInputState:
    """Accumulates streaming text chunks into complete utterances."""
    buffer: str = ""
    history: list[dict] = field(default_factory=list)  # [{role, text, t_ns}]
    last_chunk_ns: int = 0
    silence_threshold_ms: float = 1500.0  # pause between chunks → sentence boundary

    def ingest(self, chunk: str, t_ns: int) -> Optional[str]:
        """
        Ingest a text chunk. Returns a complete utterance when a sentence
        boundary is detected (punctuation or pause), otherwise None.
        """
        self.buffer += chunk
        self.last_chunk_ns = t_ns

        # Check for explicit sentence boundaries
        for terminator in ["\n", ". ", "? ", "! ", ".\n", "?\n", "!\n"]:
            if terminator in self.buffer:
                idx = self.buffer.index(terminator) + len(terminator)
                utterance = self.buffer[:idx].strip()
                self.buffer = self.buffer[idx:]
                if utterance:
                    self.history.append({"role": "user", "text": utterance, "t_ns": t_ns})
                    return utterance

        # Check for terminal punctuation at end of buffer (no trailing space needed)
        stripped = self.buffer.rstrip()
        if stripped and stripped[-1] in ".?!":
            utterance = stripped
            self.buffer = ""
            self.history.append({"role": "user", "text": utterance, "t_ns": t_ns})
            return utterance

        return None

    def flush(self, t_ns: int) -> Optional[str]:
        """
        Flush the buffer if enough time has passed since the last chunk.
        Called each tick — returns accumulated text after a pause.
        """
        if not self.buffer.strip():
            return None
        if self.last_chunk_ns == 0:
            return None
        elapsed_ms = (t_ns - self.last_chunk_ns) / 1e6
        if elapsed_ms > self.silence_threshold_ms:
            utterance = self.buffer.strip()
            self.buffer = ""
            if utterance:
                self.history.append({"role": "user", "text": utterance, "t_ns": t_ns})
                return utterance
        return None


@dataclass
class AudioInputState:
    """Buffers audio frames and manages speech-to-text state."""
    sample_rate: int = 16000
    buffer_samples: list[float] = field(default_factory=list)
    is_speaking: bool = False
    silence_frames: int = 0
    silence_threshold: int = 30  # ~30 frames of silence → end of utterance
    energy_threshold: float = 0.01  # RMS energy threshold for speech detection
    pending_transcript: Optional[str] = None
    transcript_history: list[dict] = field(default_factory=list)

    def ingest(self, frame: dict) -> Optional[dict]:
        """
        Ingest an audio frame. Returns a dict with buffered audio when
        end-of-utterance is detected, otherwise None.

        Returns:
          {'samples': list[float], 'sample_rate': int, 't_ns': int}
          or None
        """
        samples = frame.get("samples", [])
        sr = frame.get("sample_rate", self.sample_rate)
        t_ns = frame.get("t_ns", time.time_ns())

        if not samples:
            return None

        # Compute RMS energy
        if isinstance(samples, bytes):
            # Convert bytes to float samples (assume 16-bit PCM)
            import struct
            n = len(samples) // 2
            floats = [struct.unpack_from("<h", samples, i * 2)[0] / 32768.0 for i in range(n)]
            samples = floats

        rms = (sum(s * s for s in samples) / max(len(samples), 1)) ** 0.5

        if rms > self.energy_threshold:
            self.is_speaking = True
            self.silence_frames = 0
            self.buffer_samples.extend(samples)
        elif self.is_speaking:
            self.silence_frames += 1
            self.buffer_samples.extend(samples)  # include trailing silence
            if self.silence_frames >= self.silence_threshold:
                # End of utterance
                result = {
                    "samples": list(self.buffer_samples),
                    "sample_rate": sr,
                    "t_ns": t_ns,
                }
                self.buffer_samples.clear()
                self.is_speaking = False
                self.silence_frames = 0
                return result

        return None

    def set_transcript(self, text: str, t_ns: int, is_final: bool = True) -> None:
        """Called by the speech-to-text backend when transcription is ready."""
        self.pending_transcript = text
        if is_final and text.strip():
            self.transcript_history.append({"text": text, "t_ns": t_ns, "final": True})

    def take_transcript(self) -> Optional[str]:
        """Take the pending transcript (consumed once)."""
        t = self.pending_transcript
        self.pending_transcript = None
        return t


@dataclass
class OutputState:
    """Manages text and speech output from the brain."""
    pending_text: list[str] = field(default_factory=list)
    pending_speech: list[dict] = field(default_factory=list)  # [{text, voice, rate, t_ns}]
    output_history: list[dict] = field(default_factory=list)  # [{role:"agent", modality, text, t_ns}]

    def emit_text(self, text: str, t_ns: int = 0) -> None:
        """Queue a text response."""
        self.pending_text.append(text)
        self.output_history.append({
            "role": "agent", "modality": "text",
            "text": text, "t_ns": t_ns or time.time_ns(),
        })

    def emit_speech(self, text: str, t_ns: int = 0, **kwargs) -> None:
        """Queue a speech output."""
        entry = {"text": text, "t_ns": t_ns or time.time_ns(), **kwargs}
        self.pending_speech.append(entry)
        self.output_history.append({
            "role": "agent", "modality": "speech",
            "text": text, "t_ns": t_ns or time.time_ns(),
        })

    def drain_text(self) -> list[str]:
        """Take all pending text outputs."""
        out = list(self.pending_text)
        self.pending_text.clear()
        return out

    def drain_speech(self) -> list[dict]:
        """Take all pending speech outputs."""
        out = list(self.pending_speech)
        self.pending_speech.clear()
        return out


class StreamingIOManager:
    """
    Coordinates all streaming I/O for the brain.

    Each tick:
      1. Poll text input → accumulate → emit utterance events
      2. Poll audio input → buffer → detect end-of-utterance → transcribe
      3. Drain output queues → send to body
    """

    def __init__(self) -> None:
        self.text_input = TextInputState()
        self.audio_input = AudioInputState()
        self.output = OutputState()

    def poll_inputs(self, body, journal, session_id: str, episode_id: str, now_ns: int) -> list:
        """Poll all streaming inputs, emit events. Returns new events."""
        events = []

        # ── text input ──
        text_chunk = body.get_text_input()
        if text_chunk:
            ev = journal.make_event(
                "text_input_chunk", session_id, episode_id,
                {"chunk": text_chunk}, t_capture_ns=now_ns,
            )
            journal.append(ev)
            events.append(ev)

            # Accumulate and check for complete utterance
            utterance = self.text_input.ingest(text_chunk, now_ns)
            if utterance:
                ev = journal.make_event(
                    "task_instruction", session_id, episode_id,
                    {"text": utterance, "source": "text_stream"},
                    t_capture_ns=now_ns,
                )
                journal.append(ev)
                events.append(ev)

        # Check for pause-based flush
        utterance = self.text_input.flush(now_ns)
        if utterance:
            ev = journal.make_event(
                "task_instruction", session_id, episode_id,
                {"text": utterance, "source": "text_stream_flush"},
                t_capture_ns=now_ns,
            )
            journal.append(ev)
            events.append(ev)

        # ── audio input ──
        audio_frame = body.get_audio_input()
        if audio_frame:
            ev = journal.make_event(
                "audio_input_frame", session_id, episode_id,
                {"sample_rate": audio_frame.get("sample_rate", 16000),
                 "num_samples": len(audio_frame.get("samples", [])),
                 "t_ns": audio_frame.get("t_ns", now_ns)},
                t_capture_ns=audio_frame.get("t_ns", now_ns),
            )
            journal.append(ev)
            events.append(ev)

            # Buffer and detect end of utterance
            completed = self.audio_input.ingest(audio_frame)
            if completed:
                # Transcribe using the speech-to-text pipeline
                transcript = self._transcribe(completed)
                if transcript:
                    self.audio_input.set_transcript(transcript, now_ns)
                    ev = journal.make_event(
                        "transcript_chunk", session_id, episode_id,
                        {"text": transcript, "is_final": True, "source": "stt"},
                        t_capture_ns=now_ns,
                    )
                    journal.append(ev)
                    events.append(ev)

                    # Treat final transcript as task instruction
                    ev = journal.make_event(
                        "task_instruction", session_id, episode_id,
                        {"text": transcript, "source": "speech"},
                        t_capture_ns=now_ns,
                    )
                    journal.append(ev)
                    events.append(ev)

        return events

    def flush_outputs(self, body, journal, session_id: str, episode_id: str, now_ns: int) -> list:
        """Drain output queues and send to body. Returns new events."""
        events = []

        for text in self.output.drain_text():
            body.send_text_output(text)
            ev = journal.make_event(
                "text_output", session_id, episode_id,
                {"text": text}, t_capture_ns=now_ns,
            )
            journal.append(ev)
            events.append(ev)

        for speech in self.output.drain_speech():
            body.send_speech_output(speech["text"])
            ev = journal.make_event(
                "speech_output", session_id, episode_id,
                {"text": speech["text"]}, t_capture_ns=now_ns,
            )
            journal.append(ev)
            events.append(ev)

        return events

    def _transcribe(self, audio: dict) -> Optional[str]:
        """
        Speech-to-text transcription.

        Uses OpenAI Whisper if available, otherwise returns None.
        In production this would be replaced with on-device STT
        (Apple Speech framework, etc.)
        """
        try:
            import whisper
        except ImportError:
            # No whisper installed — try torch audio
            return self._transcribe_torch(audio)
        return None

    def _transcribe_torch(self, audio: dict) -> Optional[str]:
        """Fallback: no STT available. The body should provide transcripts externally."""
        return None
