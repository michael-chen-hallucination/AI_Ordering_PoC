"""Streaming speech recognition and telephony speech synthesis."""

from __future__ import annotations

import audioop
import base64
import io
import json
import time
from threading import Lock
from typing import Any

from gtts import gTTS
from pydub import AudioSegment


class SpeechError(RuntimeError):
    """Raised when speech recognition or synthesis cannot continue."""


class StreamingTranscriber:
    def __init__(
        self,
        recognizer: Any,
        *,
        silence_seconds: float,
        sample_rate: int = 16_000,
    ) -> None:
        if silence_seconds <= 0:
            raise ValueError("silence_seconds must be greater than zero")
        self._recognizer = recognizer
        self._silence_seconds = silence_seconds
        self._sample_rate = sample_rate
        self._pending_text = ""
        self._last_partial = ""
        self._last_speech_at: float | None = None

    def push_ulaw(self, payload: bytes, now: float | None = None) -> str | None:
        timestamp = time.monotonic() if now is None else now
        try:
            linear = audioop.ulaw2lin(payload, 2)
            if self._sample_rate != 8_000:
                linear = audioop.ratecv(
                    linear, 2, 1, 8_000, self._sample_rate, None
                )[0]
            accepted = self._recognizer.AcceptWaveform(linear)
            raw_result = (
                self._recognizer.Result()
                if accepted
                else self._recognizer.PartialResult()
            )
            result = json.loads(raw_result)
        except (audioop.error, json.JSONDecodeError, TypeError, ValueError):
            raise SpeechError("invalid recognizer output") from None
        except Exception:
            raise SpeechError("speech recognition failed") from None

        if not isinstance(result, dict):
            raise SpeechError("invalid recognizer output")

        if accepted:
            final_text = result.get("text", "")
            if not isinstance(final_text, str):
                raise SpeechError("invalid recognizer output")
            final_text = final_text.strip()
            if final_text:
                self._pending_text = final_text
                self._last_speech_at = timestamp
                self._last_partial = ""
        else:
            partial = result.get("partial", "")
            if not isinstance(partial, str):
                raise SpeechError("invalid recognizer output")
            partial = partial.strip()
            if partial and partial != self._last_partial:
                self._last_speech_at = timestamp
            self._last_partial = partial

        if (
            self._pending_text
            and self._last_speech_at is not None
            and timestamp - self._last_speech_at >= self._silence_seconds
        ):
            text = self._pending_text
            self._pending_text = ""
            self._last_partial = ""
            self._last_speech_at = None
            return text
        return None

    def close(self) -> None:
        try:
            self._recognizer.FinalResult()
        except Exception:
            raise SpeechError("speech recognizer cleanup failed") from None


class VoskTranscriberFactory:
    def __init__(
        self,
        model_path: str,
        *,
        silence_seconds: float = 1.0,
        sample_rate: int = 16_000,
    ) -> None:
        self._model_path = model_path
        self._silence_seconds = silence_seconds
        self._sample_rate = sample_rate
        self._model: Any | None = None
        self._model_lock = Lock()

    def create(self) -> StreamingTranscriber:
        try:
            import vosk
        except (ImportError, ModuleNotFoundError):
            raise SpeechError(
                'Vosk is not installed; install the "[speech]" extra'
            ) from None

        try:
            with self._model_lock:
                if self._model is None:
                    self._model = vosk.Model(self._model_path)
            recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
        except Exception:
            raise SpeechError("Vosk model could not be loaded") from None

        return StreamingTranscriber(
            recognizer,
            silence_seconds=self._silence_seconds,
            sample_rate=self._sample_rate,
        )


class SpeechSynthesizer:
    def __init__(self, *, language: str = "en") -> None:
        self._language = language

    def synthesize_ulaw(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            raise ValueError("text cannot be empty")
        try:
            mp3_buffer = io.BytesIO()
            gTTS(text=normalized, lang=self._language).write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            audio = AudioSegment.from_file(mp3_buffer, format="mp3")
            audio = audio.set_frame_rate(8_000).set_channels(1).set_sample_width(2)
            mulaw = audioop.lin2ulaw(audio.raw_data, 2)
            return base64.b64encode(mulaw).decode("ascii")
        except Exception:
            raise SpeechError("speech synthesis failed") from None

