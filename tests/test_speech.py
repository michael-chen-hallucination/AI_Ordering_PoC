import base64
import json
import sys
from collections import deque
from types import SimpleNamespace

import pytest

import ai_ordering.adapters.speech as speech_module
from ai_ordering.adapters.speech import (
    SpeechError,
    SpeechSynthesizer,
    StreamingTranscriber,
    VoskTranscriberFactory,
)


class ScriptedRecognizer:
    def __init__(self, events):
        self.events = deque(events)
        self.current = None
        self.audio = []
        self.closed = False

    def AcceptWaveform(self, audio):
        self.audio.append(audio)
        self.current = self.events.popleft()
        return self.current[0]

    def Result(self):
        return self.current[1]

    def PartialResult(self):
        return self.current[1]

    def FinalResult(self):
        self.closed = True
        return json.dumps({"text": ""})


def test_final_text_is_returned_only_after_the_silence_threshold():
    recognizer = ScriptedRecognizer(
        [
            (True, json.dumps({"text": "two cokes"})),
            (False, json.dumps({"partial": ""})),
            (False, json.dumps({"partial": ""})),
        ]
    )
    transcriber = StreamingTranscriber(recognizer, silence_seconds=1.0)

    assert transcriber.push_ulaw(b"\xff" * 160, now=10.0) is None
    assert transcriber.push_ulaw(b"\xff" * 160, now=10.5) is None
    assert transcriber.push_ulaw(b"\xff" * 160, now=11.1) == "two cokes"


def test_new_partial_speech_resets_the_silence_clock():
    recognizer = ScriptedRecognizer(
        [
            (False, json.dumps({"partial": "two"})),
            (True, json.dumps({"text": "two cokes"})),
            (False, json.dumps({"partial": ""})),
            (False, json.dumps({"partial": ""})),
        ]
    )
    transcriber = StreamingTranscriber(recognizer, silence_seconds=1.0)

    assert transcriber.push_ulaw(b"\xff" * 160, now=1.0) is None
    assert transcriber.push_ulaw(b"\xff" * 160, now=1.8) is None
    assert transcriber.push_ulaw(b"\xff" * 160, now=2.5) is None
    assert transcriber.push_ulaw(b"\xff" * 160, now=2.9) == "two cokes"


def test_twilio_audio_is_decoded_and_resampled_before_recognition():
    recognizer = ScriptedRecognizer([(False, json.dumps({"partial": "listening"}))])
    transcriber = StreamingTranscriber(
        recognizer, silence_seconds=1.0, sample_rate=16_000
    )

    transcriber.push_ulaw(b"\xff" * 160, now=1.0)

    assert len(recognizer.audio) == 1
    assert len(recognizer.audio[0]) > 160


def test_blank_final_result_is_not_emitted():
    recognizer = ScriptedRecognizer(
        [
            (True, json.dumps({"text": "   "})),
            (False, json.dumps({"partial": ""})),
        ]
    )
    transcriber = StreamingTranscriber(recognizer, silence_seconds=0.1)

    assert transcriber.push_ulaw(b"\xff" * 160, now=1.0) is None
    assert transcriber.push_ulaw(b"\xff" * 160, now=2.0) is None


def test_malformed_recognizer_json_is_mapped():
    recognizer = ScriptedRecognizer([(True, "not-json")])
    transcriber = StreamingTranscriber(recognizer, silence_seconds=1.0)

    with pytest.raises(SpeechError, match="recognizer output"):
        transcriber.push_ulaw(b"\xff" * 160, now=1.0)


def test_close_finalizes_the_recognizer():
    recognizer = ScriptedRecognizer([])

    StreamingTranscriber(recognizer, silence_seconds=1.0).close()

    assert recognizer.closed is True


def test_vosk_model_is_loaded_lazily_and_reused(monkeypatch):
    calls = []

    def model(path):
        calls.append(("model", path))
        return "loaded-model"

    def recognizer(loaded_model, sample_rate):
        calls.append(("recognizer", loaded_model, sample_rate))
        return ScriptedRecognizer([])

    monkeypatch.setitem(
        sys.modules,
        "vosk",
        SimpleNamespace(Model=model, KaldiRecognizer=recognizer),
    )
    factory = VoskTranscriberFactory("models/test", sample_rate=16_000)
    assert calls == []

    factory.create()
    factory.create()

    assert calls == [
        ("model", "models/test"),
        ("recognizer", "loaded-model", 16_000),
        ("recognizer", "loaded-model", 16_000),
    ]


def test_missing_vosk_dependency_has_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vosk", None)

    with pytest.raises(SpeechError, match=r"\[speech\]"):
        VoskTranscriberFactory("models/test").create()


class FakeTTS:
    def __init__(self, *, text, lang):
        assert text == "Order confirmed"
        assert lang == "en"

    def write_to_fp(self, destination):
        destination.write(b"fake-mp3")


class FakeAudio:
    def __init__(self):
        self.raw_data = b"\x00\x00\xff\x7f"

    def set_frame_rate(self, value):
        assert value == 8_000
        return self

    def set_channels(self, value):
        assert value == 1
        return self

    def set_sample_width(self, value):
        assert value == 2
        return self


def test_synthesizer_returns_base64_mulaw_without_temporary_files(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(speech_module, "gTTS", FakeTTS)
    monkeypatch.setattr(
        speech_module.AudioSegment,
        "from_file",
        lambda source, format: FakeAudio(),
    )

    encoded = SpeechSynthesizer().synthesize_ulaw("Order confirmed")

    assert len(base64.b64decode(encoded)) == 2
    assert list(tmp_path.iterdir()) == []


def test_synthesis_failure_does_not_leak_provider_details(monkeypatch):
    class FailingTTS:
        def __init__(self, **kwargs):
            raise RuntimeError("private provider response")

    monkeypatch.setattr(speech_module, "gTTS", FailingTTS)

    with pytest.raises(SpeechError, match="synthesis failed") as error:
        SpeechSynthesizer().synthesize_ulaw("Order confirmed")

    assert "private provider response" not in str(error.value)
