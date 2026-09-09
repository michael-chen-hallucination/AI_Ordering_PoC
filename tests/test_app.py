import json
from collections import deque

import pytest

from ai_ordering.adapters.bedrock import BedrockError
from ai_ordering.app import MediaStreamHandler, create_app
from ai_ordering.config import Settings
from ai_ordering.models import AssistantTurn


class RecordingConversationService:
    def __init__(self, response=None, error=None):
        self.response = response or AssistantTurn("Anything else?")
        self.error = error
        self.started = []
        self.transcripts = []
        self.stopped = []

    def start(self, stream_id):
        self.started.append(stream_id)
        return AssistantTurn("Welcome! What can I get for you?")

    def handle_transcript(self, stream_id, transcript):
        self.transcripts.append((stream_id, transcript))
        if self.error:
            raise self.error
        return self.response

    def stop(self, stream_id):
        self.stopped.append(stream_id)


class ScriptedTranscriber:
    def __init__(self, results):
        self.results = deque(results)
        self.payloads = []
        self.closed = False

    def push_ulaw(self, payload):
        self.payloads.append(payload)
        return self.results.popleft()

    def close(self):
        self.closed = True


class QueueTranscriberFactory:
    def __init__(self, transcribers):
        self.transcribers = deque(transcribers)
        self.created = []

    def create(self):
        transcriber = self.transcribers.popleft()
        self.created.append(transcriber)
        return transcriber


class RecordingSynthesizer:
    def __init__(self):
        self.texts = []

    def synthesize_ulaw(self, text):
        self.texts.append(text)
        return f"encoded:{text}"


@pytest.fixture
def transport():
    service = RecordingConversationService()
    transcriber = ScriptedTranscriber(["two cokes"])
    factory = QueueTranscriberFactory([transcriber])
    synthesizer = RecordingSynthesizer()
    handler = MediaStreamHandler(service, factory, synthesizer)
    return handler, service, transcriber, factory, synthesizer


def start_event(stream_id="call-a"):
    return json.dumps(
        {"event": "start", "streamSid": stream_id, "start": {"streamSid": stream_id}}
    )


def media_event(stream_id="call-a", payload="/w=="):
    return json.dumps(
        {
            "event": "media",
            "streamSid": stream_id,
            "media": {"payload": payload},
        }
    )


def stop_event(stream_id="call-a"):
    return json.dumps({"event": "stop", "streamSid": stream_id})


def decoded_message(message):
    return json.loads(message)


def test_health_endpoint_does_not_require_cloud_credentials():
    app = create_app(Settings.from_env({}))

    response = app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_call_webhook_returns_a_bidirectional_media_stream():
    settings = Settings.from_env(
        {"PUBLIC_WEBSOCKET_URL": "wss://voice.example.com/media"}
    )
    app = create_app(settings)

    response = app.test_client().post("/call")

    assert response.status_code == 200
    assert response.content_type.startswith("text/xml")
    assert b"<Connect>" in response.data
    assert b"wss://voice.example.com/media" in response.data


def test_call_webhook_derives_the_stream_url_from_request_host():
    app = create_app(Settings.from_env({}))

    response = app.test_client().post("/call", base_url="https://public.example.com")

    assert b"wss://public.example.com/media" in response.data


def test_start_creates_a_session_and_returns_welcome_audio(transport):
    handler, service, _, _, synthesizer = transport

    outbound = handler.handle(start_event())

    assert service.started == ["call-a"]
    assert synthesizer.texts == ["Welcome! What can I get for you?"]
    assert decoded_message(outbound[0]) == {
        "event": "media",
        "streamSid": "call-a",
        "media": {"payload": "encoded:Welcome! What can I get for you?"},
    }


def test_media_audio_is_transcribed_and_answered(transport):
    handler, service, transcriber, _, synthesizer = transport
    handler.handle(start_event())

    outbound = handler.handle(media_event())

    assert transcriber.payloads == [b"\xff"]
    assert service.transcripts == [("call-a", "two cokes")]
    assert synthesizer.texts[-1] == "Anything else?"
    assert decoded_message(outbound[0])["media"]["payload"] == (
        "encoded:Anything else?"
    )


def test_partial_transcript_produces_no_outbound_audio():
    service = RecordingConversationService()
    transcriber = ScriptedTranscriber([None])
    handler = MediaStreamHandler(
        service,
        QueueTranscriberFactory([transcriber]),
        RecordingSynthesizer(),
    )
    handler.handle(start_event())

    assert handler.handle(media_event()) == []
    assert service.transcripts == []


def test_stop_closes_transcriber_and_conversation_session(transport):
    handler, service, transcriber, _, _ = transport
    handler.handle(start_event())

    assert handler.handle(stop_event()) == []

    assert transcriber.closed is True
    assert service.stopped == ["call-a"]
    assert handler.active_stream_count == 0


def test_two_streams_receive_distinct_transcribers():
    first = ScriptedTranscriber([None])
    second = ScriptedTranscriber([None])
    factory = QueueTranscriberFactory([first, second])
    handler = MediaStreamHandler(
        RecordingConversationService(), factory, RecordingSynthesizer()
    )

    handler.handle(start_event("call-a"))
    handler.handle(start_event("call-b"))
    handler.handle(media_event("call-a"))
    handler.handle(media_event("call-b"))

    assert first.payloads == [b"\xff"]
    assert second.payloads == [b"\xff"]


@pytest.mark.parametrize(
    "message",
    [
        "not-json",
        "[]",
        json.dumps({"event": "start", "start": {}}),
        json.dumps({"event": "media", "streamSid": "unknown"}),
        media_event(payload="not-base64"),
        json.dumps({"event": "unexpected", "streamSid": "call-a"}),
    ],
)
def test_malformed_or_unknown_events_are_ignored(transport, message):
    handler, _, _, _, _ = transport

    assert handler.handle(message) == []


def test_known_processing_error_returns_a_safe_retry_message():
    service = RecordingConversationService(
        error=BedrockError("private provider details")
    )
    transcriber = ScriptedTranscriber(["two cokes"])
    synthesizer = RecordingSynthesizer()
    handler = MediaStreamHandler(
        service, QueueTranscriberFactory([transcriber]), synthesizer
    )
    handler.handle(start_event())

    outbound = handler.handle(media_event())

    assert synthesizer.texts[-1] == (
        "Sorry, I couldn't process that. Please try again."
    )
    assert "private provider details" not in outbound[0]


def test_disconnect_is_safe_for_an_unknown_stream(transport):
    handler, service, _, _, _ = transport

    handler.disconnect("missing")

    assert service.stopped == ["missing"]
