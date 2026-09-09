"""Flask and Twilio transport for the AI voice ordering PoC."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from threading import Lock
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_sock import ConnectionClosed, Sock
from twilio.twiml.voice_response import Connect, VoiceResponse

from ai_ordering.adapters.bedrock import BedrockAssistant, BedrockError
from ai_ordering.adapters.order_gateway import HttpOrderGateway, OrderGatewayError
from ai_ordering.adapters.speech import (
    SpeechError,
    SpeechSynthesizer,
    VoskTranscriberFactory,
)
from ai_ordering.config import Settings
from ai_ordering.conversation import ConversationService, UnknownSessionError
from ai_ordering.models import ModelOutputError

SAFE_RETRY_MESSAGE = "Sorry, I couldn't process that. Please try again."
KNOWN_PROCESSING_ERRORS = (
    BedrockError,
    ModelOutputError,
    OrderGatewayError,
    SpeechError,
    UnknownSessionError,
)


class MediaStreamHandler:
    """Translate Twilio Media Stream events into application operations."""

    def __init__(
        self,
        conversation_service: Any,
        transcriber_factory: Any,
        synthesizer: Any,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._conversation_service = conversation_service
        self._transcriber_factory = transcriber_factory
        self._synthesizer = synthesizer
        self._logger = logger or logging.getLogger(__name__)
        self._transcribers: dict[str, Any] = {}
        self._lock = Lock()

    @property
    def active_stream_count(self) -> int:
        with self._lock:
            return len(self._transcribers)

    def handle(self, message: str) -> list[str]:
        try:
            event = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            self._logger.warning("ignored malformed media stream message")
            return []
        if not isinstance(event, dict):
            return []

        event_type = event.get("event")
        stream_id = self._stream_id(event)
        if not isinstance(event_type, str) or not stream_id:
            return []

        try:
            if event_type == "start":
                return self._start(stream_id)
            if event_type == "media":
                return self._media(stream_id, event)
            if event_type == "stop":
                self.disconnect(stream_id)
        except KNOWN_PROCESSING_ERRORS:
            self._logger.warning(
                "media stream event failed",
                extra={"event_type": event_type, "stream_id": stream_id},
            )
            if event_type == "media" and self._has_stream(stream_id):
                return self._safe_retry(stream_id)
        return []

    def disconnect(self, stream_id: str) -> None:
        with self._lock:
            transcriber = self._transcribers.pop(stream_id, None)
        if transcriber is not None:
            try:
                transcriber.close()
            except SpeechError:
                self._logger.warning(
                    "speech recognizer cleanup failed", extra={"stream_id": stream_id}
                )
        self._conversation_service.stop(stream_id)

    def _start(self, stream_id: str) -> list[str]:
        if self._has_stream(stream_id):
            self.disconnect(stream_id)
        transcriber = self._transcriber_factory.create()
        with self._lock:
            self._transcribers[stream_id] = transcriber
        try:
            welcome = self._conversation_service.start(stream_id)
            return [self._media_message(stream_id, welcome.message)]
        except Exception:
            self.disconnect(stream_id)
            raise

    def _media(self, stream_id: str, event: dict[str, object]) -> list[str]:
        transcriber = self._transcriber(stream_id)
        if transcriber is None:
            return []
        media = event.get("media")
        if not isinstance(media, dict):
            return []
        payload = media.get("payload")
        if not isinstance(payload, str):
            return []
        try:
            audio = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return []

        transcript = transcriber.push_ulaw(audio)
        if transcript is None:
            return []
        turn = self._conversation_service.handle_transcript(stream_id, transcript)
        return [self._media_message(stream_id, turn.message)]

    def _safe_retry(self, stream_id: str) -> list[str]:
        try:
            return [self._media_message(stream_id, SAFE_RETRY_MESSAGE)]
        except SpeechError:
            return []

    def _media_message(self, stream_id: str, text: str) -> str:
        payload = self._synthesizer.synthesize_ulaw(text)
        return json.dumps(
            {
                "event": "media",
                "streamSid": stream_id,
                "media": {"payload": payload},
            },
            separators=(",", ":"),
        )

    def _transcriber(self, stream_id: str) -> Any | None:
        with self._lock:
            return self._transcribers.get(stream_id)

    def _has_stream(self, stream_id: str) -> bool:
        return self._transcriber(stream_id) is not None

    @staticmethod
    def _stream_id(event: dict[str, object]) -> str:
        value = event.get("streamSid")
        if isinstance(value, str) and value.strip():
            return value
        start = event.get("start")
        if isinstance(start, dict):
            value = start.get("streamSid")
            if isinstance(value, str) and value.strip():
                return value
        return ""


def build_default_handler(settings: Settings) -> MediaStreamHandler:
    assistant = BedrockAssistant(
        settings.bedrock_model_id,
        settings.menu,
        region=settings.aws_region,
    )
    order_gateway = HttpOrderGateway(
        settings.order_api_url,
        token=settings.order_api_token,
        timeout_seconds=settings.request_timeout_seconds,
    )
    conversations = ConversationService(assistant, order_gateway)
    transcribers = VoskTranscriberFactory(
        settings.vosk_model_path,
        silence_seconds=settings.silence_seconds,
    )
    return MediaStreamHandler(conversations, transcribers, SpeechSynthesizer())


def create_app(
    settings: Settings | None = None,
    handler: MediaStreamHandler | None = None,
) -> Flask:
    resolved_settings = settings or Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, resolved_settings.log_level, logging.INFO)
    )
    app = Flask(__name__)
    sock = Sock(app)
    handler_lock = Lock()
    shared_handler = handler

    def get_handler() -> MediaStreamHandler:
        nonlocal shared_handler
        with handler_lock:
            if shared_handler is None:
                shared_handler = build_default_handler(resolved_settings)
            return shared_handler

    @app.get("/healthz")
    def health() -> Response:
        return jsonify(status="ok")

    @app.post("/call")
    def call() -> Response:
        response = VoiceResponse()
        connect = Connect()
        connect.stream(url=resolved_settings.websocket_url(request.host_url))
        response.append(connect)
        return Response(str(response), status=200, content_type="text/xml")

    @sock.route("/media")
    def media(ws: Any) -> None:
        active_handler = get_handler()
        stream_id = ""
        try:
            while True:
                message = ws.receive()
                if message is None:
                    break
                try:
                    event = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    event = {}
                if isinstance(event, dict):
                    stream_id = MediaStreamHandler._stream_id(event) or stream_id
                for outbound in active_handler.handle(message):
                    ws.send(outbound)
                if isinstance(event, dict) and event.get("event") == "stop":
                    stream_id = ""
                    break
        except ConnectionClosed:
            pass
        finally:
            if stream_id:
                active_handler.disconnect(stream_id)

    return app
