"""Per-call conversation orchestration with isolated state."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol

from ai_ordering.models import AssistantTurn, ChatMessage, Order


class UnknownSessionError(LookupError):
    """Raised when a media event arrives outside an active call."""


class Assistant(Protocol):
    def respond(self, history: Sequence[ChatMessage]) -> AssistantTurn: ...


class OrderSink(Protocol):
    def submit(self, order: Order, idempotency_key: str) -> None: ...


@dataclass
class CallSession:
    history: list[ChatMessage] = field(default_factory=list)
    submitted_keys: set[str] = field(default_factory=set)
    lock: RLock = field(default_factory=RLock)


class ConversationService:
    """Coordinates assistant turns and side effects for independent calls."""

    def __init__(
        self,
        assistant: Assistant,
        order_sink: OrderSink,
        welcome_message: str = "Welcome! What can I get for you?",
    ) -> None:
        self._assistant = assistant
        self._order_sink = order_sink
        self._welcome_message = welcome_message
        self._sessions: dict[str, CallSession] = {}
        self._sessions_lock = RLock()

    @property
    def active_session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    def start(self, stream_id: str) -> AssistantTurn:
        if not stream_id.strip():
            raise ValueError("stream_id cannot be empty")
        with self._sessions_lock:
            self._sessions[stream_id] = CallSession()
        return AssistantTurn(self._welcome_message)

    def handle_transcript(self, stream_id: str, transcript: str) -> AssistantTurn:
        normalized = transcript.strip()
        if not normalized:
            raise ValueError("transcript cannot be empty")
        session = self._session(stream_id)

        with session.lock:
            session.history.append(ChatMessage("user", normalized))
            turn = self._assistant.respond(tuple(session.history))
            session.history.append(ChatMessage("assistant", turn.message))
            if turn.order is not None:
                key = self._idempotency_key(stream_id, turn.order)
                if key not in session.submitted_keys:
                    self._order_sink.submit(turn.order, key)
                    session.submitted_keys.add(key)
            return turn

    def stop(self, stream_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(stream_id, None)

    def _session(self, stream_id: str) -> CallSession:
        with self._sessions_lock:
            session = self._sessions.get(stream_id)
        if session is None:
            raise UnknownSessionError(f"unknown call session: {stream_id}")
        return session

    @staticmethod
    def _idempotency_key(stream_id: str, order: Order) -> str:
        content = f"{stream_id}:{order.canonical_json()}".encode()
        return hashlib.sha256(content).hexdigest()[:32]

