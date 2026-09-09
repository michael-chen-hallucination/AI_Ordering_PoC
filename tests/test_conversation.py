from collections import deque

import pytest

from ai_ordering.conversation import ConversationService, UnknownSessionError
from ai_ordering.models import (
    DEFAULT_MENU,
    AssistantTurn,
    ChatMessage,
    parse_assistant_payload,
)


class RecordingAssistant:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.histories = []

    def respond(self, history):
        self.histories.append(list(history))
        return self.responses.popleft()


class RecordingOrderSink:
    def __init__(self):
        self.calls = []

    def submit(self, order, idempotency_key):
        self.calls.append((order, idempotency_key))


def completed_turn(message="Your order is confirmed"):
    return parse_assistant_payload(
        '{"message":"' + message + '","order":{"complete":true,'
        '"items":[{"name":"Coke","quantity":2}]}}',
        DEFAULT_MENU,
    )


def customer_texts(history):
    return [message.content for message in history if message.role == "user"]


def test_start_returns_welcome_without_calling_the_assistant():
    assistant = RecordingAssistant([])
    service = ConversationService(assistant, RecordingOrderSink())

    turn = service.start("call-a")

    assert turn == AssistantTurn("Welcome! What can I get for you?")
    assert assistant.histories == []
    assert service.active_session_count == 1


def test_transcript_before_start_is_rejected():
    service = ConversationService(RecordingAssistant([]), RecordingOrderSink())

    with pytest.raises(UnknownSessionError, match="call-a"):
        service.handle_transcript("call-a", "two cokes")


def test_blank_transcript_does_not_call_the_assistant():
    assistant = RecordingAssistant([])
    service = ConversationService(assistant, RecordingOrderSink())
    service.start("call-a")

    with pytest.raises(ValueError, match="transcript"):
        service.handle_transcript("call-a", "   ")

    assert assistant.histories == []


def test_two_calls_do_not_share_history():
    assistant = RecordingAssistant(
        [
            AssistantTurn("Anything else?"),
            AssistantTurn("Would you like anything else?"),
        ]
    )
    service = ConversationService(assistant, RecordingOrderSink())
    service.start("call-a")
    service.start("call-b")

    service.handle_transcript("call-a", "two cokes")
    service.handle_transcript("call-b", "one burger")

    assert customer_texts(assistant.histories[0]) == ["two cokes"]
    assert customer_texts(assistant.histories[1]) == ["one burger"]


def test_prior_turns_are_available_to_the_same_call():
    assistant = RecordingAssistant(
        [AssistantTurn("Anything else?"), AssistantTurn("Got it")]
    )
    service = ConversationService(assistant, RecordingOrderSink())
    service.start("call-a")

    service.handle_transcript("call-a", "one coke")
    service.handle_transcript("call-a", "make that two")

    assert assistant.histories[1] == [
        ChatMessage("user", "one coke"),
        ChatMessage("assistant", "Anything else?"),
        ChatMessage("user", "make that two"),
    ]


def test_incomplete_turn_does_not_submit_an_order():
    sink = RecordingOrderSink()
    service = ConversationService(
        RecordingAssistant([AssistantTurn("Anything else?")]), sink
    )
    service.start("call-a")

    service.handle_transcript("call-a", "one coke")

    assert sink.calls == []


def test_completed_order_is_submitted_once_per_call():
    sink = RecordingOrderSink()
    service = ConversationService(
        RecordingAssistant([completed_turn(), completed_turn("Still confirmed")]), sink
    )
    service.start("call-a")

    service.handle_transcript("call-a", "two cokes")
    service.handle_transcript("call-a", "yes, two cokes")

    assert len(sink.calls) == 1
    assert len(sink.calls[0][1]) == 32


def test_same_order_in_different_calls_has_a_different_idempotency_key():
    sink = RecordingOrderSink()
    service = ConversationService(
        RecordingAssistant([completed_turn(), completed_turn()]), sink
    )
    service.start("call-a")
    service.start("call-b")

    service.handle_transcript("call-a", "two cokes")
    service.handle_transcript("call-b", "two cokes")

    assert sink.calls[0][1] != sink.calls[1][1]


def test_stop_removes_the_call_session():
    service = ConversationService(RecordingAssistant([]), RecordingOrderSink())
    service.start("call-a")

    service.stop("call-a")

    assert service.active_session_count == 0
    with pytest.raises(UnknownSessionError):
        service.handle_transcript("call-a", "one coke")
