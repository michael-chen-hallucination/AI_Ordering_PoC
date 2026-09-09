import boto3
import pytest
from botocore.exceptions import EndpointConnectionError

from ai_ordering.adapters.bedrock import BedrockAssistant, BedrockError
from ai_ordering.models import DEFAULT_MENU, ChatMessage, ModelOutputError

VALID_RESPONSE = (
    '{"message":"Your order is confirmed",'
    '"order":{"complete":true,"items":[{"name":"Coke","quantity":2}]}}'
)


class RecordingBedrockClient:
    def __init__(self, response=None, error=None):
        self.response = (
            {"output": {"message": {"content": [{"text": VALID_RESPONSE}]}}}
            if response is None
            else response
        )
        self.error = error
        self.calls = []

    def converse(self, **request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.response


def test_respond_translates_history_and_uses_deterministic_settings():
    client = RecordingBedrockClient()
    assistant = BedrockAssistant(
        "test-model", DEFAULT_MENU, region="us-west-2", client=client
    )

    turn = assistant.respond(
        [
            ChatMessage("user", "What is on the menu?"),
            ChatMessage("assistant", "We have Coke and sandwiches."),
            ChatMessage("user", "two cokes"),
        ]
    )

    assert turn.order.items[0].quantity == 2
    assert client.calls == [
        {
            "modelId": "test-model",
            "system": [{"text": assistant.system_prompt}],
            "messages": [
                {"role": "user", "content": [{"text": "What is on the menu?"}]},
                {
                    "role": "assistant",
                    "content": [{"text": "We have Coke and sandwiches."}],
                },
                {"role": "user", "content": [{"text": "two cokes"}]},
            ],
            "inferenceConfig": {
                "maxTokens": 400,
                "temperature": 0.1,
                "topP": 0.9,
            },
        }
    ]


def test_system_prompt_contains_menu_and_structured_output_contract():
    assistant = BedrockAssistant(
        "test-model", DEFAULT_MENU, client=RecordingBedrockClient()
    )

    assert "Coke: $2.00" in assistant.system_prompt
    assert '"complete": true' in assistant.system_prompt
    assert '"order": null' in assistant.system_prompt
    assert "exact menu name" in assistant.system_prompt


def test_fenced_json_response_is_validated():
    client = RecordingBedrockClient(
        {
            "output": {
                "message": {
                    "content": [{"text": f"```json\n{VALID_RESPONSE}\n```"}]
                }
            }
        }
    )

    turn = BedrockAssistant("test-model", DEFAULT_MENU, client=client).respond(
        [ChatMessage("user", "two cokes")]
    )

    assert turn.order.total == 4


def test_hallucinated_item_is_rejected_by_the_domain_boundary():
    response = (
        '{"message":"Done","order":{"complete":true,'
        '"items":[{"name":"Pizza","quantity":1}]}}'
    )
    client = RecordingBedrockClient(
        {"output": {"message": {"content": [{"text": response}]}}}
    )

    with pytest.raises(ModelOutputError, match="not on the menu"):
        BedrockAssistant("test-model", DEFAULT_MENU, client=client).respond(
            [ChatMessage("user", "one pizza")]
        )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"output": {}},
        {"output": {"message": {}}},
        {"output": {"message": {"content": []}}},
        {"output": {"message": {"content": [{"image": {}}]}}},
        {"output": {"message": {"content": [{"text": ""}]}}},
    ],
)
def test_malformed_bedrock_response_is_mapped(response):
    assistant = BedrockAssistant(
        "test-model", DEFAULT_MENU, client=RecordingBedrockClient(response)
    )

    with pytest.raises(BedrockError, match="text response"):
        assistant.respond([ChatMessage("user", "one coke")])


def test_sdk_error_is_mapped_without_leaking_endpoint():
    client = RecordingBedrockClient(
        error=EndpointConnectionError(endpoint_url="https://private.internal")
    )
    assistant = BedrockAssistant("test-model", DEFAULT_MENU, client=client)

    with pytest.raises(BedrockError, match="request failed") as error:
        assistant.respond([ChatMessage("user", "one coke")])

    assert "private.internal" not in str(error.value)


def test_missing_model_id_fails_during_construction():
    with pytest.raises(ValueError, match="model ID"):
        BedrockAssistant(" ", DEFAULT_MENU)


def test_boto3_client_is_created_lazily(monkeypatch):
    created = []
    client = RecordingBedrockClient()

    def fake_client(service_name, *, region_name):
        created.append((service_name, region_name))
        return client

    monkeypatch.setattr(boto3, "client", fake_client)
    assistant = BedrockAssistant("test-model", DEFAULT_MENU, region="us-west-2")
    assert created == []

    assistant.respond([ChatMessage("user", "two cokes")])

    assert created == [("bedrock-runtime", "us-west-2")]
