"""Amazon Bedrock Converse adapter with validated structured output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ai_ordering.models import (
    AssistantTurn,
    ChatMessage,
    Menu,
    parse_assistant_payload,
)


class BedrockError(RuntimeError):
    """Raised when Amazon Bedrock does not return usable text."""


class BedrockAssistant:
    def __init__(
        self,
        model_id: str,
        menu: Menu,
        *,
        region: str = "us-east-1",
        client: Any | None = None,
    ) -> None:
        self._model_id = model_id.strip()
        if not self._model_id:
            raise ValueError("Bedrock model ID cannot be empty")
        self._menu = menu
        self._region = region
        self._client = client
        self._system_prompt = self._build_system_prompt(menu)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def respond(self, history: Sequence[ChatMessage]) -> AssistantTurn:
        messages = [
            {"role": message.role, "content": [{"text": message.content}]}
            for message in history
        ]
        try:
            response = self._bedrock_client().converse(
                modelId=self._model_id,
                system=[{"text": self._system_prompt}],
                messages=messages,
                inferenceConfig={
                    "maxTokens": 400,
                    "temperature": 0.1,
                    "topP": 0.9,
                },
            )
        except (BotoCoreError, ClientError):
            raise BedrockError("Bedrock request failed") from None

        return parse_assistant_payload(self._response_text(response), self._menu)

    def _bedrock_client(self) -> Any:
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime", region_name=self._region
            )
        return self._client

    @staticmethod
    def _response_text(response: object) -> str:
        try:
            content = response["output"]["message"]["content"]
        except (KeyError, TypeError):
            raise BedrockError("Bedrock returned no text response") from None
        if not isinstance(content, list):
            raise BedrockError("Bedrock returned no text response")
        text = "".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        ).strip()
        if not text:
            raise BedrockError("Bedrock returned no text response")
        return text

    @staticmethod
    def _build_system_prompt(menu: Menu) -> str:
        return f"""You are a concise phone ordering assistant.

Only discuss and accept items on this menu:
{menu.as_prompt_text()}

Return one JSON object and no prose outside it. Use exactly this shape:
{{
  "message": "short spoken reply",
  "order": {{
    "complete": true,
    "items": [{{"name": "exact menu name", "quantity": 1}}]
  }}
}}

Use "order": null until the customer has confirmed a complete order.
Never invent menu items or prices. Keep the spoken message brief."""

