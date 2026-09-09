"""Environment-backed application configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from ai_ordering.models import DEFAULT_MENU, Menu


class ConfigurationError(ValueError):
    """Raised when an environment variable cannot be used safely."""


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name, str(default))
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} must be a number") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return parsed


def _menu_from_env(values: Mapping[str, str]) -> Menu:
    raw = values.get("MENU_JSON")
    if raw is None or not raw.strip():
        return DEFAULT_MENU
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigurationError("MENU_JSON must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ConfigurationError("MENU_JSON must be a JSON object")
    try:
        return Menu.from_mapping(decoded)
    except ValueError as error:
        raise ConfigurationError(f"MENU_JSON is invalid: {error}") from error


@dataclass(frozen=True)
class Settings:
    aws_region: str = "us-east-1"
    bedrock_model_id: str = ""
    vosk_model_path: str = "models/vosk-model-small-en-us-0.15"
    order_api_url: str = ""
    order_api_token: str = ""
    public_websocket_url: str = ""
    silence_seconds: float = 1.0
    request_timeout_seconds: float = 10.0
    log_level: str = "INFO"
    menu: Menu = DEFAULT_MENU

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if environ is None else environ
        return cls(
            aws_region=values.get("AWS_REGION", "us-east-1").strip(),
            bedrock_model_id=values.get("BEDROCK_MODEL_ID", "").strip(),
            vosk_model_path=values.get(
                "VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15"
            ).strip(),
            order_api_url=values.get("ORDER_API_URL", "").strip(),
            order_api_token=values.get("ORDER_API_TOKEN", "").strip(),
            public_websocket_url=values.get("PUBLIC_WEBSOCKET_URL", "").strip(),
            silence_seconds=_positive_float(values, "SILENCE_SECONDS", 1.0),
            request_timeout_seconds=_positive_float(
                values, "REQUEST_TIMEOUT_SECONDS", 10.0
            ),
            log_level=values.get("LOG_LEVEL", "INFO").strip().upper(),
            menu=_menu_from_env(values),
        )

    def websocket_url(self, request_host: str) -> str:
        if self.public_websocket_url:
            return self.public_websocket_url

        parsed = urlsplit(request_host)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunsplit((scheme, parsed.netloc, "/media", "", ""))
