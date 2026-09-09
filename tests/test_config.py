import pytest

from ai_ordering.config import ConfigurationError, Settings


def test_defaults_are_safe_for_a_local_checkout():
    settings = Settings.from_env({})

    assert settings.aws_region == "us-east-1"
    assert settings.bedrock_model_id == ""
    assert settings.vosk_model_path.endswith("vosk-model-small-en-us-0.15")
    assert settings.order_api_url == ""
    assert settings.silence_seconds == 1.0
    assert settings.request_timeout_seconds == 10.0
    assert settings.menu.price_for("coke") == "2.00"


def test_environment_overrides_runtime_configuration():
    settings = Settings.from_env(
        {
            "AWS_REGION": "us-west-2",
            "BEDROCK_MODEL_ID": "test-model",
            "VOSK_MODEL_PATH": "/tmp/vosk",
            "ORDER_API_URL": "https://orders.example.com",
            "ORDER_API_TOKEN": "test-token",
            "PUBLIC_WEBSOCKET_URL": "wss://voice.example.com/custom",
            "SILENCE_SECONDS": "1.5",
            "REQUEST_TIMEOUT_SECONDS": "3.25",
            "LOG_LEVEL": "debug",
            "MENU_JSON": '{"Tea":"1.75"}',
        }
    )

    assert settings.aws_region == "us-west-2"
    assert settings.bedrock_model_id == "test-model"
    assert settings.vosk_model_path == "/tmp/vosk"
    assert settings.order_api_token == "test-token"
    assert settings.silence_seconds == 1.5
    assert settings.request_timeout_seconds == 3.25
    assert settings.log_level == "DEBUG"
    assert settings.menu.price_for("tea") == "1.75"


@pytest.mark.parametrize(
    ("request_host", "expected"),
    [
        ("https://voice.example.com/", "wss://voice.example.com/media"),
        ("http://localhost:8443/", "ws://localhost:8443/media"),
    ],
)
def test_websocket_url_is_derived_from_the_request_host(request_host, expected):
    assert Settings.from_env({}).websocket_url(request_host) == expected


def test_explicit_websocket_url_takes_precedence():
    settings = Settings.from_env(
        {"PUBLIC_WEBSOCKET_URL": "wss://public.example.com/twilio"}
    )

    assert settings.websocket_url("http://localhost:8443/") == (
        "wss://public.example.com/twilio"
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SILENCE_SECONDS", "0"),
        ("REQUEST_TIMEOUT_SECONDS", "not-a-number"),
        ("MENU_JSON", "[]"),
    ],
)
def test_invalid_environment_values_fail_with_the_variable_name(name, value):
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({name: value})

