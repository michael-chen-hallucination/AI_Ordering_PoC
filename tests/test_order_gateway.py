import pytest
import requests

from ai_ordering.adapters.order_gateway import HttpOrderGateway, OrderGatewayError
from ai_ordering.models import DEFAULT_MENU, parse_assistant_payload


class FakeResponse:
    def __init__(self, status_code=201, text="response body must stay private"):
        self.status_code = status_code
        self.text = text


class RecordingSession:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.calls = []

    def post(self, url, *, json, headers, timeout):
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def order():
    return parse_assistant_payload(
        '{"message":"Done","order":{"complete":true,'
        '"items":[{"name":"Coke","quantity":2}]}}',
        DEFAULT_MENU,
    ).order


def test_submit_sends_validated_totals_timeout_and_idempotency(order):
    session = RecordingSession()
    gateway = HttpOrderGateway(
        "https://orders.example.com/v1/orders",
        token="test-token",
        timeout_seconds=3.5,
        session=session,
    )

    gateway.submit(order, "call-key")

    assert session.calls == [
        {
            "url": "https://orders.example.com/v1/orders",
            "json": {
                "items": [
                    {
                        "name": "Coke",
                        "quantity": 2,
                        "unit_price": "2.00",
                        "line_total": "4.00",
                    }
                ],
                "total": "4.00",
            },
            "headers": {
                "Authorization": "Bearer test-token",
                "Idempotency-Key": "call-key",
            },
            "timeout": 3.5,
        }
    ]


def test_submit_omits_authorization_when_no_token_is_configured(order):
    session = RecordingSession()
    gateway = HttpOrderGateway("https://orders.example.com", session=session)

    gateway.submit(order, "call-key")

    assert session.calls[0]["headers"] == {"Idempotency-Key": "call-key"}


@pytest.mark.parametrize("status_code", [400, 401, 429, 500])
def test_non_success_response_is_mapped_without_response_body(order, status_code):
    session = RecordingSession(FakeResponse(status_code, text="customer-private-data"))
    gateway = HttpOrderGateway("https://orders.example.com", session=session)

    with pytest.raises(OrderGatewayError, match=str(status_code)) as error:
        gateway.submit(order, "call-key")

    assert "customer-private-data" not in str(error.value)


def test_timeout_is_mapped_without_leaking_the_token(order):
    session = RecordingSession(error=requests.Timeout("secret-token"))
    gateway = HttpOrderGateway(
        "https://orders.example.com", token="secret-token", session=session
    )

    with pytest.raises(OrderGatewayError, match="timed out") as error:
        gateway.submit(order, "call-key")

    assert "secret-token" not in str(error.value)


def test_request_error_is_mapped_without_the_upstream_message(order):
    session = RecordingSession(error=requests.RequestException("private hostname"))
    gateway = HttpOrderGateway("https://orders.example.com", session=session)

    with pytest.raises(OrderGatewayError, match="could not be reached") as error:
        gateway.submit(order, "call-key")

    assert "private hostname" not in str(error.value)


def test_missing_order_url_fails_during_construction():
    with pytest.raises(ValueError, match="order API URL"):
        HttpOrderGateway("   ")
