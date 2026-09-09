"""HTTP adapter for submitting validated orders."""

from __future__ import annotations

import requests

from ai_ordering.models import Order


class OrderGatewayError(RuntimeError):
    """Raised when a validated order cannot be delivered safely."""


class HttpOrderGateway:
    def __init__(
        self,
        url: str,
        *,
        token: str = "",
        timeout_seconds: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self._url = url.strip()
        if not self._url:
            raise ValueError("order API URL cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self._token = token.strip()
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def submit(self, order: Order, idempotency_key: str) -> None:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key cannot be empty")
        headers = {"Idempotency-Key": idempotency_key}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            response = self._session.post(
                self._url,
                json=order.as_dict(),
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except requests.Timeout:
            raise OrderGatewayError("order service request timed out") from None
        except requests.RequestException:
            raise OrderGatewayError("order service could not be reached") from None

        if not 200 <= response.status_code < 300:
            raise OrderGatewayError(
                f"order service returned HTTP {response.status_code}"
            )

