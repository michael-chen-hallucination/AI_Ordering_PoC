"""Domain models and the trust boundary for language-model output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


class ModelOutputError(ValueError):
    """Raised when an assistant response cannot be trusted as an order."""


@dataclass(frozen=True)
class MenuItem:
    name: str
    price: Decimal


@dataclass(frozen=True)
class Menu:
    items: tuple[MenuItem, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> Menu:
        if not values:
            raise ValueError("menu must contain at least one item")

        items: list[MenuItem] = []
        seen: set[str] = set()
        for raw_name, raw_price in values.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("menu item names cannot be empty")
            key = name.casefold()
            if key in seen:
                raise ValueError(f"duplicate menu item: {name}")
            try:
                price = Decimal(str(raw_price))
            except (InvalidOperation, ValueError) as error:
                raise ValueError(f"invalid price for {name}") from error
            if not price.is_finite() or price < 0:
                raise ValueError(f"invalid price for {name}")
            items.append(MenuItem(name=name, price=price.quantize(Decimal("0.01"))))
            seen.add(key)
        return cls(tuple(items))

    def item_for(self, name: str) -> MenuItem:
        key = name.strip().casefold()
        for item in self.items:
            if item.name.casefold() == key:
                return item
        raise ModelOutputError(f"{name.strip() or 'item'} is not on the menu")

    def price_for(self, name: str) -> str:
        return format(self.item_for(name).price, ".2f")

    def as_prompt_text(self) -> str:
        return "\n".join(
            f"- {item.name}: ${item.price:.2f}" for item in self.items
        )


DEFAULT_MENU = Menu.from_mapping(
    {"Coke": "2.00", "Chicken Sandwich": "4.51", "Hamburger": "5.29"}
)


@dataclass(frozen=True)
class OrderLine:
    name: str
    quantity: int
    unit_price: Decimal

    @property
    def line_total(self) -> Decimal:
        return self.unit_price * self.quantity

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "unit_price": format(self.unit_price, ".2f"),
            "line_total": format(self.line_total, ".2f"),
        }


@dataclass(frozen=True)
class Order:
    items: tuple[OrderLine, ...]

    @property
    def total(self) -> Decimal:
        return sum((item.line_total for item in self.items), Decimal("0.00"))

    def as_dict(self) -> dict[str, object]:
        return {
            "items": [item.as_dict() for item in self.items],
            "total": format(self.total, ".2f"),
        }

    def canonical_json(self) -> str:
        payload = {
            "items": [
                {
                    "name": item.name,
                    "quantity": item.quantity,
                    "unit_price": format(item.unit_price, ".2f"),
                }
                for item in sorted(self.items, key=lambda value: value.name.casefold())
            ]
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class AssistantTurn:
    message: str
    order: Order | None = None


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        if not self.content.strip():
            raise ValueError("message content cannot be empty")


def _strip_json_fence(raw: str) -> str:
    value = raw.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        raise ModelOutputError("assistant returned an incomplete JSON fence")
    return "\n".join(lines[1:-1]).strip()


def _load_payload(raw: str) -> dict[str, object]:
    try:
        payload = json.loads(_strip_json_fence(raw))
    except (json.JSONDecodeError, TypeError) as error:
        raise ModelOutputError("assistant returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ModelOutputError("assistant response must be a JSON object")
    return payload


def parse_assistant_payload(raw: str, menu: Menu) -> AssistantTurn:
    """Validate model output and replace model-supplied prices with menu prices."""

    payload = _load_payload(raw)
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ModelOutputError("assistant response requires a message")

    order_payload = payload.get("order")
    if order_payload is None:
        return AssistantTurn(message=message.strip())
    if not isinstance(order_payload, dict):
        raise ModelOutputError("order must be an object or null")

    complete = order_payload.get("complete")
    if not isinstance(complete, bool):
        raise ModelOutputError("order.complete must be a boolean")
    if not complete:
        return AssistantTurn(message=message.strip())

    raw_items = order_payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ModelOutputError("a complete order requires at least one item")

    quantities: dict[str, int] = {}
    canonical_items: dict[str, MenuItem] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ModelOutputError("each order item must be an object")
        name = raw_item.get("name")
        quantity = raw_item.get("quantity")
        if not isinstance(name, str) or not name.strip():
            raise ModelOutputError("each order item requires a name")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ModelOutputError("item quantity must be a positive integer")
        menu_item = menu.item_for(name)
        key = menu_item.name.casefold()
        canonical_items[key] = menu_item
        quantities[key] = quantities.get(key, 0) + quantity

    lines = tuple(
        OrderLine(
            name=canonical_items[key].name,
            quantity=quantity,
            unit_price=canonical_items[key].price,
        )
        for key, quantity in sorted(quantities.items())
    )
    return AssistantTurn(message=message.strip(), order=Order(lines))
