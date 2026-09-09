import json
from decimal import Decimal

import pytest

from ai_ordering.models import Menu, ModelOutputError, parse_assistant_payload


@pytest.fixture
def menu():
    return Menu.from_mapping(
        {"Coke": "2.00", "Chicken Sandwich": "4.51", "Hamburger": "5.29"}
    )


def test_model_price_is_replaced_by_server_menu_price(menu):
    turn = parse_assistant_payload(
        '{"message":"Confirmed","order":{"complete":true,'
        '"items":[{"name":"Coke","quantity":2,"price":"0.01"}]}}',
        menu,
    )

    assert turn.message == "Confirmed"
    assert turn.order.items[0].unit_price == Decimal("2.00")
    assert turn.order.total == Decimal("4.00")


def test_markdown_fenced_json_is_accepted(menu):
    turn = parse_assistant_payload(
        """```json
        {"message":"What else?","order":null}
        ```""",
        menu,
    )

    assert turn.message == "What else?"
    assert turn.order is None


def test_menu_lookup_is_case_insensitive_but_keeps_canonical_name(menu):
    turn = parse_assistant_payload(
        '{"message":"Done","order":{"complete":true,'
        '"items":[{"name":"hamburger","quantity":1}]}}',
        menu,
    )

    assert turn.order.items[0].name == "Hamburger"


def test_duplicate_items_are_merged(menu):
    turn = parse_assistant_payload(
        '{"message":"Done","order":{"complete":true,"items":['
        '{"name":"Coke","quantity":1},{"name":"coke","quantity":2}]}}',
        menu,
    )

    assert len(turn.order.items) == 1
    assert turn.order.items[0].quantity == 3
    assert turn.order.total == Decimal("6.00")


def test_unknown_menu_item_is_rejected(menu):
    with pytest.raises(ModelOutputError, match="Pizza.*not on the menu"):
        parse_assistant_payload(
            '{"message":"Okay","order":{"complete":true,'
            '"items":[{"name":"Pizza","quantity":1}]}}',
            menu,
        )


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True, "2"])
def test_invalid_quantities_are_rejected(menu, quantity):
    raw = json.dumps(
        {
            "message": "Okay",
            "order": {
                "complete": True,
                "items": [{"name": "Coke", "quantity": quantity}],
            },
        }
    )

    with pytest.raises(ModelOutputError, match="quantity"):
        parse_assistant_payload(raw, menu)


def test_complete_order_requires_at_least_one_item(menu):
    with pytest.raises(ModelOutputError, match="at least one item"):
        parse_assistant_payload(
            '{"message":"Done","order":{"complete":true,"items":[]}}', menu
        )


def test_malformed_json_is_reported_without_echoing_the_payload(menu):
    raw = "customer private transcript is not json"

    with pytest.raises(ModelOutputError) as error:
        parse_assistant_payload(raw, menu)

    assert raw not in str(error.value)


def test_canonical_json_is_stable_for_idempotency(menu):
    first = parse_assistant_payload(
        '{"message":"Done","order":{"complete":true,"items":['
        '{"name":"Coke","quantity":1},{"name":"Hamburger","quantity":1}]}}',
        menu,
    ).order
    second = parse_assistant_payload(
        '{"message":"Different wording","order":{"complete":true,"items":['
        '{"name":"Hamburger","quantity":1},{"name":"Coke","quantity":1}]}}',
        menu,
    ).order

    assert first.canonical_json() == second.canonical_json()
