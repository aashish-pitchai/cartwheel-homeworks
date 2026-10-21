"""Tests for the HW1 Part A additional tools in agent/tools.py.

Offline, no API keys. Each tool is checked for the shared result shape, the
same scope rules as get_order or list_my_orders, and the seeded data-quality
records (orders 8001 to 8003), which must be reported rather than used to
compute dates.
"""

from __future__ import annotations

import pytest

from agent import tools
from agent.agent import TOOLS_BY_ROLE
from agent.auth import AuthContext

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
SHOPPER_2 = AuthContext(user_id=2, role="shopper")
MERCHANT_STORE_1 = AuthContext(user_id=9001, role="merchant", store_id=1)
MERCHANT_STORE_2 = AuthContext(user_id=9002, role="merchant", store_id=2)
SUPPORT = AuthContext(user_id=9501, role="support")

NEW_TOOLS = (
    "check_return_eligibility",
    "track_shipment",
    "get_store_info",
    "summarize_order_history",
)


def assert_error(result: dict, code: str) -> None:
    """Every failure has exactly ok, error, and reason."""
    assert set(result) == {"ok", "error", "reason"}
    assert result["ok"] is False
    assert result["error"] == code
    assert isinstance(result["reason"], str) and result["reason"]


def test_new_tools_are_registered_for_every_role() -> None:
    for role, role_tools in TOOLS_BY_ROLE.items():
        names = {tool.name for tool in role_tools}
        assert set(NEW_TOOLS) <= names, role


# ----- check_return_eligibility -------------------------------------------


def test_eligibility_inside_the_platform_window(world: dict) -> None:
    result = tools.check_return_eligibility(SHOPPER_1, 4127)
    assert result["ok"] is True
    assert result["refund_eligible"] is True
    assert result["return_window_days"] == 30
    assert result["window_source"] == "platform_default"
    assert result["return_deadline"] == "2026-07-19"
    assert result["days_remaining"] == 18
    assert result["policy_ids"] == ["cw-returns"]
    assert result["data_issues"] == []


def test_eligibility_outside_the_window(world: dict) -> None:
    result = tools.check_return_eligibility(SHOPPER_1, 3980)
    assert result["ok"] is True
    assert result["refund_eligible"] is False
    assert result["days_remaining"] == 0


@pytest.mark.parametrize("tool", [tools.check_return_eligibility, tools.track_shipment])
def test_order_tools_follow_the_get_order_scope(world: dict, tool) -> None:
    assert_error(tool(SHOPPER_1, 999_999), "not_found")
    assert_error(tool(SHOPPER_2, 4127), "permission_denied")
    assert_error(tool(MERCHANT_STORE_2, 4127), "permission_denied")
    assert tool(MERCHANT_STORE_1, 4127)["ok"] is True
    assert tool(SUPPORT, 4127)["ok"] is True


@pytest.mark.parametrize(
    "order_id, code",
    [
        (8001, "reversed_chronology"),
        (8002, "missing_required_date"),
        (8003, "store_mismatch"),
    ],
)
def test_order_tools_report_defective_records(world: dict, order_id, code) -> None:
    eligibility = tools.check_return_eligibility(SUPPORT, order_id)
    assert [issue["code"] for issue in eligibility["data_issues"]] == [code]
    assert eligibility["return_deadline"] is None
    assert eligibility["days_remaining"] is None

    shipment = tools.track_shipment(SUPPORT, order_id)
    assert [issue["code"] for issue in shipment["data_issues"]] == [code]
    assert shipment["expected_delivery_by"] is None
    assert shipment["overdue"] is False


# ----- track_shipment -----------------------------------------------------


def test_track_shipment_for_a_delivered_order(world: dict) -> None:
    result = tools.track_shipment(SHOPPER_1, 4127)
    assert result["ok"] is True
    assert result["status"] == "delivered"
    assert result["store_name"] == "Blue Heron Ceramics"
    assert (result["shipped_at"], result["delivered_at"]) == ("2026-06-16", "2026-06-19")
    assert result["expected_ship_by"] is None
    assert result["expected_delivery_by"] is None
    assert result["policy_ids"] == ["cw-shipping"]


def test_track_shipment_estimates_use_the_shipping_facts(world: dict) -> None:
    # Order 10 is placed on 2026-07-01 and order 335 shipped on 2026-06-30.
    placed = tools.track_shipment(SUPPORT, 10)
    assert placed["status"] == "placed"
    assert placed["expected_ship_by"] == "2026-07-04"  # + 3 handling days
    assert placed["expected_delivery_by"] == "2026-07-11"  # + 7 transit days

    shipped = tools.track_shipment(SUPPORT, 335)
    assert shipped["status"] == "shipped"
    assert shipped["expected_ship_by"] is None
    assert shipped["expected_delivery_by"] == "2026-07-07"


# ----- get_store_info -----------------------------------------------------


def test_store_info_reports_overrides(world: dict) -> None:
    juniper = tools.get_store_info(SHOPPER_1, "Juniper Home Goods")
    assert juniper["ok"] is True
    assert juniper["return_window_days"] == 14
    assert juniper["window_source"] == "store_override"
    assert "store-juniper-home-goods-policy" in juniper["policy_ids"]

    cascade = tools.get_store_info(SHOPPER_1, "5")
    assert cascade["name"] == "Cascade Audio"
    assert cascade["restocking_fee_opt_in"] is True
    assert cascade["restocking_fee_max_percent"] == 15

    blue_heron = tools.get_store_info(SHOPPER_1, "blue-heron-ceramics")
    assert blue_heron["window_source"] == "platform_default"
    assert blue_heron["restocking_fee_max_percent"] is None
    assert blue_heron["policy_ids"] == ["cw-store-overrides"]


def test_store_info_errors(world: dict) -> None:
    assert_error(tools.get_store_info(SHOPPER_1, "  "), "invalid_argument")
    assert_error(tools.get_store_info(SHOPPER_1, "No Such Store"), "not_found")
    assert_error(tools.get_store_info(SHOPPER_1, "999"), "not_found")


# ----- summarize_order_history --------------------------------------------


def test_summary_covers_the_whole_shopper_scope(world: dict) -> None:
    result = tools.summarize_order_history(SHOPPER_1)
    assert result["ok"] is True
    assert result["scope"] == "shopper"
    assert result["customer_user_id"] == 1
    assert result["order_count"] == sum(b["count"] for b in result["by_status"].values())
    assert {4127, 4455} <= set(result["refund_eligible_order_ids"])
    assert 3980 not in result["refund_eligible_order_ids"]
    assert len(result["recent_orders"]) == tools.RECENT_ORDERS_IN_SUMMARY
    assert tools.summarize_order_history(SHOPPER_1, 1) == result


def test_summary_scope_rules(world: dict) -> None:
    assert_error(tools.summarize_order_history(SHOPPER_1, 2), "permission_denied")
    assert_error(tools.summarize_order_history(MERCHANT_STORE_2, 1), "permission_denied")
    assert_error(tools.summarize_order_history(SUPPORT), "invalid_argument")
    assert_error(tools.summarize_order_history(SUPPORT, 123_456), "not_found")
    assert_error(tools.summarize_order_history(SUPPORT, 9002), "invalid_argument")

    merchant = tools.summarize_order_history(MERCHANT_STORE_2)
    assert merchant["scope"] == "store"
    assert merchant["store_id"] == 2
    assert merchant["customer_user_id"] is None
    assert all(order["store_id"] == 2 for order in merchant["recent_orders"])

    support = tools.summarize_order_history(SUPPORT, 1)
    assert support["order_count"] == tools.summarize_order_history(SHOPPER_1)["order_count"]
