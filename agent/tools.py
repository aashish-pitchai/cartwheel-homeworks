"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from rapidfuzz import fuzz, utils

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.config import load_facts
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch
from seed.eligibility import effective_return_window_days

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20
MAX_FIND_ORDER_RESULTS = 5
# Minimum rapidfuzz WRatio (0 to 100) for a product title to count as a match.
FIND_ORDER_MIN_SCORE = 80


def _error(code: str, reason: str) -> dict[str, Any]:
    """The structured failure every homework tool returns."""
    return {"ok": False, "error": code, "reason": reason}


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id. Risk tier: read.

    Every role may read every policy doc (the corpus is public help-center
    content), so this tool needs no permission check.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.

    Implementation notes:
        agent.helpcenter.load_policy_docs() returns every parsed doc.
    """
    for doc in load_policy_docs():
        if doc.policy_id == policy_id:
            return {
                "ok": True,
                "policy_id": doc.policy_id,
                "title": doc.title,
                "audience": doc.audience,
                "body": doc.body,
            }
    return _error("not_found", f"no policy with id '{policy_id}'")


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog. Risk tier: read.

    Every role may search products. Matching is deterministic keyword
    matching, not semantic search: a product matches when every whitespace
    token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.

    Implementation notes:
        agent.db.list_products(conn, store_id) gives the candidate set.
        Use `with db.connection() as conn:` to close the database automatically.
    """
    tokens = query.lower().split()
    if not tokens:
        return _error("invalid_argument", "empty query")
    if max_price_usd is not None and max_price_usd <= 0:
        return _error("invalid_argument", "max_price_usd must be positive")
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))
    with db.connection() as conn:
        store_id = None
        if store is not None:
            match = db.get_store_by_name(conn, store)
            if match is None:
                return _error("not_found", f"no store named '{store}'")
            store_id = match.id
        candidates = db.list_products(conn, store_id)
    matches = []
    for product in candidates:
        # A defective row can carry an empty title or description.
        text = f"{product.title or ''} {product.description or ''}".lower()
        if not all(token in text for token in tokens):
            continue
        if max_price_usd is not None and product.price_usd > max_price_usd:
            continue
        matches.append(product)
    matches.sort(key=lambda product: (product.price_usd, product.id))
    products = [
        {
            "product_id": product.id,
            "store_id": product.store_id,
            "title": product.title,
            "price_usd": product.price_usd,
        }
        for product in matches[:limit]
    ]
    return {"ok": True, "products": products, "count": len(products)}


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders in the caller's own scope. Risk tier: read.

    Role behavior, straight from the access matrix in SPEC.md:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders (ctx.store_id).
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead, so return {"ok": False,
          "error": "invalid_argument", "reason": ...} saying exactly that.

    Returns:
        For shopper and merchant: {"ok": True, "orders": [...],
        "count": <len(orders)>} where each order is
        agent.db.Order.to_public_dict() and the list holds at most
        DEFAULT_ORDER_LIMIT orders, newest first (agent.db.list_orders_for_user
        and list_orders_for_store already sort and limit this way).

    Implementation notes:
        No permission check is needed beyond the role dispatch, because the
        scope is baked into which query you run. That is the point of the
        tool: the model cannot ask for someone else's orders through it.
    """
    if ctx.role == "support":
        return _error(
            "invalid_argument",
            "support staff have no orders of their own; "
            "look up a specific order with get_order instead",
        )
    with db.connection() as conn:
        if ctx.role == "shopper":
            orders = db.list_orders_for_user(conn, ctx.user_id, DEFAULT_ORDER_LIMIT)
        else:  # merchant; AuthContext guarantees a store_id
            orders = db.list_orders_for_store(conn, ctx.store_id, DEFAULT_ORDER_LIMIT)
    payload = [order.to_public_dict() for order in orders]
    return {"ok": True, "orders": payload, "count": len(payload)}


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order. Risk tier: write.

    This is the homework's write tool, and it must enforce two independent
    rules in this order:

    1. The access matrix (scope): use agent.auth.can_cancel_order. Shoppers
       may cancel only their own orders, merchants only their own store's
       orders, support any order. On failure return
       agent.auth.permission_denied(...) with a reason naming the role and
       the order id. Scope is checked before the status rule so that an
       out-of-scope caller learns nothing about the order's state.
    2. The pre-shipment rule (facts.yaml `cancel_cutoff`): only orders whose
       status is exactly "placed" can be cancelled, for every role. If the
       order is in scope but its status is not "placed", return
       {"ok": False, "error": "not_eligible", "reason": ...} that names the
       current status and states that orders can be cancelled only before
       shipment.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.

    Implementation notes:
        Fetch with agent.db.get_order. Note the argument order of
        can_cancel_order(ctx, order_user_id, order_store_id).

    The Module 4 kill switch is checked first (before the scope and
    status rules and before your code), so that a paused write tool touches
    nothing. It is provided; the default ("off") returns None and falls
    through to your implementation.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return _error("not_found", f"no order #{order_id}")
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not cancel order #{order_id}"
            )
        if order.status != "placed":
            return _error(
                "not_eligible",
                f"order #{order_id} has status '{order.status}'; "
                f"orders can be cancelled only before shipment",
            )
        db.set_order_status(conn, order_id, "cancelled")
    return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search the caller's orders by product name. Risk tier: read.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and searches the authenticated user's orders for products whose name
    matches. Use fuzzy string matching (e.g., thefuzz.fuzz.partial_ratio
    or case-insensitive substring matching) to find orders whose product name is close to the
    query.

    Access rules: a shopper searches only the shopper's own orders, a
    merchant searches orders from the merchant's store, and support staff
    can search any orders. Use agent.db.list_order_search_candidates with
    user_id=ctx.user_id for shoppers, store_id=ctx.store_id for merchants,
    or all_orders=True only for support. Derive the scope from ctx, never
    from the query; reject unsupported roles or missing required identity.
    Use agent.db.list_products to map product IDs to product titles.

    The helper returns the complete authorised scope, newest first with
    order ID descending as the tie-breaker. Match product names first,
    preserve that order, then return at most five matches. Do not search
    only the 20 most recent orders. Convert matches with to_public_dict().

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), each as the dict returned by agent.db. If no orders
        match, return {"ok": True, "orders": []}.
    """
    # The scope comes from the authenticated context only, with no fallback
    # to an unrestricted search.
    if ctx.role == "shopper":
        scope: dict[str, Any] = {"user_id": ctx.user_id}
    elif ctx.role == "merchant" and ctx.store_id is not None:
        scope = {"store_id": ctx.store_id}
    elif ctx.role == "support":
        scope = {"all_orders": True}
    else:
        return permission_denied(
            f"role '{ctx.role}' (user {ctx.user_id}) has no order search scope"
        )
    with db.connection() as conn:
        candidates = db.list_order_search_candidates(conn, **scope)
        titles = {product.id: product.title or "" for product in db.list_products(conn)}
    # Score each product in scope once. WRatio tolerates typos and filler
    # words ("heavy duty vaes I bought"). Keep only the best-scoring products,
    # so "walnut novel" does not also return a "Walnut Wooden Train".
    scores = {
        product_id: fuzz.WRatio(query, titles.get(product_id, ""), processor=utils.default_process)
        for product_id in {order.product_id for order in candidates}
    }
    best = max(scores.values(), default=0)
    if best < FIND_ORDER_MIN_SCORE:
        return {"ok": True, "orders": []}
    matches = [order for order in candidates if scores[order.product_id] == best]
    return {
        "ok": True,
        "orders": [order.to_public_dict() for order in matches[:MAX_FIND_ORDER_RESULTS]],
    }


# ---------------------------------------------------------------------------
# Additional tools (HW1 Part A). Same conventions as the tools above: read
# tools, scope checked with agent.auth before anything about an order is
# returned, and structured results. None of them writes to the database.
# ---------------------------------------------------------------------------

RECENT_ORDERS_IN_SUMMARY = 5


def _order_data_issues(conn: Any, order: db.Order) -> list[dict[str, str]]:
    """Inconsistencies in an order record that make its dates untrustworthy.

    The seed plants defective records on purpose (the data_quality_cases
    table). A tool reports them instead of computing a deadline or an
    estimate from bad data, so the model can say the record is inconsistent
    and escalate. The codes match data_quality_cases.defect_type.
    """
    issues = []
    if order.shipped_at and order.delivered_at and order.shipped_at > order.delivered_at:
        issues.append({
            "code": "reversed_chronology",
            "detail": f"shipped_at {order.shipped_at} is after delivered_at {order.delivered_at}",
        })
    if order.status == "delivered" and order.delivered_at is None:
        issues.append({
            "code": "missing_required_date",
            "detail": "status is 'delivered' but delivered_at is missing",
        })
    products = {product.id: product for product in db.list_products(conn, order.store_id)}
    if order.product_id not in products:
        issues.append({
            "code": "store_mismatch",
            "detail": f"product {order.product_id} does not belong to store {order.store_id}",
        })
    return issues


def _store_policy_id(store: db.Store) -> str | None:
    """The store's own policy doc id, or None when the store has none."""
    policy_id = f"store-{store.slug}-policy"
    return policy_id if any(doc.policy_id == policy_id for doc in load_policy_docs()) else None


def _load_order_in_scope(
    conn: Any, ctx: AuthContext, order_id: int, action: str
) -> db.Order | dict[str, Any]:
    """The order, or the not_found / permission_denied result get_order uses."""
    order = db.get_order(conn, order_id)
    if order is None:
        return _error("not_found", f"no order #{order_id}")
    if not can_view_order(ctx, order.user_id, order.store_id):
        return permission_denied(
            f"role '{ctx.role}' (user {ctx.user_id}) may not {action} order #{order_id}"
        )
    return order


def check_return_eligibility(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Explain whether an order can be returned or refunded. Risk tier: read.

    Scope is the same as get_order (agent.auth.can_view_order), checked
    before anything about the order is returned.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found", ...}.
        Out of scope: agent.auth.permission_denied(...).
        On success: {"ok": True, "order_id", "status", "refund_eligible",
        "delivered_at", "return_window_days", "window_source",
        "return_deadline", "days_remaining", "restocking_fee_may_apply",
        "restocking_fee_max_percent", "policy_ids", "data_issues"}.
        `refund_eligible` is the stamped flag issue_refund enforces.
        `return_deadline` and `days_remaining` are None when the order has
        no delivery date or when `data_issues` is non-empty.
    """
    facts = load_facts()
    with db.connection() as conn:
        order = _load_order_in_scope(conn, ctx, order_id, "check")
        if isinstance(order, dict):
            return order
        store = db.get_store(conn, order.store_id)
        as_of = db.world_asof(conn)
        data_issues = _order_data_issues(conn, order)
    override = store.return_window_days_override if store else None
    window = effective_return_window_days(facts["return_window_days"], override)
    deadline: date | None = None
    if order.delivered_at is not None and not data_issues:
        deadline = order.delivered_at + timedelta(days=window)
    fee_opt_in = bool(store and store.restocking_fee_opt_in)
    policy_ids = ["cw-returns"]
    store_policy = _store_policy_id(store) if store else None
    if store_policy:
        policy_ids.append(store_policy)
    if fee_opt_in:
        policy_ids.append("cw-restocking-fees")
    return {
        "ok": True,
        "order_id": order.id,
        "status": order.status,
        "refund_eligible": order.refund_eligible,
        "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
        "return_window_days": window,
        "window_source": "store_override" if override is not None else "platform_default",
        "return_deadline": deadline.isoformat() if deadline else None,
        "days_remaining": max((deadline - as_of).days, 0) if deadline else None,
        "restocking_fee_may_apply": fee_opt_in,
        "restocking_fee_max_percent": facts["restocking_fee_max_percent"] if fee_opt_in else None,
        "policy_ids": policy_ids,
        "data_issues": data_issues,
    }


def track_shipment(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Report an order's shipment stage and expected dates. Risk tier: read.

    Scope is the same as get_order (agent.auth.can_view_order), checked
    before anything about the order is returned. Cartwheel stores no carrier
    tracking numbers, so none are returned.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found", ...}.
        Out of scope: agent.auth.permission_denied(...).
        On success: {"ok": True, "order_id", "status", "store_name",
        "ordered_at", "shipped_at", "delivered_at", "expected_ship_by",
        "expected_delivery_by", "overdue", "policy_ids", "data_issues"}.
        Expected dates use facts.yaml shipping_handling_days_max and
        shipping_transit_days_max: ship_by is set only for a placed order,
        delivery_by only for a placed or shipped order. Both are None, and
        `overdue` is False, when `data_issues` is non-empty.
    """
    facts = load_facts()
    handling = timedelta(days=facts["shipping_handling_days_max"])
    transit = timedelta(days=facts["shipping_transit_days_max"])
    with db.connection() as conn:
        order = _load_order_in_scope(conn, ctx, order_id, "track")
        if isinstance(order, dict):
            return order
        store = db.get_store(conn, order.store_id)
        as_of = db.world_asof(conn)
        data_issues = _order_data_issues(conn, order)
    ship_by = delivery_by = None
    if not data_issues:
        if order.status == "placed":
            ship_by = order.ordered_at + handling
            delivery_by = ship_by + transit
        elif order.status == "shipped" and order.shipped_at is not None:
            delivery_by = order.shipped_at + transit
    overdue = bool(
        (ship_by and as_of > ship_by) or (delivery_by and as_of > delivery_by)
    )
    return {
        "ok": True,
        "order_id": order.id,
        "status": order.status,
        "store_name": store.name if store else None,
        "ordered_at": order.ordered_at.isoformat(),
        "shipped_at": order.shipped_at.isoformat() if order.shipped_at else None,
        "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
        "expected_ship_by": ship_by.isoformat() if ship_by else None,
        "expected_delivery_by": delivery_by.isoformat() if delivery_by else None,
        "overdue": overdue,
        "policy_ids": ["cw-shipping"],
        "data_issues": data_issues,
    }


def get_store_info(ctx: AuthContext, store: str) -> dict[str, Any]:
    """Look up a store's public information and policy overrides. Risk tier: read.

    Store information is public help-center content, so every role may read
    every store and no permission check is needed (as with get_policy).

    Args:
        ctx: The caller's auth context.
        store: A store name or slug (agent.db.get_store_by_name), or a numeric
            store id such as the store_id on an order.

    Returns:
        Empty store: {"ok": False, "error": "invalid_argument", ...}.
        Unknown store: {"ok": False, "error": "not_found", ...}.
        On success: {"ok": True, "store_id", "name", "slug", "category",
        "return_window_days", "window_source", "restocking_fee_opt_in",
        "restocking_fee_max_percent", "policy_ids"}, where policy_ids lists
        the store's own policy doc when it has one.
    """
    store = store.strip()
    if not store:
        return _error("invalid_argument", "empty store")
    facts = load_facts()
    with db.connection() as conn:
        match = db.get_store(conn, int(store)) if store.isdigit() else None
        match = match or db.get_store_by_name(conn, store)
    if match is None:
        return _error("not_found", f"no store matching '{store}'")
    override = match.return_window_days_override
    policy_ids = ["cw-store-overrides"]
    store_policy = _store_policy_id(match)
    if store_policy:
        policy_ids.append(store_policy)
    return {
        "ok": True,
        "store_id": match.id,
        "name": match.name,
        "slug": match.slug,
        "category": match.category,
        "return_window_days": effective_return_window_days(facts["return_window_days"], override),
        "window_source": "store_override" if override is not None else "platform_default",
        "restocking_fee_opt_in": match.restocking_fee_opt_in,
        "restocking_fee_max_percent": (
            facts["restocking_fee_max_percent"] if match.restocking_fee_opt_in else None
        ),
        "policy_ids": policy_ids,
    }


def summarize_order_history(
    ctx: AuthContext, customer_user_id: int | None = None
) -> dict[str, Any]:
    """Summarize every order in one scope. Risk tier: read.

    Role behavior, following the access matrix in SPEC.md:
        - shopper: the caller's own orders. Naming any other customer
          returns permission_denied.
        - merchant: the caller's store's orders. Naming a customer returns
          permission_denied, because order records shown to merchants do not
          identify customers.
        - support: the named customer's orders. customer_user_id is required
          (invalid_argument without it), must exist (not_found), and must be
          a shopper (invalid_argument otherwise).

    Unlike list_my_orders, the summary covers the whole scope, not only the
    20 most recent orders.

    Returns:
        {"ok": True, "scope": "shopper" | "store", "customer_user_id",
        "store_id", "order_count", "total_usd", "first_ordered_at",
        "last_ordered_at", "by_status", "refund_eligible_order_ids",
        "open_order_ids", "recent_orders"}. by_status maps each status to
        {"count", "total_usd"}; recent_orders holds the newest
        RECENT_ORDERS_IN_SUMMARY orders as to_public_dict(). An empty scope
        is still a success with order_count 0.
    """
    if ctx.role == "shopper":
        if customer_user_id not in (None, ctx.user_id):
            return permission_denied(
                f"role 'shopper' (user {ctx.user_id}) may not view the orders of "
                f"user {customer_user_id}"
            )
        scope: dict[str, Any] = {"user_id": ctx.user_id}
    elif ctx.role == "merchant" and ctx.store_id is not None:
        if customer_user_id is not None:
            return permission_denied(
                f"role 'merchant' (user {ctx.user_id}) may summarize only store "
                f"{ctx.store_id}'s orders, not a customer's history"
            )
        scope = {"store_id": ctx.store_id}
    elif ctx.role == "support":
        if customer_user_id is None:
            return _error(
                "invalid_argument",
                "support staff must name the customer_user_id to summarize",
            )
        scope = {"user_id": customer_user_id}
    else:
        return permission_denied(
            f"role '{ctx.role}' (user {ctx.user_id}) has no order history scope"
        )
    with db.connection() as conn:
        if ctx.role == "support":
            customer = db.get_user(conn, customer_user_id)
            if customer is None:
                return _error("not_found", f"no user #{customer_user_id}")
            if customer.role != "shopper":
                return _error(
                    "invalid_argument",
                    f"user #{customer_user_id} is a {customer.role}, not a shopper",
                )
        orders = db.list_order_search_candidates(conn, **scope)
    by_status: dict[str, dict[str, Any]] = {}
    for order in orders:
        bucket = by_status.setdefault(order.status, {"count": 0, "total_usd": 0.0})
        bucket["count"] += 1
        bucket["total_usd"] = round(bucket["total_usd"] + order.total_usd, 2)
    return {
        "ok": True,
        "scope": "store" if "store_id" in scope else "shopper",
        "customer_user_id": scope.get("user_id"),
        "store_id": scope.get("store_id"),
        "order_count": len(orders),
        "total_usd": round(sum(order.total_usd for order in orders), 2),
        "first_ordered_at": min(o.ordered_at for o in orders).isoformat() if orders else None,
        "last_ordered_at": max(o.ordered_at for o in orders).isoformat() if orders else None,
        "by_status": dict(sorted(by_status.items())),
        "refund_eligible_order_ids": [o.id for o in orders if o.refund_eligible],
        "open_order_ids": [o.id for o in orders if o.status in ("placed", "shipped")],
        "recent_orders": [o.to_public_dict() for o in orders[:RECENT_ORDERS_IN_SUMMARY]],
    }
