from __future__ import annotations

from dataclasses import asdict
from typing import Any

from packages.environment.service import EnvironmentService

from .models import ToolDefinition
from .registry import ToolRegistry


def build_tool_registry(environment: EnvironmentService) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_customer",
            description="Read a customer from the authoritative environment.",
            input_schema=_schema("customer_id"),
            read_only=True,
        ),
        lambda arguments: asdict(environment.get_customer(arguments["customer_id"])),
    )
    registry.register(
        ToolDefinition(
            name="get_subscription",
            description="Read a subscription from the authoritative environment.",
            input_schema=_schema("subscription_id"),
            read_only=True,
        ),
        lambda arguments: asdict(
            environment.get_subscription(arguments["subscription_id"])
        ),
    )
    registry.register(
        ToolDefinition(
            name="get_billing_status",
            description="Read billing status from the authoritative environment.",
            input_schema=_schema("billing_id"),
            read_only=True,
        ),
        lambda arguments: _billing_output(
            environment, arguments["billing_id"]
        ),
    )
    registry.register(
        ToolDefinition(
            name="cancel_subscription",
            description="Cancel a subscription when its billing state permits it.",
            input_schema=_schema("subscription_id"),
            read_only=False,
        ),
        lambda arguments: _cancel_subscription(
            environment, arguments["subscription_id"]
        ),
    )
    return registry


def _schema(argument_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "required": [argument_name],
        "properties": {
            argument_name: {"type": "string"},
        },
    }


def _billing_output(environment: EnvironmentService, billing_id: str) -> dict[str, Any]:
    billing = environment.get_billing(billing_id)
    return {
        "id": billing.id,
        "subscription_id": billing.subscription_id,
        "status": billing.status.value,
    }


def _cancel_subscription(
    environment: EnvironmentService,
    subscription_id: str,
) -> dict[str, Any]:
    environment.get_subscription(subscription_id)
    billing = next(
        (
            account
            for account in environment.snapshot().billing_accounts.values()
            if account.subscription_id == subscription_id
        ),
        None,
    )
    if billing is None:
        raise ValueError(f"No billing account for subscription: {subscription_id}")
    if billing.status.value == "cancelled":
        raise ValueError(f"Billing account is already cancelled: {billing.id}")

    event = environment.cancel_subscription(subscription_id=subscription_id)
    return {
        "subscription": asdict(environment.get_subscription(subscription_id)),
        "event": {
            "id": event.id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "target": event.target,
            "field": event.field,
            "old_value": event.old_value,
            "new_value": event.new_value,
            "metadata": event.metadata,
        },
    }
