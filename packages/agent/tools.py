from __future__ import annotations

from dataclasses import asdict
from typing import Any

from packages.environment.models import BillingStatus, SubscriptionStatus
from packages.environment.service import EnvironmentService

from .models import ToolDefinition, ToolErrorCode, ToolError, UnknownCustomerError, UnknownSubscriptionError, UnknownBillingAccountError, SubscriptionAlreadyCancelledError, BillingAccountAlreadyCancelledError, NoBillingAccountForSubscriptionError, InvalidCancellationStateError
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
        lambda arguments: _get_customer(environment, arguments["customer_id"]),
    )
    registry.register(
        ToolDefinition(
            name="get_subscription",
            description="Read a subscription from the authoritative environment.",
            input_schema=_schema("subscription_id"),
            read_only=True,
        ),
        lambda arguments: _get_subscription(environment, arguments["subscription_id"]),
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
    registry.register(
        ToolDefinition(
            name="reconcile_subscription_state",
            description="Reconcile a subscription after authoritative billing cancellation.",
            input_schema=_schema("subscription_id"),
            read_only=False,
        ),
        lambda arguments: _reconcile_subscription(
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


def _get_customer(environment: EnvironmentService, customer_id: str) -> dict[str, Any]:
    try:
        return asdict(environment.get_customer(customer_id))
    except ValueError as exc:
        raise UnknownCustomerError(customer_id) from exc


def _get_subscription(environment: EnvironmentService, subscription_id: str) -> dict[str, Any]:
    try:
        return asdict(environment.get_subscription(subscription_id))
    except ValueError as exc:
        raise UnknownSubscriptionError(subscription_id) from exc


def _billing_output(environment: EnvironmentService, billing_id: str) -> dict[str, Any]:
    try:
        billing = environment.get_billing(billing_id)
    except ValueError as exc:
        raise UnknownBillingAccountError(billing_id) from exc
    return {
        "id": billing.id,
        "subscription_id": billing.subscription_id,
        "status": billing.status.value,
    }


def _cancel_subscription(
    environment: EnvironmentService,
    subscription_id: str,
) -> dict[str, Any]:
    try:
        environment.get_subscription(subscription_id)
    except ValueError as exc:
        raise UnknownSubscriptionError(subscription_id) from exc
    billing = next(
        (
            account
            for account in environment.snapshot().billing_accounts.values()
            if account.subscription_id == subscription_id
        ),
        None,
    )
    if billing is None:
        raise NoBillingAccountForSubscriptionError(subscription_id)
    if billing.status == BillingStatus.CANCELLED:
        raise BillingAccountAlreadyCancelledError(billing.id)

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


def _reconcile_subscription(
    environment: EnvironmentService,
    subscription_id: str,
) -> dict[str, Any]:
    try:
        event = environment.reconcile_subscription_state(subscription_id=subscription_id)
    except ValueError as exc:
        error_msg = str(exc)
        if "Unknown subscription" in error_msg:
            raise UnknownSubscriptionError(subscription_id) from exc
        if "already cancelled" in error_msg:
            raise SubscriptionAlreadyCancelledError(subscription_id) from exc
        if "No billing account" in error_msg:
            raise NoBillingAccountForSubscriptionError(subscription_id) from exc
        if "not cancelled" in error_msg:
            raise InvalidCancellationStateError("Billing account is not cancelled") from exc
        raise
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
