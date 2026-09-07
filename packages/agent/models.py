from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from packages.environment.models import DefaultClock, DefaultIdProvider, IdProvider, Clock


_DEFAULT_CLOCK = DefaultClock()
_DEFAULT_ID_PROVIDER = DefaultIdProvider()


def utc_now() -> datetime:
    return _DEFAULT_CLOCK.now()


def generate_id(prefix: str = "") -> str:
    return _DEFAULT_ID_PROVIDER.generate(prefix)


class ToolErrorCode(StrEnum):
    """Structured error codes for tool failures."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_CUSTOMER = "unknown_customer"
    UNKNOWN_SUBSCRIPTION = "unknown_subscription"
    UNKNOWN_BILLING_ACCOUNT = "unknown_billing_account"
    INVALID_CANCELLATION_STATE = "invalid_cancellation_state"
    ALREADY_CANCELLED = "already_cancelled"
    MISSING_BILLING_ACCOUNT = "missing_billing_account"
    NO_BILLING_ACCOUNT_FOR_SUBSCRIPTION = "no_billing_account_for_subscription"
    SUBSCRIPTION_ALREADY_CANCELLED = "subscription_already_cancelled"
    BILLING_ACCOUNT_ALREADY_CANCELLED = "billing_account_already_cancelled"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"


class ToolError(Exception):
    """Base exception for tool errors with structured error codes."""

    def __init__(self, message: str, error_code: ToolErrorCode) -> None:
        super().__init__(message)
        self.error_code = error_code


class UnknownCustomerError(ToolError):
    def __init__(self, customer_id: str) -> None:
        super().__init__(f"Unknown customer: {customer_id}", ToolErrorCode.UNKNOWN_CUSTOMER)


class UnknownSubscriptionError(ToolError):
    def __init__(self, subscription_id: str) -> None:
        super().__init__(f"Unknown subscription: {subscription_id}", ToolErrorCode.UNKNOWN_SUBSCRIPTION)


class UnknownBillingAccountError(ToolError):
    def __init__(self, billing_id: str) -> None:
        super().__init__(f"Unknown billing account: {billing_id}", ToolErrorCode.UNKNOWN_BILLING_ACCOUNT)


class SubscriptionAlreadyCancelledError(ToolError):
    def __init__(self, subscription_id: str) -> None:
        super().__init__(f"Subscription already cancelled: {subscription_id}", ToolErrorCode.SUBSCRIPTION_ALREADY_CANCELLED)


class BillingAccountAlreadyCancelledError(ToolError):
    def __init__(self, billing_id: str) -> None:
        super().__init__(f"Billing account is already cancelled: {billing_id}", ToolErrorCode.BILLING_ACCOUNT_ALREADY_CANCELLED)


class NoBillingAccountForSubscriptionError(ToolError):
    def __init__(self, subscription_id: str) -> None:
        super().__init__(f"No billing account for subscription: {subscription_id}", ToolErrorCode.NO_BILLING_ACCOUNT_FOR_SUBSCRIPTION)


class InvalidCancellationStateError(ToolError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ToolErrorCode.INVALID_CANCELLATION_STATE)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    read_only: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_schema",
            MappingProxyType(deepcopy(dict(self.input_schema))),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(deepcopy(dict(self.arguments))),
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    output: Any
    error: str | None
    error_code: ToolErrorCode | None
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", deepcopy(self.output))


@dataclass(frozen=True, slots=True)
class ToolTraceEntry:
    call: ToolCall
    result: ToolResult


@dataclass
class ToolTrace:
    entries: list[ToolTraceEntry] = field(default_factory=list)

    def append(self, call: ToolCall, result: ToolResult) -> None:
        self.entries.append(ToolTraceEntry(call=call, result=result))

    def snapshot(self) -> "ToolTrace":
        return ToolTrace(
            entries=[
                ToolTraceEntry(
                    call=ToolCall(
                        call_id=entry.call.call_id,
                        tool_name=entry.call.tool_name,
                        arguments=entry.call.arguments,
                        timestamp=entry.call.timestamp,
                    ),
                    result=ToolResult(
                        call_id=entry.result.call_id,
                        tool_name=entry.result.tool_name,
                        success=entry.result.success,
                        output=entry.result.output,
                        error=entry.result.error,
                        error_code=entry.result.error_code,
                        timestamp=entry.result.timestamp,
                    ),
                )
                for entry in self.entries
            ]
        )
