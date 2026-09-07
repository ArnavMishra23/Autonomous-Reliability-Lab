from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol


class IdProvider(Protocol):
    def generate(self, prefix: str = "") -> str:
        """Generate a new unique identifier."""
        ...


class Clock(Protocol):
    def now(self) -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        ...


class DefaultIdProvider:
    """Default implementation using uuid4."""

    def __init__(self) -> None:
        from uuid import uuid4
        self._uuid4 = uuid4

    def generate(self, prefix: str = "") -> str:
        return f"{prefix}{self._uuid4().hex[:8]}"


class DefaultClock:
    """Default implementation using datetime.now(timezone.utc)."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class DeterministicIdProvider:
    """Deterministic ID provider for testing/replay.

    Generates sequential IDs with a given prefix and namespace.
    """

    def __init__(self, *, namespace: str = "default") -> None:
        self._namespace = namespace
        self._counters: dict[str, int] = {}

    def generate(self, prefix: str = "") -> str:
        key = f"{self._namespace}:{prefix}"
        self._counters[key] = self._counters.get(key, 0) + 1
        return f"{prefix}{key}_{self._counters[key]:08d}"

    def reset(self) -> None:
        self._counters.clear()


class DeterministicClock:
    """Deterministic clock for testing/replay.

    Returns a fixed base time plus an incrementing offset.
    """

    def __init__(self, *, base: datetime | None = None, tick_seconds: float = 1.0) -> None:
        self._base = base or datetime(2000, 1, 1, tzinfo=timezone.utc)
        self._tick = tick_seconds
        self._counter = 0

    def now(self) -> datetime:
        from datetime import timedelta
        result = self._base + timedelta(seconds=self._counter * self._tick)
        self._counter += 1
        return result

    def reset(self) -> None:
        self._counter = 0


_DEFAULT_CLOCK = DefaultClock()


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return _DEFAULT_CLOCK.now()


class SubscriptionStatus(StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class BillingStatus(StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"


class EventType(StrEnum):
    STATE_CHANGE = "state_change"
    CUSTOMER_CREATED = "customer_created"
    SUBSCRIPTION_CREATED = "subscription_created"
    BILLING_ACCOUNT_CREATED = "billing_account_created"


@dataclass(frozen=True, slots=True)
class EnvironmentMutation:
    """A small, serializable description of one state transition."""

    operation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(deepcopy(dict(self.parameters))),
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """A safe, value-only representation of observable environment state."""

    observation_id: str
    timestamp: datetime
    source: str
    entity_type: str
    entity_id: str
    field: str
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", deepcopy(self.value))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )


@dataclass(slots=True)
class Customer:
    id: str
    name: str
    email: str


@dataclass(slots=True)
class Subscription:
    id: str
    customer_id: str
    status: SubscriptionStatus
    plan: str


@dataclass(slots=True)
class BillingAccount:
    id: str
    subscription_id: str
    status: BillingStatus


@dataclass(slots=True)
class EnvironmentEvent:
    id: str
    event_type: EventType
    timestamp: datetime
    target: str
    field: str
    old_value: Any
    new_value: Any
    metadata: dict[str, Any] = field(default_factory=dict)