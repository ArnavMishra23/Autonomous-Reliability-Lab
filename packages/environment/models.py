from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


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