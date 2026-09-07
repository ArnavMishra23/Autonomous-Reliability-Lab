from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


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