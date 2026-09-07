from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .models import (
    BillingAccount,
    Customer,
    EnvironmentEvent,
    Subscription,
)


@dataclass
class EnvironmentState:
    """
    Authoritative state of the simulated world.

    This object represents reality as known by the environment itself.
    The target agent will never directly mutate this object.
    """

    customers: dict[str, Customer] = field(default_factory=dict)
    subscriptions: dict[str, Subscription] = field(default_factory=dict)
    billing_accounts: dict[str, BillingAccount] = field(default_factory=dict)
    events: list[EnvironmentEvent] = field(default_factory=list)

    def snapshot(self) -> "EnvironmentState":
        """Return a deep copy suitable for deterministic replay."""
        return deepcopy(self)