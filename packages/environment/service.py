from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from uuid import uuid4

from .models import (
    BillingAccount,
    BillingStatus,
    Customer,
    EnvironmentMutation,
    EnvironmentEvent,
    EventType,
    Observation,
    Subscription,
    SubscriptionStatus,
    utc_now,
)
from .state import EnvironmentState


class EnvironmentService:
    """Controls the authoritative simulated environment."""

    def __init__(self, state: EnvironmentState | None = None) -> None:
        self._state = deepcopy(state) if state is not None else EnvironmentState()

    @property
    def state(self) -> EnvironmentState:
        """Return a safe copy rather than exposing authoritative state."""
        return self._state.snapshot()

    def create_customer(
        self,
        *,
        name: str,
        email: str,
    ) -> Customer:
        customer = Customer(
            id=f"cust_{uuid4().hex[:8]}",
            name=name,
            email=email,
        )
        self._state.customers[customer.id] = customer
        return deepcopy(customer)

    def create_subscription(
        self,
        *,
        customer_id: str,
        plan: str,
        status: SubscriptionStatus = SubscriptionStatus.ACTIVE,
    ) -> Subscription:
        if customer_id not in self._state.customers:
            raise ValueError(f"Unknown customer: {customer_id}")

        subscription = Subscription(
            id=f"sub_{uuid4().hex[:8]}",
            customer_id=customer_id,
            status=status,
            plan=plan,
        )

        self._state.subscriptions[subscription.id] = subscription
        return deepcopy(subscription)

    def create_billing_account(
        self,
        *,
        subscription_id: str,
        status: BillingStatus = BillingStatus.ACTIVE,
    ) -> BillingAccount:
        if subscription_id not in self._state.subscriptions:
            raise ValueError(f"Unknown subscription: {subscription_id}")

        billing = BillingAccount(
            id=f"bill_{uuid4().hex[:8]}",
            subscription_id=subscription_id,
            status=status,
        )

        self._state.billing_accounts[billing.id] = billing
        return deepcopy(billing)

    def change_billing_status(
        self,
        *,
        billing_id: str,
        new_status: BillingStatus,
        metadata: dict | None = None,
    ) -> EnvironmentEvent:
        return self._change_billing_status(
            billing_id=billing_id,
            new_status=new_status,
            metadata=metadata,
        )

    def _change_billing_status(
        self,
        *,
        billing_id: str,
        new_status: BillingStatus,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> EnvironmentEvent:
        billing = self._state.billing_accounts.get(billing_id)

        if billing is None:
            raise ValueError(f"Unknown billing account: {billing_id}")

        old_status = billing.status

        updated = replace(billing, status=new_status)
        self._state.billing_accounts[billing_id] = updated

        event = EnvironmentEvent(
            id=event_id or f"evt_{uuid4().hex[:8]}",
            event_type=EventType.STATE_CHANGE,
            timestamp=timestamp or utc_now(),
            target=billing_id,
            field="status",
            old_value=old_status.value,
            new_value=new_status.value,
            metadata=deepcopy(metadata or {}),
        )

        self._state.events.append(event)

        return deepcopy(event)

    def get_customer(self, customer_id: str) -> Customer:
        try:
            return deepcopy(self._state.customers[customer_id])
        except KeyError as exc:
            raise ValueError(f"Unknown customer: {customer_id}") from exc

    def get_subscription(self, subscription_id: str) -> Subscription:
        try:
            return deepcopy(self._state.subscriptions[subscription_id])
        except KeyError as exc:
            raise ValueError(f"Unknown subscription: {subscription_id}") from exc

    def get_billing(self, billing_id: str) -> BillingAccount:
        try:
            return deepcopy(self._state.billing_accounts[billing_id])
        except KeyError as exc:
            raise ValueError(f"Unknown billing account: {billing_id}") from exc

    def snapshot(self) -> EnvironmentState:
        return self._state.snapshot()

    def restore(self, snapshot: EnvironmentState) -> None:
        """Restore authoritative state without retaining snapshot references."""
        if not isinstance(snapshot, EnvironmentState):
            raise TypeError("snapshot must be an EnvironmentState")
        self._state = deepcopy(snapshot)

    def observe_customer(self, customer_id: str) -> Observation:
        customer = self.get_customer(customer_id)
        return self._observation(
            entity_type="customer",
            entity_id=customer.id,
            value=asdict(customer),
        )

    def observe_subscription(self, subscription_id: str) -> Observation:
        subscription = self.get_subscription(subscription_id)
        return self._observation(
            entity_type="subscription",
            entity_id=subscription.id,
            value=asdict(subscription),
        )

    def observe_billing(self, billing_id: str) -> Observation:
        billing = self.get_billing(billing_id)
        return self._observation(
            entity_type="billing_account",
            entity_id=billing.id,
            value=asdict(billing),
        )

    def _observation(
        self,
        *,
        entity_type: str,
        entity_id: str,
        value: dict[str, Any],
    ) -> Observation:
        return Observation(
            observation_id=f"obs_{uuid4().hex[:8]}",
            timestamp=utc_now(),
            source="environment",
            entity_type=entity_type,
            entity_id=entity_id,
            field="state",
            value=value,
        )

    def observable_state(self) -> EnvironmentState:
        """Return a safe snapshot of all observable state."""
        return self.snapshot()

    @classmethod
    def replay(
        cls,
        snapshot: EnvironmentState,
        mutations: Iterable[EnvironmentMutation],
    ) -> "EnvironmentService":
        """Replay mutations from a snapshot with deterministic replay artifacts."""
        service = cls(snapshot)
        for index, mutation in enumerate(mutations):
            service._apply_mutation(mutation, index=index)
        return service

    def _apply_mutation(self, mutation: EnvironmentMutation, *, index: int) -> Any:
        parameters = deepcopy(dict(mutation.parameters))
        operation = mutation.operation

        if operation == "create_customer":
            return self._create_customer_with_id(
                entity_id=parameters.pop("customer_id", f"cust_replay_{index:08d}"),
                **parameters,
            )
        if operation == "create_subscription":
            return self._create_subscription_with_id(
                entity_id=parameters.pop("subscription_id", f"sub_replay_{index:08d}"),
                **parameters,
            )
        if operation == "create_billing_account":
            return self._create_billing_with_id(
                entity_id=parameters.pop("billing_id", f"bill_replay_{index:08d}"),
                **parameters,
            )
        if operation == "change_billing_status":
            return self._change_billing_status(
                **parameters,
                event_id=f"evt_replay_{index:08d}",
                timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc)
                + timedelta(seconds=index),
            )
        raise ValueError(f"Unknown mutation operation: {operation}")

    def _create_customer_with_id(self, *, entity_id: str, name: str, email: str) -> Customer:
        customer = Customer(id=entity_id, name=name, email=email)
        self._state.customers[customer.id] = customer
        return deepcopy(customer)

    def _create_subscription_with_id(
        self,
        *,
        entity_id: str,
        customer_id: str,
        plan: str,
        status: SubscriptionStatus = SubscriptionStatus.ACTIVE,
    ) -> Subscription:
        if customer_id not in self._state.customers:
            raise ValueError(f"Unknown customer: {customer_id}")
        subscription = Subscription(
            id=entity_id,
            customer_id=customer_id,
            status=status,
            plan=plan,
        )
        self._state.subscriptions[subscription.id] = subscription
        return deepcopy(subscription)

    def _create_billing_with_id(
        self,
        *,
        entity_id: str,
        subscription_id: str,
        status: BillingStatus = BillingStatus.ACTIVE,
    ) -> BillingAccount:
        if subscription_id not in self._state.subscriptions:
            raise ValueError(f"Unknown subscription: {subscription_id}")
        billing = BillingAccount(
            id=entity_id,
            subscription_id=subscription_id,
            status=status,
        )
        self._state.billing_accounts[billing.id] = billing
        return deepcopy(billing)