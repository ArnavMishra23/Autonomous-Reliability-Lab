from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable, Type, TypeVar

from .models import (
    BillingAccount,
    BillingStatus,
    Customer,
    DefaultClock,
    DefaultIdProvider,
    EnvironmentMutation,
    EnvironmentEvent,
    EventType,
    IdProvider,
    Clock,
    Observation,
    Subscription,
    SubscriptionStatus,
    utc_now,
)
from .state import EnvironmentState

T = TypeVar("T", bound=Enum)


def _normalize_enum(value: T | str, enum_type: Type[T], field_name: str) -> T:
    """Normalize a string or enum value to the given enum type."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            valid_values = [e.value for e in enum_type]
            raise ValueError(
                f"Invalid {field_name}: '{value}'. Valid values: {valid_values}"
            ) from exc
    raise TypeError(
        f"{field_name} must be a {enum_type.__name__} or string, got {type(value).__name__}"
    )


class EnvironmentService:
    """Controls the authoritative simulated environment."""

    def __init__(
        self,
        state: EnvironmentState | None = None,
        *,
        id_provider: IdProvider | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._state = deepcopy(state) if state is not None else EnvironmentState()
        self._id_provider = id_provider or DefaultIdProvider()
        self._clock = clock or DefaultClock()

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
            id=self._id_provider.generate("cust_"),
            name=name,
            email=email,
        )
        self._state.customers[customer.id] = customer
        event = EnvironmentEvent(
            id=self._id_provider.generate("evt_"),
            event_type=EventType.CUSTOMER_CREATED,
            timestamp=self._clock.now(),
            target=customer.id,
            field="customer",
            old_value=None,
            new_value={"name": name, "email": email},
            metadata={"source": "creation"},
        )
        self._state.events.append(event)
        return deepcopy(customer)

    def create_subscription(
        self,
        *,
        customer_id: str,
        plan: str,
        status: SubscriptionStatus | str = SubscriptionStatus.ACTIVE,
    ) -> Subscription:
        if customer_id not in self._state.customers:
            raise ValueError(f"Unknown customer: {customer_id}")

        normalized_status = _normalize_enum(status, SubscriptionStatus, "subscription status")
        subscription = Subscription(
            id=self._id_provider.generate("sub_"),
            customer_id=customer_id,
            status=normalized_status,
            plan=plan,
        )

        self._state.subscriptions[subscription.id] = subscription
        event = EnvironmentEvent(
            id=self._id_provider.generate("evt_"),
            event_type=EventType.SUBSCRIPTION_CREATED,
            timestamp=self._clock.now(),
            target=subscription.id,
            field="subscription",
            old_value=None,
            new_value={"customer_id": customer_id, "plan": plan, "status": normalized_status.value},
            metadata={"source": "creation"},
        )
        self._state.events.append(event)
        return deepcopy(subscription)

    def create_billing_account(
        self,
        *,
        subscription_id: str,
        status: BillingStatus | str = BillingStatus.ACTIVE,
    ) -> BillingAccount:
        if subscription_id not in self._state.subscriptions:
            raise ValueError(f"Unknown subscription: {subscription_id}")

        normalized_status = _normalize_enum(status, BillingStatus, "billing status")
        billing = BillingAccount(
            id=self._id_provider.generate("bill_"),
            subscription_id=subscription_id,
            status=normalized_status,
        )

        self._state.billing_accounts[billing.id] = billing
        event = EnvironmentEvent(
            id=self._id_provider.generate("evt_"),
            event_type=EventType.BILLING_ACCOUNT_CREATED,
            timestamp=self._clock.now(),
            target=billing.id,
            field="billing_account",
            old_value=None,
            new_value={"subscription_id": subscription_id, "status": normalized_status.value},
            metadata={"source": "creation"},
        )
        self._state.events.append(event)
        return deepcopy(billing)

    def change_billing_status(
        self,
        *,
        billing_id: str,
        new_status: BillingStatus | str,
        metadata: dict | None = None,
    ) -> EnvironmentEvent:
        normalized_status = _normalize_enum(new_status, BillingStatus, "billing status")
        return self._change_billing_status(
            billing_id=billing_id,
            new_status=normalized_status,
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
            id=event_id or self._id_provider.generate("evt_"),
            event_type=EventType.STATE_CHANGE,
            timestamp=timestamp or self._clock.now(),
            target=billing_id,
            field="status",
            old_value=old_status.value,
            new_value=new_status.value,
            metadata=deepcopy(metadata or {}),
        )

        self._state.events.append(event)

        return deepcopy(event)

    def cancel_subscription(self, *, subscription_id: str) -> EnvironmentEvent:
        """Cancel a subscription when its associated billing state permits it."""
        subscription = self._state.subscriptions.get(subscription_id)
        if subscription is None:
            raise ValueError(f"Unknown subscription: {subscription_id}")
        if subscription.status == SubscriptionStatus.CANCELLED:
            raise ValueError(f"Subscription already cancelled: {subscription_id}")

        billing = next(
            (
                account
                for account in self._state.billing_accounts.values()
                if account.subscription_id == subscription_id
            ),
            None,
        )
        if billing is None:
            raise ValueError(f"No billing account for subscription: {subscription_id}")
        if billing.status == BillingStatus.CANCELLED:
            raise ValueError(
                f"Billing account is already cancelled: {billing.id}"
            )

        self._state.subscriptions[subscription_id] = replace(
            subscription,
            status=SubscriptionStatus.CANCELLED,
        )
        event = EnvironmentEvent(
            id=self._id_provider.generate("evt_"),
            event_type=EventType.STATE_CHANGE,
            timestamp=self._clock.now(),
            target=subscription_id,
            field="status",
            old_value=subscription.status.value,
            new_value=SubscriptionStatus.CANCELLED.value,
        )
        self._state.events.append(event)
        return deepcopy(event)

    def reconcile_subscription_state(self, *, subscription_id: str) -> EnvironmentEvent:
        """Resolve a subscription after authoritative billing cancellation."""
        subscription = self._state.subscriptions.get(subscription_id)
        if subscription is None:
            raise ValueError(f"Unknown subscription: {subscription_id}")
        if subscription.status == SubscriptionStatus.CANCELLED:
            raise ValueError(f"Subscription already cancelled: {subscription_id}")

        billing = next(
            (
                account
                for account in self._state.billing_accounts.values()
                if account.subscription_id == subscription_id
            ),
            None,
        )
        if billing is None:
            raise ValueError(f"No billing account for subscription: {subscription_id}")
        if billing.status != BillingStatus.CANCELLED:
            raise ValueError(f"Billing account is not cancelled: {billing.id}")

        self._state.subscriptions[subscription_id] = replace(
            subscription,
            status=SubscriptionStatus.CANCELLED,
        )
        event = EnvironmentEvent(
            id=self._id_provider.generate("evt_"),
            event_type=EventType.STATE_CHANGE,
            timestamp=self._clock.now(),
            target=subscription_id,
            field="status",
            old_value=subscription.status.value,
            new_value=SubscriptionStatus.CANCELLED.value,
            metadata={"source": "billing_reconciliation"},
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
            observation_id=self._id_provider.generate("obs_"),
            timestamp=self._clock.now(),
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
        *,
        id_provider: IdProvider | None = None,
        clock: Clock | None = None,
    ) -> "EnvironmentService":
        """Replay mutations from a snapshot with deterministic replay artifacts."""
        service = cls(snapshot, id_provider=id_provider, clock=clock)
        for index, mutation in enumerate(mutations):
            service._apply_mutation(mutation, index=index)
        return service

    def _apply_mutation(self, mutation: EnvironmentMutation, *, index: int) -> Any:
        parameters = deepcopy(dict(mutation.parameters))
        operation = mutation.operation

        if operation == "create_customer":
            return self._create_customer_with_id(
                entity_id=parameters.pop("customer_id", self._id_provider.generate("cust_replay_")),
                name=parameters.pop("name"),
                email=parameters.pop("email"),
            )
        if operation == "create_subscription":
            return self._create_subscription_with_id(
                entity_id=parameters.pop("subscription_id", self._id_provider.generate("sub_replay_")),
                customer_id=parameters.pop("customer_id"),
                plan=parameters.pop("plan"),
                status=parameters.pop("status", SubscriptionStatus.ACTIVE),
            )
        if operation == "create_billing_account":
            return self._create_billing_with_id(
                entity_id=parameters.pop("billing_id", self._id_provider.generate("bill_replay_")),
                subscription_id=parameters.pop("subscription_id"),
                status=parameters.pop("status", BillingStatus.ACTIVE),
            )
        if operation == "change_billing_status":
            return self._change_billing_status(
                **parameters,
                event_id=self._id_provider.generate("evt_replay_"),
                timestamp=self._clock.now(),
            )
        raise ValueError(f"Unknown mutation operation: {operation}")

    def _create_customer_with_id(self, *, entity_id: str, name: str, email: str) -> Customer:
        customer = Customer(id=entity_id, name=name, email=email)
        self._state.customers[customer.id] = customer
        event = EnvironmentEvent(
            id=self._id_provider.generate("evt_"),
            event_type=EventType.CUSTOMER_CREATED,
            timestamp=self._clock.now(),
            target=customer.id,
            field="customer",
            old_value=None,
            new_value={"name": name, "email": email},
            metadata={"source": "replay"},
        )
        self._state.events.append(event)
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
        event = EnvironmentEvent(
            id=self._id_provider.generate("evt_"),
            event_type=EventType.SUBSCRIPTION_CREATED,
            timestamp=self._clock.now(),
            target=subscription.id,
            field="subscription",
            old_value=None,
            new_value={"customer_id": customer_id, "plan": plan, "status": status.value},
            metadata={"source": "replay"},
        )
        self._state.events.append(event)
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
        event = EnvironmentEvent(
            id=self._id_provider.generate("evt_"),
            event_type=EventType.BILLING_ACCOUNT_CREATED,
            timestamp=self._clock.now(),
            target=billing.id,
            field="billing_account",
            old_value=None,
            new_value={"subscription_id": subscription_id, "status": status.value},
            metadata={"source": "replay"},
        )
        self._state.events.append(event)
        return deepcopy(billing)