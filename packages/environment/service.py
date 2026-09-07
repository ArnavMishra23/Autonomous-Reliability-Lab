from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from .models import (
    BillingAccount,
    BillingStatus,
    Customer,
    EnvironmentEvent,
    EventType,
    Subscription,
    SubscriptionStatus,
    utc_now,
)
from .state import EnvironmentState


class EnvironmentService:
    """Controls the authoritative simulated environment."""

    def __init__(self, state: EnvironmentState | None = None) -> None:
        self._state = state or EnvironmentState()

    @property
    def state(self) -> EnvironmentState:
        return self._state

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
        return customer

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
        return subscription

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
        return billing

    def change_billing_status(
        self,
        *,
        billing_id: str,
        new_status: BillingStatus,
        metadata: dict | None = None,
    ) -> EnvironmentEvent:
        billing = self._state.billing_accounts.get(billing_id)

        if billing is None:
            raise ValueError(f"Unknown billing account: {billing_id}")

        old_status = billing.status

        updated = replace(billing, status=new_status)
        self._state.billing_accounts[billing_id] = updated

        event = EnvironmentEvent(
            id=f"evt_{uuid4().hex[:8]}",
            event_type=EventType.STATE_CHANGE,
            timestamp=utc_now(),
            target=billing_id,
            field="status",
            old_value=old_status.value,
            new_value=new_status.value,
            metadata=metadata or {},
        )

        self._state.events.append(event)

        return event

    def get_customer(self, customer_id: str) -> Customer:
        try:
            return self._state.customers[customer_id]
        except KeyError as exc:
            raise ValueError(f"Unknown customer: {customer_id}") from exc

    def get_subscription(self, subscription_id: str) -> Subscription:
        try:
            return self._state.subscriptions[subscription_id]
        except KeyError as exc:
            raise ValueError(f"Unknown subscription: {subscription_id}") from exc

    def get_billing(self, billing_id: str) -> BillingAccount:
        try:
            return self._state.billing_accounts[billing_id]
        except KeyError as exc:
            raise ValueError(f"Unknown billing account: {billing_id}") from exc

    def snapshot(self) -> EnvironmentState:
        return self._state.snapshot()