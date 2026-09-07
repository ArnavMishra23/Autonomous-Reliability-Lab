import pytest

from packages.environment.models import BillingStatus
from packages.environment.models import EnvironmentMutation
from packages.environment.service import EnvironmentService


def test_billing_state_change_creates_event() -> None:
    environment = EnvironmentService()

    customer = environment.create_customer(
        name="Maya",
        email="maya@example.com",
    )

    subscription = environment.create_subscription(
        customer_id=customer.id,
        plan="pro",
    )

    billing = environment.create_billing_account(
        subscription_id=subscription.id,
    )

    event = environment.change_billing_status(
        billing_id=billing.id,
        new_status=BillingStatus.CANCELLED,
    )

    assert environment.get_billing(billing.id).status == BillingStatus.CANCELLED
    assert event.old_value == "active"
    assert event.new_value == "cancelled"
    assert len(environment.state.events) == 1


def test_snapshot_is_independent() -> None:
    environment = EnvironmentService()

    customer = environment.create_customer(
        name="Arnav",
        email="arnav@example.com",
    )

    snapshot = environment.snapshot()

    environment.create_subscription(
        customer_id=customer.id,
        plan="pro",
    )

    assert len(snapshot.subscriptions) == 0
    assert len(environment.state.subscriptions) == 1


def test_snapshot_mutation_restore_round_trip() -> None:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    snapshot = environment.snapshot()

    environment.change_billing_status(
        billing_id=billing.id,
        new_status=BillingStatus.CANCELLED,
    )
    environment.restore(snapshot)

    assert environment.snapshot() == snapshot
    assert environment.get_billing(billing.id).status == BillingStatus.ACTIVE


def test_restore_isolated_from_snapshot_and_future_mutations() -> None:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Arnav", email="arnav@example.com")
    snapshot = environment.snapshot()

    environment.restore(snapshot)
    environment.create_subscription(customer_id=customer.id, plan="pro")

    assert snapshot.subscriptions == {}
    assert environment.snapshot().subscriptions != snapshot.subscriptions


def test_restore_restores_event_history_without_creating_event() -> None:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    environment.change_billing_status(
        billing_id=billing.id,
        new_status=BillingStatus.CANCELLED,
        metadata={"reason": "test"},
    )
    snapshot = environment.snapshot()
    environment.change_billing_status(
        billing_id=billing.id,
        new_status=BillingStatus.PAST_DUE,
    )

    environment.restore(snapshot)

    assert environment.snapshot().events == snapshot.events
    assert len(environment.state.events) == 1


def test_event_and_state_references_are_safe_copies() -> None:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    event = environment.change_billing_status(
        billing_id=billing.id,
        new_status=BillingStatus.CANCELLED,
        metadata={"source": "test"},
    )

    event.metadata["source"] = "changed"
    state = environment.state
    state.events[0].metadata["source"] = "changed again"

    assert environment.state.events[0].metadata["source"] == "test"


def test_structured_observation_is_not_internal_state() -> None:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")

    observation = environment.observe_customer(customer.id)
    observation.value["name"] = "changed"

    assert observation.entity_type == "customer"
    assert observation.entity_id == customer.id
    assert observation.source == "environment"
    assert observation.field == "state"
    assert environment.get_customer(customer.id).name == "Maya"


def test_replay_from_same_snapshot_is_equivalent() -> None:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    snapshot = environment.snapshot()
    mutations = [
        EnvironmentMutation(
            operation="change_billing_status",
            parameters={
                "billing_id": billing.id,
                "new_status": BillingStatus.CANCELLED,
                "metadata": {"reason": "replay"},
            },
        )
    ]

    first = EnvironmentService.replay(snapshot, mutations)
    second = EnvironmentService.replay(snapshot, mutations)

    assert first.snapshot() == second.snapshot()
    assert first.get_billing(billing.id).status == BillingStatus.CANCELLED


def test_invalid_entity_ids_raise_value_error() -> None:
    environment = EnvironmentService()

    with pytest.raises(ValueError, match="Unknown customer"):
        environment.get_customer("missing")
    with pytest.raises(ValueError, match="Unknown subscription"):
        environment.get_subscription("missing")
    with pytest.raises(ValueError, match="Unknown billing account"):
        environment.get_billing("missing")
    with pytest.raises(ValueError, match="Unknown customer"):
        environment.observe_customer("missing")
    with pytest.raises(ValueError, match="Unknown subscription"):
        environment.observe_subscription("missing")
    with pytest.raises(ValueError, match="Unknown billing account"):
        environment.observe_billing("missing")