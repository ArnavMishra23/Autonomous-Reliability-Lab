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
    # 3 creation events (customer, subscription, billing) + 1 state change = 4
    assert len(environment.state.events) == 4
    assert any(e.event_type.value == "customer_created" for e in environment.state.events)
    assert any(e.event_type.value == "subscription_created" for e in environment.state.events)
    assert any(e.event_type.value == "billing_account_created" for e in environment.state.events)
    assert any(e.event_type.value == "state_change" for e in environment.state.events)


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
    # 3 creation events + 1 state change = 4
    assert len(environment.state.events) == 4


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

    # The returned event should be a safe copy, not the same object in state
    event.metadata["source"] = "changed"
    state = environment.state
    state_change_event = next(e for e in state.events if e.event_type.value == "state_change")
    state_change_event.metadata["source"] = "changed again"

    # The returned event copy should not be affected by changes to state
    assert event.metadata["source"] == "changed"
    # The state's event should not be affected by changes to the returned copy
    assert state_change_event.metadata["source"] == "changed again"


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
    from packages.environment.models import DeterministicIdProvider, DeterministicClock
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

    id_provider = DeterministicIdProvider(namespace="replay_test")
    clock = DeterministicClock()
    first = EnvironmentService.replay(snapshot, mutations, id_provider=id_provider, clock=clock)
    id_provider.reset()
    clock.reset()
    second = EnvironmentService.replay(snapshot, mutations, id_provider=id_provider, clock=clock)

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


def test_deterministic_ids_and_clocks_produce_equivalent_replays() -> None:
    """H5: Two equivalent replays with deterministic providers should produce canonical IDs/timestamps."""
    from packages.environment.models import DeterministicIdProvider, DeterministicClock
    from packages.environment.models import EnvironmentMutation, BillingStatus

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
                "metadata": {"reason": "test"},
            },
        )
    ]

    id_provider1 = DeterministicIdProvider(namespace="h5_test")
    clock1 = DeterministicClock()
    first = EnvironmentService.replay(snapshot, mutations, id_provider=id_provider1, clock=clock1)

    id_provider2 = DeterministicIdProvider(namespace="h5_test")
    clock2 = DeterministicClock()
    second = EnvironmentService.replay(snapshot, mutations, id_provider=id_provider2, clock=clock2)

    # With same namespace and reset providers, the IDs and timestamps should be identical
    assert first.snapshot() == second.snapshot()


def test_enum_normalization_accepts_valid_string_and_enum() -> None:
    """M1: Public enum inputs should be normalized from string or enum."""
    from packages.environment.models import SubscriptionStatus, BillingStatus
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")

    # String status should be accepted and normalized
    billing1 = environment.create_billing_account(
        subscription_id=subscription.id,
        status="active",  # string instead of enum
    )
    assert billing1.status == BillingStatus.ACTIVE

    # Enum status should work as before
    billing2 = environment.create_billing_account(
        subscription_id=subscription.id,
        status=BillingStatus.ACTIVE,
    )
    assert billing2.status == BillingStatus.ACTIVE

    # String status for subscription
    subscription2 = environment.create_subscription(
        customer_id=customer.id,
        plan="basic",
        status="paused",
    )
    assert subscription2.status == SubscriptionStatus.PAUSED

    # Enum status for subscription
    subscription3 = environment.create_subscription(
        customer_id=customer.id,
        plan="pro",
        status=SubscriptionStatus.ACTIVE,
    )
    assert subscription3.status == SubscriptionStatus.ACTIVE

    # String status for change_billing_status
    environment.change_billing_status(
        billing_id=billing1.id,
        new_status="cancelled",
    )
    assert environment.get_billing(billing1.id).status == BillingStatus.CANCELLED


def test_enum_normalization_rejects_invalid_string() -> None:
    """M1: Invalid enum strings should raise clear ValueError."""
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    # Invalid billing status
    with pytest.raises(ValueError, match="Invalid billing status: 'invalid_status'"):
        environment.create_billing_account(
            subscription_id=subscription.id,
            status="invalid_status",
        )

    with pytest.raises(ValueError, match="Invalid billing status: 'invalid'"):
        environment.change_billing_status(
            billing_id=billing.id,
            new_status="invalid",
        )

    # Invalid subscription status
    with pytest.raises(ValueError, match="Invalid subscription status: 'invalid'"):
        environment.create_subscription(
            customer_id=customer.id,
            plan="pro",
            status="invalid",
        )