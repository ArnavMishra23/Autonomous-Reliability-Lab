from packages.environment.models import BillingStatus
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