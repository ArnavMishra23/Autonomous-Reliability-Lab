import json

import pytest

from packages.agent.models import ToolDefinition
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.registry import ToolRegistry
from packages.agent.tools import build_tool_registry
from packages.environment.models import BillingStatus, SubscriptionStatus
from packages.environment.service import EnvironmentService


def create_environment() -> tuple[EnvironmentService, str, str, str]:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(
        customer_id=customer.id,
        plan="pro",
    )
    billing = environment.create_billing_account(subscription_id=subscription.id)
    return environment, customer.id, subscription.id, billing.id


def test_registry_registration_and_lookup() -> None:
    registry = ToolRegistry()
    definition = ToolDefinition(
        name="echo",
        description="Echo input.",
        input_schema={"type": "object", "properties": {}},
        read_only=True,
    )
    registry.register(definition, lambda arguments: {"ok": True})

    assert registry.get("echo").name == "echo"
    assert registry.list_tools()[0].read_only is True


def test_unknown_tool_handling() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unknown tool"):
        registry.get("missing")

    result = registry.execute("missing", {})
    assert result.success is False
    assert result.error == "Unknown tool: missing"


def test_get_customer_tool() -> None:
    environment, customer_id, _, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute("get_customer", {"customer_id": customer_id})

    assert result.success is True
    assert result.output["id"] == customer_id
    assert result.output["name"] == "Maya"
    assert next(tool for tool in interface.available_tools if tool.name == "get_customer").read_only


def test_get_subscription_tool() -> None:
    environment, _, subscription_id, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute("get_subscription", {"subscription_id": subscription_id})

    assert result.success is True
    assert result.output["id"] == subscription_id
    assert result.output["status"] == SubscriptionStatus.ACTIVE


def test_get_billing_status_tool() -> None:
    environment, _, _, billing_id = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute("get_billing_status", {"billing_id": billing_id})

    assert result.success is True
    assert result.output == {
        "id": billing_id,
        "subscription_id": result.output["subscription_id"],
        "status": BillingStatus.ACTIVE.value,
    }


def test_cancel_subscription_success_path() -> None:
    environment, _, subscription_id, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute(
        "cancel_subscription",
        {"subscription_id": subscription_id},
    )

    assert result.success is True
    assert result.output["subscription"]["status"] == SubscriptionStatus.CANCELLED
    assert result.output["event"]["target"] == subscription_id
    assert environment.get_subscription(subscription_id).status == SubscriptionStatus.CANCELLED
    assert len(environment.state.events) == 1


def test_cancel_subscription_validation_failure() -> None:
    environment, _, subscription_id, billing_id = create_environment()
    environment.change_billing_status(
        billing_id=billing_id,
        new_status=BillingStatus.CANCELLED,
    )
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute(
        "cancel_subscription",
        {"subscription_id": subscription_id},
    )

    assert result.success is False
    assert "already cancelled" in result.error
    assert environment.get_subscription(subscription_id).status == SubscriptionStatus.ACTIVE


def test_trace_records_call_and_result() -> None:
    environment, customer_id, _, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute("get_customer", {"customer_id": customer_id})
    entry = interface.trace.entries[0]

    assert entry.call.tool_name == "get_customer"
    assert entry.call.call_id == result.call_id
    assert entry.result.success is True


def test_tool_outputs_do_not_expose_mutable_environment_objects() -> None:
    environment, customer_id, _, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute("get_customer", {"customer_id": customer_id})
    result.output["name"] = "changed"
    trace_result = interface.trace.entries[0].result
    trace_result.output["name"] = "changed again"

    assert environment.get_customer(customer_id).name == "Maya"
    assert interface.execute("get_customer", {"customer_id": customer_id}).output["name"] == "Maya"


def test_target_agent_interface_does_not_expose_environment_state() -> None:
    environment, _, _, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    assert not hasattr(interface, "environment")
    assert not hasattr(interface, "state")
    assert not hasattr(interface, "service")


def test_sequential_tool_calls_preserve_ordering() -> None:
    environment, customer_id, subscription_id, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    interface.execute("get_customer", {"customer_id": customer_id})
    interface.execute("get_subscription", {"subscription_id": subscription_id})

    names = [entry.call.tool_name for entry in interface.trace.entries]
    assert names == ["get_customer", "get_subscription"]


def test_failed_tool_calls_produce_structured_error_results() -> None:
    environment, _, _, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute("get_customer", {})

    assert result.success is False
    assert result.output is None
    assert "Missing required arguments" in result.error
    assert interface.trace.entries[-1].result.error == result.error


def test_tool_outputs_are_json_serializable() -> None:
    environment, _, subscription_id, _ = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    result = interface.execute(
        "cancel_subscription",
        {"subscription_id": subscription_id},
    )

    json.dumps(result.output)
