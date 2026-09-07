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
    # 3 creation events (customer, subscription, billing) + 1 state change = 4
    assert len(environment.state.events) == 4


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


def test_tool_results_have_structured_error_codes() -> None:
    """M2: ToolResults should have structured error_code field."""
    from packages.agent.models import ToolErrorCode
    environment, _, subscription_id, billing_id = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))

    # Test unknown customer error
    result = interface.execute("get_customer", {"customer_id": "missing"})
    assert result.success is False
    assert result.error_code == ToolErrorCode.UNKNOWN_CUSTOMER
    assert result.error is not None

    # Test unknown subscription error
    result = interface.execute("get_subscription", {"subscription_id": "missing"})
    assert result.success is False
    assert result.error_code == ToolErrorCode.UNKNOWN_SUBSCRIPTION

    # Test unknown billing account error
    result = interface.execute("get_billing_status", {"billing_id": "missing"})
    assert result.success is False
    assert result.error_code == ToolErrorCode.UNKNOWN_BILLING_ACCOUNT

    # Test already cancelled subscription
    environment.change_billing_status(billing_id=billing_id, new_status="cancelled")
    result = interface.execute("cancel_subscription", {"subscription_id": subscription_id})
    assert result.success is False
    assert result.error_code == ToolErrorCode.BILLING_ACCOUNT_ALREADY_CANCELLED


def test_failure_classification_uses_error_codes_not_strings() -> None:
    """M2: Changing human-readable error message should not change failure classification."""
    from packages.agent.models import ToolErrorCode
    from packages.discovery.failure import DeterministicVerifier, FailureDetector
    from packages.discovery.scenario import Scenario, ScenarioRunner
    from packages.agent.target import AgentTask, MockProvider, TargetAgent
    from packages.agent.protocol import TargetAgentToolInterface
    from packages.agent.tools import build_tool_registry

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    # Make billing cancelled BEFORE the agent runs
    environment.change_billing_status(billing_id=billing.id, new_status="cancelled")

    scenario = Scenario(
        scenario_id="error-code-test",
        name="Error code test",
        description="Test error code based classification",
        task=AgentTask(task_id="task", goal="Test"),
        perturbations=(),
    )

    # Use a provider that will attempt cancellation without checking billing first
    class UnsafeProvider:
        def complete(self, **kwargs):
            prev_results = kwargs["previous_results"]
            if not prev_results:
                return ModelResponse(
                    content="Cancel immediately.",
                    tool_calls=(ToolRequest("cancel_subscription", {"subscription_id": subscription.id}),),
                )
            return ModelResponse(content="Done.", completed=True)

    from packages.agent.target import ModelResponse, ToolRequest

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        UnsafeProvider(),
    )

    run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(run)
    failure = FailureDetector().detect(run, verification)

    # Should be UNSAFE_ACTION due to BILLING_ACCOUNT_ALREADY_CANCELLED error code
    assert failure is not None
    assert failure.outcome.value == "unsafe_action"
    assert any("unsafe cancel_subscription attempt" in action for action in failure.unsafe_actions)
    # Verify error code is in the action description
    assert any("billing_account_already_cancelled" in action for action in failure.unsafe_actions)
