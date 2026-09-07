import json
from datetime import datetime, timezone

from packages.agent.models import ToolDefinition
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.target import (
    AgentRunStatus,
    AgentTask,
    MockProvider,
    ModelResponse,
    TargetAgent,
    ToolRequest,
)
from packages.agent.tools import build_tool_registry
from packages.environment.models import SubscriptionStatus
from packages.environment.service import EnvironmentService


class UnknownToolProvider:
    def complete(self, **kwargs: object) -> ModelResponse:
        return ModelResponse(
            content="Trying an unavailable tool.",
            tool_calls=(ToolRequest("missing_tool", {}),),
        )


class MalformedProvider:
    def complete(self, **kwargs: object) -> object:
        return {"content": "not a ModelResponse"}


class NeverCompleteProvider:
    def __init__(self, customer_id: str) -> None:
        self.customer_id = customer_id

    def complete(self, **kwargs: object) -> ModelResponse:
        return ModelResponse(
            content="Continue.",
            tool_calls=(ToolRequest("get_customer", {"customer_id": self.customer_id}),),
        )


class ExceptionProvider:
    def complete(self, **kwargs: object) -> ModelResponse:
        raise RuntimeError("provider unavailable")


class RecordingProvider:
    def __init__(self, customer_id: str) -> None:
        self.customer_id = customer_id
        self.previous_results: list[tuple[object, ...]] = []

    def complete(self, **kwargs: object) -> ModelResponse:
        results = kwargs["previous_results"]
        self.previous_results.append(tuple(results))
        if len(self.previous_results) == 1:
            return ModelResponse(
                content="Read customer.",
                tool_calls=(
                    ToolRequest("get_customer", {"customer_id": self.customer_id}),
                ),
            )
        return ModelResponse(content="Finished.", completed=True)


def create_environment() -> tuple[EnvironmentService, str, str, str]:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    return environment, customer.id, subscription.id, billing.id


def build_agent(environment: EnvironmentService, provider: object, max_steps: int = 10) -> TargetAgent:
    interface = TargetAgentToolInterface(build_tool_registry(environment))
    return TargetAgent(interface, provider, max_steps=max_steps)


def test_agent_task_serialization() -> None:
    task = AgentTask(
        task_id="task-1",
        goal="Process a cancellation.",
        context={"customer": "Maya"},
    )

    payload = task.to_dict()
    json.dumps(payload)
    assert payload["task_id"] == "task-1"
    assert payload["context"]["customer"] == "Maya"


def test_mock_provider_drives_expected_sequence() -> None:
    provider = MockProvider(
        customer_id="cust-1",
        subscription_id="sub-1",
        billing_id="bill-1",
    )
    task = AgentTask(task_id="task-1", goal="Cancel a subscription.")
    tools = (
        ToolDefinition("get_customer", "", {}, True),
        ToolDefinition("get_subscription", "", {}, True),
        ToolDefinition("get_billing_status", "", {}, True),
        ToolDefinition("cancel_subscription", "", {}, False),
    )

    responses = [
        provider.complete(
            system_instructions="",
            task=task,
            available_tools=tools,
            previous_results=(),
        )
    ]

    assert responses[0].tool_calls[0].tool_name == "get_customer"


def test_successful_agent_run_and_expected_tool_sequence() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    provider = MockProvider(
        customer_id=customer_id,
        subscription_id=subscription_id,
        billing_id=billing_id,
    )
    agent = build_agent(environment, provider)

    run = agent.run(
        AgentTask(
            task_id="cancel-maya",
            goal="Process Maya's subscription cancellation.",
        )
    )

    assert run.status == AgentRunStatus.SUCCEEDED
    assert [entry.call.tool_name for entry in run.trace.entries] == [
        "get_customer",
        "get_subscription",
        "get_billing_status",
        "cancel_subscription",
    ]
    assert environment.get_subscription(subscription_id).status == SubscriptionStatus.CANCELLED
    assert run.final_result is not None


def test_tool_results_are_fed_back_into_provider() -> None:
    environment, customer_id, _, _ = create_environment()
    provider = RecordingProvider(customer_id)
    run = build_agent(environment, provider).run(
        AgentTask(task_id="task-1", goal="Inspect Maya.")
    )

    assert run.status == AgentRunStatus.SUCCEEDED
    assert len(provider.previous_results) == 2
    assert provider.previous_results[0] == ()
    assert provider.previous_results[1][0].success is True


def test_unknown_requested_tool_produces_failure() -> None:
    environment, _, _, _ = create_environment()

    run = build_agent(environment, UnknownToolProvider()).run(
        AgentTask(task_id="task-1", goal="Use a missing tool.")
    )

    assert run.status == AgentRunStatus.FAILED
    assert "Unknown tool" in run.error
    assert len(run.trace.entries) == 1
    assert run.trace.entries[0].result.success is False


def test_malformed_provider_response_is_handled() -> None:
    environment, _, _, _ = create_environment()

    run = build_agent(environment, MalformedProvider()).run(
        AgentTask(task_id="task-1", goal="Malformed response.")
    )

    assert run.status == AgentRunStatus.FAILED
    assert run.error == "Malformed model response"
    assert run.trace.entries == []


def test_max_step_limit_works() -> None:
    environment, customer_id, _, _ = create_environment()

    run = build_agent(environment, NeverCompleteProvider(customer_id), max_steps=2).run(
        AgentTask(task_id="task-1", goal="Stop eventually.")
    )

    assert run.status == AgentRunStatus.FAILED
    assert "Maximum agent steps exceeded: 2" in run.error
    assert len(run.trace.entries) == 2


def test_provider_exception_produces_structured_failure() -> None:
    environment, _, _, _ = create_environment()

    run = build_agent(environment, ExceptionProvider()).run(
        AgentTask(task_id="task-1", goal="Provider error.")
    )

    assert run.status == AgentRunStatus.FAILED
    assert run.error == "Model execution failed: provider unavailable"
    assert run.finished_at is not None


def test_target_agent_does_not_expose_environment_state() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    agent = build_agent(
        environment,
        MockProvider(
            customer_id=customer_id,
            subscription_id=subscription_id,
            billing_id=billing_id,
        ),
    )

    assert not hasattr(agent, "environment")
    assert not hasattr(agent, "state")
    assert not hasattr(agent, "service")


def test_complete_agent_run_is_inspectable_and_json_serializable() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    agent = build_agent(
        environment,
        MockProvider(
            customer_id=customer_id,
            subscription_id=subscription_id,
            billing_id=billing_id,
        ),
    )

    run = agent.run(AgentTask(task_id="task-1", goal="Cancel Maya's subscription."))
    payload = run.to_dict()

    json.dumps(payload)
    assert payload["status"] == AgentRunStatus.SUCCEEDED.value
    assert payload["trace"]
    assert payload["finished_at"] is not None


def test_target_agent_multiple_runs_with_same_interface_are_isolated() -> None:
    """H4: Multiple runs with the same tool interface must not cross-contaminate traces.
    
    The agent's default_tools should not be mutated between runs. The environment
    state is expected to be managed by the caller (e.g., ScenarioRunner restores it).
    """
    environment, customer_id, subscription_id, billing_id = create_environment()
    interface = TargetAgentToolInterface(build_tool_registry(environment))
    original_state = environment.snapshot()

    agent = TargetAgent(
        interface,
        MockProvider(customer_id=customer_id, subscription_id=subscription_id, billing_id=billing_id),
    )

    # Run 1
    run1 = agent.run(AgentTask(task_id="run1", goal="Test"))
    assert run1.status == AgentRunStatus.SUCCEEDED

    # Restore environment to original state (caller's responsibility)
    environment.restore(original_state)

    # Run 2 - using same interface
    run2 = agent.run(AgentTask(task_id="run2", goal="Test"))
    assert run2.status == AgentRunStatus.SUCCEEDED

    # Traces should be separate objects (each run gets fresh trace)
    assert run1.trace is not run2.trace
    assert len(run1.trace.entries) > 0
    assert len(run2.trace.entries) > 0

    # Trace entries should be independent
    assert run1.trace.entries is not run2.trace.entries

    # The agent's default_tools should be unchanged
    assert agent.default_tools is interface


def test_target_agent_run_with_tools_does_not_mutate_default_interface() -> None:
    """H4: run_with_tools must not mutate the agent's default tool interface."""
    environment, customer_id, subscription_id, billing_id = create_environment()
    default_interface = TargetAgentToolInterface(build_tool_registry(environment))
    alternate_interface = TargetAgentToolInterface(build_tool_registry(environment))

    agent = TargetAgent(
        default_interface,
        MockProvider(customer_id=customer_id, subscription_id=subscription_id, billing_id=billing_id),
    )

    # Verify default interface is unchanged before
    assert agent.default_tools is default_interface

    # Run with alternate tools
    run = agent.run_with_tools(
        AgentTask(task_id="run1", goal="Test"),
        alternate_interface,
    )

    # Default interface should be unchanged
    assert agent.default_tools is default_interface
    assert run.status == AgentRunStatus.SUCCEEDED


def test_target_agent_tool_wrapper_does_not_mutate_default_interface() -> None:
    """H4: run_with_tool_wrapper must not mutate the agent's default tool interface."""
    environment, customer_id, subscription_id, billing_id = create_environment()
    default_interface = TargetAgentToolInterface(build_tool_registry(environment))

    agent = TargetAgent(
        default_interface,
        MockProvider(customer_id=customer_id, subscription_id=subscription_id, billing_id=billing_id),
    )

    def identity_wrapper(tools):
        return tools

    # Verify default interface is unchanged before
    assert agent.default_tools is default_interface

    # Run with wrapper
    run = agent.run_with_tool_wrapper(
        AgentTask(task_id="run1", goal="Test"),
        identity_wrapper,
    )

    # Default interface should be unchanged
    assert agent.default_tools is default_interface
    assert run.status == AgentRunStatus.SUCCEEDED


def test_target_agent_concurrent_run_simulation_is_isolated() -> None:
    """H4: Simulated concurrent runs (alternating steps) must not leak state.
    
    Each run should use a fresh environment state to avoid cross-contamination.
    The agent configuration should remain reusable.
    """
    environment, customer_id, subscription_id, billing_id = create_environment()
    interface1 = TargetAgentToolInterface(build_tool_registry(environment))
    interface2 = TargetAgentToolInterface(build_tool_registry(environment))
    original_state = environment.snapshot()

    agent = TargetAgent(
        interface1,
        MockProvider(customer_id=customer_id, subscription_id=subscription_id, billing_id=billing_id),
    )

    # Simulate interleaved execution by running with different interfaces
    run1 = agent.run(AgentTask(task_id="run1", goal="Test"))
    environment.restore(original_state)
    run2 = agent.run_with_tools(AgentTask(task_id="run2", goal="Test"), interface2)
    environment.restore(original_state)
    run3 = agent.run(AgentTask(task_id="run3", goal="Test"))

    # All should succeed independently
    assert run1.status == AgentRunStatus.SUCCEEDED
    assert run2.status == AgentRunStatus.SUCCEEDED
    assert run3.status == AgentRunStatus.SUCCEEDED

    # All traces should be independent
    assert run1.trace is not run2.trace
    assert run2.trace is not run3.trace
    assert run1.trace is not run3.trace

    # The agent's default_tools should remain unchanged
    assert agent.default_tools is interface1
