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
