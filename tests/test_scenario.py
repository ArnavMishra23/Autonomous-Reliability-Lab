import json

from packages.agent.target import AgentTask, MockProvider, TargetAgent
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.tools import build_tool_registry
from packages.discovery.scenario import (
    Perturbation,
    PerturbationError,
    PerturbationType,
    Scenario,
    ScenarioRunStatus,
    ScenarioRunner,
    Trigger,
    TriggerType,
)
from packages.environment.models import BillingStatus
from packages.environment.service import EnvironmentService


class FailingProvider:
    def complete(self, **kwargs: object) -> object:
        raise RuntimeError("agent provider failed")


def create_environment() -> tuple[EnvironmentService, str, str, str]:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    return environment, customer.id, subscription.id, billing.id


def create_agent(
    environment: EnvironmentService,
    customer_id: str,
    subscription_id: str,
    billing_id: str,
    provider: object | None = None,
) -> TargetAgent:
    interface = TargetAgentToolInterface(build_tool_registry(environment))
    model = provider or MockProvider(
        customer_id=customer_id,
        subscription_id=subscription_id,
        billing_id=billing_id,
    )
    return TargetAgent(interface, model)


def state_change(
    perturbation_id: str,
    billing_id: str,
    trigger: Trigger,
    value: BillingStatus = BillingStatus.CANCELLED,
) -> Perturbation:
    return Perturbation(
        perturbation_id=perturbation_id,
        type=PerturbationType.STATE_CHANGE,
        target=f"billing.{billing_id}.status",
        value=value,
        trigger=trigger,
    )


def test_scenario_model_serialization_round_trip() -> None:
    scenario = Scenario(
        scenario_id="scenario-1",
        name="Cancel billing during run",
        description="Change billing while the agent is working.",
        task=AgentTask(task_id="task-1", goal="Cancel Maya's subscription."),
        perturbations=(
            state_change(
                "p-1",
                "bill-1",
                Trigger(TriggerType.AGENT_STEP, step=4),
            ),
        ),
        metadata={"owner": "test"},
    )

    payload = scenario.to_dict()
    json.dumps(payload)
    restored = Scenario.from_dict(payload)

    assert restored == scenario


def test_state_change_perturbation_model() -> None:
    perturbation = state_change(
        "p-1",
        "bill-1",
        Trigger(TriggerType.BEFORE_RUN),
    )

    assert perturbation.type == PerturbationType.STATE_CHANGE
    assert perturbation.target == "billing.bill-1.status"
    assert perturbation.trigger.type == TriggerType.BEFORE_RUN


def test_scheduler_orders_before_run_and_agent_steps() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    scenario = Scenario(
        scenario_id="scenario-order",
        name="Order",
        description="Ordering",
        task=AgentTask(task_id="task-order", goal="Order"),
        perturbations=(
            state_change("step-3", billing_id, Trigger(TriggerType.AGENT_STEP, 3)),
            state_change("before", billing_id, Trigger(TriggerType.BEFORE_RUN)),
            state_change("step-1", billing_id, Trigger(TriggerType.AGENT_STEP, 1)),
        ),
    )
    agent = create_agent(environment, customer_id, subscription_id, billing_id)
    run = ScenarioRunner().run(scenario, environment, agent)

    assert [item.perturbation_id for item in run.applied_perturbations] == [
        "before",
        "step-1",
        "step-3",
    ]


def test_before_run_perturbation_is_applied() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    original = environment.snapshot()
    scenario = Scenario(
        scenario_id="scenario-before",
        name="Before run",
        description="Change before start",
        task=AgentTask(task_id="task-before", goal="Cancel Maya's subscription."),
        perturbations=(
            state_change("p-before", billing_id, Trigger(TriggerType.BEFORE_RUN)),
        ),
    )

    run = ScenarioRunner().run(
        scenario,
        environment,
        create_agent(environment, customer_id, subscription_id, billing_id),
    )

    assert run.applied_perturbations[0].step is None
    assert run.final_environment["billing_accounts"][billing_id]["status"] == "cancelled"
    assert environment.snapshot() == original


def test_agent_step_perturbation_changes_state_and_records_event() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    original = environment.snapshot()
    scenario = Scenario(
        scenario_id="scenario-step",
        name="During run",
        description="Change before cancellation decision",
        task=AgentTask(
            task_id="cancel-maya",
            goal="Process Maya's subscription cancellation.",
        ),
        perturbations=(
            state_change(
                "p-step-4",
                billing_id,
                Trigger(TriggerType.AGENT_STEP, step=4),
            ),
        ),
    )

    run = ScenarioRunner().run(
        scenario,
        environment,
        create_agent(environment, customer_id, subscription_id, billing_id),
    )

    assert run.status == ScenarioRunStatus.AGENT_FAILED
    assert run.agent_run is not None
    assert run.applied_perturbations[0].step == 4
    assert run.final_environment["billing_accounts"][billing_id]["status"] == "cancelled"
    events = run.final_environment["events"]
    assert any(
        event["target"] == billing_id
        and event["old_value"] == "active"
        and event["new_value"] == "cancelled"
        for event in events
    )
    assert environment.snapshot() == original


def test_original_environment_restored_after_agent_failure() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    original = environment.snapshot()
    scenario = Scenario(
        scenario_id="scenario-agent-failure",
        name="Agent failure",
        description="Provider failure",
        task=AgentTask(task_id="task-failure", goal="Fail."),
    )

    run = ScenarioRunner().run(
        scenario,
        environment,
        create_agent(
            environment,
            customer_id,
            subscription_id,
            billing_id,
            provider=FailingProvider(),
        ),
    )

    assert run.status == ScenarioRunStatus.AGENT_FAILED
    assert environment.snapshot() == original


def test_original_environment_restored_after_perturbation_failure() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    original = environment.snapshot()
    scenario = Scenario(
        scenario_id="scenario-perturbation-failure",
        name="Bad perturbation",
        description="Invalid target",
        task=AgentTask(task_id="task-invalid", goal="Fail perturbation."),
        perturbations=(
            Perturbation(
                perturbation_id="bad-target",
                type=PerturbationType.STATE_CHANGE,
                target="subscription.not-a-billing.status",
                value="cancelled",
                trigger=Trigger(TriggerType.BEFORE_RUN),
            ),
        ),
    )

    run = ScenarioRunner().run(
        scenario,
        environment,
        create_agent(environment, customer_id, subscription_id, billing_id),
    )

    assert run.status == ScenarioRunStatus.PERTURBATION_FAILED
    assert "bad-target" in run.error
    assert environment.snapshot() == original


def test_invalid_perturbation_target_is_rejected() -> None:
    environment, _, _, billing_id = create_environment()
    scenario = Scenario(
        scenario_id="scenario-invalid",
        name="Invalid target",
        description="Invalid target",
        task=AgentTask(task_id="task-invalid", goal="Invalid target."),
        perturbations=(
            state_change(
                "bad",
                billing_id,
                Trigger(TriggerType.BEFORE_RUN),
            ),
        ),
    )
    scenario = Scenario(
        scenario_id=scenario.scenario_id,
        name=scenario.name,
        description=scenario.description,
        task=scenario.task,
        perturbations=(
            Perturbation(
                perturbation_id="bad",
                type=PerturbationType.STATE_CHANGE,
                target="billing.invalid",
                value="cancelled",
                trigger=Trigger(TriggerType.BEFORE_RUN),
            ),
        ),
    )

    run = ScenarioRunner().run(
        scenario,
        environment,
        create_agent(environment, "missing", "missing", billing_id),
    )

    assert run.status == ScenarioRunStatus.PERTURBATION_FAILED
    assert "expected billing.<billing_id>.status" in run.error


def test_same_scenario_replays_from_restored_starting_state() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    original = environment.snapshot()
    scenario = Scenario(
        scenario_id="scenario-replay",
        name="Replay",
        description="Replay deterministic perturbation",
        task=AgentTask(task_id="task-replay", goal="Cancel Maya's subscription."),
        perturbations=(
            state_change(
                "p-replay",
                billing_id,
                Trigger(TriggerType.AGENT_STEP, step=4),
            ),
        ),
    )
    runner = ScenarioRunner()

    first = runner.run(
        scenario,
        environment,
        create_agent(environment, customer_id, subscription_id, billing_id),
    )
    second = runner.run(
        scenario,
        environment,
        create_agent(environment, customer_id, subscription_id, billing_id),
    )

    assert first.final_environment["billing_accounts"] == second.final_environment["billing_accounts"]
    assert first.final_environment["events"][0]["target"] == second.final_environment["events"][0]["target"]
    assert [item.perturbation_id for item in first.applied_perturbations] == [
        item.perturbation_id for item in second.applied_perturbations
    ]
    assert environment.snapshot() == original


def test_scenario_run_is_json_serializable() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    scenario = Scenario(
        scenario_id="scenario-json",
        name="JSON",
        description="Serializable run",
        task=AgentTask(task_id="task-json", goal="Cancel."),
        perturbations=(
            state_change(
                "p-json",
                billing_id,
                Trigger(TriggerType.AGENT_STEP, step=4),
            ),
        ),
    )

    run = ScenarioRunner().run(
        scenario,
        environment,
        create_agent(environment, customer_id, subscription_id, billing_id),
    )

    json.dumps(run.to_dict())
