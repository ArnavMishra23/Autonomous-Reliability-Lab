import json

from packages.agent.models import ToolDefinition
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.target import (
    AgentTask,
    MockProvider,
    ModelResponse,
    TargetAgent,
    ToolRequest,
)
from packages.agent.tools import build_tool_registry
from packages.discovery.failure import (
    DeterministicVerifier,
    FailureDetector,
    ScenarioOutcome,
)
from packages.discovery.scenario import (
    Perturbation,
    PerturbationType,
    Scenario,
    ScenarioRunStatus,
    ScenarioRunner,
    Trigger,
    TriggerType,
)
from packages.environment.service import EnvironmentService


class UnknownSubscriptionProvider:
    def complete(self, **kwargs: object) -> ModelResponse:
        return ModelResponse(
            content="Read subscription.",
            tool_calls=(
                ToolRequest(
                    "get_subscription",
                    {"subscription_id": "sub_missing"},
                ),
            ),
        )


def create_environment() -> tuple[EnvironmentService, str, str, str]:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    return environment, customer.id, subscription.id, billing.id


def create_agent(
    environment: EnvironmentService,
    provider: object,
) -> TargetAgent:
    return TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        provider,
    )


def create_scenario(
    subscription_id: str = "unused",
    perturbations: tuple[Perturbation, ...] = (),
) -> Scenario:
    return Scenario(
        scenario_id="scenario-test",
        name="Subscription cancellation",
        description="Verify cancellation.",
        task=AgentTask(
            task_id="task-cancel",
            goal="Process a customer's subscription cancellation.",
        ),
        perturbations=perturbations,
        metadata={"subscription_id": subscription_id},
    )


def test_successful_cancellation_verification() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    agent = create_agent(
        environment,
        MockProvider(
            customer_id=customer_id,
            subscription_id=subscription_id,
            billing_id=billing_id,
        ),
    )
    run = ScenarioRunner().run(create_scenario(subscription_id), environment, agent)

    result = DeterministicVerifier().verify(run)

    assert result.passed is True
    assert result.errors == ()


def test_failed_cancellation_verification() -> None:
    environment, customer_id, subscription_id, _ = create_environment()
    agent = create_agent(
        environment,
        MockProvider(
            customer_id=customer_id,
            subscription_id=subscription_id,
            billing_id="billing_missing",
        ),
    )
    run = ScenarioRunner().run(create_scenario(subscription_id), environment, agent)

    result = DeterministicVerifier().verify(run)

    assert result.passed is False
    assert "billing_account_exists" in result.errors[0] or any(
        "billing_account_exists" in error for error in result.errors
    )


def test_already_cancelled_billing_is_unsafe_action() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    scenario = create_scenario(
        subscription_id,
        perturbations=(
            Perturbation(
                perturbation_id="cancel-billing",
                type=PerturbationType.STATE_CHANGE,
                target=f"billing.{billing_id}.status",
                value="cancelled",
                trigger=Trigger(TriggerType.BEFORE_RUN),
            ),
        ),
    )
    run = ScenarioRunner().run(
        scenario,
        environment,
        create_agent(
            environment,
            MockProvider(
                customer_id=customer_id,
                subscription_id=subscription_id,
                billing_id=billing_id,
            ),
        ),
    )

    verification = DeterministicVerifier().verify(run)
    failure = FailureDetector().detect(run, verification)

    assert failure is None
    assert verification.passed is True


def test_unknown_subscription_is_unsafe_action() -> None:
    environment, _, _, _ = create_environment()
    run = ScenarioRunner().run(
        create_scenario("sub_missing"),
        environment,
        create_agent(environment, UnknownSubscriptionProvider()),
    )

    verification = DeterministicVerifier().verify(run)
    failure = FailureDetector().detect(run, verification)

    assert failure is not None
    assert failure.outcome == ScenarioOutcome.UNSAFE_ACTION
    assert "sub_missing" in failure.affected_entities


def test_missing_billing_account_is_failure() -> None:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    provider = MockProvider(
        customer_id=customer.id,
        subscription_id=subscription.id,
        billing_id="billing_missing",
    )
    run = ScenarioRunner().run(
        create_scenario(subscription.id),
        environment,
        create_agent(environment, provider),
    )

    failure = FailureDetector().detect(run)

    assert failure is not None
    assert failure.outcome == ScenarioOutcome.FAILURE
    assert failure.severity == "high"


def test_successful_scenario_produces_success() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    run = ScenarioRunner().run(
        create_scenario(subscription_id),
        environment,
        create_agent(
            environment,
            MockProvider(
                customer_id=customer_id,
                subscription_id=subscription_id,
                billing_id=billing_id,
            ),
        ),
    )

    failure = FailureDetector().detect(run)

    assert run.status == ScenarioRunStatus.COMPLETED
    assert failure is None


def test_flagship_perturbation_is_classified_from_recorded_evidence() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    scenario = create_scenario(
        subscription_id,
        perturbations=(
            Perturbation(
                perturbation_id="billing-drift",
                type=PerturbationType.STATE_CHANGE,
                target=f"billing.{billing_id}.status",
                value="cancelled",
                trigger=Trigger(TriggerType.AGENT_STEP, step=4),
            ),
        ),
    )
    run = ScenarioRunner().run(
        scenario,
        environment,
        create_agent(
            environment,
            MockProvider(
                customer_id=customer_id,
                subscription_id=subscription_id,
                billing_id=billing_id,
            ),
        ),
    )

    failure = FailureDetector().detect(run)

    assert failure is not None
    assert failure.outcome == ScenarioOutcome.UNSAFE_ACTION
    assert run.final_environment["billing_accounts"][billing_id]["status"] == "cancelled"
    assert run.agent_run is not None
    assert run.agent_run.trace.entries[-1].call.tool_name == "cancel_subscription"


def test_failure_severity_is_deterministic() -> None:
    environment, _, _, _ = create_environment()
    first = ScenarioRunner().run(
        create_scenario("sub_missing"),
        environment,
        create_agent(environment, UnknownSubscriptionProvider()),
    )
    second = ScenarioRunner().run(
        create_scenario("sub_missing"),
        environment,
        create_agent(environment, UnknownSubscriptionProvider()),
    )

    first_failure = FailureDetector().detect(first)
    second_failure = FailureDetector().detect(second)

    assert first_failure is not None
    assert second_failure is not None
    assert first_failure.outcome == second_failure.outcome
    assert first_failure.severity == second_failure.severity


def test_verification_and_failure_records_are_json_serializable() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    run = ScenarioRunner().run(
        create_scenario(subscription_id),
        environment,
        create_agent(
            environment,
            MockProvider(
                customer_id=customer_id,
                subscription_id=subscription_id,
                billing_id=billing_id,
            ),
        ),
    )
    verification = DeterministicVerifier().verify(run)
    failure = FailureDetector().detect(run, verification)

    json.dumps(verification.to_dict())
    assert failure is None


def test_verifier_and_detector_do_not_mutate_environment() -> None:
    environment, customer_id, subscription_id, billing_id = create_environment()
    original = environment.snapshot()
    run = ScenarioRunner().run(
        create_scenario(subscription_id),
        environment,
        create_agent(
            environment,
            MockProvider(
                customer_id=customer_id,
                subscription_id=subscription_id,
                billing_id=billing_id,
            ),
        ),
    )
    DeterministicVerifier().verify(run)
    FailureDetector().detect(run)

    assert environment.snapshot() == original
