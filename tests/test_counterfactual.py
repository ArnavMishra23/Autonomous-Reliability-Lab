import json
from datetime import datetime, timezone

from packages.agent import (
    AgentTask,
    MockProvider,
    TargetAgent,
    TargetAgentToolInterface,
    build_tool_registry,
)
from packages.analysis import CausalAnalyzer
from packages.counterfactual import (
    CounterfactualExecutor,
    CounterfactualStatus,
    Intervention,
    InterventionType,
    generate_candidates,
)
from packages.discovery.failure import DeterministicVerifier, FailureDetector
from packages.discovery.scenario import (
    Perturbation,
    PerturbationType,
    Scenario,
    ScenarioRunner,
    Trigger,
    TriggerType,
)
from packages.environment.service import EnvironmentService


def create_flagship() -> tuple[EnvironmentService, Scenario, TargetAgent, str, str]:
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)
    scenario = Scenario(
        scenario_id="flagship-counterfactual",
        name="Stale billing",
        description="Billing changes before cancellation.",
        task=AgentTask(
            task_id="task-counterfactual",
            goal="Process Maya's subscription cancellation.",
        ),
        perturbations=(
            Perturbation(
                perturbation_id="billing-drift",
                type=PerturbationType.STATE_CHANGE,
                target=f"billing.{billing.id}.status",
                value="cancelled",
                trigger=Trigger(TriggerType.AGENT_STEP, step=4),
            ),
        ),
    )
    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        MockProvider(
            customer_id=customer.id,
            subscription_id=subscription.id,
            billing_id=billing.id,
        ),
    )
    return environment, scenario, agent, subscription.id, billing.id


def get_original_run(environment, scenario, agent):
    return ScenarioRunner().run(scenario, environment, agent)


def test_intervention_and_candidate_serialization() -> None:
    intervention = Intervention(
        intervention_id="i1",
        intervention_type=InterventionType.REQUIRED_OBSERVATION,
        description="Observe billing before cancel.",
        target="cancel_subscription",
        value="get_billing_status",
        metadata={"billing_id": "bill_1"},
    )
    candidate = __import__(
        "packages.counterfactual.models", fromlist=["CounterfactualCandidate"]
    ).CounterfactualCandidate("c1", intervention, "hypothesis_stale_observation")

    json.dumps(intervention.to_dict())
    json.dumps(candidate.to_dict())
    assert intervention.intervention_type == InterventionType.REQUIRED_OBSERVATION


def test_stale_observation_hypothesis_generates_candidate() -> None:
    environment, scenario, agent, _, billing_id = create_flagship()
    run = get_original_run(environment, scenario, agent)
    verification = DeterministicVerifier().verify(run)
    failure = FailureDetector().detect(run, verification)
    analysis = CausalAnalyzer().analyze(run, verification, failure)

    candidates = generate_candidates(analysis)

    assert len(candidates) == 1
    assert candidates[0].intervention.value == "get_billing_status"
    assert candidates[0].intervention.target == "cancel_subscription"
    assert candidates[0].intervention.metadata["billing_id"] == billing_id


def test_mock_provider_cancels_after_active_observation() -> None:
    environment, scenario, agent, _, _ = create_flagship()

    run = get_original_run(environment, scenario, agent)

    assert run.agent_run is not None
    assert [entry.call.tool_name for entry in run.agent_run.trace.entries] == [
        "get_customer",
        "get_subscription",
        "get_billing_status",
        "cancel_subscription",
    ]


def test_mock_provider_completes_after_cancelled_observation() -> None:
    environment, scenario, agent, _, billing_id = create_flagship()
    scenario = Scenario(
        scenario_id=scenario.scenario_id,
        name=scenario.name,
        description=scenario.description,
        task=scenario.task,
        perturbations=(
            Perturbation(
                perturbation_id="already-cancelled",
                type=PerturbationType.STATE_CHANGE,
                target=f"billing.{billing_id}.status",
                value="cancelled",
                trigger=Trigger(TriggerType.BEFORE_RUN),
            ),
        ),
    )

    run = get_original_run(environment, scenario, agent)

    assert run.agent_run is not None
    assert run.agent_run.status.value == "succeeded"
    assert [entry.call.tool_name for entry in run.agent_run.trace.entries] == [
        "get_customer",
        "get_subscription",
        "get_billing_status",
        "reconcile_subscription_state",
    ]


def test_insufficient_analysis_produces_no_candidate() -> None:
    analysis = CausalAnalyzer().analyze(
        __import__("packages.discovery.scenario", fromlist=["ScenarioRun"]).ScenarioRun(
            scenario_id="s",
            run_id="r",
            status=__import__("packages.discovery.scenario", fromlist=["ScenarioRunStatus"]).ScenarioRunStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
        )
    )

    assert generate_candidates(analysis) == ()


def test_counterfactual_uses_same_scenario_and_restores_environment() -> None:
    environment, scenario, agent, _, _ = create_flagship()
    original_state = environment.snapshot()
    original_run = get_original_run(environment, scenario, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    analysis = CausalAnalyzer().analyze(original_run, verification, failure)
    candidate = generate_candidates(analysis)[0]

    result = CounterfactualExecutor().execute(
        scenario,
        original_state,
        environment,
        agent,
        candidate,
        original_run_id=original_run.run_id,
    )

    assert result.original_run_id == original_run.run_id
    assert result.counterfactual_run_id is not None
    assert result.environment_restored is True
    assert environment.snapshot() == original_state
    assert result.status == CounterfactualStatus.VALIDATED


def test_required_observation_is_inserted_through_tool_trace() -> None:
    environment, scenario, agent, _, _ = create_flagship()
    original_run = get_original_run(environment, scenario, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    candidate = generate_candidates(CausalAnalyzer().analyze(original_run, verification, failure))[0]
    result = CounterfactualExecutor().execute(
        scenario,
        environment.snapshot(),
        environment,
        agent,
        candidate,
    )

    assert result.verification is not None
    assert result.failure is None
    assert result.status == CounterfactualStatus.VALIDATED
    assert result.metadata["final_subscription_status"] == "cancelled"
    assert result.metadata["final_billing_status"] == "cancelled"
    assert result.metadata["tool_sequence"] == [
        "get_customer",
        "get_subscription",
        "get_billing_status",
        "get_billing_status",
        "reconcile_subscription_state",
    ]


def test_flagship_intervention_does_not_fake_success() -> None:
    environment, scenario, agent, _, _ = create_flagship()
    original = get_original_run(environment, scenario, agent)
    verification = DeterministicVerifier().verify(original)
    failure = FailureDetector().detect(original, verification)
    candidate = generate_candidates(CausalAnalyzer().analyze(original, verification, failure))[0]

    result = CounterfactualExecutor().execute(
        scenario,
        environment,
        environment.snapshot(),
        environment,
        agent,
        candidate,
    ) if False else CounterfactualExecutor().execute(
        scenario,
        environment.snapshot(),
        environment,
        agent,
        candidate,
    )

    assert result.status == CounterfactualStatus.VALIDATED
    assert result.verification is not None
    assert result.verification.passed is True


def test_repeated_validation_is_deterministic_and_serializable() -> None:
    environment, scenario, agent, _, _ = create_flagship()
    original = get_original_run(environment, scenario, agent)
    verification = DeterministicVerifier().verify(original)
    failure = FailureDetector().detect(original, verification)
    candidate = generate_candidates(CausalAnalyzer().analyze(original, verification, failure))[0]

    result = CounterfactualExecutor().validate_repeatedly(
        scenario,
        environment.snapshot(),
        environment,
        agent,
        candidate,
        runs=2,
    )

    assert result.validation_runs == 2
    assert result.failed_runs == 0
    assert result.successful_runs == 2
    assert result.environment_restored is True
    json.dumps(result.to_dict())


def test_counterfactual_does_not_expose_environment_to_agent() -> None:
    environment, scenario, agent, _, _ = create_flagship()
    assert not hasattr(agent, "environment")
    assert not hasattr(agent, "state")
    assert not hasattr(agent, "service")
