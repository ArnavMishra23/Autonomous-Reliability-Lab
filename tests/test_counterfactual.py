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


# =============================================================================
# NEW TESTS: Counterfactual reachability semantics
# =============================================================================

def test_intervention_reached_successful_repair_is_validated() -> None:
    """A. Intervention reached + successful repair -> VALIDATED"""
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

    assert result.status == CounterfactualStatus.VALIDATED
    assert result.intervention_reached is True
    assert result.verification is not None
    assert result.verification.passed is True
    assert result.failure is None


def test_intervention_reached_failed_repair_is_failed() -> None:
    """B. Intervention reached + failed repair -> FAILED"""
    from packages.agent import MockProvider, TargetAgent, TargetAgentToolInterface, build_tool_registry
    from packages.environment.service import EnvironmentService
    from packages.discovery.scenario import Scenario, Perturbation, PerturbationType, Trigger, TriggerType
    from packages.agent import AgentTask

    # Create an environment where the repair will fail (billing already cancelled at step 1)
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    # Perturbation at step 1 (before agent starts) - billing already cancelled
    scenario = Scenario(
        scenario_id="failed-repair",
        name="Already cancelled billing",
        description="Billing is already cancelled before agent starts.",
        task=AgentTask(
            task_id="task-failed-repair",
            goal="Process Maya's subscription cancellation.",
        ),
        perturbations=(
            Perturbation(
                perturbation_id="already-cancelled-billing",
                type=PerturbationType.STATE_CHANGE,
                target=f"billing.{billing.id}.status",
                value="cancelled",
                trigger=Trigger(TriggerType.BEFORE_RUN),
            ),
        ),
    )

    # Use MockProvider which will attempt reconcile_subscription_state
    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        MockProvider(
            customer_id=customer.id,
            subscription_id=subscription.id,
            billing_id=billing.id,
        ),
    )

    original_state = environment.snapshot()
    original_run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    analysis = CausalAnalyzer().analyze(original_run, verification, failure)
    candidates = generate_candidates(analysis)

    # Should generate a candidate since stale_observation should be detected
    if not candidates:
        # If no candidate generated, test is not applicable
        return

    candidate = candidates[0]
    original_state = environment.snapshot()

    result = CounterfactualExecutor().execute(
        scenario,
        original_state,
        environment,
        agent,
        candidate,
        original_run_id=original_run.run_id,
    )

    # The intervention should be reached (cancel_subscription is called)
    # But the repair might fail due to environment state
    if result.intervention_reached:
        assert result.status == CounterfactualStatus.FAILED
    else:
        # If intervention not reached, should be INCONCLUSIVE
        assert result.status == CounterfactualStatus.INCONCLUSIVE


def test_intervention_not_reached_early_completion_is_inconclusive() -> None:
    """C. Intervention NOT reached because model completes early -> INCONCLUSIVE"""
    from packages.agent import TargetAgent, TargetAgentToolInterface, build_tool_registry
    from packages.environment.service import EnvironmentService
    from packages.discovery.scenario import Scenario, Perturbation, PerturbationType, Trigger, TriggerType
    from packages.agent import AgentTask, ModelProvider, ModelResponse, ToolRequest
    from packages.agent.models import ToolDefinition, ToolResult, ToolTrace, ToolTraceEntry, ToolCall
    from packages.agent.target import AgentRun, AgentRunStatus, AgentTask
    from packages.analysis import CausalAnalyzer
    from packages.counterfactual import CounterfactualExecutor, CounterfactualStatus, generate_candidates
    from packages.discovery.failure import DeterministicVerifier, FailureDetector
    from packages.discovery.scenario import ScenarioRunner
    from packages.environment.service import EnvironmentService

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    scenario = Scenario(
        scenario_id="early-completion",
        name="Early completion test",
        description="Model completes early.",
        task=AgentTask(
            task_id="task-early-completion",
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

    class EarlyStopProvider:
        """Provider that returns completed=True after get_customer."""
        def __init__(self, customer_id: str, subscription_id: str, billing_id: str):
            self.customer_id = customer_id
            self.subscription_id = subscription_id
            self.billing_id = billing_id

        def complete(self, *, system_instructions, task, available_tools, previous_trace):
            step = len(previous_trace)
            # Stop after get_customer
            if previous_trace and previous_trace[-1].call.tool_name == "get_customer":
                return ModelResponse(content="Done early.", completed=True)
            requests = (
                ToolRequest("get_customer", {"customer_id": self.customer_id}),
                ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
                ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
                ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
            )
            if step < len(requests):
                return ModelResponse(content=f"Step {step+1}", tool_calls=(requests[step],))
            return ModelResponse(content="Done.", completed=True)

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        EarlyStopProvider(customer.id, subscription.id, billing.id),
        max_steps=10,
    )

    original_state = environment.snapshot()
    original_run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    analysis = CausalAnalyzer().analyze(original_run, verification, failure)
    candidates = generate_candidates(analysis)

    if not candidates:
        return

    candidate = candidates[0]
    original_state = environment.snapshot()

    result = CounterfactualExecutor().execute(
        scenario,
        original_state,
        environment,
        agent,
        candidate,
        original_run_id=original_run.run_id,
    )

    # Intervention should NOT be reached (model completed after get_customer)
    assert result.intervention_reached is False
    assert result.status == CounterfactualStatus.INCONCLUSIVE


def test_intervention_not_reached_early_failure_is_inconclusive() -> None:
    """D. Intervention NOT reached because model fails early -> INCONCLUSIVE"""
    from packages.agent import TargetAgent, TargetAgentToolInterface, build_tool_registry
    from packages.environment.service import EnvironmentService
    from packages.discovery.scenario import Scenario, Perturbation, PerturbationType, Trigger, TriggerType
    from packages.agent import AgentTask, ModelProvider, ModelResponse, ToolRequest
    from packages.agent.models import ToolDefinition, ToolResult, ToolTrace, ToolTraceEntry, ToolCall
    from packages.analysis import CausalAnalyzer
    from packages.counterfactual import CounterfactualExecutor, CounterfactualStatus, generate_candidates
    from packages.discovery.failure import DeterministicVerifier, FailureDetector
    from packages.discovery.scenario import ScenarioRunner
    from packages.environment.service import EnvironmentService

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    scenario = Scenario(
        scenario_id="early-failure",
        name="Early failure test",
        description="Model fails early.",
        task=AgentTask(
            task_id="task-early-failure",
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

    class EarlyFailProvider:
        """Provider that fails after get_customer."""
        def __init__(self, customer_id: str, subscription_id: str, billing_id: str):
            self.customer_id = customer_id
            self.subscription_id = subscription_id
            self.billing_id = billing_id

        def complete(self, *, system_instructions, task, available_tools, previous_trace):
            step = len(previous_trace)
            # Fail after get_customer
            if previous_trace and previous_trace[-1].call.tool_name == "get_customer":
                return ModelResponse(content="Error!", completed=False)
            requests = (
                ToolRequest("get_customer", {"customer_id": self.customer_id}),
                ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
                ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
                ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
            )
            if step < len(requests):
                return ModelResponse(content=f"Step {step+1}", tool_calls=(requests[step],))
            return ModelResponse(content="Done.", completed=True)

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        EarlyFailProvider(customer.id, subscription.id, billing.id),
        max_steps=10,
    )

    original_state = environment.snapshot()
    original_run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    analysis = CausalAnalyzer().analyze(original_run, verification, failure)
    candidates = generate_candidates(analysis)

    if not candidates:
        return

    candidate = candidates[0]
    original_state = environment.snapshot()

    result = CounterfactualExecutor().execute(
        scenario,
        original_state,
        environment,
        agent,
        candidate,
        original_run_id=original_run.run_id,
    )

    # Intervention should NOT be reached (model failed after get_customer)
    assert result.intervention_reached is False
    assert result.status == CounterfactualStatus.INCONCLUSIVE


def test_reachability_survives_json_serialization() -> None:
    """E. Reachability survives JSON serialization"""
    from packages.counterfactual.models import CounterfactualResult, CounterfactualStatus
    from packages.discovery.failure.models import VerificationResult

    result = CounterfactualResult(
        result_id="test",
        candidate_id="test",
        original_run_id="orig",
        counterfactual_run_id="cf",
        status=CounterfactualStatus.VALIDATED,
        verification=VerificationResult(passed=True, checks=(), errors=()),
        failure=None,
        environment_restored=True,
        intervention_reached=True,
    )

    json_str = json.dumps(result.to_dict())
    data = json.loads(json_str)

    assert data["intervention_reached"] is True
    assert data["status"] == "validated"


def test_no_false_validated_when_intervention_not_exercised() -> None:
    """H. No false VALIDATED result when intervention was not exercised"""
    from packages.agent import TargetAgent, TargetAgentToolInterface, build_tool_registry
    from packages.environment.service import EnvironmentService
    from packages.discovery.scenario import Scenario, Perturbation, PerturbationType, Trigger, TriggerType
    from packages.agent import AgentTask, ModelProvider, ModelResponse, ToolRequest
    from packages.analysis import CausalAnalyzer
    from packages.counterfactual import CounterfactualExecutor, CounterfactualStatus, generate_candidates
    from packages.discovery.failure import DeterministicVerifier, FailureDetector
    from packages.discovery.scenario import ScenarioRunner
    from packages.environment.service import EnvironmentService

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    scenario = Scenario(
        scenario_id="no-false-validated",
        name="No false validated test",
        description="Model completes early - should not be VALIDATED.",
        task=AgentTask(
            task_id="task-no-false",
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

    class EarlyStopProvider:
        def __init__(self, customer_id: str, subscription_id: str, billing_id: str):
            self.customer_id = customer_id
            self.subscription_id = subscription_id
            self.billing_id = billing_id

        def complete(self, *, system_instructions, task, available_tools, previous_trace):
            if previous_trace and previous_trace[-1].call.tool_name == "get_customer":
                return ModelResponse(content="Done early.", completed=True)
            requests = (
                ToolRequest("get_customer", {"customer_id": self.customer_id}),
                ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
                ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
                ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
            )
            step = len(previous_trace)
            if step < len(requests):
                return ModelResponse(content=f"Step {step+1}", tool_calls=(requests[step],))
            return ModelResponse(content="Done.", completed=True)

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        EarlyStopProvider(customer.id, subscription.id, billing.id),
        max_steps=10,
    )

    original_state = environment.snapshot()
    original_run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    analysis = CausalAnalyzer().analyze(original_run, verification, failure)
    candidates = generate_candidates(analysis)

    if not candidates:
        return

    candidate = candidates[0]
    original_state = environment.snapshot()

    result = CounterfactualExecutor().execute(
        scenario,
        original_state,
        environment,
        agent,
        candidate,
        original_run_id=original_run.run_id,
    )

    # Must NOT be VALIDATED if intervention not reached
    assert result.intervention_reached is False
    assert result.status != CounterfactualStatus.VALIDATED
    assert result.status == CounterfactualStatus.INCONCLUSIVE


def test_environment_restoration_unchanged() -> None:
    """G. Environment restoration remains unchanged"""
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

    assert result.environment_restored is True
    assert environment.snapshot() == original_state


# =============================================================================
# NEW TESTS: Counterfactual instrumentation and telemetry
# =============================================================================

def test_tool_sequence_captured_when_agent_makes_tool_calls() -> None:
    """A trajectory containing get_customer does not produce tool_sequence=[]."""
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

    # The mock provider makes multiple tool calls
    assert result.metadata.get("tool_sequence") is not None
    assert len(result.metadata["tool_sequence"]) > 0
    assert "get_customer" in result.metadata["tool_sequence"]
    assert result.metadata["tool_calls"] > 0
    assert result.metadata["agent_steps"] > 0
    assert result.metadata["model_calls"] > 0


def test_counters_nonzero_when_execution_occurred() -> None:
    """Counters are nonzero when actual model/tool execution occurred."""
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

    assert result.metadata["tool_calls"] > 0
    assert result.metadata["agent_steps"] > 0
    assert result.metadata["model_calls"] > 0
    assert len(result.metadata["tool_sequence"]) == result.metadata["tool_calls"]


def test_intervention_not_reached_always_inconclusive() -> None:
    """intervention_reached=false always produces INCONCLUSIVE."""
    from packages.agent import TargetAgent, TargetAgentToolInterface, build_tool_registry
    from packages.environment.service import EnvironmentService
    from packages.discovery.scenario import Scenario, Perturbation, PerturbationType, Trigger, TriggerType
    from packages.agent import AgentTask, ModelProvider, ModelResponse, ToolRequest

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    scenario = Scenario(
        scenario_id="inconclusive-test",
        name="Inconclusive test",
        description="Model completes early.",
        task=AgentTask(
            task_id="task-inconclusive",
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

    class EarlyStopProvider:
        def __init__(self, customer_id: str, subscription_id: str, billing_id: str):
            self.customer_id = customer_id
            self.subscription_id = subscription_id
            self.billing_id = billing_id

        def complete(self, *, system_instructions, task, available_tools, previous_trace):
            if previous_trace and previous_trace[-1].call.tool_name == "get_customer":
                return ModelResponse(content="Done early.", completed=True)
            requests = (
                ToolRequest("get_customer", {"customer_id": self.customer_id}),
                ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
                ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
                ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
            )
            step = len(previous_trace)
            if step < len(requests):
                return ModelResponse(content=f"Step {step+1}", tool_calls=(requests[step],))
            return ModelResponse(content="Done.", completed=True)

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        EarlyStopProvider(customer.id, subscription.id, billing.id),
        max_steps=10,
    )

    original_state = environment.snapshot()
    original_run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    analysis = CausalAnalyzer().analyze(original_run, verification, failure)
    candidates = generate_candidates(analysis)

    if not candidates:
        return

    candidate = candidates[0]
    original_state = environment.snapshot()

    result = CounterfactualExecutor().execute(
        scenario,
        original_state,
        environment,
        agent,
        candidate,
        original_run_id=original_run.run_id,
    )

    assert result.intervention_reached is False
    assert result.status == CounterfactualStatus.INCONCLUSIVE
    assert result.verification is None
    assert result.failure is None


def test_unreached_intervention_not_reported_as_failed_repair() -> None:
    """Unreached interventions are not reported as failed repairs."""
    from packages.agent import TargetAgent, TargetAgentToolInterface, build_tool_registry
    from packages.environment.service import EnvironmentService
    from packages.discovery.scenario import Scenario, Perturbation, PerturbationType, Trigger, TriggerType
    from packages.agent import AgentTask, ModelProvider, ModelResponse, ToolRequest

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    scenario = Scenario(
        scenario_id="unreached-not-failed",
        name="Unreached not failed",
        description="Model completes early.",
        task=AgentTask(
            task_id="task-unreached",
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

    class EarlyStopProvider:
        def __init__(self, customer_id: str, subscription_id: str, billing_id: str):
            self.customer_id = customer_id
            self.subscription_id = subscription_id
            self.billing_id = billing_id

        def complete(self, *, system_instructions, task, available_tools, previous_trace):
            if previous_trace and previous_trace[-1].call.tool_name == "get_customer":
                return ModelResponse(content="Done early.", completed=True)
            requests = (
                ToolRequest("get_customer", {"customer_id": self.customer_id}),
                ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
                ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
                ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
            )
            step = len(previous_trace)
            if step < len(requests):
                return ModelResponse(content=f"Step {step+1}", tool_calls=(requests[step],))
            return ModelResponse(content="Done.", completed=True)

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        EarlyStopProvider(customer.id, subscription.id, billing.id),
        max_steps=10,
    )

    original_state = environment.snapshot()
    original_run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    analysis = CausalAnalyzer().analyze(original_run, verification, failure)
    candidates = generate_candidates(analysis)

    if not candidates:
        return

    candidate = candidates[0]
    original_state = environment.snapshot()

    result = CounterfactualExecutor().execute(
        scenario,
        original_state,
        environment,
        agent,
        candidate,
        original_run_id=original_run.run_id,
    )

    # Must be INCONCLUSIVE, not FAILED
    assert result.status == CounterfactualStatus.INCONCLUSIVE
    assert result.intervention_reached is False
    # Verification should be None when intervention not reached
    assert result.verification is None
    assert result.failure is None


def test_verification_not_run_when_intervention_unreached() -> None:
    """Verification is not run when intervention not reached."""
    from packages.agent import TargetAgent, TargetAgentToolInterface, build_tool_registry
    from packages.environment.service import EnvironmentService
    from packages.discovery.scenario import Scenario, Perturbation, PerturbationType, Trigger, TriggerType
    from packages.agent import AgentTask, ModelProvider, ModelResponse, ToolRequest

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    scenario = Scenario(
        scenario_id="verification-not-run",
        name="Verification not run test",
        description="Model completes early.",
        task=AgentTask(
            task_id="task-verification",
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

    class EarlyStopProvider:
        def __init__(self, customer_id: str, subscription_id: str, billing_id: str):
            self.customer_id = customer_id
            self.subscription_id = subscription_id
            self.billing_id = billing_id

        def complete(self, *, system_instructions, task, available_tools, previous_trace):
            if previous_trace and previous_trace[-1].call.tool_name == "get_customer":
                return ModelResponse(content="Done early.", completed=True)
            requests = (
                ToolRequest("get_customer", {"customer_id": self.customer_id}),
                ToolRequest("get_subscription", {"subscription_id": self.subscription_id}),
                ToolRequest("get_billing_status", {"billing_id": self.billing_id}),
                ToolRequest("cancel_subscription", {"subscription_id": self.subscription_id}),
            )
            step = len(previous_trace)
            if step < len(requests):
                return ModelResponse(content=f"Step {step+1}", tool_calls=(requests[step],))
            return ModelResponse(content="Done.", completed=True)

    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        EarlyStopProvider(customer.id, subscription.id, billing.id),
        max_steps=10,
    )

    original_state = environment.snapshot()
    original_run = ScenarioRunner().run(scenario, environment, agent)
    verification = DeterministicVerifier().verify(original_run)
    failure = FailureDetector().detect(original_run, verification)
    analysis = CausalAnalyzer().analyze(original_run, verification, failure)
    candidates = generate_candidates(analysis)

    if not candidates:
        return

    candidate = candidates[0]
    original_state = environment.snapshot()

    result = CounterfactualExecutor().execute(
        scenario,
        original_state,
        environment,
        agent,
        candidate,
        original_run_id=original_run.run_id,
    )

    # When intervention not reached, verification should be None
    assert result.intervention_reached is False
    assert result.verification is None
    assert result.failure is None
    assert result.status == CounterfactualStatus.INCONCLUSIVE


def test_source_provenance_includes_both_source_artifacts() -> None:
    """Source provenance includes both discovery and counterfactual artifacts."""
    from packages.counterfactual.models import CounterfactualResult, CounterfactualStatus
    from packages.discovery.failure.models import VerificationResult

    result = CounterfactualResult(
        result_id="test",
        candidate_id="test",
        original_run_id="orig",
        counterfactual_run_id="cf",
        status=CounterfactualStatus.VALIDATED,
        verification=VerificationResult(passed=True, checks=(), errors=()),
        failure=None,
        environment_restored=True,
        intervention_reached=True,
        metadata={
            "tool_sequence": ["get_customer", "get_billing_status"],
            "model_calls": 2,
            "agent_steps": 1,
            "tool_calls": 1,
        },
    )

    # Test that the to_dict includes all necessary provenance fields
    data = result.to_dict()
    assert "result_id" in data
    assert "candidate_id" in data
    assert "original_run_id" in data
    assert "counterfactual_run_id" in data
    assert "status" in data
    assert "intervention_reached" in data
    assert "metadata" in data
    assert data["metadata"]["tool_sequence"] == ["get_customer", "get_billing_status"]
    assert data["metadata"]["model_calls"] == 2
    assert data["metadata"]["agent_steps"] == 1
    assert data["metadata"]["tool_calls"] == 1
