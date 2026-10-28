"""End-to-end offline regression integration test using real Failure A artifact."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from packages.analysis import CausalAnalyzer
from packages.analysis.models import CausalAnalysis
from packages.benchmark.regression import (
    RegressionSuite,
    RegressionTestGenerator,
    RegressionTestRunner,
    RegressionStatus,
    RegressionTest,
    RegressionRunResult,
)
from packages.counterfactual import (
    CounterfactualExecutor,
    CounterfactualStatus,
    generate_candidates,
)
from packages.counterfactual.models import CounterfactualCandidate, Intervention
from packages.discovery.failure import DeterministicVerifier, FailureDetector
from packages.discovery.failure.models import (
    FailureRecord,
    VerificationResult,
    ScenarioOutcome,
    CheckResult,
)
from packages.discovery.scenario.models import (
    ScenarioRun,
    ScenarioRunStatus,
    AppliedPerturbation,
)
from packages.discovery.scenario import (
    Scenario,
    Perturbation,
    PerturbationType,
    Trigger,
    TriggerType,
    ScenarioRunner,
)
from packages.agent.target import (
    AgentRun,
    AgentRunStatus,
    ToolCall,
    ToolResult,
    ToolTrace,
    ToolTraceEntry,
    TargetAgent,
)
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.tools import build_tool_registry
from packages.agent import (
    AgentTask,
    MockProvider,
    ModelProvider,
    ModelResponse,
    ToolRequest,
)
from packages.agent.models import ToolDefinition, ToolResult, ToolTrace, ToolTraceEntry, ToolCall, ToolErrorCode
from packages.environment.service import EnvironmentService
from packages.environment.state import EnvironmentState
from packages.environment.models import (
    EnvironmentMutation,
    SubscriptionStatus,
    BillingStatus,
    DeterministicIdProvider,
    DeterministicClock,
)
from packages.agent.models import ToolErrorCode


# =============================================================================
# CONSTANTS
# =============================================================================

ARTIFACT_PATH = Path(__file__).resolve().parent / "fixtures" / "discovery" / "discovery10_20260911T002503Z_discovery_2117fc60.json"
FAILURE_A_CANDIDATE_ID = "candidate_billing_status_cancelled_trigger_step_4"
FAILURE_A_ID = "failure_ad13bb52"
DISCOVERY_RUN_ID = "discovery_2117fc60"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_artifact() -> dict:
    """Load the real discovery artifact."""
    with ARTIFACT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_scenario_run(artifact: dict, candidate_id: str) -> ScenarioRun:
    """Extract and reconstruct ScenarioRun from artifact for a specific candidate."""
    for evidence in artifact.get("scenario_evidence", []):
        if evidence.get("candidate_id") == candidate_id:
            scenario_run_data = evidence.get("scenario_run", {})
            agent_run_data = evidence.get("agent_run")
            tool_trace_data = evidence.get("tool_trace", [])

            applied_perturbations = []
            for p in scenario_run_data.get("applied_perturbations", []):
                applied_perturbations.append(AppliedPerturbation(
                    perturbation_id=p["perturbation_id"],
                    target=p["target"],
                    value=p["value"],
                    step=p["step"],
                    event_id=p["event_id"],
                    applied_at=datetime.fromisoformat(p["applied_at"]),
                ))

            agent_run = None
            if agent_run_data:
                trace_entries = []
                for t in tool_trace_data:
                    call = ToolCall(
                        call_id=t["call_id"],
                        tool_name=t["tool_name"],
                        arguments=t["arguments"],
                        timestamp=datetime.fromisoformat(t["call_timestamp"]),
                        step=t.get("step", 0) if "step" in t else 0,
                    )
                    result = ToolResult(
                        call_id=t["result_call_id"],
                        tool_name=t["result_tool_name"],
                        success=t["result_success"],
                        output=t["result_output"],
                        error=t["result_error"],
                        error_code=ToolErrorCode(t["result_error_code"]) if t.get("result_error_code") else None,
                        timestamp=datetime.fromisoformat(t["result_timestamp"]),
                    )
                    trace_entries.append(ToolTraceEntry(call=call, result=result))

                agent_run = AgentRun(
                    run_id=agent_run_data["run_id"],
                    task_id=agent_run_data["task_id"],
                    status=AgentRunStatus(agent_run_data["status"]),
                    started_at=datetime.fromisoformat(agent_run_data["started_at"]),
                    finished_at=datetime.fromisoformat(agent_run_data["finished_at"]) if agent_run_data.get("finished_at") else None,
                    trace=ToolTrace(entries=trace_entries),
                    final_result=agent_run_data.get("final_result"),
                    error=agent_run_data.get("error"),
                )

            final_env = scenario_run_data.get("final_environment", {})

            return ScenarioRun(
                scenario_id=scenario_run_data["scenario_id"],
                run_id=scenario_run_data["run_id"],
                status=ScenarioRunStatus(scenario_run_data["status"]),
                started_at=datetime.fromisoformat(scenario_run_data["started_at"]),
                ended_at=datetime.fromisoformat(scenario_run_data["ended_at"]),
                applied_perturbations=tuple(applied_perturbations),
                agent_run=agent_run,
                final_environment=final_env,
                error=scenario_run_data.get("error"),
            )

    raise ValueError(f"Candidate {candidate_id} not found in scenario_evidence")


def extract_verification_failure(artifact: dict, candidate_id: str) -> tuple[VerificationResult | None, FailureRecord | None]:
    """Extract verification and failure from discovery_result for a candidate."""
    for result in artifact.get("discovery_result", {}).get("results", []):
        if result.get("candidate", {}).get("candidate_id") == candidate_id:
            verification_data = result.get("verification")
            failure_data = result.get("failure")

            verification = None
            if verification_data:
                checks = []
                for c in verification_data.get("checks", []):
                    checks.append(CheckResult(
                        name=c["name"],
                        passed=c["passed"],
                        expected=c["expected"],
                        actual=c["actual"],
                        severity=c["severity"],
                    ))
                verification = VerificationResult(
                    passed=verification_data.get("passed", False),
                    checks=tuple(checks),
                    errors=tuple(verification_data.get("errors", [])),
                    metadata=verification_data.get("metadata", {}),
                )

            failure = None
            if failure_data:
                failure = FailureRecord(
                    failure_id=failure_data["failure_id"],
                    scenario_id=failure_data["scenario_id"],
                    run_id=failure_data["run_id"],
                    outcome=ScenarioOutcome(failure_data["outcome"]),
                    severity=failure_data["severity"],
                    summary=failure_data["summary"],
                    affected_entities=tuple(failure_data.get("affected_entities", [])),
                    failed_checks=tuple(failure_data.get("failed_checks", [])),
                    unsafe_actions=tuple(failure_data.get("unsafe_actions", [])),
                    first_detected_event_id=failure_data.get("first_detected_event_id"),
                    metadata=failure_data.get("metadata", {}),
                )

            return verification, failure

    return None, None


def reconstruct_starting_state(artifact: dict) -> tuple[EnvironmentService, EnvironmentState]:
    """Reconstruct the exact starting EnvironmentState from the artifact."""
    for evidence in artifact.get("scenario_evidence", []):
        final_env = evidence.get("scenario_run", {}).get("final_environment", {})
        events = final_env.get("events", [])
        if events:
            break
    else:
        raise ValueError("No events found in artifact")

    creation_events = [e for e in events if e.get("event_type") in (
        "customer_created", "subscription_created", "billing_account_created"
    )]

    mutations = []
    for event in creation_events:
        if event["event_type"] == "customer_created":
            mutations.append(EnvironmentMutation(
                operation="create_customer",
                parameters={
                    "customer_id": event["target"],
                    "name": event["new_value"]["name"],
                    "email": event["new_value"]["email"],
                }
            ))
        elif event["event_type"] == "subscription_created":
            status_str = event["new_value"]["status"]
            status = SubscriptionStatus(status_str) if isinstance(status_str, str) else status_str
            mutations.append(EnvironmentMutation(
                operation="create_subscription",
                parameters={
                    "subscription_id": event["target"],
                    "customer_id": event["new_value"]["customer_id"],
                    "plan": event["new_value"]["plan"],
                    "status": status,
                }
            ))
        elif event["event_type"] == "billing_account_created":
            status_str = event["new_value"]["status"]
            status = BillingStatus(status_str) if isinstance(status_str, str) else status_str
            mutations.append(EnvironmentMutation(
                operation="create_billing_account",
                parameters={
                    "billing_id": event["target"],
                    "subscription_id": event["new_value"]["subscription_id"],
                    "status": status,
                }
            ))

    base_time = datetime.fromisoformat(creation_events[0]["timestamp"])
    id_provider = DeterministicIdProvider(namespace="replay")
    clock = DeterministicClock(base=base_time, tick_seconds=0.001)

    environment = EnvironmentService.replay(
        EnvironmentState(),
        mutations,
        id_provider=id_provider,
        clock=clock,
    )

    starting_state = environment.snapshot()
    return environment, starting_state


def reconstruct_scenario_from_artifact(artifact: dict, candidate_id: str) -> Scenario:
    """Reconstruct the exact Scenario from artifact for a specific candidate."""
    for evidence in artifact.get("scenario_evidence", []):
        if evidence.get("candidate_id") == candidate_id:
            scenario_run_data = evidence.get("scenario_run", {})
            applied_perturbations = scenario_run_data.get("applied_perturbations", [])

            perturbations = []
            for p in applied_perturbations:
                trigger_type = TriggerType.AGENT_STEP if p.get("step") is not None else TriggerType.BEFORE_RUN
                trigger = Trigger(type=trigger_type, step=p.get("step"))
                target = p["target"]
                value = p["value"]

                perturbations.append(Perturbation(
                    perturbation_id=p["perturbation_id"],
                    type=PerturbationType.STATE_CHANGE,
                    target=target,
                    value=value,
                    trigger=trigger,
                ))

            agent_run_data = evidence.get("agent_run")
            task_id = agent_run_data.get("task_id", "task-counterfactual") if agent_run_data else "task-counterfactual"

            return Scenario(
                scenario_id=scenario_run_data["scenario_id"],
                name="Stale billing observation",
                description="Billing becomes cancelled at step 4 before cancellation attempt",
                task=AgentTask(
                    task_id=task_id,
                goal="Process the customer's subscription cancellation.",
                ),
                perturbations=tuple(perturbations),
            )

    raise ValueError(f"Candidate {candidate_id} not found in scenario_evidence")


def _make_updated_candidate(candidate: CounterfactualCandidate, billing_id: str) -> CounterfactualCandidate:
    """Create a candidate with updated intervention metadata for the new environment."""
    updated_intervention = candidate.intervention.__class__(
        intervention_id=candidate.intervention.intervention_id,
        intervention_type=candidate.intervention.intervention_type,
        description=candidate.intervention.description,
        target=candidate.intervention.target,
        value=candidate.intervention.value,
        metadata={**candidate.intervention.metadata, "billing_id": billing_id},
    )
    return candidate.__class__(
        candidate_id=candidate.candidate_id,
        intervention=updated_intervention,
        source_hypothesis_id=candidate.source_hypothesis_id,
        metadata=candidate.metadata,
    )


def _setup_counterfactual_env(artifact: dict, failure_a_candidate_id: str):
    """Set up environment, scenario, and agent for counterfactual testing using reconstructed state."""
    environment, starting_state = reconstruct_starting_state(artifact)
    scenario = reconstruct_scenario_from_artifact(artifact, failure_a_candidate_id)

    agent = TargetAgent(
        TargetAgentToolInterface(build_tool_registry(environment)),
        MockProvider(
            customer_id="cust_bcdfbba2",
            subscription_id="sub_ab88faa5",
            billing_id="bill_f464af3e",
        ),
    )

    return environment, scenario, agent, starting_state, "bill_f464af3e"


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestRealFailureRegressionIntegration:
    """End-to-end regression integration test using real Failure A artifact."""

    FAILURE_A_CANDIDATE_ID = "candidate_billing_status_cancelled_trigger_step_4"
    FAILURE_A_ID = "failure_ad13bb52"
    DISCOVERY_RUN_ID = "discovery_2117fc60"

    @classmethod
    def setup_class(cls):
        """Set up all shared state once for the entire test class."""
        # Load artifact
        cls.artifact = load_artifact()
        
        # 1. Extract scenario run and evidence
        cls.scenario_run = extract_scenario_run(cls.artifact, cls.FAILURE_A_CANDIDATE_ID)
        cls.verification, cls.failure = extract_verification_failure(cls.artifact, cls.FAILURE_A_CANDIDATE_ID)

        # Verify we loaded the correct failure
        assert cls.scenario_run.scenario_id == "scenario_candidate_billing_status_cancelled_trigger_step_4"
        assert cls.failure is not None
        assert cls.failure.failure_id == cls.FAILURE_A_ID
        assert cls.failure.outcome == ScenarioOutcome.UNSAFE_ACTION

        # 1. REAL FAILURE ANALYSIS
        analyzer = CausalAnalyzer()
        cls.analysis = analyzer.analyze(cls.scenario_run, cls.verification, cls.failure)

        # Confirm hypothesis
        stale_hypotheses = [h for h in cls.analysis.hypotheses if h.hypothesis_id == "hypothesis_stale_observation"]
        assert len(stale_hypotheses) == 1
        assert stale_hypotheses[0].confidence == 1.0
        assert "no longer current" in stale_hypotheses[0].statement

        # Generate candidate
        cls.candidate = generate_candidates(cls.analysis)[0]

        # 2. OFFLINE COUNTERFACTUAL VALIDATION
        environment, scenario, agent, starting_state, billing_id = _setup_counterfactual_env(
            cls.artifact, cls.FAILURE_A_CANDIDATE_ID
        )
        cls.environment = environment
        cls.scenario = scenario
        cls.agent = agent
        cls.starting_state = starting_state

        updated_candidate = _make_updated_candidate(cls.candidate, billing_id)
        executor = CounterfactualExecutor()
        cls.counterfactual_result = executor.execute(
            scenario,
            starting_state,
            environment,
            agent,
            updated_candidate,
            original_run_id=cls.scenario_run.run_id,
        )

        # Verify counterfactual is VALIDATED
        assert cls.counterfactual_result.status == CounterfactualStatus.VALIDATED
        assert cls.counterfactual_result.verification is not None
        assert cls.counterfactual_result.verification.passed is True
        assert cls.counterfactual_result.failure is None
        assert cls.counterfactual_result.environment_restored is True
        assert cls.counterfactual_result.intervention_reached is True

        # 3. REGRESSION TEST GENERATION
        generator = RegressionTestGenerator()
        cls.regression = generator.generate(
            scenario=cls.scenario,
            failure=cls.failure,
            analysis=cls.analysis,
            counterfactual=cls.counterfactual_result,
            candidate=cls.candidate,
            discovery_run_id="discovery_2117fc60",
        )

        # Store for later use
        cls.DISCOVERY_RUN_ID = "discovery_2117fc60"
        cls.FAILURE_A_ID = "failure_ad13bb52"
        cls.FAILURE_A_CANDIDATE_ID = "candidate_billing_status_cancelled_trigger_step_4"


    # -------------------------------------------------------------------------
    # 1. REAL FAILURE ANALYSIS
    # -------------------------------------------------------------------------

    def test_real_failure_produces_stale_observation_hypothesis(self):
        """1. Real Failure A produces hypothesis_stale_observation."""
        stale_hypotheses = [h for h in self.analysis.hypotheses if h.hypothesis_id == "hypothesis_stale_observation"]
        assert len(stale_hypotheses) == 1
        assert stale_hypotheses[0].confidence == 1.0
        assert "no longer current" in stale_hypotheses[0].statement

    def test_real_failure_generates_counterfactual_candidate(self):
        """1. Real Failure A generates the required-observation candidate."""
        candidate = self.candidate
        assert candidate.candidate_id == "candidate_required_billing_observation"
        assert candidate.intervention.intervention_type.value == "required_observation"
        assert candidate.intervention.target == "cancel_subscription"
        assert candidate.intervention.value == "get_billing_status"
        assert candidate.intervention.metadata["billing_id"] == "bill_f464af3e"
        assert candidate.source_hypothesis_id == "hypothesis_stale_observation"

    # -------------------------------------------------------------------------
    # 2. OFFLINE COUNTERFACTUAL VALIDATION
    # -------------------------------------------------------------------------

    def test_offline_counterfactual_execution_is_validated(self):
        """2. Offline counterfactual executes and is VALIDATED with MockProvider."""
        result = self.counterfactual_result
        assert result.status == CounterfactualStatus.VALIDATED
        assert result.verification is not None
        assert result.verification.passed is True
        assert result.failure is None
        assert result.environment_restored is True
        assert result.intervention_reached is True

    # -------------------------------------------------------------------------
    # 3. REGRESSION TEST GENERATION
    # -------------------------------------------------------------------------

    def test_regression_test_generated_with_correct_provenance(self):
        """3. RegressionTest generated with correct expectations and provenance."""
        regression = self.regression
        assert regression.scenario.scenario_id == self.scenario.scenario_id
        assert regression.expected_original_outcome.outcome == ScenarioOutcome.UNSAFE_ACTION
        assert regression.expected_repaired_outcome.outcome == ScenarioOutcome.SUCCESS
        assert regression.validated_intervention.intervention_type.value == "required_observation"
        assert regression.validated_intervention.target == "cancel_subscription"
        assert regression.validated_intervention.value == "get_billing_status"

        # Verify provenance
        assert regression.provenance.discovery_run_id == "discovery_2117fc60"
        assert regression.provenance.scenario_id == self.scenario.scenario_id
        assert regression.provenance.failure_id == "failure_ad13bb52"
        assert regression.provenance.analysis_id == self.analysis.analysis_id
        assert regression.provenance.counterfactual_result_id == self.counterfactual_result.result_id
        assert regression.provenance.intervention_id == self.candidate.intervention.intervention_id

    def test_regression_preserves_exact_scenario(self):
        """3. Regression test preserves exact scenario (no silent regeneration)."""
        assert self.regression.scenario.scenario_id == self.scenario.scenario_id
        assert self.regression.scenario.to_dict() == self.scenario.to_dict()
        assert self.regression.scenario.perturbations == self.scenario.perturbations

    def test_regression_preserves_failure_provenance(self):
        """3. Regression test preserves failure provenance."""
        assert self.regression.provenance.failure_id == "failure_ad13bb52"
        assert self.regression.expected_original_outcome.outcome == self.failure.outcome

    def test_regression_preserves_counterfactual_provenance(self):
        """3. Regression test preserves counterfactual provenance."""
        assert self.regression.provenance.counterfactual_result_id == self.counterfactual_result.result_id
        assert self.regression.provenance.intervention_id == self.candidate.intervention.intervention_id

    # -------------------------------------------------------------------------
    # 4. ORIGINAL REPLAY
    # -------------------------------------------------------------------------

    def test_original_replay_reproduces_unsafe_action(self):
        """4. Original replay reproduces the unsafe_action outcome."""
        runner = RegressionTestRunner()
        original_result = runner.run_original(self.regression, self.environment, self.agent, self.starting_state)

        assert original_result.status == RegressionStatus.PASSED
        assert original_result.actual_outcome == ScenarioOutcome.UNSAFE_ACTION
        assert original_result.expected_outcome == ScenarioOutcome.UNSAFE_ACTION
        assert original_result.environment_restored is True

    def test_original_replay_restores_environment(self):
        """4. Original replay restores environment to starting state."""
        runner = RegressionTestRunner()
        runner.run_original(self.regression, self.environment, self.agent, self.starting_state)
        assert self.environment.snapshot() == self.starting_state

    # -------------------------------------------------------------------------
    # 5. REPAIRED REPLAY
    # -------------------------------------------------------------------------

    def test_repaired_replay_achieves_success(self):
        """5. Repaired replay achieves success with intervention."""
        runner = RegressionTestRunner()
        repaired_result = runner.run_repaired(self.regression, self.environment, self.agent, self.starting_state)
        assert repaired_result.status == RegressionStatus.PASSED
        assert repaired_result.actual_outcome == ScenarioOutcome.SUCCESS
        assert repaired_result.expected_outcome == ScenarioOutcome.SUCCESS
        assert repaired_result.environment_restored is True

    def test_repaired_replay_has_expected_behavior(self):
        """5. Repaired replay shows expected behavior (fresh observation + reconciliation)."""
        runner = RegressionTestRunner()
        repaired_result = runner.run_repaired(self.regression, self.environment, self.agent, self.starting_state)
        tool_sequence = repaired_result.metadata.get("tool_sequence", [])
        assert "get_billing_status" in tool_sequence
        assert "reconcile_subscription_state" in tool_sequence
        assert "cancel_subscription" not in tool_sequence

    def test_repaired_replay_restores_environment(self):
        """5. Repaired replay restores environment to starting state."""
        runner = RegressionTestRunner()
        runner.run_repaired(self.regression, self.environment, self.agent, self.starting_state)
        assert self.environment.snapshot() == self.starting_state

    # -------------------------------------------------------------------------
    # 6. REGRESSION SUITE
    # -------------------------------------------------------------------------

    def test_regression_suite_discovers_and_runs_test(self):
        """6. RegressionSuite discovers test and runs both phases successfully."""
        suite = RegressionSuite()
        suite.add_test(self.regression)
        result = suite.run_all(self.environment, self.agent, self.starting_state)

        assert len(result.results) == 2
        assert all(item.status == RegressionStatus.PASSED for item in result.results)

        original_results = [r for r in result.results if r.phase == "original"]
        assert len(original_results) == 1
        assert original_results[0].status == RegressionStatus.PASSED
        assert original_results[0].actual_outcome == ScenarioOutcome.UNSAFE_ACTION

        repaired_results = [r for r in result.results if r.phase == "repaired"]
        assert len(repaired_results) == 1
        assert repaired_results[0].status == RegressionStatus.PASSED
        assert repaired_results[0].actual_outcome == ScenarioOutcome.SUCCESS

    def test_regression_suite_no_regression_reported(self):
        """6. No regression reported when both phases pass."""
        suite = RegressionSuite()
        suite.add_test(self.regression)
        result = suite.run_all(self.environment, self.agent, self.starting_state)
        # No regression should be reported (both phases pass)
        assert all(item.status == RegressionStatus.PASSED for item in result.results)

    def test_regression_suite_restores_environment(self):
        """6. RegressionSuite restores environment after running."""
        suite = RegressionSuite()
        suite.add_test(self.regression)
        suite.run_all(self.environment, self.agent, self.starting_state)
        assert self.environment.snapshot() == self.starting_state

    # -------------------------------------------------------------------------
    # 7. ARTIFACT SERIALIZATION
    # -------------------------------------------------------------------------

    def test_regression_test_serialization_roundtrip(self):
        """7. RegressionTest serializes to dict and YAML roundtrip."""
        payload = self.regression.to_dict()
        restored = RegressionTest.from_dict(payload)
        assert restored.to_dict() == payload
        yaml_payload = self.regression.to_yaml()
        assert RegressionTest.from_yaml(yaml_payload).to_dict() == self.regression.to_dict()

    def test_regression_suite_result_serialization(self):
        """7. RegressionSuite result serializes to dict."""
        suite = RegressionSuite()
        suite.add_test(self.regression)
        result = suite.run_all(self.environment, self.agent, self.starting_state)
        payload = result.to_dict()
        assert json.dumps(payload)

    def test_counterfactual_result_serialization(self):
        """7. CounterfactualResult serializes with intervention_reached field."""
        json_str = json.dumps(self.counterfactual_result.to_dict())
        data = json.loads(json_str)
        assert data["intervention_reached"] is True
        assert data["status"] == "validated"

    # -------------------------------------------------------------------------
    # 8. IMMUTABILITY
    # -------------------------------------------------------------------------

    def test_source_artifact_unchanged(self):
        """8. Source discovery artifact is unchanged."""
        artifact_copy = deepcopy(self.artifact)
        _ = extract_scenario_run(self.artifact, self.FAILURE_A_CANDIDATE_ID)
        _ = extract_verification_failure(self.artifact, self.FAILURE_A_CANDIDATE_ID)
        _ = reconstruct_starting_state(self.artifact)
        _ = reconstruct_scenario_from_artifact(self.artifact, self.FAILURE_A_CANDIDATE_ID)
        assert self.artifact == artifact_copy

    def test_scenario_run_unchanged(self):
        """8. Reconstructed scenario_run is unchanged."""
        assert self.scenario_run.scenario_id == "scenario_candidate_billing_status_cancelled_trigger_step_4"
        assert self.scenario_run.run_id == "scenario_run_1e86d5c1"

    def test_causal_analysis_unchanged(self):
        """8. Causal analysis is unchanged."""
        analyzer = CausalAnalyzer()
        analysis2 = analyzer.analyze(self.scenario_run, self.verification, self.failure)
        # Analysis ID is generated fresh each time, check hypotheses are identical
        assert [h.hypothesis_id for h in analysis2.hypotheses] == [h.hypothesis_id for h in self.analysis.hypotheses]
        assert [h.confidence for h in analysis2.hypotheses] == [h.confidence for h in self.analysis.hypotheses]

    def test_counterfactual_result_unchanged(self):
        """8. Counterfactual result is unchanged."""
        environment, scenario, agent, starting_state, billing_id = _setup_counterfactual_env(
            self.artifact, self.FAILURE_A_CANDIDATE_ID
        )
        analyzer = CausalAnalyzer()
        analysis = analyzer.analyze(self.scenario_run, self.verification, self.failure)
        candidates = generate_candidates(analysis)
        candidate = candidates[0]
        updated_candidate = _make_updated_candidate(candidate, billing_id)

        executor = CounterfactualExecutor()
        result2 = executor.execute(
            scenario,
            starting_state,
            environment,
            agent,
            updated_candidate,
            original_run_id=self.scenario_run.run_id,
        )
        assert result2.status == self.counterfactual_result.status
        assert result2.intervention_reached == self.counterfactual_result.intervention_reached
        assert result2.environment_restored == self.counterfactual_result.environment_restored

    def test_starting_environment_unchanged(self):
        """8. Starting environment snapshot is unchanged."""
        env2, state2 = reconstruct_starting_state(self.artifact)
        assert state2 == self.starting_state


def run_demo_offline_validation() -> dict:
    """Execute the deterministic MockProvider replay and return observed results.

    The report generator reuses this offline test harness so report status comes
    from an actual local replay and regression-suite run. No credentials or
    external services are used.
    """
    case = TestRealFailureRegressionIntegration
    case.setup_class()

    runner = RegressionTestRunner()
    original = runner.run_original(case.regression, case.environment, case.agent, case.starting_state)
    repaired = runner.run_repaired(case.regression, case.environment, case.agent, case.starting_state)

    suite = RegressionSuite()
    suite.add_test(case.regression)
    suite_result = suite.run_all(case.environment, case.agent, case.starting_state)
    phases = {item.phase: item for item in suite_result.results}
    all_phases_passed = bool(suite_result.results) and all(
        item.status == RegressionStatus.PASSED for item in suite_result.results
    )
    verification = case.counterfactual_result.verification

    return {
        "original_replay": {
            "outcome": original.actual_outcome.value,
            "status": "PASSED" if original.status == RegressionStatus.PASSED else "FAILED",
            "environment_restored": original.environment_restored,
        },
        "repaired_replay": {
            "outcome": repaired.actual_outcome.value,
            "status": "PASSED" if repaired.status == RegressionStatus.PASSED else "FAILED",
            "tool_sequence": repaired.metadata.get("tool_sequence", []),
            "environment_restored": repaired.environment_restored,
        },
        "verification": {
            "passed": bool(verification and verification.passed),
            "failure_detector": "no failure" if case.counterfactual_result.failure is None else "failure detected",
        },
        "regression_test": {
            "status": "PASSED" if all_phases_passed else "FAILED",
            "no_regression_reported": all_phases_passed,
            "phase_statuses": {phase: item.status.value for phase, item in phases.items()},
        },
        "validation_source": "executed deterministic MockProvider replay and regression suite",
    }


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
