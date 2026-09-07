from __future__ import annotations

from packages.agent.target import TargetAgent
from packages.counterfactual.executor import CounterfactualExecutor
from packages.counterfactual.models import CounterfactualCandidate
from packages.discovery.failure import DeterministicVerifier, FailureDetector
from packages.discovery.failure.models import ScenarioOutcome
from packages.discovery.scenario import ScenarioRunner
from packages.environment.service import EnvironmentService
from packages.environment.state import EnvironmentState

from .models import RegressionRunResult, RegressionStatus, RegressionTest


class RegressionTestRunner:
    """Runs original and repaired phases against one unchanged scenario."""

    def __init__(self) -> None:
        self._scenario_runner = ScenarioRunner()
        self._verifier = DeterministicVerifier()
        self._detector = FailureDetector(self._verifier)
        self._counterfactual = CounterfactualExecutor(
            runner=self._scenario_runner,
            verifier=self._verifier,
            detector=self._detector,
        )

    def run_original(
        self,
        regression_test: RegressionTest,
        environment: EnvironmentService,
        agent: TargetAgent,
        starting_state: EnvironmentState,
    ) -> RegressionRunResult:
        try:
            scenario_run = self._scenario_runner.run(
                regression_test.scenario,
                environment,
                agent,
            )
            verification = self._verifier.verify(scenario_run)
            failure = self._detector.detect(scenario_run, verification)
            actual = failure.outcome if failure else (
                ScenarioOutcome.SUCCESS if verification.passed else ScenarioOutcome.INCONCLUSIVE
            )
            status = (
                RegressionStatus.PASSED
                if actual == regression_test.expected_original_outcome.outcome
                else RegressionStatus.NOT_REPRODUCED
            )
            return RegressionRunResult(
                regression_test_id=regression_test.regression_test_id,
                phase="original",
                status=status,
                actual_outcome=actual,
                expected_outcome=regression_test.expected_original_outcome.outcome,
                scenario_run_id=scenario_run.run_id,
                failure_id=failure.failure_id if failure else None,
                environment_restored=environment.snapshot() == starting_state,
                errors=tuple(verification.errors) if status != RegressionStatus.PASSED else (),
            )
        except Exception as exc:
            environment.restore(starting_state)
            return RegressionRunResult(
                regression_test_id=regression_test.regression_test_id,
                phase="original",
                status=RegressionStatus.EXECUTION_ERROR,
                actual_outcome=None,
                expected_outcome=regression_test.expected_original_outcome.outcome,
                scenario_run_id=None,
                failure_id=None,
                environment_restored=environment.snapshot() == starting_state,
                errors=(str(exc),),
            )

    def run_repaired(
        self,
        regression_test: RegressionTest,
        environment: EnvironmentService,
        agent: TargetAgent,
        starting_state: EnvironmentState,
    ) -> RegressionRunResult:
        candidate = CounterfactualCandidate(
            candidate_id=f"candidate_{regression_test.validated_intervention.intervention_id}",
            intervention=regression_test.validated_intervention,
            source_hypothesis_id="hypothesis_stale_observation",
        )
        try:
            result = self._counterfactual.execute(
                regression_test.scenario,
                starting_state,
                environment,
                agent,
                candidate,
            )
            actual = (
                ScenarioOutcome.SUCCESS
                if result.status.value == "validated"
                else result.failure.outcome
                if result.failure
                else ScenarioOutcome.INCONCLUSIVE
            )
            status = (
                RegressionStatus.PASSED
                if actual == regression_test.expected_repaired_outcome.outcome
                and result.status.value == "validated"
                else RegressionStatus.REGRESSED
            )
            return RegressionRunResult(
                regression_test_id=regression_test.regression_test_id,
                phase="repaired",
                status=status,
                actual_outcome=actual,
                expected_outcome=regression_test.expected_repaired_outcome.outcome,
                scenario_run_id=result.counterfactual_run_id,
                failure_id=result.failure.failure_id if result.failure else None,
                environment_restored=result.environment_restored,
                errors=(
                    (str(result.metadata["error"]),)
                    if "error" in result.metadata
                    else ()
                ),
                metadata=result.metadata,
            )
        except Exception as exc:
            environment.restore(starting_state)
            return RegressionRunResult(
                regression_test_id=regression_test.regression_test_id,
                phase="repaired",
                status=RegressionStatus.EXECUTION_ERROR,
                actual_outcome=None,
                expected_outcome=regression_test.expected_repaired_outcome.outcome,
                scenario_run_id=None,
                failure_id=None,
                environment_restored=environment.snapshot() == starting_state,
                errors=(str(exc),),
            )
