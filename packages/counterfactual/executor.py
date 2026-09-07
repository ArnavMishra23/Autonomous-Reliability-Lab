from __future__ import annotations

from uuid import uuid4

from packages.agent.target import AgentRun, AgentTask, TargetAgent
from packages.discovery.failure import DeterministicVerifier, FailureDetector
from packages.discovery.scenario import Scenario, ScenarioRunner
from packages.environment.service import EnvironmentService
from packages.environment.state import EnvironmentState

from .interventions import InterventionToolInterface
from .models import (
    CounterfactualCandidate,
    CounterfactualResult,
    CounterfactualStatus,
)


class CounterfactualExecutor:
    """Replays one unchanged scenario with one tool-boundary intervention."""

    def __init__(
        self,
        *,
        runner: ScenarioRunner | None = None,
        verifier: DeterministicVerifier | None = None,
        detector: FailureDetector | None = None,
    ) -> None:
        self._runner = runner or ScenarioRunner()
        self._verifier = verifier or DeterministicVerifier()
        self._detector = detector or FailureDetector(self._verifier)

    def execute(
        self,
        scenario: Scenario,
        starting_state: EnvironmentState,
        environment: EnvironmentService,
        agent: TargetAgent,
        candidate: CounterfactualCandidate,
        *,
        original_run_id: str | None = None,
    ) -> CounterfactualResult:
        scenario_run = None
        verification = None
        failure = None
        status = CounterfactualStatus.EXECUTION_ERROR
        metadata: dict[str, object] = {}
        try:
            environment.restore(starting_state)
            scenario_run = self._runner.run(
                scenario,
                environment,
                _WrappedAgent(agent, candidate),
            )
            verification = self._verifier.verify(scenario_run)
            failure = self._detector.detect(scenario_run, verification)
            status = (
                CounterfactualStatus.VALIDATED
                if verification.passed and failure is None and scenario_run.agent_run is not None
                else CounterfactualStatus.FAILED
            )
            metadata = {
                "scenario_status": scenario_run.status.value,
                "tool_sequence": [
                    entry.call.tool_name
                    for entry in scenario_run.agent_run.trace.entries
                ]
                if scenario_run.agent_run
                else [],
                "final_subscription_status": _final_subscription_status(scenario_run),
                "final_billing_status": _final_billing_status(scenario_run),
            }
        except Exception as exc:
            metadata = {"error": str(exc)}
        try:
            environment.restore(starting_state)
            restored = True
        except Exception as exc:
            restored = False
            status = CounterfactualStatus.EXECUTION_ERROR
            metadata = {"error": f"Environment restore failed: {exc}"}
        return CounterfactualResult(
            result_id=f"result_{uuid4().hex[:8]}",
            candidate_id=candidate.candidate_id,
            original_run_id=original_run_id,
            counterfactual_run_id=scenario_run.run_id if scenario_run else None,
            status=status,
            verification=verification,
            failure=failure,
            environment_restored=restored,
            metadata=metadata,
        )

    def validate_repeatedly(
        self,
        scenario: Scenario,
        starting_state: EnvironmentState,
        environment: EnvironmentService,
        agent: TargetAgent,
        candidate: CounterfactualCandidate,
        runs: int,
    ) -> CounterfactualResult:
        if runs < 1:
            raise ValueError("runs must be at least 1")
        results = [
            self.execute(scenario, starting_state, environment, agent, candidate)
            for _ in range(runs)
        ]
        successful = sum(result.status == CounterfactualStatus.VALIDATED for result in results)
        final = results[-1]
        return CounterfactualResult(
            result_id=final.result_id,
            candidate_id=final.candidate_id,
            original_run_id=final.original_run_id,
            counterfactual_run_id=final.counterfactual_run_id,
            status=(
                CounterfactualStatus.VALIDATED
                if successful == runs
                else CounterfactualStatus.FAILED
            ),
            verification=final.verification,
            failure=final.failure,
            environment_restored=all(result.environment_restored for result in results),
            metadata={"repeated_validation": True},
            validation_runs=runs,
            successful_runs=successful,
            failed_runs=runs - successful,
        )


class _WrappedAgent:
    def __init__(self, agent: TargetAgent, candidate: CounterfactualCandidate) -> None:
        self._agent = agent
        self._candidate = candidate

    def run_with_step_hook(
        self,
        task: AgentTask,
        *,
        step_hook,
    ) -> AgentRun:
        return self._agent.run_with_tool_wrapper_and_step_hook(
            task,
            lambda base: InterventionToolInterface(base, self._candidate.intervention),
            step_hook=step_hook,
        )


def _final_subscription_status(scenario_run) -> str | None:
    subscriptions = scenario_run.final_environment.get("subscriptions", {})
    if not subscriptions:
        return None
    return next(iter(subscriptions.values())).get("status")


def _final_billing_status(scenario_run) -> str | None:
    billing_accounts = scenario_run.final_environment.get("billing_accounts", {})
    if not billing_accounts:
        return None
    return next(iter(billing_accounts.values())).get("status")


