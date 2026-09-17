from __future__ import annotations

from packages.environment.models import IdProvider, DefaultIdProvider
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
        id_provider: IdProvider | None = None,
    ) -> None:
        self._runner = runner or ScenarioRunner()
        self._verifier = verifier or DeterministicVerifier()
        self._detector = detector or FailureDetector(self._verifier)
        self._id_provider = id_provider or DefaultIdProvider()

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
        intervention_reached = False
        metadata: dict[str, object] = {}
        tool_sequence: list[str] = []
        model_calls = 0
        agent_steps = 0
        tool_calls = 0
        try:
            environment.restore(starting_state)
            wrapped_agent = _WrappedAgent(agent, candidate)
            scenario_run = self._runner.run(
                scenario,
                environment,
                wrapped_agent,
            )
            intervention_reached = wrapped_agent.intervention_reached

            # Extract telemetry from the agent run
            if scenario_run.agent_run is not None:
                trace_entries = scenario_run.agent_run.trace.entries
                tool_sequence = [entry.call.tool_name for entry in trace_entries]
                tool_calls = len(trace_entries)
                # Agent steps = max step number in trace, or 0 if no trace
                agent_steps = max((entry.call.step for entry in trace_entries), default=0)
                # Model calls = tool_calls + 1 (for the final response that completed or failed)
                # If the agent completed without tools, model_calls = 1
                # If the agent made tool calls and then completed/failed, model_calls = tool_calls + 1
                # If the agent failed before any tool call, model_calls = 1
                if tool_calls > 0:
                    model_calls = tool_calls + 1
                else:
                    model_calls = 1
            else:
                tool_sequence = []
                tool_calls = 0
                agent_steps = 0
                model_calls = 0

            if not intervention_reached:
                status = CounterfactualStatus.INCONCLUSIVE
                # When intervention not reached, verification is not applicable
                verification = None
                failure = None
            else:
                verification = self._verifier.verify(scenario_run)
                failure = self._detector.detect(scenario_run, verification)

                if verification.passed and failure is None and scenario_run.agent_run is not None:
                    status = CounterfactualStatus.VALIDATED
                else:
                    status = CounterfactualStatus.FAILED

            metadata = {
                "scenario_status": scenario_run.status.value,
                "tool_sequence": tool_sequence,
                "model_calls": model_calls,
                "agent_steps": agent_steps,
                "tool_calls": tool_calls,
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
            result_id=self._id_provider.generate("result_"),
            candidate_id=candidate.candidate_id,
            original_run_id=original_run_id,
            counterfactual_run_id=scenario_run.run_id if scenario_run else None,
            status=status,
            verification=verification,
            failure=failure,
            environment_restored=restored,
            intervention_reached=intervention_reached,
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
        all_reached = all(result.intervention_reached for result in results)
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
            intervention_reached=all_reached,
            metadata={"repeated_validation": True},
            validation_runs=runs,
            successful_runs=successful,
            failed_runs=runs - successful,
        )


class _WrappedAgent:
    def __init__(self, agent: TargetAgent, candidate: CounterfactualCandidate) -> None:
        self._agent = agent
        self._candidate = candidate
        self._tool_interface: InterventionToolInterface | None = None

    def run_with_step_hook(
        self,
        task: AgentTask,
        *,
        step_hook,
    ) -> AgentRun:
        def make_wrapper(base: TargetAgentToolInterface) -> InterventionToolInterface:
            self._tool_interface = InterventionToolInterface(base, self._candidate.intervention)
            return self._tool_interface

        return self._agent.run_with_tool_wrapper_and_step_hook(
            task,
            make_wrapper,
            step_hook=step_hook,
        )

    @property
    def intervention_reached(self) -> bool:
        return self._tool_interface.intervention_reached if self._tool_interface else False


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


