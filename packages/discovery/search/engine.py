from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from packages.agent.target import TargetAgent
from packages.discovery.failure.detector import FailureDetector
from packages.discovery.failure.verifier import DeterministicVerifier
from packages.discovery.scenario.runner import ScenarioRunner
from packages.environment.service import EnvironmentService

from .generator import ScenarioGenerator
from .models import (
    DiscoveryResult,
    DiscoveryStatus,
    ScenarioCandidate,
    ScenarioSpace,
)
from .state import DiscoveryState
from .strategies import SearchStrategy


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DiscoveryEngine:
    """Runs a finite baseline search over generated deterministic scenarios."""

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock

    def run(
        self,
        space: ScenarioSpace,
        generator: ScenarioGenerator,
        strategy: SearchStrategy,
        runner: ScenarioRunner,
        environment: EnvironmentService,
        agent: TargetAgent,
        failure_detector: FailureDetector,
        max_scenarios: int,
    ) -> DiscoveryResult:
        if max_scenarios < 1:
            raise ValueError("max_scenarios must be at least 1")
        started_at = self._clock()
        state = DiscoveryState(started_at=started_at)
        candidates = space.enumerate_candidates()
        verifier = DeterministicVerifier()
        status = DiscoveryStatus.COMPLETED
        error: str | None = None
        decisions: list[dict[str, object]] = []

        try:
            while state.scenarios_evaluated < max_scenarios:
                candidate = strategy.choose_next(
                    candidates,
                    state.evaluated_candidates,
                    state.results,
                )
                decision = getattr(strategy, "last_decision", None)
                if decision is not None:
                    decisions.append(decision.to_dict())
                if candidate is None:
                    status = (
                        DiscoveryStatus.NO_CANDIDATES
                        if state.scenarios_evaluated == 0
                        else DiscoveryStatus.COMPLETED
                    )
                    break
                if not space.is_valid(candidate):
                    status = DiscoveryStatus.FAILED
                    error = f"Invalid scenario candidate: {candidate.candidate_id}"
                    break
                try:
                    scenario = generator.generate(candidate)
                    scenario_run = runner.run(scenario, environment, agent)
                    verification = verifier.verify(scenario_run)
                    failure = failure_detector.detect(scenario_run, verification)
                    state.record(candidate, scenario_run, verification, failure)
                except Exception as exc:
                    status = DiscoveryStatus.FAILED
                    error = f"Unrecoverable discovery error for {candidate.candidate_id}: {exc}"
                    break
            else:
                status = DiscoveryStatus.MAX_SCENARIOS_REACHED
        except Exception as exc:
            status = DiscoveryStatus.FAILED
            error = str(exc)

        state.finished_at = self._clock()
        return DiscoveryResult(
            discovery_run_id=f"discovery_{uuid4().hex[:8]}",
            status=status,
            scenarios_evaluated=state.scenarios_evaluated,
            outcome_counts=state.outcome_counts,
            unique_failure_count=len({failure.failure_id for failure in state.failure_records}),
            failure_ids=tuple(failure.failure_id for failure in state.failure_records),
            results=state.results,
            search_decisions=tuple(decisions),
            started_at=state.started_at,
            finished_at=state.finished_at,
            error=error,
        )

