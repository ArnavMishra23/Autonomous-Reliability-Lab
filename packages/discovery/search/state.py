from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping

from packages.discovery.failure.models import FailureRecord, ScenarioOutcome, VerificationResult
from packages.discovery.scenario.models import ScenarioRun

from .models import ScenarioCandidate


class DiscoveryState:
    """Mutable internal state for one discovery run."""

    def __init__(self, *, started_at: datetime | None = None) -> None:
        self._evaluated_candidates: set[str] = set()
        self._results: list[dict[str, Any]] = []
        self._failure_records: list[FailureRecord] = []
        self._outcome_counts: dict[str, int] = {}
        self.started_at = started_at
        self.finished_at: datetime | None = None

    @property
    def evaluated_candidates(self) -> frozenset[str]:
        return frozenset(self._evaluated_candidates)

    @property
    def scenarios_evaluated(self) -> int:
        return len(self._results)

    @property
    def outcome_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._outcome_counts))

    @property
    def failure_records(self) -> tuple[FailureRecord, ...]:
        return tuple(self._failure_records)

    @property
    def results(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(deepcopy(result) for result in self._results)

    def record(
        self,
        candidate: ScenarioCandidate,
        scenario_run: ScenarioRun,
        verification: VerificationResult,
        failure: FailureRecord | None,
    ) -> bool:
        if candidate.candidate_id in self._evaluated_candidates:
            return False
        self._evaluated_candidates.add(candidate.candidate_id)
        outcome = (
            failure.outcome.value
            if failure is not None
            else ScenarioOutcome.SUCCESS.value
        )
        self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1
        if failure is not None:
            self._failure_records.append(failure)
        self._results.append(
            {
                "candidate": candidate.to_dict(),
                "scenario_id": scenario_run.scenario_id,
                "run_id": scenario_run.run_id,
                "scenario_status": scenario_run.status.value,
                "outcome": outcome,
                "verification": verification.to_dict(),
                "failure": failure.to_dict() if failure else None,
            }
        )
        return True
