from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .models import ScenarioCandidate

OUTCOME_WEIGHTS: Mapping[str, float] = {
    "unsafe_action": 10.0,
    "failure": 7.0,
    "degraded": 4.0,
    "success": 1.0,
    "inconclusive": 0.5,
}
INTERESTING_OUTCOMES = frozenset({"unsafe_action", "failure", "degraded"})
BOUNDARY_BONUS = 3.0
RELEVANCE_BONUS = 1.5
INTERACTION_BONUS = 2.0
NOVELTY_BONUS = 0.25


@dataclass(frozen=True, slots=True)
class SearchDecision:
    candidate_id: str
    score: float
    reasons: tuple[str, ...]
    score_components: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "reasons": list(self.reasons),
            "score_components": dict(self.score_components),
        }


def candidate_distance(first: ScenarioCandidate, second: ScenarioCandidate) -> int:
    """Count parameter names whose assigned values differ."""
    names = set(first.parameter_values) | set(second.parameter_values)
    return sum(
        first.parameter_values.get(name) != second.parameter_values.get(name)
        for name in names
    )


class FailureGuidedSearchStrategy:
    """Transparent active search around prior non-success outcomes."""

    def __init__(self) -> None:
        self.last_decision: SearchDecision | None = None

    def choose_next(
        self,
        candidates: Iterable[ScenarioCandidate],
        evaluated: frozenset[str] = frozenset(),
        history: Iterable[Mapping[str, Any]] = (),
    ) -> ScenarioCandidate | None:
        available = tuple(
            candidate
            for candidate in candidates
            if candidate.candidate_id not in evaluated
        )
        if not available:
            self.last_decision = None
            return None

        records = tuple(history)
        interesting = tuple(
            record for record in records if record.get("outcome") in INTERESTING_OUTCOMES
        )
        relevance = self._parameter_relevance(records)
        scored = [
            self._score(candidate, records, interesting, relevance)
            for candidate in available
        ]
        scored.sort(key=lambda item: (-item[0], item[1].key, item[1].candidate_id))
        score, candidate, reasons, components = scored[0]
        self.last_decision = SearchDecision(
            candidate_id=candidate.candidate_id,
            score=round(score, 6),
            reasons=tuple(reasons),
            score_components={
                name: round(value, 6) for name, value in components.items()
            },
        )
        return candidate

    def _score(
        self,
        candidate: ScenarioCandidate,
        records: tuple[Mapping[str, Any], ...],
        interesting: tuple[Mapping[str, Any], ...],
        relevance: Mapping[str, int],
    ) -> tuple[float, ScenarioCandidate, list[str], dict[str, float]]:
        components = {
            "neighborhood": 0.0,
            "boundary": 0.0,
            "relevance": 0.0,
            "interaction": 0.0,
            "novelty": NOVELTY_BONUS,
        }
        score = components["novelty"]
        reasons = ["novelty_bonus"]
        candidate_values = candidate.parameter_values

        for record in interesting:
            prior = self._candidate_from_record(record)
            if prior is None:
                continue
            distance = candidate_distance(candidate, prior)
            weight = OUTCOME_WEIGHTS.get(record.get("outcome", "inconclusive"), 0.0)
            contribution = weight / (1 + distance)
            components["neighborhood"] += contribution
            score += contribution
            reasons.append(
                f"near_{record.get('outcome')}_{prior.candidate_id}"
            )

        boundary_pairs = self._boundary_pairs(records)
        for left, right in boundary_pairs:
            left_candidate = self._candidate_from_record(left)
            right_candidate = self._candidate_from_record(right)
            if left_candidate is None or right_candidate is None:
                continue
            if min(
                candidate_distance(candidate, left_candidate),
                candidate_distance(candidate, right_candidate),
            ) <= 1:
                components["boundary"] += BOUNDARY_BONUS
                score += BOUNDARY_BONUS
                reasons.append("boundary_bonus")
                break

        for name, frequency in relevance.items():
            if any(
                prior.get("candidate", {}).get("parameter_values", {}).get(name)
                == candidate_values.get(name)
                for prior in interesting
            ):
                contribution = RELEVANCE_BONUS * frequency
                components["relevance"] += contribution
                score += contribution
                reasons.append(f"relevant_parameter_{name}")

        for record in interesting:
            prior = self._candidate_from_record(record)
            if prior is None:
                continue
            shared = sum(
                candidate_values.get(name) == value
                for name, value in prior.parameter_values.items()
            )
            if shared >= 2:
                components["interaction"] += INTERACTION_BONUS
                score += INTERACTION_BONUS
                reasons.append("interaction_bonus")
                break

        components["total"] = score
        return score, candidate, reasons, components

    @staticmethod
    def _candidate_from_record(
        record: Mapping[str, Any],
    ) -> ScenarioCandidate | None:
        payload = record.get("candidate")
        if not isinstance(payload, Mapping):
            return None
        values = payload.get("parameter_values")
        candidate_id = payload.get("candidate_id")
        if not isinstance(values, Mapping) or not isinstance(candidate_id, str):
            return None
        return ScenarioCandidate(candidate_id, values)

    @staticmethod
    def _boundary_pairs(
        records: tuple[Mapping[str, Any], ...],
    ) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                if left.get("outcome") == right.get("outcome"):
                    continue
                left_candidate = FailureGuidedSearchStrategy._candidate_from_record(left)
                right_candidate = FailureGuidedSearchStrategy._candidate_from_record(right)
                if left_candidate and right_candidate and candidate_distance(left_candidate, right_candidate) == 1:
                    pairs.append((left, right))
        return tuple(pairs)

    @staticmethod
    def _parameter_relevance(
        records: tuple[Mapping[str, Any], ...],
    ) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        interesting = [
            record for record in records if record.get("outcome") in INTERESTING_OUTCOMES
        ]
        successes = [record for record in records if record.get("outcome") == "success"]
        for interesting_record in interesting:
            interesting_candidate = FailureGuidedSearchStrategy._candidate_from_record(
                interesting_record
            )
            if interesting_candidate is None:
                continue
            for success_record in successes:
                success_candidate = FailureGuidedSearchStrategy._candidate_from_record(
                    success_record
                )
                if success_candidate is None or candidate_distance(interesting_candidate, success_candidate) != 1:
                    continue
                for name in interesting_candidate.parameter_values:
                    if interesting_candidate.parameter_values.get(name) != success_candidate.parameter_values.get(name):
                        counts[name] = counts.get(name, 0) + 1
        return counts
