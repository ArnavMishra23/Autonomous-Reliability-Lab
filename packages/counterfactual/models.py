from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping

from packages.discovery.failure.models import FailureRecord, VerificationResult
from packages.discovery.scenario.models import ScenarioRun


class InterventionType(StrEnum):
    REQUIRED_OBSERVATION = "required_observation"


class CounterfactualStatus(StrEnum):
    VALIDATED = "validated"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True, slots=True)
class Intervention:
    intervention_id: str
    intervention_type: InterventionType
    description: str
    target: str
    value: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervention_type", InterventionType(self.intervention_type))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "intervention_type": self.intervention_type.value,
            "description": self.description,
            "target": self.target,
            "value": self.value,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualCandidate:
    candidate_id: str
    intervention: Intervention
    source_hypothesis_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "intervention": self.intervention.to_dict(),
            "source_hypothesis_id": self.source_hypothesis_id,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    result_id: str
    candidate_id: str
    original_run_id: str | None
    counterfactual_run_id: str | None
    status: CounterfactualStatus
    verification: VerificationResult | None
    failure: FailureRecord | None
    environment_restored: bool
    intervention_reached: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validation_runs: int = 1
    successful_runs: int = 0
    failed_runs: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CounterfactualStatus(self.status))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "candidate_id": self.candidate_id,
            "original_run_id": self.original_run_id,
            "counterfactual_run_id": self.counterfactual_run_id,
            "status": self.status.value,
            "verification": self.verification.to_dict() if self.verification else None,
            "failure": self.failure.to_dict() if self.failure else None,
            "environment_restored": self.environment_restored,
            "intervention_reached": self.intervention_reached,
            "metadata": _json_value(self.metadata),
            "validation_runs": self.validation_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    return value
