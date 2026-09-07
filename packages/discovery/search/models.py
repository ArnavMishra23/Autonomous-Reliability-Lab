from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import product
from types import MappingProxyType
from typing import Any, Mapping


class DiscoveryStatus(StrEnum):
    COMPLETED = "completed"
    MAX_SCENARIOS_REACHED = "max_scenarios_reached"
    NO_CANDIDATES = "no_candidates"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ScenarioParameter:
    name: str
    parameter_type: str
    allowed_values: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parameter_type != "discrete":
            raise ValueError("Only discrete scenario parameters are supported")
        values = tuple(deepcopy(self.allowed_values))
        if not self.name:
            raise ValueError("Scenario parameter name must not be empty")
        if not values:
            raise ValueError("Scenario parameter must define allowed values")
        if len(set(_freeze(value) for value in values)) != len(values):
            raise ValueError("Scenario parameter values must be unique")
        object.__setattr__(self, "allowed_values", values)
        object.__setattr__(self, "metadata", MappingProxyType(deepcopy(dict(self.metadata))))

    def validate(self, value: Any) -> bool:
        return _freeze(value) in {_freeze(item) for item in self.allowed_values}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_type": self.parameter_type,
            "allowed_values": [_json_value(value) for value in self.allowed_values],
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScenarioCandidate:
    candidate_id: str
    parameter_values: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_values", MappingProxyType(deepcopy(dict(self.parameter_values))))
        object.__setattr__(self, "metadata", MappingProxyType(deepcopy(dict(self.metadata))))

    @property
    def key(self) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted((name, _freeze(value)) for name, value in self.parameter_values.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parameter_values": _json_value(self.parameter_values),
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScenarioSpace:
    parameters: tuple[ScenarioParameter, ...]

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise ValueError("Scenario parameter names must be unique")
        object.__setattr__(self, "parameters", parameters)

    def validate_assignment(self, values: Mapping[str, Any]) -> bool:
        if set(values) != {parameter.name for parameter in self.parameters}:
            return False
        return all(
            parameter.validate(values[parameter.name])
            for parameter in self.parameters
        )

    def is_valid(self, candidate: ScenarioCandidate) -> bool:
        return self.validate_assignment(candidate.parameter_values)

    def enumerate_candidates(self) -> tuple[ScenarioCandidate, ...]:
        if not self.parameters:
            return ()
        candidates: list[ScenarioCandidate] = []
        value_sets = [parameter.allowed_values for parameter in self.parameters]
        for index, values in enumerate(product(*value_sets)):
            assignments = {
                parameter.name: deepcopy(value)
                for parameter, value in zip(self.parameters, values)
            }
            candidate_id = "candidate_" + "_".join(
                f"{name}_{_slug(value)}" for name, value in assignments.items()
            )
            candidates.append(
                ScenarioCandidate(
                    candidate_id=candidate_id,
                    parameter_values=assignments,
                    metadata={"ordinal": index},
                )
            )
        return tuple(candidates)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    discovery_run_id: str
    status: DiscoveryStatus
    scenarios_evaluated: int
    outcome_counts: Mapping[str, int]
    unique_failure_count: int
    failure_ids: tuple[str, ...]
    results: tuple[Mapping[str, Any], ...] = ()
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovery_run_id": self.discovery_run_id,
            "status": self.status.value,
            "scenarios_evaluated": self.scenarios_evaluated,
            "outcome_counts": dict(self.outcome_counts),
            "unique_failure_count": self.unique_failure_count,
            "failure_ids": list(self.failure_ids),
            "results": [_json_value(result) for result in self.results],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _slug(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value
