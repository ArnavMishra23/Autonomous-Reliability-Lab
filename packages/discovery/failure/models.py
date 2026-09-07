from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class ScenarioOutcome(StrEnum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILURE = "failure"
    UNSAFE_ACTION = "unsafe_action"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    expected: Any
    actual: Any
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": _json_value(self.expected),
            "actual": _json_value(self.actual),
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    passed: bool
    checks: tuple[CheckResult, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "errors": list(self.errors),
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FailureRecord:
    failure_id: str
    scenario_id: str
    run_id: str
    outcome: ScenarioOutcome
    severity: str
    summary: str
    affected_entities: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    unsafe_actions: tuple[str, ...] = ()
    first_detected_event_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", ScenarioOutcome(self.outcome))
        object.__setattr__(self, "affected_entities", tuple(self.affected_entities))
        object.__setattr__(self, "failed_checks", tuple(self.failed_checks))
        object.__setattr__(self, "unsafe_actions", tuple(self.unsafe_actions))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "outcome": self.outcome.value,
            "severity": self.severity,
            "summary": self.summary,
            "affected_entities": list(self.affected_entities),
            "failed_checks": list(self.failed_checks),
            "unsafe_actions": list(self.unsafe_actions),
            "first_detected_event_id": self.first_detected_event_id,
            "metadata": _json_value(self.metadata),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value
