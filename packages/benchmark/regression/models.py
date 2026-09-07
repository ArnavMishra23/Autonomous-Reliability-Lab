from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from packages.counterfactual.models import Intervention
from packages.discovery.failure.models import ScenarioOutcome
from packages.discovery.scenario.models import Scenario


class RegressionStatus(StrEnum):
    PASSED = "passed"
    NOT_REPRODUCED = "not_reproduced"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True, slots=True)
class Expectation:
    outcome: ScenarioOutcome
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", ScenarioOutcome(self.outcome))

    def to_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome.value, "description": self.description}


@dataclass(frozen=True, slots=True)
class Provenance:
    discovery_run_id: str | None = None
    scenario_id: str | None = None
    failure_id: str | None = None
    analysis_id: str | None = None
    counterfactual_result_id: str | None = None
    intervention_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovery_run_id": self.discovery_run_id,
            "scenario_id": self.scenario_id,
            "failure_id": self.failure_id,
            "analysis_id": self.analysis_id,
            "counterfactual_result_id": self.counterfactual_result_id,
            "intervention_id": self.intervention_id,
        }


@dataclass(frozen=True, slots=True)
class RegressionTest:
    regression_test_id: str
    name: str
    description: str
    scenario: Scenario
    expected_original_outcome: Expectation
    validated_intervention: Intervention
    expected_repaired_outcome: Expectation
    provenance: Provenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "regression_test_id": self.regression_test_id,
            "name": self.name,
            "description": self.description,
            "scenario": self.scenario.to_dict(),
            "expected_original_outcome": self.expected_original_outcome.to_dict(),
            "validated_intervention": self.validated_intervention.to_dict(),
            "expected_repaired_outcome": self.expected_repaired_outcome.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegressionTest":
        intervention_payload = payload["validated_intervention"]
        return cls(
            regression_test_id=payload["regression_test_id"],
            name=payload["name"],
            description=payload["description"],
            scenario=Scenario.from_dict(payload["scenario"]),
            expected_original_outcome=_expectation(payload["expected_original_outcome"]),
            validated_intervention=Intervention(
                intervention_id=intervention_payload["intervention_id"],
                intervention_type=intervention_payload["intervention_type"],
                description=intervention_payload["description"],
                target=intervention_payload["target"],
                value=intervention_payload["value"],
                metadata=intervention_payload.get("metadata", {}),
            ),
            expected_repaired_outcome=_expectation(payload["expected_repaired_outcome"]),
            provenance=Provenance(**payload.get("provenance", {})),
            metadata=payload.get("metadata", {}),
        )

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, value: str) -> "RegressionTest":
        import yaml

        return cls.from_dict(yaml.safe_load(value))


@dataclass(frozen=True, slots=True)
class RegressionRunResult:
    regression_test_id: str
    phase: str
    status: RegressionStatus
    actual_outcome: ScenarioOutcome | None
    expected_outcome: ScenarioOutcome
    scenario_run_id: str | None
    failure_id: str | None
    environment_restored: bool
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RegressionStatus(self.status))
        if self.actual_outcome is not None:
            object.__setattr__(self, "actual_outcome", ScenarioOutcome(self.actual_outcome))
        object.__setattr__(self, "expected_outcome", ScenarioOutcome(self.expected_outcome))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "regression_test_id": self.regression_test_id,
            "phase": self.phase,
            "status": self.status.value,
            "actual_outcome": self.actual_outcome.value if self.actual_outcome else None,
            "expected_outcome": self.expected_outcome.value,
            "scenario_run_id": self.scenario_run_id,
            "failure_id": self.failure_id,
            "environment_restored": self.environment_restored,
            "errors": list(self.errors),
            "metadata": _json_value(self.metadata),
        }


def _expectation(value: Mapping[str, Any]) -> Expectation:
    return Expectation(outcome=value["outcome"], description=value["description"])


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value
