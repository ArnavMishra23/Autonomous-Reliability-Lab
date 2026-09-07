from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from packages.agent.target import AgentRun, AgentTask


class PerturbationType(StrEnum):
    STATE_CHANGE = "state_change"


class TriggerType(StrEnum):
    BEFORE_RUN = "before_run"
    AGENT_STEP = "agent_step"


class ScenarioRunStatus(StrEnum):
    COMPLETED = "completed"
    AGENT_FAILED = "agent_failed"
    PERTURBATION_FAILED = "perturbation_failed"
    RUNNER_FAILED = "runner_failed"


@dataclass(frozen=True, slots=True)
class Trigger:
    type: TriggerType
    step: int | None = None

    def __post_init__(self) -> None:
        trigger_type = TriggerType(self.type)
        object.__setattr__(self, "type", trigger_type)
        if trigger_type == TriggerType.BEFORE_RUN and self.step is not None:
            raise ValueError("before_run triggers cannot define a step")
        if trigger_type == TriggerType.AGENT_STEP and (
            self.step is None or self.step < 1
        ):
            raise ValueError("agent_step triggers require a positive step")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type.value}
        if self.step is not None:
            result["step"] = self.step
        return result


@dataclass(frozen=True, slots=True)
class Perturbation:
    perturbation_id: str
    type: PerturbationType
    target: str
    value: Any
    trigger: Trigger
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", PerturbationType(self.type))
        object.__setattr__(self, "metadata", MappingProxyType(deepcopy(dict(self.metadata))))
        object.__setattr__(self, "value", deepcopy(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "type": self.type.value,
            "target": self.target,
            "value": _json_value(self.value),
            "trigger": self.trigger.to_dict(),
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    name: str
    description: str
    task: AgentTask
    perturbations: tuple[Perturbation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "perturbations", tuple(self.perturbations))
        object.__setattr__(self, "metadata", MappingProxyType(deepcopy(dict(self.metadata))))
        identifiers = [item.perturbation_id for item in self.perturbations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Scenario perturbation IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "task": self.task.to_dict(),
            "perturbations": [item.to_dict() for item in self.perturbations],
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Scenario":
        task_payload = payload["task"]
        perturbations = tuple(
            Perturbation(
                perturbation_id=item["perturbation_id"],
                type=item["type"],
                target=item["target"],
                value=item["value"],
                trigger=Trigger(
                    type=item["trigger"]["type"],
                    step=item["trigger"].get("step"),
                ),
                metadata=item.get("metadata", {}),
            )
            for item in payload.get("perturbations", [])
        )
        return cls(
            scenario_id=payload["scenario_id"],
            name=payload["name"],
            description=payload["description"],
            task=AgentTask(
                task_id=task_payload["task_id"],
                goal=task_payload["goal"],
                context=task_payload.get("context", {}),
            ),
            perturbations=perturbations,
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class AppliedPerturbation:
    perturbation_id: str
    target: str
    value: Any
    step: int | None
    event_id: str
    applied_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "target": self.target,
            "value": _json_value(self.value),
            "step": self.step,
            "event_id": self.event_id,
            "applied_at": self.applied_at.isoformat(),
        }


@dataclass
class ScenarioRun:
    scenario_id: str
    run_id: str
    status: ScenarioRunStatus
    started_at: datetime
    ended_at: datetime
    applied_perturbations: tuple[AppliedPerturbation, ...] = ()
    agent_run: AgentRun | None = None
    final_environment: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "applied_perturbations": [
                item.to_dict() for item in self.applied_perturbations
            ],
            "agent_run": self.agent_run.to_dict() if self.agent_run else None,
            "final_environment": _json_value(self.final_environment),
            "error": self.error,
        }


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(
            {key: getattr(value, key) for key in value.__dataclass_fields__}
        )
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    return value
