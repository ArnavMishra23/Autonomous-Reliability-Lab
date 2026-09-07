from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Mapping
from uuid import uuid4

from packages.agent.target import AgentRunStatus, TargetAgent
from packages.environment.models import EnvironmentEvent
from packages.environment.service import EnvironmentService

from .models import (
    AppliedPerturbation,
    Scenario,
    ScenarioRun,
    ScenarioRunStatus,
)
from .perturbations import PerturbationError, apply_perturbation
from .scheduler import ScenarioScheduler


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ScenarioRunner:
    """Executes one deterministic scenario and restores its starting world."""

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock

    def run(
        self,
        scenario: Scenario,
        environment: EnvironmentService,
        agent: TargetAgent,
    ) -> ScenarioRun:
        original_state = environment.snapshot()
        started_at = self._clock()
        applied: list[AppliedPerturbation] = []
        agent_run = None
        final_environment: Mapping[str, Any] = {}
        status = ScenarioRunStatus.RUNNER_FAILED
        error: str | None = None
        perturbation_error: str | None = None
        scheduler: ScenarioScheduler | None = None

        def apply_for_step(step: int | None) -> None:
            nonlocal perturbation_error
            perturbations = (
                scheduler.before_run()
                if step is None
                else scheduler.at_agent_step(step)
            )
            for perturbation in perturbations:
                try:
                    applied.append(
                        apply_perturbation(
                            environment,
                            perturbation,
                            step=step,
                            applied_at=self._clock(),
                        )
                    )
                except Exception as exc:
                    perturbation_error = (
                        f"Perturbation {perturbation.perturbation_id} failed: {exc}"
                    )
                    raise

        try:
            scheduler = ScenarioScheduler(scenario)
            apply_for_step(None)
            agent_run = agent.run_with_step_hook(
                scenario.task,
                step_hook=apply_for_step,
            )
            final_environment = _environment_to_dict(environment.snapshot())
            if perturbation_error is not None:
                status = ScenarioRunStatus.PERTURBATION_FAILED
                error = perturbation_error
            elif agent_run.status == AgentRunStatus.FAILED:
                status = ScenarioRunStatus.AGENT_FAILED
                error = agent_run.error
            else:
                status = ScenarioRunStatus.COMPLETED
        except PerturbationError as exc:
            status = ScenarioRunStatus.PERTURBATION_FAILED
            error = perturbation_error or str(exc)
            final_environment = _environment_to_dict(environment.snapshot())
        except Exception as exc:
            if perturbation_error is not None:
                status = ScenarioRunStatus.PERTURBATION_FAILED
                error = perturbation_error
            else:
                status = ScenarioRunStatus.RUNNER_FAILED
                error = str(exc)
            final_environment = _environment_to_dict(environment.snapshot())
        finally:
            try:
                environment.restore(original_state)
            except Exception as exc:
                status = ScenarioRunStatus.RUNNER_FAILED
                error = f"Failed to restore environment: {exc}"

        return ScenarioRun(
            scenario_id=scenario.scenario_id,
            run_id=f"scenario_run_{uuid4().hex[:8]}",
            status=status,
            started_at=started_at,
            ended_at=self._clock(),
            applied_perturbations=tuple(applied),
            agent_run=agent_run,
            final_environment=final_environment,
            error=error,
        )

def _environment_to_dict(environment_state: Any) -> dict[str, Any]:
    return {
        "customers": {
            identifier: _json_value(customer)
            for identifier, customer in environment_state.customers.items()
        },
        "subscriptions": {
            identifier: _json_value(subscription)
            for identifier, subscription in environment_state.subscriptions.items()
        },
        "billing_accounts": {
            identifier: _json_value(billing)
            for identifier, billing in environment_state.billing_accounts.items()
        },
        "events": [_json_value(event) for event in environment_state.events],
    }


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    return value
