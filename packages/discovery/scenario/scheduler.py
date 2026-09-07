from __future__ import annotations

from .models import Perturbation, Scenario, TriggerType


class ScenarioScheduler:
    """Returns declared perturbations in deterministic execution order."""

    def __init__(self, scenario: Scenario) -> None:
        self._perturbations = tuple(scenario.perturbations)
        self._validate()

    def before_run(self) -> tuple[Perturbation, ...]:
        return tuple(
            perturbation
            for perturbation in self._perturbations
            if perturbation.trigger.type == TriggerType.BEFORE_RUN
        )

    def at_agent_step(self, step: int) -> tuple[Perturbation, ...]:
        return tuple(
            perturbation
            for perturbation in self._perturbations
            if perturbation.trigger.type == TriggerType.AGENT_STEP
            and perturbation.trigger.step == step
        )

    def _validate(self) -> None:
        for perturbation in self._perturbations:
            if perturbation.trigger.type == TriggerType.AGENT_STEP:
                if perturbation.trigger.step is None or perturbation.trigger.step < 1:
                    raise ValueError(
                        f"Invalid agent step for perturbation: {perturbation.perturbation_id}"
                    )
