from __future__ import annotations

from packages.agent.target import AgentTask
from packages.discovery.scenario.models import (
    Perturbation,
    PerturbationType,
    Scenario,
    Trigger,
    TriggerType,
)

from .models import ScenarioCandidate


class ScenarioGenerator:
    """Builds declarative Customer Operations scenarios from candidates."""

    def __init__(
        self,
        *,
        customer_id: str,
        subscription_id: str,
        billing_id: str,
    ) -> None:
        self.customer_id = customer_id
        self.subscription_id = subscription_id
        self.billing_id = billing_id

    def generate(self, candidate: ScenarioCandidate) -> Scenario:
        billing_status = candidate.parameter_values.get("billing_status")
        trigger_step = candidate.parameter_values.get("trigger_step")
        if billing_status not in {"active", "cancelled"}:
            raise ValueError(f"Unsupported billing_status: {billing_status}")
        if not isinstance(trigger_step, int) or not 1 <= trigger_step <= 5:
            raise ValueError(f"Unsupported trigger_step: {trigger_step}")

        perturbations: tuple[Perturbation, ...] = ()
        if billing_status == "cancelled":
            perturbations = (
                Perturbation(
                    perturbation_id=f"{candidate.candidate_id}_billing_change",
                    type=PerturbationType.STATE_CHANGE,
                    target=f"billing.{self.billing_id}.status",
                    value="cancelled",
                    trigger=Trigger(TriggerType.AGENT_STEP, step=trigger_step),
                    metadata={"candidate_id": candidate.candidate_id},
                ),
            )

        return Scenario(
            scenario_id=f"scenario_{candidate.candidate_id}",
            name="Customer Operations subscription cancellation",
            description="Baseline finite-space cancellation experiment.",
            task=AgentTask(
                task_id=f"task_{candidate.candidate_id}",
                goal="Process Maya's subscription cancellation.",
                context={"customer_id": self.customer_id},
            ),
            perturbations=perturbations,
            metadata={
                "candidate_id": candidate.candidate_id,
                "subscription_id": self.subscription_id,
                "billing_status": billing_status,
                "trigger_step": trigger_step,
            },
        )
