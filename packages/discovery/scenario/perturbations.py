from __future__ import annotations

from datetime import datetime

from packages.environment.models import BillingStatus
from packages.environment.service import EnvironmentService

from .models import AppliedPerturbation, Perturbation, PerturbationType


class PerturbationError(ValueError):
    """Raised when a declared perturbation cannot be applied."""


def apply_perturbation(
    environment: EnvironmentService,
    perturbation: Perturbation,
    *,
    step: int | None,
    applied_at: datetime,
) -> AppliedPerturbation:
    if perturbation.type != PerturbationType.STATE_CHANGE:
        raise PerturbationError(
            f"Unsupported perturbation type: {perturbation.type.value}"
        )

    resource_type, resource_id, field = _parse_target(perturbation.target)
    if resource_type != "billing" or field != "status":
        raise PerturbationError(
            "State-change target must use billing.<billing_id>.status"
        )

    try:
        new_status = BillingStatus(perturbation.value)
    except ValueError as exc:
        raise PerturbationError(
            f"Invalid billing status: {perturbation.value}"
        ) from exc

    event = environment.change_billing_status(
        billing_id=resource_id,
        new_status=new_status,
        metadata={
            "source": "scenario",
            "perturbation_id": perturbation.perturbation_id,
            **dict(perturbation.metadata),
        },
    )
    return AppliedPerturbation(
        perturbation_id=perturbation.perturbation_id,
        target=perturbation.target,
        value=new_status.value,
        step=step,
        event_id=event.id,
        applied_at=applied_at,
    )


def _parse_target(target: str) -> tuple[str, str, str]:
    parts = target.split(".")
    if len(parts) != 3 or any(not part for part in parts):
        raise PerturbationError(
            "Invalid perturbation target; expected billing.<billing_id>.status"
        )
    return parts[0], parts[1], parts[2]
