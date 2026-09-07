from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from packages.agent.models import ToolDefinition, ToolResult, ToolTrace
from packages.agent.protocol import TargetAgentToolInterface
from packages.analysis.models import CausalAnalysis

from .models import CounterfactualCandidate, Intervention, InterventionType


class InterventionToolInterface:
    """Tool boundary wrapper that inserts declared observations before actions."""

    def __init__(self, base: TargetAgentToolInterface, intervention: Intervention) -> None:
        self._base = base
        self._intervention = intervention

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]:
        return self._base.available_tools

    @property
    def trace(self) -> ToolTrace:
        return self._base.trace

    def reset_trace(self) -> None:
        self._base.reset_trace()

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolResult:
        if (
            self._intervention.intervention_type == InterventionType.REQUIRED_OBSERVATION
            and tool_name == self._intervention.target
        ):
            observation_tool = self._intervention.value
            observation_arguments = self._observation_arguments(arguments)
            observation_result = self._base.execute(observation_tool, observation_arguments)
            if not observation_result.success:
                return observation_result
            # Return the fresh evidence as this turn's result. The agent/provider
            # gets another decision cycle and the original action is not forced.
            return observation_result
        return self._base.execute(tool_name, arguments)

    def _observation_arguments(self, action_arguments: Mapping[str, Any]) -> dict[str, Any]:
        billing_id = self._intervention.metadata.get("billing_id")
        if not isinstance(billing_id, str):
            raise ValueError("required_observation intervention requires billing_id metadata")
        return {"billing_id": billing_id}


def generate_candidates(
    analysis: CausalAnalysis,
    *,
    billing_id: str | None = None,
) -> tuple[CounterfactualCandidate, ...]:
    """Generate the one supported repair only for stale-observation evidence."""
    hypothesis = next(
        (
            item
            for item in analysis.hypotheses
            if item.hypothesis_id == "hypothesis_stale_observation"
        ),
        None,
    )
    if hypothesis is None:
        return ()
    resolved_billing_id = billing_id or _billing_id_from_analysis(analysis)
    if resolved_billing_id is None:
        return ()
    intervention = Intervention(
        intervention_id="intervention_required_billing_observation",
        intervention_type=InterventionType.REQUIRED_OBSERVATION,
        description="Require a fresh billing observation before cancellation.",
        target="cancel_subscription",
        value="get_billing_status",
        metadata={"billing_id": resolved_billing_id},
    )
    return (
        CounterfactualCandidate(
            candidate_id="candidate_required_billing_observation",
            intervention=intervention,
            source_hypothesis_id=hypothesis.hypothesis_id,
        ),
    )


def _billing_id_from_analysis(analysis: CausalAnalysis) -> str | None:
    for node in analysis.graph.nodes:
        if node.node_type == "reality_change" and node.entity_type == "billing":
            return node.entity_id
    return None
