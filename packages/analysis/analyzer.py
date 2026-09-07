from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Mapping
from uuid import uuid4

from packages.discovery.failure.models import FailureRecord, VerificationResult
from packages.discovery.scenario.models import ScenarioRun

from .graph import add_edge, build_graph
from .models import (
    CausalAnalysis,
    CausalNode,
    EvidenceReference,
    FailureHypothesis,
)


class CausalAnalyzer:
    """Builds an evidence-backed causal representation without hidden reasoning."""

    def analyze(
        self,
        scenario_run: ScenarioRun,
        verification: VerificationResult | None = None,
        failure: FailureRecord | None = None,
    ) -> CausalAnalysis:
        run_payload = deepcopy(scenario_run.to_dict())
        verification_payload = verification.to_dict() if verification else None
        failure_payload = failure.to_dict() if failure else None
        timeline = self.extract_timeline(scenario_run)
        nodes: list[CausalNode] = []
        edges = []
        evidence: list[EvidenceReference] = []

        reality_nodes = self._reality_nodes(run_payload, evidence)
        nodes.extend(reality_nodes)
        observation_nodes, observation_evidence = self._observation_nodes(
            run_payload, timeline, evidence
        )
        nodes.extend(observation_nodes)
        evidence.extend(observation_evidence)
        action_nodes, action_evidence = self._action_nodes(run_payload, timeline, evidence)
        nodes.extend(action_nodes)
        evidence.extend(action_evidence)
        result_nodes = self._tool_result_nodes(run_payload, timeline, evidence)
        nodes.extend(result_nodes)
        verification_node = self._verification_node(verification_payload, evidence)
        if verification_node:
            nodes.append(verification_node)
        outcome_node = self._outcome_node(run_payload, failure_payload)
        nodes.append(outcome_node)

        for reality in reality_nodes:
            for observation in observation_nodes:
                if self._contradicts(reality, observation):
                    add_edge(edges, reality, observation, "contradicted")
        for action in action_nodes:
            preceding = [
                observation
                for observation in observation_nodes
                if observation.metadata.get("sequence", -1)
                < action.metadata.get("sequence", -1)
            ]
            if preceding:
                basis = CausalNode(
                    node_id=f"decision_basis_{action.node_id}",
                    node_type="decision_basis",
                    timestamp=action.timestamp,
                    source="observable_execution_context",
                    entity_type=action.entity_type,
                    entity_id=action.entity_id,
                    value={"observations": [node.node_id for node in preceding]},
                )
                nodes.append(basis)
                for observation in preceding:
                    add_edge(edges, observation, basis, "observed")
                add_edge(edges, basis, action, "followed_by")
        for action in action_nodes:
            for result in result_nodes:
                if result.metadata.get("call_id") == action.metadata.get("call_id"):
                    add_edge(edges, action, result, "resulted_in")
        if verification_node:
            for result in result_nodes:
                add_edge(edges, result, verification_node, "associated_with")
        for action in action_nodes:
            if verification_node:
                add_edge(edges, action, verification_node, "associated_with")
        if verification_node:
            add_edge(edges, verification_node, outcome_node, "resulted_in")

        divergence, consequential = self._key_nodes(nodes, reality_nodes, observation_nodes, action_nodes)
        hypotheses = self._hypotheses(
            reality_nodes,
            observation_nodes,
            action_nodes,
            result_nodes,
            verification_node,
            failure,
        )
        primary_failure = failure.summary if failure else (
            "; ".join(verification.errors) if verification and verification.errors else None
        )
        return CausalAnalysis(
            analysis_id=f"analysis_{uuid4().hex[:8]}",
            scenario_id=scenario_run.scenario_id,
            run_id=scenario_run.run_id,
            primary_failure=primary_failure,
            first_reality_divergence=divergence.node_id if divergence else None,
            first_consequential_action=consequential.node_id if consequential else None,
            graph=build_graph(nodes, edges),
            hypotheses=tuple(hypotheses),
            evidence=tuple(evidence),
            metadata={"timeline_length": len(timeline), "evidence_only": True},
        )

    def extract_timeline(self, scenario_run: ScenarioRun) -> tuple[Mapping[str, Any], ...]:
        payload = deepcopy(scenario_run.to_dict())
        timeline: list[dict[str, Any]] = []
        for index, event in enumerate(payload.get("final_environment", {}).get("events", ())):
            timeline.append({"sequence": index, "kind": "environment_event", "value": event})
        trace = payload.get("agent_run", {}).get("trace", []) if payload.get("agent_run") else []
        base = len(timeline)
        for index, entry in enumerate(trace):
            timeline.append({"sequence": base + index * 2, "kind": "tool_call", "value": entry["call"]})
            timeline.append({"sequence": base + index * 2 + 1, "kind": "tool_result", "value": entry["result"]})
        return tuple(timeline)

    @staticmethod
    def _reality_nodes(payload: Mapping[str, Any], evidence: list[EvidenceReference]) -> list[CausalNode]:
        nodes = []
        for index, event in enumerate(payload.get("final_environment", {}).get("events", ())):
            event_id = str(event.get("id", f"event_{index}"))
            reference = EvidenceReference(
                evidence_id=f"evidence_{event_id}",
                source_type="environment_event",
                source_id=event_id,
                description="Authoritative environment event",
            )
            evidence.append(reference)
            nodes.append(CausalNode(
                node_id=f"reality_{event_id}",
                node_type="reality_change",
                timestamp=_timestamp(event.get("timestamp")),
                source="environment",
                entity_type="billing" if str(event.get("target", "")).startswith("bill_") else None,
                entity_id=event.get("target"),
                value={"field": event.get("field"), "old": event.get("old_value"), "new": event.get("new_value")},
                metadata={"sequence": index, "evidence_id": reference.evidence_id},
            ))
        return nodes

    @staticmethod
    def _observation_nodes(
        payload: Mapping[str, Any],
        timeline: tuple[Mapping[str, Any], ...],
        evidence: list[EvidenceReference],
    ) -> tuple[list[CausalNode], list[EvidenceReference]]:
        nodes = []
        references = []
        for item in timeline:
            if item["kind"] != "tool_result":
                continue
            result = item["value"]
            if not result.get("success"):
                continue
            call = next(
                (entry["value"] for entry in timeline if entry["kind"] == "tool_call" and entry["sequence"] == item["sequence"] - 1),
                {},
            )
            if call.get("tool_name") not in {"get_customer", "get_subscription", "get_billing_status"}:
                continue
            call_id = str(call.get("call_id"))
            reference = EvidenceReference(
                evidence_id=f"evidence_{call_id}",
                source_type="tool_result",
                source_id=call_id,
                description=f"Observable result from {call.get('tool_name')}",
                timestamp=_timestamp(result.get("timestamp")),
            )
            references.append(reference)
            output = result.get("output")
            entity_type = call.get("tool_name", "").removeprefix("get_")
            entity_id = output.get("id") if isinstance(output, Mapping) else None
            nodes.append(CausalNode(
                node_id=f"observation_{call_id}",
                node_type="observation",
                timestamp=_timestamp(result.get("timestamp")),
                source="tool_protocol",
                entity_type=entity_type,
                entity_id=entity_id,
                value=deepcopy(output),
                metadata={"sequence": item["sequence"], "call_id": call_id, "evidence_id": reference.evidence_id},
            ))
        return nodes, references

    @staticmethod
    def _action_nodes(
        payload: Mapping[str, Any],
        timeline: tuple[Mapping[str, Any], ...],
        evidence: list[EvidenceReference],
    ) -> tuple[list[CausalNode], list[EvidenceReference]]:
        nodes = []
        references = []
        for item in timeline:
            if item["kind"] != "tool_call":
                continue
            call = item["value"]
            if call.get("tool_name") != "cancel_subscription":
                continue
            call_id = str(call.get("call_id"))
            reference = EvidenceReference(
                evidence_id=f"evidence_{call_id}",
                source_type="tool_call",
                source_id=call_id,
                description="Consequential cancellation tool call",
                timestamp=_timestamp(call.get("timestamp")),
            )
            references.append(reference)
            nodes.append(CausalNode(
                node_id=f"action_{call_id}",
                node_type="action",
                timestamp=_timestamp(call.get("timestamp")),
                source="tool_protocol",
                entity_type="subscription",
                entity_id=call.get("arguments", {}).get("subscription_id"),
                value=deepcopy(call.get("arguments", {})),
                metadata={"sequence": item["sequence"], "call_id": call_id, "evidence_id": reference.evidence_id},
            ))
        return nodes, references

    @staticmethod
    def _tool_result_nodes(
        payload: Mapping[str, Any],
        timeline: tuple[Mapping[str, Any], ...],
        evidence: list[EvidenceReference],
    ) -> list[CausalNode]:
        nodes = []
        for item in timeline:
            if item["kind"] != "tool_result":
                continue
            result = item["value"]
            call_id = str(result.get("call_id"))
            reference = EvidenceReference(
                evidence_id=f"evidence_result_{call_id}",
                source_type="tool_result",
                source_id=call_id,
                description="Observable tool result",
                timestamp=_timestamp(result.get("timestamp")),
            )
            evidence.append(reference)
            nodes.append(CausalNode(
                node_id=f"tool_result_{call_id}",
                node_type="tool_result",
                timestamp=_timestamp(result.get("timestamp")),
                source="tool_protocol",
                value={"success": result.get("success"), "output": result.get("output"), "error": result.get("error")},
                metadata={"sequence": item["sequence"], "call_id": call_id, "evidence_id": reference.evidence_id},
            ))
        return nodes

    @staticmethod
    def _verification_node(payload: Mapping[str, Any] | None, evidence: list[EvidenceReference]) -> CausalNode | None:
        if payload is None:
            return None
        evidence_id = "evidence_verification"
        reference = EvidenceReference(evidence_id, "verification", "verification", "Deterministic verification result")
        evidence.append(reference)
        return CausalNode("verification", "verification", None, "deterministic_verifier", value=deepcopy(payload), metadata={"evidence_id": evidence_id})

    @staticmethod
    def _outcome_node(payload: Mapping[str, Any], failure: Mapping[str, Any] | None) -> CausalNode:
        return CausalNode("outcome", "outcome", None, "failure_detector", value={"scenario_status": payload.get("status"), "failure": deepcopy(failure)})

    @staticmethod
    def _contradicts(reality: CausalNode, observation: CausalNode) -> bool:
        return (
            reality.entity_id == observation.entity_id
            and isinstance(reality.value, Mapping)
            and isinstance(observation.value, Mapping)
            and reality.value.get("field") == "status"
            and observation.value.get("status") is not None
            and reality.value.get("old") != reality.value.get("new")
            and observation.value.get("status") == reality.value.get("old")
        )

    @staticmethod
    def _key_nodes(nodes: list[CausalNode], realities: list[CausalNode], observations: list[CausalNode], actions: list[CausalNode]) -> tuple[CausalNode | None, CausalNode | None]:
        divergence = next(
            (reality for reality in sorted(realities, key=lambda node: node.metadata.get("sequence", 0)) if any(CausalAnalyzer._contradicts(reality, observation) for observation in observations)),
            None,
        )
        consequential = next(
            (action for action in sorted(actions, key=lambda node: node.metadata.get("sequence", 0)) if divergence and action.metadata.get("sequence", 0) > divergence.metadata.get("sequence", 0)),
            None,
        )
        return divergence, consequential

    @staticmethod
    def _hypotheses(
        realities: list[CausalNode],
        observations: list[CausalNode],
        actions: list[CausalNode],
        results: list[CausalNode],
        verification: CausalNode | None,
        failure: FailureRecord | None,
    ) -> list[FailureHypothesis]:
        if not realities or not observations or not actions:
            return []
        supporting = tuple(
            EvidenceReference(f"hypothesis_{index}", "causal_node", node.node_id, "Observable evidence")
            for index, node in enumerate((*realities[:1], *observations[:1], *actions[:1], *results[:1]))
        )
        verification_failed = (
            verification is not None
            and isinstance(verification.value, Mapping)
            and not verification.value.get("passed", True)
        )
        score = sum(
            (bool(realities), bool(observations), bool(actions), bool(results), verification_failed)
        ) / 5
        return [FailureHypothesis(
            hypothesis_id="hypothesis_stale_observation",
            statement="The agent acted on state that was no longer current.",
            confidence=round(score, 3),
            supporting_evidence=supporting,
            metadata={"score_is_evidence_coverage_not_probability": True, "failure_id": failure.failure_id if failure else None},
        )]


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
