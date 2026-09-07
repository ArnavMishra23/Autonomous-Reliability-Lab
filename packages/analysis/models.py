from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    evidence_id: str
    source_type: str
    source_id: str
    description: str
    timestamp: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "description": self.description,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CausalNode:
    node_id: str
    node_type: str
    timestamp: datetime | None
    source: str
    entity_type: str | None = None
    entity_id: str | None = None
    value: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "value": _json_value(self.value),
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CausalEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation: str
    evidence: tuple[EvidenceReference, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relation": self.relation,
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CausalGraph:
    nodes: tuple[CausalNode, ...] = ()
    edges: tuple[CausalEdge, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True, slots=True)
class FailureHypothesis:
    hypothesis_id: str
    statement: str
    confidence: float
    supporting_evidence: tuple[EvidenceReference, ...] = ()
    contradicting_evidence: tuple[EvidenceReference, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "confidence": self.confidence,
            "supporting_evidence": [item.to_dict() for item in self.supporting_evidence],
            "contradicting_evidence": [item.to_dict() for item in self.contradicting_evidence],
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CausalAnalysis:
    analysis_id: str
    scenario_id: str
    run_id: str
    primary_failure: str | None
    first_reality_divergence: str | None
    first_consequential_action: str | None
    graph: CausalGraph
    hypotheses: tuple[FailureHypothesis, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "primary_failure": self.primary_failure,
            "first_reality_divergence": self.first_reality_divergence,
            "first_consequential_action": self.first_consequential_action,
            "graph": self.graph.to_dict(),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": _json_value(self.metadata),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value
