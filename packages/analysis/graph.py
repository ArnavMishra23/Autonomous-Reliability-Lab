from __future__ import annotations

from .models import CausalEdge, CausalGraph, CausalNode, EvidenceReference


def add_edge(
    edges: list[CausalEdge],
    source: CausalNode,
    target: CausalNode,
    relation: str,
    evidence: tuple[EvidenceReference, ...] = (),
) -> None:
    edges.append(
        CausalEdge(
            edge_id=f"edge_{len(edges) + 1}",
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            relation=relation,
            evidence=evidence,
        )
    )


def build_graph(nodes: list[CausalNode], edges: list[CausalEdge]) -> CausalGraph:
    return CausalGraph(nodes=tuple(nodes), edges=tuple(edges))
