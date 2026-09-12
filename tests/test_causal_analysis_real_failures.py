#!/usr/bin/env python
"""Offline causal analysis of real Nemotron failures using existing CausalAnalyzer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from packages.analysis.analyzer import CausalAnalyzer
from packages.analysis.models import CausalAnalysis, CausalNode, CausalEdge
from packages.discovery.scenario.models import ScenarioRun, ScenarioRunStatus, AppliedPerturbation
from packages.agent.target import AgentRun, AgentRunStatus, ToolCall, ToolResult, ToolTrace, ToolTraceEntry
from packages.discovery.failure.models import FailureRecord, VerificationResult, ScenarioOutcome, CheckResult
from packages.environment.models import DefaultIdProvider
from packages.agent.models import ToolErrorCode


def load_artifact() -> dict:
    """Load the real discovery artifact."""
    artifact_path = Path(__file__).resolve().parents[1] / "artifacts" / "live-discovery" / "discovery10_20260911T002503Z_discovery_2117fc60.json"
    with artifact_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_scenario_run(artifact: dict, candidate_id: str) -> ScenarioRun:
    """Extract and reconstruct ScenarioRun from artifact for a specific candidate."""
    # Find in scenario_evidence
    for evidence in artifact.get("scenario_evidence", []):
        if evidence.get("candidate_id") == candidate_id:
            scenario_run_data = evidence.get("scenario_run", {})
            agent_run_data = evidence.get("agent_run")
            tool_trace_data = evidence.get("tool_trace", [])
            
            # Reconstruct AppliedPerturbations
            applied_perturbations = []
            for p in scenario_run_data.get("applied_perturbations", []):
                applied_perturbations.append(AppliedPerturbation(
                    perturbation_id=p["perturbation_id"],
                    target=p["target"],
                    value=p["value"],
                    step=p["step"],
                    event_id=p["event_id"],
                    applied_at=datetime.fromisoformat(p["applied_at"]),
                ))
            
            # Reconstruct AgentRun with ToolTrace
            agent_run = None
            if agent_run_data:
                trace_entries = []
                for t in tool_trace_data:
                    call = ToolCall(
                        call_id=t["call_id"],
                        tool_name=t["tool_name"],
                        arguments=t["arguments"],
                        timestamp=datetime.fromisoformat(t["call_timestamp"]),
                        step=t.get("step", 0) if "step" in t else 0,
                    )
                    result = ToolResult(
                        call_id=t["result_call_id"],
                        tool_name=t["result_tool_name"],
                        success=t["result_success"],
                        output=t["result_output"],
                        error=t["result_error"],
                        error_code=ToolErrorCode(t["result_error_code"]) if t.get("result_error_code") else None,
                        timestamp=datetime.fromisoformat(t["result_timestamp"]),
                    )
                    trace_entries.append(ToolTraceEntry(call=call, result=result))
                
                agent_run = AgentRun(
                    run_id=agent_run_data["run_id"],
                    task_id=agent_run_data["task_id"],
                    status=AgentRunStatus(agent_run_data["status"]),
                    started_at=datetime.fromisoformat(agent_run_data["started_at"]),
                    finished_at=datetime.fromisoformat(agent_run_data["finished_at"]) if agent_run_data.get("finished_at") else None,
                    trace=ToolTrace(entries=trace_entries),
                    final_result=agent_run_data.get("final_result"),
                    error=agent_run_data.get("error"),
                )
            
            # Reconstruct final_environment
            final_env = scenario_run_data.get("final_environment", {})
            
            return ScenarioRun(
                scenario_id=scenario_run_data["scenario_id"],
                run_id=scenario_run_data["run_id"],
                status=ScenarioRunStatus(scenario_run_data["status"]),
                started_at=datetime.fromisoformat(scenario_run_data["started_at"]),
                ended_at=datetime.fromisoformat(scenario_run_data["ended_at"]),
                applied_perturbations=tuple(applied_perturbations),
                agent_run=agent_run,
                final_environment=final_env,
                error=scenario_run_data.get("error"),
            )
    
    raise ValueError(f"Candidate {candidate_id} not found in scenario_evidence")


def extract_verification_failure(artifact: dict, candidate_id: str) -> tuple[VerificationResult | None, FailureRecord | None]:
    """Extract verification and failure from discovery_result for a candidate."""
    for result in artifact.get("discovery_result", {}).get("results", []):
        if result.get("candidate", {}).get("candidate_id") == candidate_id:
            verification_data = result.get("verification")
            failure_data = result.get("failure")
            
            verification = None
            if verification_data:
                checks = []
                for c in verification_data.get("checks", []):
                    checks.append(CheckResult(
                        name=c["name"],
                        passed=c["passed"],
                        expected=c["expected"],
                        actual=c["actual"],
                        severity=c["severity"],
                    ))
                verification = VerificationResult(
                    passed=verification_data.get("passed", False),
                    checks=tuple(checks),
                    errors=tuple(verification_data.get("errors", [])),
                    metadata=verification_data.get("metadata", {}),
                )
            
            failure = None
            if failure_data:
                failure = FailureRecord(
                    failure_id=failure_data["failure_id"],
                    scenario_id=failure_data["scenario_id"],
                    run_id=failure_data["run_id"],
                    outcome=ScenarioOutcome(failure_data["outcome"]),
                    severity=failure_data["severity"],
                    summary=failure_data["summary"],
                    affected_entities=tuple(failure_data.get("affected_entities", [])),
                    failed_checks=tuple(failure_data.get("failed_checks", [])),
                    unsafe_actions=tuple(failure_data.get("unsafe_actions", [])),
                    first_detected_event_id=failure_data.get("first_detected_event_id"),
                    metadata=failure_data.get("metadata", {}),
                )
            
            return verification, failure
    
    return None, None


def print_analysis(analysis: CausalAnalysis, label: str) -> None:
    """Print the causal analysis results."""
    print(f"\n{'='*60}")
    print(f"{label} CAUSAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Analysis ID: {analysis.analysis_id}")
    print(f"Scenario: {analysis.scenario_id}")
    print(f"Run: {analysis.run_id}")
    print(f"Primary Failure: {analysis.primary_failure}")
    print(f"First Reality Divergence: {analysis.first_reality_divergence}")
    print(f"First Consequential Action: {analysis.first_consequential_action}")
    print(f"\nGraph Nodes ({len(analysis.graph.nodes)}):")
    for node in analysis.graph.nodes:
        print(f"  {node.node_id} [{node.node_type}] - {node.entity_type}:{node.entity_id} @ {node.timestamp}")
        if node.value:
            val_str = str(node.value)[:120]
            print(f"    value: {val_str}")
    print(f"\nGraph Edges ({len(analysis.graph.edges)}):")
    for edge in analysis.graph.edges:
        print(f"  {edge.source_node_id} --[{edge.relation}]--> {edge.target_node_id}")
        if edge.evidence:
            for ev in edge.evidence:
                print(f"    evidence: {ev.evidence_id} ({ev.source_type}:{ev.source_id})")
    
    print(f"\nHypotheses ({len(analysis.hypotheses)}):")
    for hyp in analysis.hypotheses:
        print(f"  {hyp.hypothesis_id}: {hyp.statement}")
        print(f"    confidence: {hyp.confidence}")
        print(f"    supporting: {[e.evidence_id for e in hyp.supporting_evidence]}")
        print(f"    metadata: {hyp.metadata}")
    
    print(f"\nEvidence References ({len(analysis.evidence)}):")
    for ev in analysis.evidence:
        print(f"  {ev.evidence_id} [{ev.source_type}:{ev.source_id}] - {ev.description}")


def main():
    artifact = load_artifact()
    
    # Define the two failure candidates
    failure_a_candidate = "candidate_billing_status_cancelled_trigger_step_4"
    failure_b_candidate = "candidate_billing_status_active_trigger_step_4"
    
    print("Loading real discovery artifact...")
    print(f"Artifact: discovery10_20260911T002503Z_discovery_2117fc60.json")
    print(f"Model: nvidia/nemotron-3-super-120b-a12b")
    
    # Extract ScenarioRun and verification/failure for Failure A
    print(f"\n{'='*60}")
    print("EXTRACTING FAILURE A")
    print(f"{'='*60}")
    scenario_run_a = extract_scenario_run(artifact, failure_a_candidate)
    verification_a, failure_a = extract_verification_failure(artifact, failure_a_candidate)
    
    print(f"Scenario: {scenario_run_a.scenario_id}")
    print(f"Status: {scenario_run_a.status.value}")
    print(f"Agent Run: {scenario_run_a.agent_run.run_id if scenario_run_a.agent_run else 'None'}")
    print(f"Agent Status: {scenario_run_a.agent_run.status.value if scenario_run_a.agent_run else 'None'}")
    print(f"Verification: {'passed' if verification_a and verification_a.passed else 'failed' if verification_a else 'None'}")
    print(f"Failure: {failure_a.outcome.value if failure_a else 'None'} ({failure_a.summary if failure_a else 'N/A'})")
    
    # Extract ScenarioRun and verification/failure for Failure B
    print(f"\n{'='*60}")
    print("EXTRACTING FAILURE B")
    print(f"{'='*60}")
    scenario_run_b = extract_scenario_run(artifact, failure_b_candidate)
    verification_b, failure_b = extract_verification_failure(artifact, failure_b_candidate)
    
    print(f"Scenario: {scenario_run_b.scenario_id}")
    print(f"Status: {scenario_run_b.status.value}")
    print(f"Agent Run: {scenario_run_b.agent_run.run_id if scenario_run_b.agent_run else 'None'}")
    print(f"Agent Status: {scenario_run_b.agent_run.status.value if scenario_run_b.agent_run else 'None'}")
    print(f"Verification: {'passed' if verification_b and verification_b.passed else 'failed' if verification_b else 'None'}")
    print(f"Failure: {failure_b.outcome.value if failure_b else 'None'} ({failure_b.summary if failure_b else 'N/A'})")
    
    # Run CausalAnalyzer on Failure A
    print(f"\n{'='*60}")
    print("RUNNING CAUSAL ANALYZER ON FAILURE A")
    print(f"{'='*60}")
    analyzer = CausalAnalyzer()
    analysis_a = analyzer.analyze(scenario_run_a, verification_a, failure_a)
    print_analysis(analysis_a, "FAILURE A")
    
    # Run CausalAnalyzer on Failure B
    print(f"\n{'='*60}")
    print("RUNNING CAUSAL ANALYZER ON FAILURE B")
    print(f"{'='*60}")
    analysis_b = analyzer.analyze(scenario_run_b, verification_b, failure_b)
    print_analysis(analysis_b, "FAILURE B")
    
    # Save analyses for reference
    output_dir = Path(__file__).resolve().parents[1] / "tests" / "analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with (output_dir / "failure_a_analysis.json").open("w") as f:
        json.dump(analysis_a.to_dict(), f, indent=2)
    with (output_dir / "failure_b_analysis.json").open("w") as f:
        json.dump(analysis_b.to_dict(), f, indent=2)
    
    print(f"\nAnalyses saved to {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())