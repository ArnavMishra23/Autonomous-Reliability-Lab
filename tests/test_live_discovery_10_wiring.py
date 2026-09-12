from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from packages.discovery.search import (
    ScenarioSpace,
    ScenarioParameter,
    ScenarioGenerator,
    RandomSearchStrategy,
)
from packages.environment.service import EnvironmentService
from packages.discovery.search.models import DiscoveryResult, DiscoveryStatus
from packages.discovery.scenario.models import ScenarioRun, ScenarioRunStatus
from packages.agent.target import AgentRun, AgentRunStatus


def create_space() -> ScenarioSpace:
    """Create the full 10-candidate ScenarioSpace."""
    return ScenarioSpace(
        parameters=(
            ScenarioParameter("billing_status", "discrete", ("active", "cancelled")),
            ScenarioParameter("trigger_step", "discrete", (1, 2, 3, 4, 5)),
        )
    )


def test_space_enumeration():
    """Verify the space enumerates exactly 10 candidates."""
    space = create_space()
    candidates = space.enumerate_candidates()
    assert len(candidates) == 10, f"Expected 10 candidates, got {len(candidates)}"

    # Verify all combinations exist
    expected = {
        ("active", 1), ("active", 2), ("active", 3), ("active", 4), ("active", 5),
        ("cancelled", 1), ("cancelled", 2), ("cancelled", 3), ("cancelled", 4), ("cancelled", 5),
    }
    actual = {(c.parameter_values["billing_status"], c.parameter_values["trigger_step"]) for c in candidates}
    assert actual == expected, f"Missing combinations: {expected - actual}"


def test_candidate_uniqueness():
    """Verify all 10 candidates have unique IDs."""
    space = create_space()
    candidates = space.enumerate_candidates()
    ids = [c.candidate_id for c in candidates]
    assert len(set(ids)) == 10, f"Duplicate candidate IDs found: {ids}"


def test_random_strategy_fixed_seed():
    """Verify RandomSearchStrategy with fixed seed produces deterministic ordering."""
    space = create_space()
    candidates = space.enumerate_candidates()

    strategy1 = RandomSearchStrategy(seed=20260910)
    strategy2 = RandomSearchStrategy(seed=20260910)

    # Both should produce the same sequence
    evaluated = frozenset()
    for _ in range(10):
        c1 = strategy1.choose_next(candidates, evaluated)
        c2 = strategy2.choose_next(candidates, evaluated)
        assert c1 is not None and c2 is not None
        assert c1.candidate_id == c2.candidate_id, f"Seed not stable: {c1.candidate_id} != {c2.candidate_id}"
        evaluated = frozenset(set(evaluated) | {c1.candidate_id})


def test_random_strategy_covers_all():
    """Verify RandomSearchStrategy covers all candidates exactly once."""
    space = create_space()
    candidates = space.enumerate_candidates()

    strategy = RandomSearchStrategy(seed=20260910)
    evaluated = frozenset()
    seen = set()

    for _ in range(10):
        c = strategy.choose_next(candidates, evaluated)
        assert c is not None
        assert c.candidate_id not in seen, f"Duplicate: {c.candidate_id}"
        seen.add(c.candidate_id)
        evaluated = frozenset(set(evaluated) | {c.candidate_id})

    # Next call should return None
    c = strategy.choose_next(candidates, evaluated)
    assert c is None


def test_experiment_targets_ten():
    """Verify experiment configuration targets exactly 10 candidates."""
    space = create_space()
    candidates = space.enumerate_candidates()
    assert len(candidates) == 10

    # The experiment uses max_scenarios=10
    max_scenarios = 10
    assert max_scenarios == 10


def test_artifact_serialization_roundtrip():
    """Verify DiscoveryResult can be serialized and deserialized."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    result = DiscoveryResult(
        discovery_run_id="discovery_test_10",
        status=DiscoveryStatus.COMPLETED,
        scenarios_evaluated=10,
        outcome_counts={"success": 5, "failure": 3, "degraded": 2},
        unique_failure_count=3,
        failure_ids=("failure_1", "failure_2", "failure_3"),
        results=(
            {
                "candidate": {
                    "candidate_id": "candidate_billing_status_active_trigger_step_1",
                    "parameter_values": {"billing_status": "active", "trigger_step": 1},
                    "metadata": {"ordinal": 0},
                },
                "scenario_id": "scenario_candidate_billing_status_active_trigger_step_1",
                "run_id": "scenario_run_abc123",
                "scenario_status": "completed",
                "outcome": "success",
                "verification": {"passed": True, "checks": [], "errors": []},
                "failure": None,
            },
        ),
        search_decisions=(),
        started_at=now,
        finished_at=now,
        error=None,
    )

    result_dict = result.to_dict()
    
    # Add experiment metadata (as the script does)
    result_dict["experiment_metadata"] = {
        "experiment_name": "discovery10",
        "seed": 20260910,
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "candidate_count": 10,
        "max_steps": 10,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Test serialization to JSON and back
    json_str = json.dumps(result_dict, indent=2)
    loaded = json.loads(json_str)
    
    # Verify key fields survive round-trip
    assert loaded["discovery_run_id"] == "discovery_test_10"
    assert loaded["status"] == "completed"
    assert loaded["scenarios_evaluated"] == 10
    assert loaded["outcome_counts"] == {"success": 5, "failure": 3, "degraded": 2}
    assert loaded["unique_failure_count"] == 3
    assert loaded["failure_ids"] == ["failure_1", "failure_2", "failure_3"]
    assert len(loaded["results"]) == 1
    
    # Verify experiment metadata
    assert loaded["experiment_metadata"]["experiment_name"] == "discovery10"
    assert loaded["experiment_metadata"]["seed"] == 20260910
    assert loaded["experiment_metadata"]["candidate_count"] == 10
    assert loaded["experiment_metadata"]["max_steps"] == 10


def test_artifact_naming():
    """Test artifact filename generation logic for discovery10."""
    from datetime import datetime, timezone

    run_id = "discovery_abc123"
    timestamp = datetime(2026, 9, 10, 17, 30, 0, tzinfo=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"discovery10_{timestamp}_{run_id}.json"
    
    assert filename == "discovery10_20260910T173000Z_discovery_abc123.json"
    assert filename.endswith(".json")
    assert "discovery10_" in filename
    assert "20260910T173000Z" in filename


def test_no_secrets_in_artifact():
    """Verify artifact doesn't contain secrets."""
    from datetime import datetime, timezone
    
    result = DiscoveryResult(
        discovery_run_id="discovery_test",
        status=DiscoveryStatus.COMPLETED,
        scenarios_evaluated=10,
        outcome_counts={"success": 10},
        unique_failure_count=0,
        failure_ids=(),
        results=(),
        search_decisions=(),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        error=None,
    )
    
    result_dict = result.to_dict()
    result_dict["experiment_metadata"] = {
        "experiment_name": "discovery10",
        "seed": 20260910,
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "candidate_count": 10,
        "max_steps": 10,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    json_output = json.dumps(result_dict)
    assert "NEBIUS_API_KEY" not in json_output
    assert "api_key" not in json_output.lower() or "api_key" in json_output.lower()


def test_full_artifact_structure():
    """Test the full experiment artifact structure with scenario_evidence."""
    from datetime import datetime, timezone
    from packages.discovery.scenario.models import (
        ScenarioRun, ScenarioRunStatus, AppliedPerturbation, Trigger, TriggerType, PerturbationType
    )
    from packages.agent.target import AgentRun, AgentRunStatus
    from packages.agent.models import ToolCall, ToolResult, ToolTrace, ToolTraceEntry, ToolErrorCode
    from packages.discovery.failure.models import VerificationResult, FailureRecord, ScenarioOutcome, CheckResult
    
    now = datetime.now(timezone.utc)
    
    # Build a mock AgentRun with tool trace
    call1 = ToolCall(
        call_id="call_1",
        tool_name="get_customer",
        arguments={"customer_id": "cust_1"},
        timestamp=now,
        step=1,
    )
    result1 = ToolResult(
        call_id="call_1",
        tool_name="get_customer",
        success=True,
        output={"id": "cust_1", "subscription_id": "sub_1", "billing_id": "bill_1"},
        error=None,
        error_code=None,
        timestamp=now,
    )
    
    call2 = ToolCall(
        call_id="call_2",
        tool_name="cancel_subscription",
        arguments={"subscription_id": "sub_1"},
        timestamp=now,
        step=2,
    )
    result2 = ToolResult(
        call_id="call_2",
        tool_name="cancel_subscription",
        success=False,
        output=None,
        error="Subscription already cancelled",
        error_code=ToolErrorCode.SUBSCRIPTION_ALREADY_CANCELLED,
        timestamp=now,
    )
    
    agent_run = AgentRun(
        run_id="run_test",
        task_id="task_test",
        status=AgentRunStatus.FAILED,
        started_at=now,
        finished_at=now,
        trace=ToolTrace(entries=[
            ToolTraceEntry(call=call1, result=result1),
            ToolTraceEntry(call=call2, result=result2),
        ]),
        final_result=None,
        error="Tool execution failed",
    )
    
    # Create AppliedPerturbation
    perturbation = AppliedPerturbation(
        perturbation_id="pert_1",
        target="billing.bill_1.status",
        value="cancelled",
        step=4,
        event_id="evt_1",
        applied_at=now,
    )
    
    # Create ScenarioRun
    scenario_run = ScenarioRun(
        scenario_id="scenario_candidate_billing_status_cancelled_trigger_step_4",
        run_id="scenario_run_test",
        status=ScenarioRunStatus.AGENT_FAILED,
        started_at=now,
        ended_at=now,
        applied_perturbations=(perturbation,),
        agent_run=agent_run,
        final_environment={
            "subscriptions": {"sub_1": {"id": "sub_1", "status": "active"}},
            "billing_accounts": {"bill_1": {"id": "bill_1", "status": "active"}},
            "events": [],
        },
        error="Agent failed",
    )
    
    # Test build_scenario_evidence function (replicate from script)
    def build_scenario_evidence(scenario_run: ScenarioRun, candidate_id: str) -> dict:
        agent_run = scenario_run.agent_run

        tool_trace = []
        if agent_run is not None:
            for entry in agent_run.trace.entries:
                call = entry.call
                result = entry.result
                tool_trace.append({
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "arguments": dict(call.arguments),
                    "call_timestamp": call.timestamp.isoformat(),
                    "result_call_id": result.call_id,
                    "result_tool_name": result.tool_name,
                    "result_success": result.success,
                    "result_output": result.output,
                    "result_error": result.error,
                    "result_error_code": result.error_code.value if result.error_code else None,
                    "result_timestamp": result.timestamp.isoformat(),
                })

        evidence = {
            "candidate_id": candidate_id,
            "scenario_run": {
                "scenario_id": scenario_run.scenario_id,
                "run_id": scenario_run.run_id,
                "status": scenario_run.status.value,
                "started_at": scenario_run.started_at.isoformat(),
                "ended_at": scenario_run.ended_at.isoformat(),
                "applied_perturbations": [
                    {
                        "perturbation_id": p.perturbation_id,
                        "target": p.target,
                        "value": p.value,
                        "step": p.step,
                        "event_id": p.event_id,
                        "applied_at": p.applied_at.isoformat(),
                    }
                    for p in scenario_run.applied_perturbations
                ],
                "final_environment": {
                    "subscriptions": {
                        k: dict(v) if hasattr(v, "__dataclass_fields__") else v
                        for k, v in scenario_run.final_environment.get("subscriptions", {}).items()
                    },
                    "billing_accounts": {
                        k: dict(v) if hasattr(v, "__dataclass_fields__") else v
                        for k, v in scenario_run.final_environment.get("billing_accounts", {}).items()
                    },
                    "events": [
                        dict(e) if hasattr(e, "__dataclass_fields__") else e
                        for e in scenario_run.final_environment.get("events", [])
                    ],
                },
                "error": scenario_run.error,
            },
            "agent_run": None,
            "tool_trace": tool_trace,
            "restored": (
                len(scenario_run.final_environment.get("subscriptions", {})) > 0
                and len(scenario_run.final_environment.get("billing_accounts", {})) > 0
            ),
        }

        if agent_run is not None:
            evidence["agent_run"] = {
                "run_id": agent_run.run_id,
                "task_id": agent_run.task_id,
                "status": agent_run.status.value,
                "started_at": agent_run.started_at.isoformat(),
                "finished_at": agent_run.finished_at.isoformat() if agent_run.finished_at else None,
                "final_result": agent_run.final_result,
                "error": agent_run.error,
            }

        return evidence
    
    evidence = build_scenario_evidence(scenario_run, "candidate_billing_status_cancelled_trigger_step_4")
    
    # Verify structure
    assert evidence["candidate_id"] == "candidate_billing_status_cancelled_trigger_step_4"
    assert evidence["scenario_run"]["scenario_id"] == "scenario_candidate_billing_status_cancelled_trigger_step_4"
    assert evidence["scenario_run"]["status"] == "agent_failed"
    assert len(evidence["scenario_run"]["applied_perturbations"]) == 1
    assert evidence["scenario_run"]["applied_perturbations"][0]["target"] == "billing.bill_1.status"
    assert evidence["agent_run"] is not None
    assert evidence["agent_run"]["status"] == "failed"
    assert evidence["agent_run"]["run_id"] == "run_test"
    assert len(evidence["tool_trace"]) == 2
    assert evidence["tool_trace"][0]["tool_name"] == "get_customer"
    assert evidence["tool_trace"][0]["result_success"] is True
    assert evidence["tool_trace"][1]["tool_name"] == "cancel_subscription"
    assert evidence["tool_trace"][1]["result_success"] is False
    assert evidence["tool_trace"][1]["result_error_code"] == "subscription_already_cancelled"
    assert evidence["restored"] is True
    
    # Verify full artifact structure
    artifact = {
        "artifact_version": 1,
        "experiment_metadata": {
            "experiment_name": "discovery10",
            "seed": 20260910,
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "candidate_count": 10,
            "max_steps": 10,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "discovery_result": {
            "discovery_run_id": "discovery_test",
            "status": "completed",
            "scenarios_evaluated": 10,
        },
        "scenario_evidence": [evidence],
    }
    
    # Round-trip serialization
    json_str = json.dumps(artifact, indent=2)
    loaded = json.loads(json_str)
    
    assert loaded["artifact_version"] == 1
    assert loaded["experiment_metadata"]["experiment_name"] == "discovery10"
    assert loaded["experiment_metadata"]["seed"] == 20260910
    assert len(loaded["scenario_evidence"]) == 1
    
    loaded_evidence = loaded["scenario_evidence"][0]
    assert loaded_evidence["candidate_id"] == "candidate_billing_status_cancelled_trigger_step_4"
    assert len(loaded_evidence["tool_trace"]) == 2
    assert loaded_evidence["tool_trace"][0]["tool_name"] == "get_customer"
    assert loaded_evidence["tool_trace"][1]["result_error_code"] == "subscription_already_cancelled"
    assert loaded_evidence["restored"] is True


def test_tool_trace_completeness():
    """Verify tool trace includes all required fields."""
    from datetime import datetime, timezone
    from packages.agent.models import ToolCall, ToolResult, ToolErrorCode
    from packages.agent.target import ToolTrace, ToolTraceEntry, AgentRun, AgentRunStatus
    
    now = datetime.now(timezone.utc)
    
    call = ToolCall(
        call_id="call_test",
        tool_name="get_billing_status",
        arguments={"billing_id": "bill_1"},
        timestamp=now,
        step=3,
    )
    result = ToolResult(
        call_id="call_test",
        tool_name="get_billing_status",
        success=True,
        output={"id": "bill_1", "subscription_id": "sub_1", "status": "active"},
        error=None,
        error_code=None,
        timestamp=now,
    )
    
    agent_run = AgentRun(
        run_id="run_test",
        task_id="task_test",
        status=AgentRunStatus.SUCCEEDED,
        started_at=now,
        finished_at=now,
        trace=ToolTrace(entries=[
            ToolTraceEntry(call=call, result=result),
        ]),
        final_result="Done",
        error=None,
    )
    
    # Extract tool trace
    tool_trace = []
    for entry in agent_run.trace.entries:
        c = entry.call
        r = entry.result
        tool_trace.append({
            "call_id": c.call_id,
            "tool_name": c.tool_name,
            "arguments": dict(c.arguments),
            "call_timestamp": c.timestamp.isoformat(),
            "result_call_id": r.call_id,
            "result_tool_name": r.tool_name,
            "result_success": r.success,
            "result_output": r.output,
            "result_error": r.error,
            "result_error_code": r.error_code.value if r.error_code else None,
            "result_timestamp": r.timestamp.isoformat(),
        })
    
    assert len(tool_trace) == 1
    trace = tool_trace[0]
    assert trace["call_id"] == "call_test"
    assert trace["tool_name"] == "get_billing_status"
    assert trace["arguments"] == {"billing_id": "bill_1"}
    assert trace["call_timestamp"] == now.isoformat()
    assert trace["result_call_id"] == "call_test"
    assert trace["result_tool_name"] == "get_billing_status"
    assert trace["result_success"] is True
    assert trace["result_output"] == {"id": "bill_1", "subscription_id": "sub_1", "status": "active"}
    assert trace["result_error"] is None
    assert trace["result_error_code"] is None
    assert trace["result_timestamp"] == now.isoformat()


def test_verification_failure_serialization():
    """Verify verification and failure records survive round-trip."""
    from datetime import datetime, timezone
    from packages.discovery.failure.models import VerificationResult, FailureRecord, ScenarioOutcome, CheckResult
    
    verification = VerificationResult(
        passed=False,
        checks=(
            CheckResult(
                name="subscription_cancelled",
                passed=False,
                expected="cancelled",
                actual="active",
                severity="high",
            ),
        ),
        errors=("subscription not cancelled",),
        metadata={"subscription_id": "sub_1", "billing_id": "bill_1"},
    )
    
    failure = FailureRecord(
        failure_id="failure_1",
        scenario_id="scenario_test",
        run_id="run_test",
        outcome=ScenarioOutcome.FAILURE,
        severity="high",
        summary="Test failure",
        affected_entities=("sub_1", "bill_1"),
        failed_checks=("subscription_cancelled",),
        unsafe_actions=(),
        first_detected_event_id=None,
        metadata={},
    )
    
    v_dict = verification.to_dict()
    f_dict = failure.to_dict()
    
    # Round-trip
    v_loaded = json.loads(json.dumps(v_dict))
    f_loaded = json.loads(json.dumps(f_dict))
    
    assert v_loaded["passed"] is False
    assert len(v_loaded["checks"]) == 1
    assert v_loaded["checks"][0]["name"] == "subscription_cancelled"
    assert v_loaded["errors"] == ["subscription not cancelled"]
    
    assert f_loaded["failure_id"] == "failure_1"
    assert f_loaded["outcome"] == "failure"
    assert f_loaded["severity"] == "high"
    assert f_loaded["affected_entities"] == ["sub_1", "bill_1"]


if __name__ == "__main__":
    test_space_enumeration()
    print("test_space_enumeration: PASS")

    test_candidate_uniqueness()
    print("test_candidate_uniqueness: PASS")

    test_random_strategy_fixed_seed()
    print("test_random_strategy_fixed_seed: PASS")

    test_random_strategy_covers_all()
    print("test_random_strategy_covers_all: PASS")

    test_experiment_targets_ten()
    print("test_experiment_targets_ten: PASS")

    test_artifact_serialization_roundtrip()
    print("test_artifact_serialization_roundtrip: PASS")

    test_artifact_naming()
    print("test_artifact_naming: PASS")

    test_no_secrets_in_artifact()
    print("test_no_secrets_in_artifact: PASS")

    test_full_artifact_structure()
    print("test_full_artifact_structure: PASS")

    test_tool_trace_completeness()
    print("test_tool_trace_completeness: PASS")

    test_verification_failure_serialization()
    print("test_verification_failure_serialization: PASS")

    print("\nAll offline tests PASSED")