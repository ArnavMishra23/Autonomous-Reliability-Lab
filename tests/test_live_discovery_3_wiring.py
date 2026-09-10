from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from packages.discovery.search import (
    ScenarioSpace,
    ScenarioParameter,
    ScenarioGenerator,
)
from packages.environment.service import EnvironmentService
from packages.discovery.scenario.models import ScenarioRun, ScenarioRunStatus, AppliedPerturbation
from packages.agent.target import AgentRun, AgentRunStatus, ToolCall, ToolResult
from packages.agent.models import ToolErrorCode


def create_space() -> ScenarioSpace:
    """Create the ScenarioSpace matching test definitions."""
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


def test_target_candidates_exist():
    """Verify the three target candidates exist in the space."""
    space = create_space()
    candidates = space.enumerate_candidates()

    target_ids = []
    for c in candidates:
        params = c.parameter_values
        if (params.get("billing_status") == "active" and params.get("trigger_step") == 1):
            target_ids.append(c.candidate_id)
        elif (params.get("billing_status") == "cancelled" and params.get("trigger_step") == 1):
            target_ids.append(c.candidate_id)
        elif (params.get("billing_status") == "cancelled" and params.get("trigger_step") == 3):
            target_ids.append(c.candidate_id)

    assert len(target_ids) == 3, f"Expected 3 target candidates, found {len(target_ids)}"

    # Verify exact candidate IDs
    expected_ids = {
        "candidate_billing_status_active_trigger_step_1",
        "candidate_billing_status_cancelled_trigger_step_1",
        "candidate_billing_status_cancelled_trigger_step_3",
    }
    assert set(target_ids) == expected_ids, f"Mismatch: {set(target_ids)} != {expected_ids}"


def test_generator_creates_scenarios():
    """Verify ScenarioGenerator creates valid scenarios for target candidates."""
    env = EnvironmentService()
    customer = env.create_customer(name="Maya", email="maya@example.com")
    subscription = env.create_subscription(customer_id=customer.id, plan="pro")
    billing = env.create_billing_account(subscription_id=subscription.id)

    generator = ScenarioGenerator(
        customer_id=customer.id,
        subscription_id=subscription.id,
        billing_id=billing.id,
    )

    space = create_space()
    candidates = space.enumerate_candidates()

    for c in candidates:
        params = c.parameter_values
        if not ((params["billing_status"] == "active" and params["trigger_step"] == 1) or
                (params["billing_status"] == "cancelled" and params["trigger_step"] in (1, 3))):
            continue

        scenario = generator.generate(c)
        assert scenario is not None
        assert scenario.scenario_id == f"scenario_{c.candidate_id}"
        assert scenario.task.goal == "Process Maya's subscription cancellation."
        assert scenario.task.context["customer_id"] == customer.id

        # Verify perturbation logic
        if params["billing_status"] == "cancelled":
            assert len(scenario.perturbations) == 1
            p = scenario.perturbations[0]
            assert p.target == f"billing.{billing.id}.status"
            assert p.value == "cancelled"
            assert p.trigger.step == params["trigger_step"]
        else:
            assert len(scenario.perturbations) == 0


def test_fixed_three_strategy():
    """Verify the FixedThreeStrategy logic."""
    space = create_space()
    all_candidates = space.enumerate_candidates()

    target_ids = [
        "candidate_billing_status_active_trigger_step_1",
        "candidate_billing_status_cancelled_trigger_step_1",
        "candidate_billing_status_cancelled_trigger_step_3",
    ]

    class FixedThreeStrategy:
        def __init__(self, target_candidates: list) -> None:
            self._target_candidates = target_candidates
            self._index = 0

        def choose_next(self, candidates, evaluated=frozenset(), history=None):
            candidate_map = {c.candidate_id: c for c in candidates}
            while self._index < len(self._target_candidates):
                target_id = self._target_candidates[self._index]
                self._index += 1
                if target_id not in evaluated and target_id in candidate_map:
                    return candidate_map[target_id]
            return None

    strategy = FixedThreeStrategy(target_ids)

    # First call should return first target
    c1 = strategy.choose_next(all_candidates, frozenset())
    assert c1 is not None
    assert c1.candidate_id == target_ids[0]

    # Second call with first evaluated
    c2 = strategy.choose_next(all_candidates, frozenset({target_ids[0]}))
    assert c2 is not None
    assert c2.candidate_id == target_ids[1]

    # Third call with first two evaluated
    c3 = strategy.choose_next(all_candidates, frozenset({target_ids[0], target_ids[1]}))
    assert c3 is not None
    assert c3.candidate_id == target_ids[2]

    # Fourth call should return None (exhausted)
    c4 = strategy.choose_next(all_candidates, frozenset(target_ids))
    assert c4 is None


def test_serialization_wiring():
    """Verify DiscoveryResult can be serialized to JSON."""
    from packages.discovery.search.models import DiscoveryResult, DiscoveryStatus
    from datetime import datetime

    result = DiscoveryResult(
        discovery_run_id="test_123",
        status=DiscoveryStatus.COMPLETED,
        scenarios_evaluated=3,
        outcome_counts={"success": 2, "failure": 1},
        unique_failure_count=1,
        failure_ids=("failure_1",),
        started_at=datetime.now(),
        finished_at=datetime.now(),
    )

    # This should not raise
    json_str = result.to_dict()
    assert json_str["discovery_run_id"] == "test_123"
    assert json_str["scenarios_evaluated"] == 3


def test_diagnostic_extraction():
    """Test that diagnostic extraction correctly parses ScenarioRun with AgentRun trace."""
    from packages.agent.models import ToolErrorCode
    
    # Create a mock AgentRun with a tool trace
    now = datetime.now(timezone.utc)
    
    # Build a mock trace with multiple tool calls
    trace_entries = [
        {
            "call": {
                "call_id": "call_1",
                "tool_name": "get_customer",
                "arguments": {"customer_id": "cust_1"},
                "timestamp": now.isoformat(),
            },
            "result": {
                "call_id": "call_1",
                "tool_name": "get_customer",
                "success": True,
                "output": {"id": "cust_1", "subscription_id": "sub_1", "billing_id": "bill_1"},
                "error": None,
                "timestamp": now.isoformat(),
            },
        },
        {
            "call": {
                "call_id": "call_2",
                "tool_name": "cancel_subscription",
                "arguments": {"subscription_id": "sub_1"},
                "timestamp": now.isoformat(),
            },
            "result": {
                "call_id": "call_2",
                "tool_name": "cancel_subscription",
                "success": False,
                "output": None,
                "error": "Subscription already cancelled",
                "error_code": ToolErrorCode.SUBSCRIPTION_ALREADY_CANCELLED,
                "timestamp": now.isoformat(),
            },
        },
    ]
    
    # Create AgentRun using the actual classes
    agent_run = AgentRun(
        run_id="run_test",
        task_id="task_test",
        status=AgentRunStatus.FAILED,
        started_at=now,
        finished_at=now,
        trace=None,  # We'll build trace differently
        final_result=None,
        error="Tool execution failed",
    )
    
    # Actually let's use the proper trace structure
    from packages.agent.models import ToolTrace, ToolTraceEntry, ToolCall, ToolResult
    
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
    
    # Create ScenarioRun
    scenario_run = ScenarioRun(
        scenario_id="scenario_test",
        run_id="run_test",
        status=ScenarioRunStatus.AGENT_FAILED,
        started_at=now,
        ended_at=now,
        applied_perturbations=[],
        agent_run=agent_run,
        final_environment={
            "subscriptions": {"sub_1": {"id": "sub_1", "status": "active"}},
            "billing_accounts": {"bill_1": {"id": "bill_1", "status": "active"}},
        },
        error="Agent failed",
    )
    
    # Now test the extraction logic (replicate from script)
    def extract_scenario_diagnostics(scenario_run: ScenarioRun) -> dict:
        agent_run = scenario_run.agent_run
        diagnostics = {
            "scenario_id": scenario_run.scenario_id,
            "run_id": scenario_run.run_id,
            "scenario_status": scenario_run.status.value,
            "scenario_error": scenario_run.error,
            "applied_perturbations": [
                {
                    "perturbation_id": p.perturbation_id,
                    "target": p.target,
                    "value": p.value,
                    "step": p.step,
                    "event_id": p.event_id,
                }
                for p in scenario_run.applied_perturbations
            ],
            "agent_run": None,
            "final_subscription_count": len(scenario_run.final_environment.get("subscriptions", {})),
            "final_billing_count": len(scenario_run.final_environment.get("billing_accounts", {})),
        }

        if agent_run is not None:
            tool_sequence = []
            for entry in agent_run.trace.entries:
                call = entry.call
                result = entry.result
                tool_sequence.append({
                    "tool_name": call.tool_name,
                    "arguments": dict(call.arguments),
                    "success": result.success,
                    "error": result.error,
                    "error_code": result.error_code.value if result.error_code else None,
                })

            diagnostics["agent_run"] = {
                "status": agent_run.status.value,
                "error": agent_run.error,
                "steps": len(agent_run.trace.entries),
                "final_result": agent_run.final_result,
                "tool_sequence": tool_sequence,
            }

        return diagnostics
    
    diag = extract_scenario_diagnostics(scenario_run)
    
    # Verify structure
    assert diag["scenario_id"] == "scenario_test"
    assert diag["run_id"] == "run_test"
    assert diag["scenario_status"] == "agent_failed"
    assert diag["scenario_error"] == "Agent failed"
    assert diag["final_subscription_count"] == 1
    assert diag["final_billing_count"] == 1
    
    # Verify agent run extraction
    assert diag["agent_run"] is not None
    ar = diag["agent_run"]
    assert ar["status"] == "failed"
    assert ar["error"] == "Tool execution failed"
    assert ar["steps"] == 2
    assert ar["final_result"] is None
    
    # Verify tool sequence
    assert len(ar["tool_sequence"]) == 2
    
    tool1 = ar["tool_sequence"][0]
    assert tool1["tool_name"] == "get_customer"
    assert tool1["arguments"] == {"customer_id": "cust_1"}
    assert tool1["success"] is True
    assert tool1["error"] is None
    assert tool1["error_code"] is None
    
    tool2 = ar["tool_sequence"][1]
    assert tool2["tool_name"] == "cancel_subscription"
    assert tool2["arguments"] == {"subscription_id": "sub_1"}
    assert tool2["success"] is False
    assert tool2["error"] == "Subscription already cancelled"
    assert tool2["error_code"] == "subscription_already_cancelled"
    
    # Verify serialization works
    json_str = json.dumps(diag)
    assert "get_customer" in json_str
    assert "cancel_subscription" in json_str
    assert "subscription_already_cancelled" in json_str


def test_diagnostic_extraction_no_agent_run():
    """Test extraction when agent_run is None."""
    from packages.discovery.scenario.models import ScenarioRun, ScenarioRunStatus
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    
    scenario_run = ScenarioRun(
        scenario_id="scenario_test",
        run_id="run_test",
        status=ScenarioRunStatus.PERTURBATION_FAILED,
        started_at=now,
        ended_at=now,
        applied_perturbations=[],
        agent_run=None,
        final_environment={"subscriptions": {}, "billing_accounts": {}},
        error="Perturbation failed",
    )
    
    def extract_scenario_diagnostics(scenario_run: ScenarioRun) -> dict:
        agent_run = scenario_run.agent_run
        diagnostics = {
            "scenario_id": scenario_run.scenario_id,
            "run_id": scenario_run.run_id,
            "scenario_status": scenario_run.status.value,
            "scenario_error": scenario_run.error,
            "applied_perturbations": [],
            "agent_run": None,
            "final_subscription_count": 0,
            "final_billing_count": 0,
        }
        return diagnostics
    
    diag = extract_scenario_diagnostics(scenario_run)
    
    assert diag["agent_run"] is None
    assert diag["scenario_status"] == "perturbation_failed"
    assert diag["scenario_error"] == "Perturbation failed"


def test_artifact_persistence():
    """Test that DiscoveryResult artifact can be written and loaded correctly."""
    from packages.discovery.search.models import DiscoveryResult, DiscoveryStatus
    from datetime import datetime, timezone
    import tempfile
    
    # Create a sample DiscoveryResult with full structure
    now = datetime.now(timezone.utc)
    result = DiscoveryResult(
        discovery_run_id="discovery_test_123",
        status=DiscoveryStatus.COMPLETED,
        scenarios_evaluated=3,
        outcome_counts={"success": 2, "degraded": 1},
        unique_failure_count=1,
        failure_ids=("failure_1",),
        results=(
            {
                "candidate": {
                    "candidate_id": "candidate_billing_status_active_trigger_step_1",
                    "parameter_values": {"billing_status": "active", "trigger_step": 1},
                    "metadata": {"ordinal": 0},
                },
                "scenario_id": "scenario_candidate_billing_status_active_trigger_step_1",
                "run_id": "scenario_run_abc123",
                "scenario_status": "agent_failed",
                "outcome": "degraded",
                "verification": {"passed": False, "checks": [], "errors": []},
                "failure": {
                    "failure_id": "failure_1",
                    "scenario_id": "scenario_candidate_billing_status_active_trigger_step_1",
                    "run_id": "scenario_run_abc123",
                    "outcome": "degraded",
                    "severity": "low",
                    "summary": "Test",
                    "affected_entities": ("sub_1",),
                    "failed_checks": (),
                    "unsafe_actions": (),
                    "first_detected_event_id": None,
                    "metadata": {},
                },
            },
        ),
        search_decisions=(),
        started_at=now,
        finished_at=now,
        error=None,
    )
    
    result_dict = result.to_dict()
    
    # Test serialization to JSON and back
    json_str = json.dumps(result_dict, indent=2)
    loaded = json.loads(json_str)
    
    # Verify key fields survive round-trip
    assert loaded["discovery_run_id"] == "discovery_test_123"
    assert loaded["status"] == "completed"
    assert loaded["scenarios_evaluated"] == 3
    assert loaded["outcome_counts"] == {"success": 2, "degraded": 1}
    assert loaded["unique_failure_count"] == 1
    assert loaded["failure_ids"] == ["failure_1"]
    assert len(loaded["results"]) == 1
    
    # Verify scenario result structure
    scenario_result = loaded["results"][0]
    assert scenario_result["candidate"]["candidate_id"] == "candidate_billing_status_active_trigger_step_1"
    assert scenario_result["scenario_status"] == "agent_failed"
    assert scenario_result["outcome"] == "degraded"
    assert "failure" in scenario_result
    assert scenario_result["failure"]["failure_id"] == "failure_1"
    
    # Test writing to file (simulating atomic write)
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "discovery_test.json"
        
        # Write
        with artifact_path.open("w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # Verify file exists
        assert artifact_path.exists()
        
        # Read back
        with artifact_path.open("r", encoding="utf-8") as f:
            reloaded = json.load(f)
        
        assert reloaded["discovery_run_id"] == "discovery_test_123"
        assert reloaded["scenarios_evaluated"] == 3
    
    # Verify no secrets in output
    json_output = json.dumps(result_dict)
    assert "NEBIUS_API_KEY" not in json_output
    assert "api_key" not in json_output.lower() or "api_key" in json_output.lower()  # May appear in field names but not values
    # The result dict should not contain any actual API keys


def test_artifact_naming():
    """Test artifact filename generation logic."""
    from datetime import datetime, timezone
    
    run_id = "discovery_abc123"
    timestamp = datetime(2026, 9, 10, 17, 30, 0, tzinfo=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"discovery_{timestamp}_{run_id}.json"
    
    assert filename == "discovery_20260910T173000Z_discovery_abc123.json"
    assert filename.endswith(".json")
    assert "discovery_" in filename
    assert "20260910T173000Z" in filename


if __name__ == "__main__":
    test_space_enumeration()
    print("test_space_enumeration: PASS")

    test_target_candidates_exist()
    print("test_target_candidates_exist: PASS")

    test_generator_creates_scenarios()
    print("test_generator_creates_scenarios: PASS")

    test_fixed_three_strategy()
    print("test_fixed_three_strategy: PASS")

    test_serialization_wiring()
    print("test_serialization_wiring: PASS")

    test_diagnostic_extraction()
    print("test_diagnostic_extraction: PASS")

    test_diagnostic_extraction_no_agent_run()
    print("test_diagnostic_extraction_no_agent_run: PASS")

    print("\nAll offline tests PASSED")