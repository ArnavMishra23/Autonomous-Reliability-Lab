"""Direct unit tests for hypothesis_redundant_action semantics."""

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from packages.analysis import CausalAnalyzer, FailureHypothesis
from packages.discovery.failure import FailureDetector, DeterministicVerifier
from packages.discovery.failure.models import FailureRecord, ScenarioOutcome, VerificationResult
from packages.discovery.scenario.models import (
    ScenarioRun,
    ScenarioRunStatus,
    AppliedPerturbation,
    Trigger,
    TriggerType,
    Perturbation,
    PerturbationType,
)
from packages.agent import AgentTask, AgentRun, AgentRunStatus
from packages.agent.models import ToolTrace, ToolTraceEntry, ToolCall, ToolResult, generate_id, utc_now
from packages.environment.models import (
    BillingAccount,
    BillingStatus,
    Customer,
    Subscription,
    SubscriptionStatus,
    EnvironmentEvent,
    EventType,
    DefaultIdProvider,
)
from packages.environment.service import EnvironmentService


def _base_time() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_scenario_run(
    trace_entries: list[ToolTraceEntry],
    final_env_events: list[EnvironmentEvent] | None = None,
    applied_perturbations: list[AppliedPerturbation] | None = None,
    status: ScenarioRunStatus = ScenarioRunStatus.AGENT_FAILED,
) -> ScenarioRun:
    """Construct a synthetic ScenarioRun with the given trace and environment."""
    run = AgentRun(
        run_id=generate_id("run_"),
        task_id="task_test",
        status=AgentRunStatus.FAILED,
        started_at=_base_time(),
        finished_at=_base_time() + timedelta(minutes=1),
        trace=ToolTrace(entries=trace_entries),
        error="Test failure",
    )

    if final_env_events is None:
        final_env_events = []

    return ScenarioRun(
        scenario_id="scenario_test",
        run_id=run.run_id,
        status=status,
        started_at=run.started_at,
        ended_at=run.finished_at or _base_time() + timedelta(minutes=1),
        applied_perturbations=tuple(applied_perturbations or []),
        agent_run=run,
        final_environment={"events": [asdict(e) for e in final_env_events]},
        error="Test failure",
    )


def _make_env_event(
    event_id: str,
    target: str,
    field: str,
    old_value: Any,
    new_value: Any,
    sequence: int,
) -> EnvironmentEvent:
    """Create an environment event with a controlled timestamp."""
    base = _base_time() + timedelta(seconds=sequence)
    return EnvironmentEvent(
        id=event_id,
        event_type=EventType.STATE_CHANGE,
        timestamp=base,
        target=target,
        field=field,
        old_value=old_value,
        new_value=new_value,
        metadata={"source": "test"},
    )


def _make_tool_call(
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    sequence: int,
    step: int,
) -> ToolCall:
    """Create a tool call with a controlled timestamp."""
    base = _base_time() + timedelta(seconds=sequence)
    return ToolCall(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
        timestamp=base,
        step=step,
    )


def _make_tool_result(
    call_id: str,
    tool_name: str,
    success: bool,
    output: Any = None,
    error: str | None = None,
    error_code: str | None = None,
    sequence: int = 0,
) -> ToolResult:
    """Create a tool result with a controlled timestamp."""
    base = _base_time() + timedelta(seconds=sequence)
    return ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        success=success,
        output=output,
        error=error,
        error_code=error_code,
        timestamp=base,
    )


def _make_trace_entry(call: ToolCall, result: ToolResult) -> ToolTraceEntry:
    return ToolTraceEntry(call=call, result=result)


class TestRedundantActionHappyPath:
    """Test 1: Happy path for hypothesis_redundant_action."""

    def test_redundant_action_detected_with_confidence_1_0(self) -> None:
        sub_id = "sub_test_1"

        # Sequence:
        # 0: get_subscription (observation - active)
        # 1: cancel_subscription (first action - succeeds)
        # 2: get_subscription (confirming observation - cancelled)
        # 3: cancel_subscription (repeated action - fails with already_cancelled)
        # Env event: subscription status active -> cancelled (authoritative)

        call_0 = _make_tool_call("call_0", "get_subscription", {"subscription_id": sub_id}, 0, 1)
        result_0 = _make_tool_result("call_0", "get_subscription", True, {"id": sub_id, "status": "active"}, sequence=1)

        call_1 = _make_tool_call("call_1", "cancel_subscription", {"subscription_id": sub_id}, 2, 2)
        result_1 = _make_tool_result("call_1", "cancel_subscription", True, {"subscription": {"id": sub_id, "status": "cancelled"}}, sequence=3)

        call_2 = _make_tool_call("call_2", "get_subscription", {"subscription_id": sub_id}, 4, 3)
        result_2 = _make_tool_result("call_2", "get_subscription", True, {"id": sub_id, "status": "cancelled"}, sequence=5)

        call_3 = _make_tool_call("call_3", "cancel_subscription", {"subscription_id": sub_id}, 6, 4)
        result_3 = _make_tool_result(
            "call_3", "cancel_subscription", False, None,
            error="Subscription already cancelled: sub_test_1",
            error_code="already_cancelled",
            sequence=7,
        )

        env_event = _make_env_event("evt_1", sub_id, "status", "active", "cancelled", 10)

        trace = [
            _make_trace_entry(call_0, result_0),
            _make_trace_entry(call_1, result_1),
            _make_trace_entry(call_2, result_2),
            _make_trace_entry(call_3, result_3),
        ]

        run = _make_scenario_run(trace, final_env_events=[env_event])

        verifier = DeterministicVerifier()
        verification = verifier.verify(run)
        detector = FailureDetector()
        failure = detector.detect(run, verification)

        analysis = CausalAnalyzer().analyze(run, verification, failure)

        redundant_hypotheses = [h for h in analysis.hypotheses if h.hypothesis_id == "hypothesis_redundant_action"]
        assert len(redundant_hypotheses) == 1
        hyp = redundant_hypotheses[0]
        assert hyp.confidence == 1.0
        assert "repeated a successful state-changing action" in hyp.statement
        assert len(hyp.supporting_evidence) == 5

        # Verify evidence references the correct 5 nodes
        evidence_node_ids = [e.source_id for e in hyp.supporting_evidence]
        assert any("action_call_1" in eid for eid in evidence_node_ids), "First successful action"
        assert any("reality_" in eid for eid in evidence_node_ids), "State transition"
        assert any("observation_call_2" in eid for eid in evidence_node_ids), "Confirming observation"
        assert any("action_call_3" in eid for eid in evidence_node_ids), "Repeated action"
        assert any("tool_result_call_3" in eid for eid in evidence_node_ids), "Repeated action failure"

    def test_hypothesis_is_json_serializable(self) -> None:
        """Test 7: Verify hypothesis_redundant_action remains JSON serializable."""
        sub_id = "sub_test_2"

        call_0 = _make_tool_call("call_0", "get_subscription", {"subscription_id": sub_id}, 0, 1)
        result_0 = _make_tool_result("call_0", "get_subscription", True, {"id": sub_id, "status": "active"}, sequence=1)

        call_1 = _make_tool_call("call_1", "cancel_subscription", {"subscription_id": sub_id}, 2, 2)
        result_1 = _make_tool_result("call_1", "cancel_subscription", True, {"subscription": {"id": sub_id, "status": "cancelled"}}, sequence=3)

        call_2 = _make_tool_call("call_2", "get_subscription", {"subscription_id": sub_id}, 4, 3)
        result_2 = _make_tool_result("call_2", "get_subscription", True, {"id": sub_id, "status": "cancelled"}, sequence=5)

        call_3 = _make_tool_call("call_3", "cancel_subscription", {"subscription_id": sub_id}, 6, 4)
        result_3 = _make_tool_result(
            "call_3", "cancel_subscription", False, None,
            error="Subscription already cancelled: sub_test_2",
            error_code="already_cancelled",
            sequence=7,
        )

        env_event = _make_env_event("evt_2", sub_id, "status", "active", "cancelled", 10)

        trace = [
            _make_trace_entry(call_0, result_0),
            _make_trace_entry(call_1, result_1),
            _make_trace_entry(call_2, result_2),
            _make_trace_entry(call_3, result_3),
        ]

        run = _make_scenario_run(trace, final_env_events=[env_event])

        verifier = DeterministicVerifier()
        verification = verifier.verify(run)
        detector = FailureDetector()
        failure = detector.detect(run, verification)

        analysis = CausalAnalyzer().analyze(run, verification, failure)

        redundant_hypotheses = [h for h in analysis.hypotheses if h.hypothesis_id == "hypothesis_redundant_action"]
        assert len(redundant_hypotheses) == 1

        # Full analysis serialization
        json.dumps(analysis.to_dict())

        # Individual hypothesis serialization
        json.dumps(redundant_hypotheses[0].to_dict())


class TestFailureAMustNotMatch:
    """Test 2: Failure A (stale_observation) must NOT produce hypothesis_redundant_action."""

    def test_stale_observation_pattern_does_not_produce_redundant_action(self) -> None:
        """
        Pattern:
        - active billing observation
        - billing active -> cancelled reality change
        - cancel_subscription failure
        
        This is the classic stale_observation case, not redundant_action.
        """
        sub_id = "sub_test_3"
        billing_id = "bill_test_3"

        # Agent observes billing as active
        call_0 = _make_tool_call("call_0", "get_billing_status", {"billing_id": billing_id}, 0, 1)
        result_0 = _make_tool_result("call_0", "get_billing_status", True, {"id": billing_id, "subscription_id": sub_id, "status": "active"}, sequence=1)

        # Then environment changes billing to cancelled (perturbation)
        env_event = _make_env_event("evt_3", billing_id, "status", "active", "cancelled", 5)

        # Agent attempts cancel_subscription (which fails because billing is cancelled)
        call_1 = _make_tool_call("call_1", "cancel_subscription", {"subscription_id": sub_id}, 10, 2)
        result_1 = _make_tool_result(
            "call_1", "cancel_subscription", False, None,
            error="Billing account is already cancelled: bill_test_3",
            error_code="billing_account_already_cancelled",
            sequence=11,
        )

        trace = [
            _make_trace_entry(call_0, result_0),
            _make_trace_entry(call_1, result_1),
        ]

        run = _make_scenario_run(trace, final_env_events=[env_event])

        verifier = DeterministicVerifier()
        verification = verifier.verify(run)
        detector = FailureDetector()
        failure = detector.detect(run, verification)

        analysis = CausalAnalyzer().analyze(run, verification, failure)

        # Must NOT produce hypothesis_redundant_action
        redundant_hypotheses = [h for h in analysis.hypotheses if h.hypothesis_id == "hypothesis_redundant_action"]
        assert len(redundant_hypotheses) == 0, "hypothesis_redundant_action must not match stale observation pattern"

        # Must produce hypothesis_stale_observation
        stale_hypotheses = [h for h in analysis.hypotheses if h.hypothesis_id == "hypothesis_stale_observation"]
        assert len(stale_hypotheses) == 1, "hypothesis_stale_observation must be produced"
        assert "no longer current" in stale_hypotheses[0].statement


class TestTransientFailureMustNotMatch:
    """Test 3: Transient failure must NOT produce hypothesis_redundant_action."""

    def test_transient_failure_does_not_produce_redundant_action(self) -> None:
        """
        Pattern:
        - First cancel_subscription succeeds
        - Later same cancel_subscription fails with transient/unrelated error (timeout)
        """
        sub_id = "sub_test_4"
        billing_id = "bill_test_4"

        call_0 = _make_tool_call("call_0", "get_subscription", {"subscription_id": sub_id}, 0, 1)
        result_0 = _make_tool_result("call_0", "get_subscription", True, {"id": sub_id, "status": "active"}, sequence=1)

        call_1 = _make_tool_call("call_1", "cancel_subscription", {"subscription_id": sub_id}, 2, 2)
        result_1 = _make_tool_result("call_1", "cancel_subscription", True, {"subscription": {"id": sub_id, "status": "cancelled"}}, sequence=3)

        # State transition
        env_event = _make_env_event("evt_4", sub_id, "status", "active", "cancelled", 10)

        # Confirming observation
        call_2 = _make_tool_call("call_2", "get_billing_status", {"billing_id": billing_id}, 4, 3)
        result_2 = _make_tool_result("call_2", "get_billing_status", True, {"id": billing_id, "subscription_id": sub_id, "status": "cancelled"}, sequence=5)

        # Repeated action fails with TRANSIENT error (not "already cancelled")
        call_3 = _make_tool_call("call_3", "cancel_subscription", {"subscription_id": sub_id}, 6, 4)
        result_3 = _make_tool_result(
            "call_3", "cancel_subscription", False, None,
            error="Request timeout",
            error_code="tool_execution_failed",
            sequence=7,
        )

        trace = [
            _make_trace_entry(call_0, result_0),
            _make_trace_entry(call_1, result_1),
            _make_trace_entry(call_2, result_2),
            _make_trace_entry(call_3, result_3),
        ]

        run = _make_scenario_run(trace, final_env_events=[env_event])

        verifier = DeterministicVerifier()
        verification = verifier.verify(run)
        detector = FailureDetector()
        failure = detector.detect(run, verification)

        analysis = CausalAnalyzer().analyze(run, verification, failure)

        redundant_hypotheses = [h for h in analysis.hypotheses if h.hypothesis_id == "hypothesis_redundant_action"]
        assert len(redundant_hypotheses) == 0, "hypothesis_redundant_action must not match transient failure"


class TestRepeatedReadMustNotMatch:
    """Test 4: Repeated read-only tool call must NOT produce hypothesis_redundant_action."""

    def test_repeated_read_only_does_not_produce_redundant_action(self) -> None:
        """
        Pattern:
        - Multiple get_subscription calls (read-only)
        - No state-changing action repeated
        """
        sub_id = "sub_test_5"
        billing_id = "bill_test_5"

        call_0 = _make_tool_call("call_0", "get_subscription", {"subscription_id": sub_id}, 0, 1)
        result_0 = _make_tool_result("call_0", "get_subscription", True, {"id": sub_id, "status": "active"}, sequence=1)

        call_1 = _make_tool_call("call_1", "get_subscription", {"subscription_id": sub_id}, 2, 2)
        result_1 = _make_tool_result("call_1", "get_subscription", True, {"id": sub_id, "status": "active"}, sequence=3)

        call_2 = _make_tool_call("call_2", "get_subscription", {"subscription_id": sub_id}, 4, 3)
        result_2 = _make_tool_result("call_2", "get_subscription", True, {"id": sub_id, "status": "active"}, sequence=5)

        trace = [
            _make_trace_entry(call_0, result_0),
            _make_trace_entry(call_1, result_1),
            _make_trace_entry(call_2, result_2),
        ]

        run = _make_scenario_run(trace, final_env_events=[])

        verifier = DeterministicVerifier()
        verification = verifier.verify(run)
        detector = FailureDetector()
        failure = detector.detect(run, verification)

        analysis = CausalAnalyzer().analyze(run, verification, failure)

        redundant_hypotheses = [h for h in analysis.hypotheses if h.hypothesis_id == "hypothesis_redundant_action"]
        assert len(redundant_hypotheses) == 0, "hypothesis_redundant_action must not match repeated read-only calls"


class TestStateChangedBackMustNotMatch:
    """Test 5: State changed back before repeated action must NOT produce hypothesis_redundant_action."""

    def test_state_changed_back_does_not_produce_redundant_action(self) -> None:
        """
        Pattern:
        - Successful cancellation
        - State transitions to cancelled
        - State changes BACK to active (reinstated)
        - Repeated cancel_subscription fails (but not because already cancelled)
        """
        sub_id = "sub_test_6"
        billing_id = "bill_test_6"

        call_0 = _make_tool_call("call_0", "get_subscription", {"subscription_id": sub_id}, 0, 1)
        result_0 = _make_tool_result("call_0", "get_subscription", True, {"id": sub_id, "status": "active"}, sequence=1)

        call_1 = _make_tool_call("call_1", "cancel_subscription", {"subscription_id": sub_id}, 2, 2)
        result_1 = _make_tool_result("call_1", "cancel_subscription", True, {"subscription": {"id": sub_id, "status": "cancelled"}}, sequence=3)

        # First state transition: active -> cancelled
        env_event_1 = _make_env_event("evt_6a", sub_id, "status", "active", "cancelled", 10)

        # Confirming observation
        call_2 = _make_tool_call("call_2", "get_subscription", {"subscription_id": sub_id}, 4, 3)
        result_2 = _make_tool_result("call_2", "get_subscription", True, {"id": sub_id, "status": "cancelled"}, sequence=5)

        # State changes BACK: cancelled -> active (reinstated)
        env_event_2 = _make_env_event("evt_6b", sub_id, "status", "cancelled", "active", 20)

        # Repeated action - but state is no longer cancelled, so failure is different
        call_3 = _make_tool_call("call_3", "cancel_subscription", {"subscription_id": sub_id}, 6, 4)
        result_3 = _make_tool_result(
            "call_3", "cancel_subscription", False, None,
            error="Invalid cancellation state: billing is active",
            error_code="invalid_cancellation_state",
            sequence=7,
        )

        trace = [
            _make_trace_entry(call_0, result_0),
            _make_trace_entry(call_1, result_1),
            _make_trace_entry(call_2, result_2),
            _make_trace_entry(call_3, result_3),
        ]

        run = _make_scenario_run(trace, final_env_events=[env_event_1, env_event_2])

        verifier = DeterministicVerifier()
        verification = verifier.verify(run)
        detector = FailureDetector()
        failure = detector.detect(run, verification)

        analysis = CausalAnalyzer().analyze(run, verification, failure)

        redundant_hypotheses = [h for h in analysis.hypotheses if h.hypothesis_id == "hypothesis_redundant_action"]
        assert len(redundant_hypotheses) == 0, "hypothesis_redundant_action must not match when state changed back"


class TestEvidenceReferences:
    """Test 6: Verify evidence references are correct."""

    def test_evidence_references_match_five_pieces(self) -> None:
        """
        Verify the hypothesis references evidence corresponding to:
        - first successful action
        - state transition
        - confirming observation
        - repeated action
        - repeated-action failure
        """
        sub_id = "sub_test_7"

        call_0 = _make_tool_call("call_0", "get_subscription", {"subscription_id": sub_id}, 0, 1)
        result_0 = _make_tool_result("call_0", "get_subscription", True, {"id": sub_id, "status": "active"}, sequence=1)

        call_1 = _make_tool_call("call_1", "cancel_subscription", {"subscription_id": sub_id}, 2, 2)
        result_1 = _make_tool_result("call_1", "cancel_subscription", True, {"subscription": {"id": sub_id, "status": "cancelled"}}, sequence=3)

        call_2 = _make_tool_call("call_2", "get_subscription", {"subscription_id": sub_id}, 4, 3)
        result_2 = _make_tool_result("call_2", "get_subscription", True, {"id": sub_id, "status": "cancelled"}, sequence=5)

        call_3 = _make_tool_call("call_3", "cancel_subscription", {"subscription_id": sub_id}, 6, 4)
        result_3 = _make_tool_result(
            "call_3", "cancel_subscription", False, None,
            error="Subscription already cancelled: sub_test_7",
            error_code="already_cancelled",
            sequence=7,
        )

        env_event = _make_env_event("evt_7", sub_id, "status", "active", "cancelled", 10)

        trace = [
            _make_trace_entry(call_0, result_0),
            _make_trace_entry(call_1, result_1),
            _make_trace_entry(call_2, result_2),
            _make_trace_entry(call_3, result_3),
        ]

        run = _make_scenario_run(trace, final_env_events=[env_event])

        verifier = DeterministicVerifier()
        verification = verifier.verify(run)
        detector = FailureDetector()
        failure = detector.detect(run, verification)

        analysis = CausalAnalyzer().analyze(run, verification, failure)

        redundant_hypotheses = [h for h in analysis.hypotheses if h.hypothesis_id == "hypothesis_redundant_action"]
        assert len(redundant_hypotheses) == 1
        hyp = redundant_hypotheses[0]

        # Must have exactly 5 supporting evidence references
        assert len(hyp.supporting_evidence) == 5

        # Map evidence to node types by source_id pattern
        evidence_sources = {e.source_type: e.source_id for e in hyp.supporting_evidence}
        node_ids = [e.source_id for e in hyp.supporting_evidence]

        # 1. First successful action (action node)
        assert any("action_call_1" in nid for nid in node_ids), "Missing first successful action evidence"

        # 2. State transition (reality_change node)
        assert any("reality_" in nid for nid in node_ids), "Missing state transition evidence"

        # 3. Confirming observation (observation node)
        assert any("observation_call_2" in nid for nid in node_ids), "Missing confirming observation evidence"

        # 4. Repeated action (action node)
        assert any("action_call_3" in nid for nid in node_ids), "Missing repeated action evidence"

        # 5. Repeated action failure (tool_result node)
        assert any("tool_result_call_3" in nid for nid in node_ids), "Missing repeated action failure evidence"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])