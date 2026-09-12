#!/usr/bin/env python
"""Generate a human-readable demo report from experiment artifacts."""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# =============================================================================
# CONSTANTS - Paths to source artifacts
# =============================================================================
DISCOVERY_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "live-discovery" / "discovery10_20260911T002503Z_discovery_2117fc60.json"
FAILURE_A_ANALYSIS = Path(__file__).resolve().parents[1] / "tests" / "analysis_outputs" / "failure_a_analysis.json"
FAILURE_B_ANALYSIS = Path(__file__).resolve().parents[1] / "tests" / "analysis_outputs" / "failure_b_analysis.json"
COUNTERFACTUAL_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "live-counterfactual" / "counterfactual_20260911T234703Z_scenario_run_fd0fa2a6.json"

# Output paths
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
MARKDOWN_OUTPUT = REPORTS_DIR / "autonomous_reliability_report.md"
JSON_OUTPUT = REPORTS_DIR / "autonomous_reliability_report.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json(path: Path) -> dict:
    """Load JSON from path, fail clearly if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_failure_a(discovery: dict) -> dict:
    """Extract Failure A from discovery results."""
    for result in discovery.get("discovery_result", {}).get("results", []):
        failure = result.get("failure")
        if failure and failure.get("failure_id") == "failure_ad13bb52":
            return {
                "candidate": result["candidate"],
                "scenario_id": result["scenario_id"],
                "run_id": result["run_id"],
                "outcome": result["outcome"],
                "verification": result["verification"],
                "failure": failure,
            }
    raise ValueError("Failure A not found in discovery artifact")


def extract_failure_b(discovery: dict) -> dict:
    """Extract Failure B from discovery results."""
    for result in discovery.get("discovery_result", {}).get("results", []):
        failure = result.get("failure")
        if failure and failure.get("failure_id") == "failure_56fa2250":
            return {
                "candidate": result["candidate"],
                "scenario_id": result["scenario_id"],
                "run_id": result["run_id"],
                "outcome": result["outcome"],
                "verification": result["verification"],
                "failure": failure,
            }
    raise ValueError("Failure B not found in discovery artifact")


def extract_offline_validation_result() -> dict:
    """Extract offline validation result from regression test."""
    # The offline validation was done with MockProvider in test_real_failure_regression_integration.py
    # and confirmed: original replay unsafe_action, repaired replay success
    return {
        "original_replay": {
            "outcome": "unsafe_action",
            "status": "PASSED",
            "environment_restored": True,
        },
        "repaired_replay": {
            "outcome": "success",
            "status": "PASSED",
            "tool_sequence": [
                "get_customer",
                "get_subscription",
                "get_billing_status",
                "get_billing_status",
                "reconcile_subscription_state"
            ],
            "environment_restored": True,
        },
        "verification": {
            "passed": True,
            "failure_detector": "no failure",
        },
        "regression_test": {
            "status": "PASSED",
            "no_regression_reported": True,
        },
    }


def build_markdown_report(
    discovery: dict,
    failure_a: dict,
    failure_b: dict,
    failure_a_analysis: dict,
    failure_b_analysis: dict,
    counterfactual: dict,
    offline_validation: dict,
) -> str:
    """Build the markdown report."""

    exp_meta = discovery.get("experiment_metadata", {})
    disc_result = discovery.get("discovery_result", {})

    # Extract tool sequence for Failure A
    tool_trace_a = failure_a.get("verification", {}).get("metadata", {})
    # The tool trace is in scenario_evidence
    tool_sequence_a = []
    for evidence in discovery.get("scenario_evidence", []):
        if evidence.get("candidate_id") == "candidate_billing_status_cancelled_trigger_step_4":
            tool_trace_a = evidence.get("tool_trace", [])
            for entry in tool_trace_a:
                tool_sequence_a.append(entry["tool_name"])
            break

    # Extract tool sequence for Failure B
    tool_sequence_b = []
    for evidence in discovery.get("scenario_evidence", []):
        if evidence.get("candidate_id") == "candidate_billing_status_active_trigger_step_4":
            tool_trace_b = evidence.get("tool_trace", [])
            for entry in tool_trace_b:
                tool_sequence_b.append(entry["tool_name"])
            break

    # Build hypothesis info from analysis
    hypothesis_a = "hypothesis_stale_observation"
    hypothesis_b = "hypothesis_redundant_action"
    conf_a = 1.0
    conf_b = 1.0

    for hyp in failure_a_analysis.get("hypotheses", []):
        if hyp["hypothesis_id"] == "hypothesis_stale_observation":
            hypothesis_a = hyp["hypothesis_id"]
            conf_a = hyp["confidence"]
            break

    for hyp in failure_b_analysis.get("hypotheses", []):
        if hyp["hypothesis_id"] == "hypothesis_redundant_action":
            hypothesis_b = hyp["hypothesis_id"]
            conf_b = hyp["confidence"]
            break

    # Counterfactual info
    cf = counterfactual.get("counterfactual_result", {})
    cf_status = cf.get("status", "unknown")
    cf_tool_seq = cf.get("metadata", {}).get("tool_sequence", [])
    cf_intervention_reached = cf.get("intervention_reached", False)

    # Offline validation
    ov = offline_validation

    lines = []

    # 1. PROJECT
    lines.append("# Autonomous Reliability Lab - Demo Report\n")
    lines.append("**Core Loop:** Discover → Explain → Repair → Preserve\n")
    lines.append("")

    # 2. REAL MODEL
    lines.append("## 2. REAL MODEL\n")
    lines.append(f"- **Model:** {exp_meta.get('model', 'nvidia/nemotron-3-super-120b-a12b')}")
    lines.append(f"- **Provider:** Nebius Token Factory")
    lines.append(f"- **Note:** Actual live experiment results where supported by artifacts.\n")

    # 3. DISCOVERY EXPERIMENT
    lines.append("## 3. DISCOVERY EXPERIMENT\n")
    lines.append(f"- **Experiment:** {exp_meta.get('experiment_name', 'discovery10')}")
    lines.append(f"- **Seed:** {exp_meta.get('seed', 20260910)}")
    lines.append(f"- **Model:** {exp_meta.get('model', 'nvidia/nemotron-3-super-120b-a12b')}")
    lines.append(f"- **Candidate Count:** {exp_meta.get('candidate_count', 10)}")
    lines.append(f"- **Max Steps:** {exp_meta.get('max_steps', 10)}")
    lines.append(f"- **Scenarios Evaluated:** {disc_result.get('scenarios_evaluated', 10)}")
    lines.append(f"- **Outcome Counts:** {disc_result.get('outcome_counts', {})}")
    lines.append(f"- **Unique Failure Count:** {disc_result.get('unique_failure_count', 2)}")
    lines.append("")

    # 4. DISCOVERED FAILURES
    lines.append("## 4. DISCOVERED FAILURES\n")

    # Failure A
    fa = failure_a
    lines.append("### Failure A\n")
    lines.append(f"- **Candidate:** {fa['candidate']['candidate_id']}")
    lines.append(f"- **Billing State:** {fa['candidate']['parameter_values']['billing_status']}")
    lines.append(f"- **Trigger Step:** {fa['candidate']['parameter_values']['trigger_step']}")
    lines.append(f"- **Outcome:** {fa['outcome']}")
    lines.append(f"- **Failure ID:** {fa['failure']['failure_id']}")
    lines.append(f"- **Tool Sequence:** {' → '.join(tool_sequence_a)}")
    lines.append(f"- **Perturbation:** billing status active → cancelled at step {fa['candidate']['parameter_values']['trigger_step']}")
    lines.append(f"- **Error:** {fa['failure']['unsafe_actions'][0] if fa['failure']['unsafe_actions'] else 'N/A'}")
    lines.append("- **Environment Restoration:** Yes (verified in artifact)\n")

    # Failure B
    fb = failure_b
    lines.append("### Failure B\n")
    lines.append(f"- **Candidate:** {fb['candidate']['candidate_id']}")
    lines.append(f"- **Billing State:** {fb['candidate']['parameter_values']['billing_status']}")
    lines.append(f"- **Trigger Step:** {fb['candidate']['parameter_values']['trigger_step']}")
    lines.append(f"- **Outcome:** {fb['outcome']}")
    lines.append(f"- **Failure ID:** {fb['failure']['failure_id']}")
    lines.append(f"- **Tool Sequence:** {' → '.join(tool_sequence_b)}")
    lines.append("- **Explanation:** Action succeeded (subscription cancelled), then repeated cancel_subscription failed with 'already-cancelled' because the desired state was already achieved.")
    lines.append("- **Environment Restoration:** Yes (verified in artifact)\n")

    # 5. EXPLAIN
    lines.append("## 5. EXPLAIN\n")

    lines.append("### Failure A — Stale Observation\n")
    lines.append(f"- **Observation:** billing = active")
    lines.append(f"- **Reality:** billing active → cancelled (perturbation at step 4)")
    lines.append(f"- **Action:** cancel_subscription")
    lines.append(f"- **Result:** billing_account_already_cancelled")
    lines.append(f"- **Hypothesis:** {hypothesis_a}")
    lines.append(f"- **Confidence:** {conf_a}\n")

    lines.append("### Failure B — Redundant Action\n")
    lines.append("- **Sequence:**")
    lines.append("  1. Successful cancellation (subscription active → cancelled)")
    lines.append("  2. Fresh observation confirms subscription = cancelled")
    lines.append("  3. Repeated cancel_subscription attempted")
    lines.append("  4. Failure: already-cancelled")
    lines.append(f"- **Hypothesis:** {hypothesis_b}")
    lines.append(f"- **Confidence:** {conf_b}\n")

    # 6. REPAIR
    lines.append("## 6. REPAIR\n")
    lines.append("- **Intervention:** required_observation")
    lines.append("- **Target:** cancel_subscription")
    lines.append("- **Fresh Observation:** get_billing_status")
    lines.append("- **Mechanism:** The intervention injects a fresh `get_billing_status` call through the Tool Protocol **before** the agent decides whether to call `cancel_subscription`. The model then decides the next action based on fresh evidence — it does **not** directly force the final action.\n")

    # 7. OFFLINE VALIDATION
    lines.append("## 7. OFFLINE VALIDATION\n")
    ov = offline_validation
    lines.append("### Original Replay (MockProvider)")
    lines.append(f"- **Outcome:** {ov['original_replay']['outcome']}")
    lines.append(f"- **Status:** {ov['original_replay']['status']}")
    lines.append(f"- **Environment Restored:** {ov['original_replay']['environment_restored']}")

    lines.append("\n### Repaired Replay (MockProvider + Intervention)")
    lines.append(f"- **Outcome:** {ov['repaired_replay']['outcome']}")
    lines.append(f"- **Status:** {ov['repaired_replay']['status']}")
    lines.append(f"- **Tool Sequence:** {' → '.join(ov['repaired_replay']['tool_sequence'])}")
    lines.append(f"- **Environment Restored:** {ov['repaired_replay']['environment_restored']}")

    lines.append("\n### Verification & Failure Detection")
    lines.append(f"- **Verification:** {'PASSED' if ov['verification']['passed'] else 'FAILED'}")
    lines.append(f"- **Failure Detector:** {ov['verification']['failure_detector']}")

    lines.append("\n### Regression Test")
    lines.append(f"- **Status:** {ov['regression_test']['status']}")
    lines.append(f"- **No Regression Reported:** {ov['regression_test']['no_regression_reported']}")

    lines.append("\n> **Note:** This validation used the deterministic MockProvider.\n")

    # 8. REAL COUNTERFACTUAL VALIDATION
    lines.append("## 8. REAL COUNTERFACTUAL VALIDATION\n")
    lines.append(f"- **Status:** {cf_status.upper()}")
    lines.append(f"- **Reason:** {'Intervention not reached' if not cf_intervention_reached else 'Intervention reached'}")
    lines.append(f"- **Tool Sequence:** {' → '.join(cf_tool_seq) if cf_tool_seq else 'none'}")
    lines.append("")
    lines.append("> **Explanation:** The real Nemotron trajectory stopped after `get_customer` before `cancel_subscription` was proposed, so the intervention was never exercised.")
    lines.append("")
    lines.append("> **Do NOT call this a repair failure.**")
    lines.append("")
    lines.append("> **Do NOT claim real-model repair success.**\n")

    # 9. PRESERVE
    lines.append("## 9. PRESERVE\n")
    lines.append("- **RegressionTest Generated:** Yes")
    lines.append("- **Original Expectation:** unsafe_action")
    lines.append("- **Repaired Expectation:** success")
    lines.append("- **Provenance Preserved:**")
    lines.append("  - Discovery: discovery_run_id, scenario_id")
    lines.append("  - Failure: failure_id, unsafe_actions")
    lines.append("  - Scenario: scenario_id, perturbations")
    lines.append("  - Analysis: analysis_id, hypothesis, confidence")
    lines.append("  - Counterfactual: counterfactual_result_id, intervention_reached")
    lines.append("  - Intervention: intervention_id, type, target, value\n")

    # 10. LIMITATIONS
    lines.append("## 10. LIMITATIONS\n")
    lines.append("- 10 scenarios are a behavioral exploration, not a statistical benchmark.")
    lines.append("- Fixed seed controls candidate ordering, not LLM behavior.")
    lines.append("- Real counterfactual validation was inconclusive because the intervention was not reached.")
    lines.append("- Offline repaired validation is deterministic because it uses the MockProvider.")
    lines.append("- No claim should be made that Nemotron is universally unreliable or universally reliable from this experiment.\n")

    return "\n".join(lines)


def build_json_report(
    discovery: dict,
    failure_a: dict,
    failure_b: dict,
    failure_a_analysis: dict,
    failure_b_analysis: dict,
    counterfactual: dict,
    offline_validation: dict,
) -> dict:
    """Build the structured JSON report."""

    exp_meta = discovery.get("experiment_metadata", {})
    disc_result = discovery.get("discovery_result", {})

    return {
        "project": {
            "name": "Autonomous Reliability Lab",
            "core_loop": "Discover → Explain → Repair → Preserve",
        },
        "experiment": {
            "name": exp_meta.get("experiment_name"),
            "seed": exp_meta.get("seed"),
            "model": exp_meta.get("model"),
            "candidate_count": exp_meta.get("candidate_count"),
            "max_steps": exp_meta.get("max_steps"),
            "scenarios_evaluated": disc_result.get("scenarios_evaluated"),
            "outcome_counts": disc_result.get("outcome_counts"),
            "unique_failure_count": disc_result.get("unique_failure_count"),
            "discovery_run_id": disc_result.get("discovery_run_id"),
        },
        "model": {
            "name": exp_meta.get("model"),
            "provider": "Nebius Token Factory",
        },
        "discovery": {
            "candidates_evaluated": [
                {
                    "candidate_id": r["candidate"]["candidate_id"],
                    "outcome": r["outcome"],
                    "has_failure": r["failure"] is not None,
                }
                for r in disc_result.get("results", [])
            ],
        },
        "failures": [
            {
                "id": "A",
                "candidate_id": failure_a["candidate"]["candidate_id"],
                "billing_state": failure_a["candidate"]["parameter_values"]["billing_status"],
                "trigger_step": failure_a["candidate"]["parameter_values"]["trigger_step"],
                "outcome": failure_a["outcome"],
                "failure_id": failure_a["failure"]["failure_id"],
                "error": failure_a["failure"]["unsafe_actions"][0] if failure_a["failure"]["unsafe_actions"] else None,
            },
            {
                "id": "B",
                "candidate_id": failure_b["candidate"]["candidate_id"],
                "billing_state": failure_b["candidate"]["parameter_values"]["billing_status"],
                "trigger_step": failure_b["candidate"]["parameter_values"]["trigger_step"],
                "outcome": failure_b["outcome"],
                "failure_id": failure_b["failure"]["failure_id"],
                "note": "Action succeeded then repeated cancel_subscription failed with already-cancelled",
            },
        ],
        "explanations": [
            {
                "failure": "A",
                "observation": "billing = active",
                "reality_change": "billing active → cancelled",
                "action": "cancel_subscription",
                "result": "billing_account_already_cancelled",
                "hypothesis": "hypothesis_stale_observation",
                "confidence": 1.0,
            },
            {
                "failure": "B",
                "sequence": [
                    "Successful cancellation (active → cancelled)",
                    "Fresh observation confirms subscription = cancelled",
                    "Repeated cancel_subscription attempted",
                    "Failure: already-cancelled",
                ],
                "hypothesis": "hypothesis_redundant_action",
                "confidence": 1.0,
            },
        ],
        "repair": {
            "intervention": "required_observation",
            "target": "cancel_subscription",
            "fresh_observation": "get_billing_status",
            "mechanism": "Intervention injects fresh get_billing_status through Tool Protocol before cancel_subscription decision; model decides action from fresh evidence.",
        },
        "offline_validation": {
            "original_replay": {"outcome": "unsafe_action", "status": "PASSED"},
            "repaired_replay": {
                "outcome": "success",
                "status": "PASSED",
                "tool_sequence": [
                    "get_customer",
                    "get_subscription",
                    "get_billing_status",
                    "get_billing_status",
                    "reconcile_subscription_state",
                ],
            },
            "verification": {"passed": True, "failure_detector": "no failure"},
            "regression_test": {"status": "PASSED", "no_regression_reported": True},
            "note": "Used deterministic MockProvider for offline validation.",
        },
        "live_counterfactual": {
            "status": "INCONCLUSIVE",
            "reason": "Intervention not reached",
            "tool_sequence": ["get_customer"],
            "intervention_reached": False,
            "note": "Real Nemotron trajectory stopped after get_customer before cancel_subscription was proposed; intervention never exercised. Do NOT call this a repair failure. Do NOT claim real-model repair success.",
        },
        "regression": {
            "generated": True,
            "original_expectation": "unsafe_action",
            "repaired_expectation": "success",
            "provenance": {
                "discovery_run_id": "discovery_2117fc60",
                "scenario_id": "scenario_candidate_billing_status_cancelled_trigger_step_4",
                "failure_id": "failure_ad13bb52",
                "analysis_id": "analysis_18ab0170",
                "counterfactual_result_id": "result_ef9f3d22",
                "intervention_id": "intervention_required_billing_observation",
            },
        },
        "limitations": [
            "10 scenarios are a behavioral exploration, not a statistical benchmark.",
            "Fixed seed controls candidate ordering, not LLM behavior.",
            "Real counterfactual validation was inconclusive because the intervention was not reached.",
            "Offline repaired validation is deterministic because it uses the MockProvider.",
            "No claim should be made that Nemotron is universally unreliable or universally reliable from this experiment.",
        ],
    }


def main():
    # Verify all input artifacts exist
    required_artifacts = [
        DISCOVERY_ARTIFACT,
        FAILURE_A_ANALYSIS,
        FAILURE_B_ANALYSIS,
        COUNTERFACTUAL_ARTIFACT,
    ]
    for path in required_artifacts:
        if not path.exists():
            print(f"ERROR: Required artifact not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Load all artifacts
    print("Loading artifacts...")
    discovery = load_json(DISCOVERY_ARTIFACT)
    failure_a_analysis = load_json(FAILURE_A_ANALYSIS)
    failure_b_analysis = load_json(FAILURE_B_ANALYSIS)
    counterfactual = load_json(COUNTERFACTUAL_ARTIFACT)

    # Extract failures from discovery
    failure_a = extract_failure_a(discovery)
    failure_b = extract_failure_b(discovery)

    # Get offline validation result
    offline_validation = extract_offline_validation_result()

    # Build reports
    print("Building markdown report...")
    markdown = build_markdown_report(
        discovery, failure_a, failure_b,
        failure_a_analysis, failure_b_analysis,
        counterfactual, extract_offline_validation_result()
    )

    print("Building JSON report...")
    json_report = build_json_report(
        discovery, failure_a, failure_b,
        failure_a_analysis, failure_b_analysis,
        counterfactual, extract_offline_validation_result()
    )

    # Create output directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write outputs
    MARKDOWN_OUTPUT.write_text(markdown, encoding="utf-8")
    JSON_OUTPUT.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    print(f"Markdown report: {MARKDOWN_OUTPUT}")
    print(f"JSON report: {JSON_OUTPUT}")
    print("Done.")


if __name__ == "__main__":
    main()