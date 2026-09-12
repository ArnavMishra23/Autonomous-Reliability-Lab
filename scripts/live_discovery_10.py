#!/usr/bin/env python
"""Live DiscoveryEngine 10-candidate experiment with real Nemotron.

This script runs the COMPLETE 10-candidate ScenarioSpace through the existing
DiscoveryEngine using the real Nebius Token Factory + NVIDIA Nemotron provider.

Usage:
    NEBIUS_API_KEY=xxx NEBIUS_MODEL=nvidia/nemotron-3-super-120b-a12b \
    python scripts/live_discovery_10.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Bootstrap: add repository root to sys.path so 'packages' imports works.
# This allows running the script directly from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Load repository-root .env before reading environment variables.
load_dotenv(_REPO_ROOT / ".env")

from packages.agent.target import TargetAgent
from packages.agent.protocol import TargetAgentToolInterface
from packages.agent.tools import build_tool_registry
from packages.agent.providers.nebius import NebiusNemotronProvider
from packages.environment.service import EnvironmentService
from packages.discovery.search import (
    DiscoveryEngine,
    RandomSearchStrategy,
    ScenarioGenerator,
    ScenarioParameter,
    ScenarioSpace,
)
from packages.discovery.failure import FailureDetector
from packages.discovery.scenario.runner import ScenarioRunner
from packages.discovery.scenario.models import ScenarioRun
from packages.agent.target import AgentRun
from packages.discovery.search.models import DiscoveryResult


# Experiment configuration
EXPERIMENT_NAME = "discovery10"
SEED = 20260910
MAX_STEPS = 10
ARTIFACT_VERSION = 1


class CapturingRunner:
    """Wrapper that delegates to ScenarioRunner but captures each ScenarioRun."""

    def __init__(self, runner: ScenarioRunner) -> None:
        self._runner = runner
        self.captured_runs: list[ScenarioRun] = []

    def run(self, scenario, environment, agent) -> ScenarioRun:
        scenario_run = self._runner.run(scenario, environment, agent)
        self.captured_runs.append(scenario_run)
        return scenario_run


def create_space() -> ScenarioSpace:
    """Create the full 10-candidate ScenarioSpace."""
    return ScenarioSpace(
        parameters=(
            ScenarioParameter("billing_status", "discrete", ("active", "cancelled")),
            ScenarioParameter("trigger_step", "discrete", (1, 2, 3, 4, 5)),
        )
    )


def build_scenario_evidence(scenario_run: ScenarioRun, candidate_id: str) -> dict:
    """Build complete observable evidence for one scenario from ScenarioRun."""
    agent_run = scenario_run.agent_run

    # Build complete tool trace
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


def print_scenario_summary(index: int, scenario_run: ScenarioRun, candidate_id: str) -> None:
    """Print compact one-line summary for a scenario."""
    agent_run = scenario_run.agent_run

    # Parse candidate params from candidate_id
    billing_status = "unknown"
    trigger_step = "unknown"
    if "billing_status_" in candidate_id:
        parts = candidate_id.split("_")
        try:
            bs_idx = parts.index("billing") + 1
            billing_status = parts[bs_idx]
        except (ValueError, IndexError):
            pass
        try:
            ts_idx = parts.index("trigger") + 2
            trigger_step = parts[ts_idx]
        except (ValueError, IndexError):
            pass

    agent_status = "none"
    steps = 0
    tool_count = 0
    if agent_run is not None:
        agent_status = agent_run.status.value
        steps = len(agent_run.trace.entries)
        tool_count = steps

    restored = "yes" if (
        len(scenario_run.final_environment.get("subscriptions", {})) > 0
        and len(scenario_run.final_environment.get("billing_accounts", {})) > 0
    ) else "no"

    outcome = "unknown"
    if scenario_run.status.value == "completed":
        outcome = "success"
    elif scenario_run.status.value == "agent_failed":
        outcome = "failure"
    elif scenario_run.status.value == "perturbation_failed":
        outcome = "perturbation_failure"
    elif scenario_run.status.value == "runner_failed":
        outcome = "runner_failure"

    print(f"  {candidate_id} | billing={billing_status} | step={trigger_step} | "
          f"scenario={scenario_run.status.value} | agent={agent_status} | "
          f"outcome={outcome} | steps={steps} | tools={tool_count} | restored={restored}")


def main() -> int:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY not set. Skipping live discovery experiment.")
        print("Set NEBIUS_API_KEY and NEBIUS_MODEL to run.")
        return 0

    model = os.getenv("NEBIUS_MODEL")
    if not model:
        print("NEBIUS_MODEL not set. Please specify a model identifier.")
        return 1

    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

    # Print experiment header
    print(f"Experiment:")
    print(f"  {EXPERIMENT_NAME}")
    print(f"Model:")
    print(f"  {model}")
    print(f"Seed:")
    print(f"  {SEED}")
    print(f"Candidates:")
    print(f"  10")
    print(f"Max steps:")
    print(f"  {MAX_STEPS}")

    # Set up the Customer Operations environment
    environment = EnvironmentService()
    customer = environment.create_customer(name="Maya", email="maya@example.com")
    subscription = environment.create_subscription(customer_id=customer.id, plan="pro")
    billing = environment.create_billing_account(subscription_id=subscription.id)

    print(f"\nEnvironment setup:")
    print(f"  Customer: {customer.id} ({customer.name})")
    print(f"  Subscription: {subscription.id} (plan: {subscription.plan})")
    print(f"  Billing: {billing.id} (status: {billing.status.value})")

    # Build tool registry and agent
    tool_registry = build_tool_registry(environment)
    tool_interface = TargetAgentToolInterface(tool_registry)

    provider = NebiusNemotronProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=120.0,  # Longer timeout for 10 scenarios
    )

    agent = TargetAgent(tool_interface, provider, max_steps=MAX_STEPS)

    # Create the full space and use RandomSearchStrategy with fixed seed
    space = create_space()
    strategy = RandomSearchStrategy(seed=SEED)

    # Create generator, runner, detector
    generator = ScenarioGenerator(
        customer_id=customer.id,
        subscription_id=subscription.id,
        billing_id=billing.id,
    )
    base_runner = ScenarioRunner()
    capturing_runner = CapturingRunner(base_runner)
    detector = FailureDetector()
    engine = DiscoveryEngine()

    # Run discovery with max_scenarios=10 (full coverage)
    print(f"\nStarting DiscoveryEngine run (max_scenarios=10)...")
    try:
        result = engine.run(
            space=space,
            generator=generator,
            strategy=strategy,
            runner=capturing_runner,
            environment=environment,
            agent=agent,
            failure_detector=detector,
            max_scenarios=10,
        )
    except Exception as exc:
        print(f"\nDiscoveryEngine FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    print(f"\n=== DiscoveryResult ===")
    print(f"  Discovery Run ID: {result.discovery_run_id}")
    print(f"  Status: {result.status.value}")
    print(f"  Scenarios Evaluated: {result.scenarios_evaluated}")
    print(f"  Outcome Counts: {dict(result.outcome_counts)}")
    print(f"  Unique Failures: {result.unique_failure_count}")
    print(f"  Failure IDs: {list(result.failure_ids)}")
    print(f"  Started: {result.started_at}")
    print(f"  Finished: {result.finished_at}")
    if result.error:
        print(f"  Error: {result.error}")

    # Print compact per-scenario summary
    print(f"\n=== Scenario Summaries ===")
    print(f"  candidate_id | billing | step | scenario | agent | outcome | steps | tools | restored")
    for scenario_run in capturing_runner.captured_runs:
        candidate_id = scenario_run.scenario_id.replace("scenario_", "")
        print_scenario_summary(0, scenario_run, candidate_id)

    # Verify serialization
    try:
        result_dict = result.to_dict()
        json.dumps(result_dict)
        print(f"\nDiscoveryResult JSON serialization: OK")
    except Exception as exc:
        print(f"\nDiscoveryResult JSON serialization FAILED: {exc}")
        return 1

    # Build comprehensive experiment artifact with full evidence
    artifact_data = build_experiment_artifact(result, capturing_runner.captured_runs, model)

    # Persist comprehensive artifact
    artifact_path = write_experiment_artifact(artifact_data, _REPO_ROOT)
    if artifact_path:
        print(f"\nArtifact:")
        print(f"  {artifact_path.relative_to(_REPO_ROOT)}")

    # Verify environment restored after all scenarios
    final_env = environment.snapshot()
    print(f"\n=== Final Environment State (after all scenarios) ===")
    for sub in final_env.subscriptions.values():
        print(f"  Subscription {sub.id}: {sub.status.value}")
    for bill in final_env.billing_accounts.values():
        print(f"  Billing {bill.id}: {bill.status.value}")

    # Check if original entities are still active
    sub_ok = final_env.subscriptions.get(subscription.id) is not None
    bill_ok = final_env.billing_accounts.get(billing.id) is not None
    if sub_ok and bill_ok:
        print("\nEnvironment restoration: OK")
    else:
        print("\nEnvironment restoration: FAILED - entities missing")

    # Verify we evaluated exactly 10 unique candidates
    evaluated_ids = set()
    for run in capturing_runner.captured_runs:
        cid = run.scenario_id.replace("scenario_", "")
        evaluated_ids.add(cid)
    
    if len(evaluated_ids) == 10:
        print(f"\nCandidate coverage: 10/10 unique candidates evaluated")
    else:
        print(f"\nWARNING: Only {len(evaluated_ids)}/10 unique candidates evaluated: {evaluated_ids}")

    print("\nLive DiscoveryEngine 10-candidate experiment completed.")
    return 0


def build_experiment_artifact(result: DiscoveryResult, captured_runs: list[ScenarioRun], model: str) -> dict:
    """Build complete experiment artifact with full evidence."""
    result_dict = result.to_dict()
    
    # Build per-scenario evidence with full traces
    scenario_evidence = []
    for scenario_run in captured_runs:
        candidate_id = scenario_run.scenario_id.replace("scenario_", "")
        evidence = build_scenario_evidence(scenario_run, candidate_id)
        
        # Add verification and failure from DiscoveryResult.results
        for r in result.results:
            if r.get("scenario_id") == scenario_run.scenario_id:
                evidence["verification"] = r.get("verification")
                evidence["failure"] = r.get("failure")
                break
        
        scenario_evidence.append(evidence)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "experiment_metadata": {
            "experiment_name": EXPERIMENT_NAME,
            "seed": SEED,
            "model": model,
            "candidate_count": 10,
            "max_steps": MAX_STEPS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "discovery_result": result_dict,
        "scenario_evidence": scenario_evidence,
    }
    
    return artifact


def write_experiment_artifact(artifact_data: dict, repo_root: Path) -> Path | None:
    """Write complete experiment artifact to a timestamped JSON file atomically."""
    try:
        # Create artifacts directory
        artifact_dir = repo_root / "artifacts" / "live-discovery"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: discovery10_<timestamp>Z_<run_id>.json
        run_id = artifact_data.get("discovery_result", {}).get("discovery_run_id", "unknown")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"discovery10_{timestamp}_{run_id}.json"
        artifact_path = artifact_dir / filename

        # Atomic write: write to temp file then rename
        temp_path = artifact_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(artifact_data, f, indent=2, ensure_ascii=False)
        temp_path.replace(artifact_path)

        return artifact_path
    except Exception as exc:
        print(f"WARNING: Failed to write experiment artifact: {exc}")
        return None


if __name__ == "__main__":
    sys.exit(main())