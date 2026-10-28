# Autonomous Reliability Lab

> Find the failures your agent doesn't know to test for.

Autonomous Reliability Lab explores failure modes in tool-using agents, explains failures from recorded evidence, tests a repair in a deterministic offline replay, and preserves the result as a regression test.

## Core loop

**Discover → Explain → Repair → Preserve → Validate**

## Architecture

| Component | Purpose |
|---|---|
| `TargetAgent` | Runs a replaceable model through the explicit tool protocol boundary |
| `ToolRegistry` / `ToolTrace` | Defines tools and records ordered calls and results |
| `EnvironmentService` | Simulated state with snapshot and restore support |
| `ScenarioRunner` / `ScenarioScheduler` | Runs scenarios and deterministic perturbations |
| `DiscoveryEngine` | Evaluates candidate scenarios using real or mock providers |
| `FailureDetector` / `DeterministicVerifier` | Classifies outcomes from structured evidence |
| `CausalAnalyzer` | Identifies observation/reality divergence and records hypotheses |
| `CounterfactualExecutor` | Replays scenarios with tool-boundary interventions |
| Regression framework | Generates and runs regression checks from a validated offline repair |
| `NebiusNemotronProvider` | OpenAI-compatible client for NVIDIA Nemotron on Nebius |

## Recorded discovery

The committed discovery fixture records 10 scenarios: 8 `success`, 1 `unsafe_action`, and 1 `degraded`.

- **Failure A:** the agent observed billing as active; the scenario’s billing state changed to cancelled at the step-four trigger, and the cancellation call returned `billing_account_already_cancelled`. The analysis records a `stale_observation` hypothesis.
- **Failure B:** the trace records a repeated cancellation call whose second call returned `invalid_arguments`. The analysis classifies it under the `redundant_action` hypothesis; this is a hypothesis, not proof of cause.

The hypothesis confidence values are the analyzer’s recorded values, not statistical probabilities.

## Offline repair validation

The deterministic MockProvider replay tests a `required_observation` intervention for `cancel_subscription`, requiring a fresh `get_billing_status` observation. The report generator executes the offline replay and regression suite and derives its displayed statuses from those results. It does not call a hosted model provider.

## Live Nemotron counterfactual

The saved counterfactual artifact records `status: failed` and `scenario_status: agent_failed`. Its tool sequence contains only `get_customer`; it contains neither the required `get_billing_status` observation nor `cancel_subscription`. The source record has no `intervention_reached` field. Based on that trace, the report derives **INCONCLUSIVE**: the repair intervention was not exercised. This result does not establish repair failure or live repair success.

## Supplemental runs

Three supplemental runs of the same setup are recorded. All are inconclusive and none reached the intervention. Their model-call counts differ:

| Trial | Status | Intervention reached | Model calls | Tool sequence |
|---|---|---:|---:|---|
| 1 | INCONCLUSIVE | false | 0 | `[]` |
| 2 | INCONCLUSIVE | false | 1 | `[]` |
| 3 | INCONCLUSIVE | false | 1 | `[]` |

These records are supplemental observations. They do not establish statistical significance, generalization, or proof of repair behavior.

## Evidence and reports

Sanitized, committed source fixtures live under `tests/fixtures/`:

- `discovery/` contains the authoritative 10-scenario run.
- `counterfactual/` contains the original live counterfactual record.
- `replications/` contains the three supplemental records.
- `analysis/` contains compact, provenance-preserving hypothesis records.

`replications/provenance.json` maps each recorded replication to its committed fixture and source evidence. It preserves the original artifact path values; Trial 1’s field named `source_counterfactual_artifact` contains the discovery artifact path in the historical record, which the mapping identifies as a mislabeled source-discovery reference.

The report generator writes these local outputs, which are ignored by Git:

- `reports/autonomous_reliability_report.json`
- `reports/autonomous_reliability_report.md`

The static browser demo reads the JSON report. It has no hosted API dependency.

## Setup and local demo

Requires Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe scripts/generate_demo_report.py
.\.venv\Scripts\python.exe -m http.server 8080 --bind 127.0.0.1
```

On macOS or Linux, use `.venv/bin/python` for the commands above.

Then open `http://127.0.0.1:8080/apps/web/`. Start the server from the repository root so the UI can fetch `reports/autonomous_reliability_report.json`.

The root requirements file contains the project’s direct test/provider dependencies rather than a full environment snapshot. `openai` and `python-dotenv` support the optional live provider; the fixture-backed report, offline replay, and UI do not require `.env` credentials or network access. The API scaffold under `apps/api/` is separate from the static demo.

## Live provider workflows

Scripts named `scripts/live_*.py` are opt-in. They contact Nebius and may consume API credits. They are not part of the offline reproduction commands above. To configure a live run, create a local `.env` from `.env.example`; never commit credentials.

## Limitations

- Ten scenarios are behavioral exploration, not a statistical benchmark.
- A fixed seed controls candidate ordering, not model behavior.
- The live counterfactual is inconclusive because the intervention was not reached.
- Offline repair validation uses the deterministic MockProvider.
- No universal reliability or unreliability claim follows from these runs.

## License

MIT. See [LICENSE](LICENSE).
