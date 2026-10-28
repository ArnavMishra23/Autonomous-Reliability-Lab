"""Fixture-backed tests for the deterministic demo report."""

import json
import re
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_demo_report import generate_reports, load_json


class TestDemoReport:
    """Generate reports into a temporary directory; never require ignored output."""

    @classmethod
    def setup_class(cls):
        cls._temp_dir = tempfile.TemporaryDirectory(prefix="arl-report-test-")
        cls.markdown_path, cls.json_path = generate_reports(Path(cls._temp_dir.name))
        cls.json_report = json.loads(cls.json_path.read_text(encoding="utf-8"))
        cls.markdown_report = cls.markdown_path.read_text(encoding="utf-8")

    @classmethod
    def teardown_class(cls):
        cls._temp_dir.cleanup()

    def test_report_loads_committed_discovery_fixture(self):
        assert self.json_report["experiment"]["name"] == "discovery10"
        assert self.json_report["experiment"]["discovery_run_id"] == "discovery_2117fc60"
        assert self.json_report["experiment"]["scenarios_evaluated"] == 10
        assert self.json_report["experiment"]["outcome_counts"] == {
            "success": 8, "unsafe_action": 1, "degraded": 1
        }

    def test_failure_a_is_derived_from_discovery_and_analysis(self):
        failure = next(item for item in self.json_report["failures"] if item["id"] == "A")
        explanation = next(item for item in self.json_report["explanations"] if item["failure"] == "A")
        assert failure["outcome"] == "unsafe_action"
        assert failure["failure_id"] == "failure_ad13bb52"
        assert explanation["hypothesis"] == "hypothesis_stale_observation"
        assert explanation["confidence"] == 1.0
        assert explanation["observation"] == "billing = active"
        assert explanation["result"] == "billing_account_already_cancelled"

    def test_failure_b_error_matches_recorded_second_call(self):
        failure = next(item for item in self.json_report["failures"] if item["id"] == "B")
        explanation = next(item for item in self.json_report["explanations"] if item["failure"] == "B")
        assert failure["outcome"] == "degraded"
        assert failure["failure_id"] == "failure_56fa2250"
        assert failure["second_cancel_error_code"] == "invalid_arguments"
        assert explanation["hypothesis"] == "hypothesis_redundant_action"
        assert explanation["confidence"] == 1.0
        assert explanation["classification_note"] == "causal hypothesis, not proof"
        assert "invalid_arguments" in explanation["sequence"][-1]

    def test_offline_status_comes_from_executed_mock_replay(self):
        validation = self.json_report["offline_validation"]
        assert validation["status"] == "PASSED"
        assert validation["original_replay"]["outcome"] == "unsafe_action"
        assert validation["repaired_replay"]["outcome"] == "success"
        assert validation["verification"]["passed"] is True
        assert validation["regression_test"]["status"] == "PASSED"
        assert validation["validation_source"].startswith("executed deterministic MockProvider")

    def test_live_counterfactual_separates_raw_status_and_interpretation(self):
        live = self.json_report["live_counterfactual"]
        assert live["recorded_status"] == "failed"
        assert live["recorded_scenario_status"] == "agent_failed"
        assert live["status"] == live["interpretation"] == "INCONCLUSIVE"
        assert live["tool_sequence"] == ["get_customer"]
        assert live["intervention_reached"] is False
        assert live["source_has_intervention_reached_field"] is False
        assert live["evidence_basis"]["required_observation_in_trace"] is False

    def test_provenance_is_preserved(self):
        provenance = self.json_report["regression"]["provenance"]
        assert provenance["discovery_run_id"] == "discovery_2117fc60"
        assert provenance["failure_id"] == "failure_ad13bb52"
        assert provenance["analysis_id"] == "analysis_18ab0170"
        assert provenance["counterfactual_result_id"] == "result_ef9f3d22"
        assert provenance["intervention_id"] == "intervention_required_billing_observation"

    def test_markdown_and_json_agree_on_live_result_and_failure_b(self):
        assert "Recorded status:** failed (`agent_failed`)" in self.markdown_report
        assert "Evidence-derived interpretation:** INCONCLUSIVE" in self.markdown_report
        assert "Second cancellation error code:** invalid_arguments" in self.markdown_report
        assert self.json_report["live_counterfactual"]["status"] in self.markdown_report
        assert "already-cancelled" not in self.markdown_report

    def test_report_schema_is_ui_compatible(self):
        for field in ("project", "experiment", "model", "discovery", "failures", "explanations",
                      "repair", "offline_validation", "live_counterfactual", "regression", "limitations"):
            assert field in self.json_report
        assert len(self.json_report["failures"]) == 2
        assert len(self.json_report["explanations"]) == 2

    def test_supplemental_fixtures_match_documented_inconclusive_runs(self):
        fixture_dir = _REPO_ROOT / "tests" / "fixtures" / "replications"
        provenance = json.loads((fixture_dir / "provenance.json").read_text(encoding="utf-8"))
        expected_calls = [0, 1, 1]
        for trial, expected_model_calls in enumerate(expected_calls, start=1):
            path = next(fixture_dir.glob(f"replication{trial}_*.json"))
            data = json.loads(path.read_text(encoding="utf-8"))
            result = data["replication_result"]
            mapping = next(item for item in provenance["trials"] if item["trial_number"] == trial)
            assert data["replication_metadata"]["trial_number"] == trial
            assert result["status"] == "inconclusive"
            assert result["intervention_reached"] is False
            assert result["model_calls"] == expected_model_calls
            assert mapping["fixture"] == path.name
            assert mapping["status"] == result["status"]
            assert mapping["model_calls"] == result["model_calls"]
            for reference in mapping["normalized_source_refs"].values():
                assert (fixture_dir / reference).is_file()

        trial_one = provenance["trials"][0]
        assert "live-discovery" in trial_one["recorded_source_paths"]["source_counterfactual_artifact"]
        assert "discovery path" in trial_one["normalization_note"]

    def test_no_credentials_or_synthetic_secret_sentinels_in_report_output(self):
        """Both report formats must not expose credential-shaped data."""
        report_text = json.dumps(self.json_report, ensure_ascii=False) + "\n" + self.markdown_report
        lower_report = report_text.lower()
        for marker in ("api_key", "apikey", "secret", "password", "authorization", "bearer"):
            assert marker not in lower_report
        patterns = (
            r"(?i)(?:api[_-]?key|password|authorization)\s*[:=]\s*[^\s,;]+",
            r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}",
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"\bAKIA[0-9A-Z]{16}\b",
            r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
        )
        assert not any(re.search(pattern, report_text) for pattern in patterns)
        assert "TEST_SECRET_SENTINEL_DO_NOT_LEAK" not in report_text

    def test_report_generation_is_deterministic_for_committed_fixtures(self):
        with tempfile.TemporaryDirectory(prefix="arl-report-repeat-") as output:
            markdown_path, json_path = generate_reports(Path(output))
            assert json.loads(json_path.read_text(encoding="utf-8")) == self.json_report
            assert markdown_path.read_text(encoding="utf-8") == self.markdown_report

    def test_missing_fixture_fails_clearly(self):
        missing = _REPO_ROOT / "tests" / "fixtures" / "discovery" / "missing.json"
        try:
            load_json(missing)
        except FileNotFoundError as exc:
            assert "Required artifact not found" in str(exc)
        else:
            raise AssertionError("Missing fixture should raise FileNotFoundError")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
