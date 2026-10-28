#!/usr/bin/env python
"""Build the minimal, fixture-backed static site used by GitHub Pages."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "apps" / "web"
REPORT_PATH = REPO_ROOT / "reports" / "autonomous_reliability_report.json"
SITE_DIR = REPO_ROOT / "_site"
OWNER_FILE = REPO_ROOT / ".arl-pages-site-owner"
OWNER_MARKER = "Autonomous Reliability Lab generated Pages output v1\n"
REPORT_URL_IN_SOURCE = 'content="../../reports/autonomous_reliability_report.json"'
REPORT_URL_IN_SITE = 'content="reports/autonomous_reliability_report.json"'


def generate_report() -> dict:
    """Run the normal fixture-backed report command and validate its claims."""
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_demo_report.py")],
        cwd=REPO_ROOT,
        check=True,
    )
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Report generator did not create {REPORT_PATH}")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    required = {
        "project", "experiment", "discovery", "failures", "explanations",
        "repair", "offline_validation", "live_counterfactual", "regression",
    }
    missing = sorted(required - report.keys())
    if missing:
        raise ValueError(f"Generated report is missing required sections: {', '.join(missing)}")
    if report["offline_validation"].get("status") != "PASSED":
        raise ValueError("The deterministic offline validation did not pass")
    if report["live_counterfactual"].get("status") != "INCONCLUSIVE":
        raise ValueError("The saved live counterfactual must remain INCONCLUSIVE")
    return report


def _check_output_ownership() -> None:
    if SITE_DIR.is_symlink():
        raise RuntimeError(f"Refusing to replace symlink output directory: {SITE_DIR}")
    if SITE_DIR.exists():
        if not OWNER_FILE.is_file() or OWNER_FILE.read_text(encoding="utf-8") != OWNER_MARKER:
            raise RuntimeError(
                f"Refusing to replace unmanaged output directory {SITE_DIR}. "
                "Move it aside before building."
            )


def build_site() -> Path:
    _check_output_ownership()
    report = generate_report()
    with tempfile.TemporaryDirectory(prefix=".arl-pages-build-", dir=REPO_ROOT) as temp:
        staged_site = Path(temp) / "site"
        staged_site.mkdir()

        # Copy only the static dashboard assets. No source code, test files,
        # original artifacts, or local report files are included.
        for filename in ("index.html", "app.js", "styles.css"):
            source = WEB_ROOT / filename
            if not source.is_file():
                raise FileNotFoundError(f"Required dashboard asset not found: {source}")
            shutil.copy2(source, staged_site / filename)

        index_path = staged_site / "index.html"
        html = index_path.read_text(encoding="utf-8")
        if html.count(REPORT_URL_IN_SOURCE) != 1:
            raise ValueError("Expected exactly one repository-demo report URL in index.html")
        html = html.replace(REPORT_URL_IN_SOURCE, REPORT_URL_IN_SITE)
        index_path.write_text(html, encoding="utf-8")

        deployed_report = staged_site / "reports" / "autonomous_reliability_report.json"
        deployed_report.parent.mkdir(parents=True)
        shutil.copy2(REPORT_PATH, deployed_report)
        (staged_site / ".nojekyll").write_text("", encoding="utf-8")

        # The ownership marker is kept beside the site, never deployed. This
        # permits safe repeat builds while refusing to delete an unrelated _site.
        if SITE_DIR.exists():
            shutil.rmtree(SITE_DIR)
        staged_site.replace(SITE_DIR)
        OWNER_FILE.write_text(OWNER_MARKER, encoding="utf-8")

    if report["live_counterfactual"].get("interpretation") != "INCONCLUSIVE":
        raise ValueError("The live counterfactual interpretation must remain INCONCLUSIVE")
    return SITE_DIR


if __name__ == "__main__":
    output = build_site()
    print(f"Built static demo at {output.relative_to(REPO_ROOT)}")
