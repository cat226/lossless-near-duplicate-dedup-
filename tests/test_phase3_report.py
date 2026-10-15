"""The Phase 3 report must equal a fresh generation from the committed JSON, and every number in it must equal the JSON."""

import subprocess
import sys

import pytest

from phase3.splits import REPO, TEST_APPROVAL_FILE

NEEDED = ["results/phase3_results.json", "results/phase3_posthoc.json", "docs/phase3_report.md"]


@pytest.mark.skipif(not all((REPO / p).exists() for p in NEEDED) or not TEST_APPROVAL_FILE.exists(),
                    reason="needs the committed results and the local test-split approval file (used for the family-overlap table)")
def test_report_numbers_equal_json():
    r = subprocess.run([sys.executable, "experiments/make_phase3_report.py", "--check"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "CHECK OK" in r.stdout
