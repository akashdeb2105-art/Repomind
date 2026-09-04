"""run_tests — the one tool that executes third-party code."""

from __future__ import annotations

import sys

from repomind.tools import RepoContext, run_tests
from repomind.tools.testing import ENABLE_ENV_VAR


def test_execution_is_refused_unless_explicitly_enabled(sample_repo: RepoContext, monkeypatch):
    """Analysing a stranger's repo must never silently execute their code."""
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)

    result = run_tests(sample_repo)

    assert result.skipped_reason and ENABLE_ENV_VAR in result.skipped_reason
    assert result.passed is False
    assert result.exit_code is None


def test_runs_a_passing_command(sample_repo: RepoContext, monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")

    result = run_tests(sample_repo, command=[sys.executable, "-c", "print('all good')"])

    assert result.passed is True
    assert result.exit_code == 0
    assert "all good" in result.stdout_tail
    assert result.timed_out is False


def test_reports_failure_without_raising(sample_repo: RepoContext, monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")

    result = run_tests(sample_repo, command=[sys.executable, "-c", "raise SystemExit(1)"])

    assert result.passed is False
    assert result.exit_code == 1


def test_a_hanging_suite_is_killed_by_the_timeout(sample_repo: RepoContext, monkeypatch):
    """The failure mode this guard exists for: a suite that never returns."""
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")

    result = run_tests(
        sample_repo,
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_s=5,
    )

    assert result.timed_out is True
    assert result.passed is False
    assert result.duration_s < 15, "must not wait for the full 60 seconds"


def test_detects_pytest_from_layout(sample_repo: RepoContext, monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")

    result = run_tests(sample_repo, timeout_s=60)

    assert result.detected_framework == "pytest"


def test_unrecognised_project_is_skipped_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")

    result = run_tests(RepoContext.create(tmp_path))

    assert result.skipped_reason == "no recognised test setup found"


def test_a_string_command_is_rejected(sample_repo: RepoContext, monkeypatch):
    """Refusing strings is what keeps a shell — and shell injection — out."""
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")

    result = run_tests(sample_repo, command="rm -rf /")  # type: ignore[arg-type]

    assert result.skipped_reason and "argument list" in result.skipped_reason
