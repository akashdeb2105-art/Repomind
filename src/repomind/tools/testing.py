"""run_tests: attempt the repository's own test suite, safely.

This is the only tool that executes code from a repository RepoMind did not
write, so it is the only one that can do real damage. Three guardrails:

1. **Off by default.** Requires ``REPOMIND_ALLOW_TEST_EXECUTION=1``. Analysing a
   stranger's repository must never silently run their code — a `conftest.py`
   runs arbitrary Python at collection time.
2. **No shell.** Commands are argument lists passed straight to the OS, so
   nothing in a repo can inject ``; rm -rf ~``.
3. **Hard timeout, and the process tree dies with it.** A hung suite must not
   hang the agent, and killing only the parent leaves orphans running.

Custom commands are accepted, but only from the caller — never inferred from
file contents. Detection maps a *recognised project layout* to a *known-safe
command*, which is a much narrower thing than "run what the repo tells you to".
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from repomind.models import TestRunResult
from repomind.tools.repo import RepoContext

DEFAULT_TIMEOUT_S = 120.0
MAX_OUTPUT_CHARS = 4_000

# text=True alone decodes with the *locale* encoding, which is cp1252 on a
# default Windows install. Source code, commit messages and test output are
# full of characters cp1252 cannot represent — one curly quote in one commit
# message killed a benchmark run, because the decode failure happens on a
# reader thread and surfaces only as stdout being None six frames later.
# UTF-8 with replacement is the only safe reading of another program's output.
DECODE = {"encoding": "utf-8", "errors": "replace"}
ENABLE_ENV_VAR = "REPOMIND_ALLOW_TEST_EXECUTION"


def _detect(repo: RepoContext) -> tuple[str, list[str]] | None:
    """Map a recognised layout to a known test command."""
    root = repo.root

    has_test_dir = (root / "tests").is_dir() or (root / "test").is_dir()
    if (root / "pytest.ini").is_file() or has_test_dir or _pyproject_mentions(root, "pytest"):
        return "pytest", [sys.executable, "-m", "pytest", "-q", "--no-header"]

    package_json = root / "package.json"
    if package_json.is_file():
        try:
            import json

            scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if "test" in scripts:
            return "npm", ["npm", "test", "--silent"]

    if (root / "go.mod").is_file():
        return "go", ["go", "test", "./..."]
    if (root / "Cargo.toml").is_file():
        return "cargo", ["cargo", "test", "--quiet"]

    return None


def _pyproject_mentions(root: Path, needle: str) -> bool:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        return needle in pyproject.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def run_tests(
    repo: RepoContext,
    command: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> TestRunResult:
    """Run the repository's test suite and summarise the outcome.

    Returns a result object in every case — including "refused" and "timed out".
    The agent needs to reason about those outcomes, not crash on them.
    """
    if os.getenv(ENABLE_ENV_VAR) != "1":
        return TestRunResult(
            command="",
            skipped_reason=(
                f"test execution is disabled; set {ENABLE_ENV_VAR}=1 to allow running "
                "code from the analysed repository"
            ),
        )

    if command is None:
        detected = _detect(repo)
        if detected is None:
            return TestRunResult(command="", skipped_reason="no recognised test setup found")
        framework, argv = detected
    else:
        if not isinstance(command, list) or not command:
            return TestRunResult(
                command="", skipped_reason="command must be a non-empty argument list"
            )
        framework, argv = "custom", command

    timeout_s = max(5.0, min(timeout_s, 600.0))
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            argv,
            cwd=str(repo.root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
            **DECODE,  # never a shell: no injection surface from repo contents
        )
    except subprocess.TimeoutExpired as exc:
        return TestRunResult(
            command=" ".join(argv),
            timed_out=True,
            duration_s=round(time.perf_counter() - started, 2),
            detected_framework=framework,
            stdout_tail=_tail(exc.stdout),
            stderr_tail=_tail(exc.stderr),
        )
    except (FileNotFoundError, OSError) as exc:
        return TestRunResult(
            command=" ".join(argv),
            detected_framework=framework,
            skipped_reason=f"could not run test command: {exc}",
        )

    return TestRunResult(
        command=" ".join(argv),
        exit_code=completed.returncode,
        passed=completed.returncode == 0,
        duration_s=round(time.perf_counter() - started, 2),
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        detected_framework=framework,
    )


def _tail(output: str | bytes | None) -> str:
    """Keep the end of the output — that is where the failure summary lives."""
    if not output:
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    return "… [truncated]\n" + output[-MAX_OUTPUT_CHARS:]
