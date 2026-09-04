"""The Critic — the anti-hallucination guardrail.

The brief calls this the single most important feature, so it gets the most
adversarial tests: plant fabrications in a draft and prove they are caught.
"""

from __future__ import annotations

from repomind.agent.nodes import apply_corrections, extract_path_claims, verify_path_claims
from repomind.agent.state import Evidence

REAL_PATHS = [
    "src/sample/main.py",
    "src/sample/core.py",
    "README.md",
    "pyproject.toml",
    "tests/test_core.py",
]


def make_evidence() -> Evidence:
    evidence = Evidence()
    evidence.record_listing(REAL_PATHS)
    evidence.record_read("src/sample/main.py", "def main(): ...")
    evidence.record_dependencies(["httpx", "pydantic"])
    return evidence


# --------------------------------------------------------------------------- #
# The headline test
# --------------------------------------------------------------------------- #


def test_critic_catches_a_deliberately_planted_hallucination():
    """The acceptance test the brief demands: inject a fake path, catch it."""
    draft = (
        "# Onboarding\n\n"
        "Start with `src/sample/main.py`, which is the entry point.\n"
        "Configuration lives in `src/sample/settings.py`.\n"  # <-- does not exist
    )

    claims = verify_path_claims(extract_path_claims(draft, "onboarding"), make_evidence())
    by_target = {c.target: c for c in claims}

    assert by_target["src/sample/main.py"].grounded is True
    assert by_target["src/sample/settings.py"].grounded is False
    assert "no tool call ever saw this path" in by_target["src/sample/settings.py"].reason


def test_a_fabricated_module_is_removed_from_the_output():
    draft = (
        "## Project structure\n\n"
        "- `src/sample/main.py` — entry point\n"
        "- `src/sample/database.py` — persistence layer\n"  # invented
        "- `README.md` — this file\n"
    )

    claims = verify_path_claims(extract_path_claims(draft, "onboarding"), make_evidence())
    corrected, removed, _ = apply_corrections(draft, claims)

    assert "database.py" not in corrected, "the fabrication must not survive"
    assert "main.py" in corrected and "README.md" in corrected, "real files must survive"
    assert any("database.py" in line for line in removed)


def test_prose_mentioning_a_fake_path_is_flagged_not_silently_deleted():
    draft = "The service is configured through `config/production.yaml` at startup.\n"

    claims = verify_path_claims(extract_path_claims(draft, "onboarding"), make_evidence())
    corrected, removed, flagged = apply_corrections(draft, claims)

    assert "unverified" in corrected
    assert removed == []
    assert flagged


# --------------------------------------------------------------------------- #
# False positives are what make a verifier useless
# --------------------------------------------------------------------------- #


def test_a_clean_document_is_left_completely_untouched():
    draft = (
        "# Onboarding\n\n"
        "This project is a sample application.\n\n"
        "Start reading at `src/sample/main.py`, then `src/sample/core.py`.\n"
        "Tests live in `tests/test_core.py` and run with pytest.\n"
    )

    claims = verify_path_claims(extract_path_claims(draft, "onboarding"), make_evidence())
    corrected, removed, flagged = apply_corrections(draft, claims)

    assert corrected == draft
    assert removed == [] and flagged == []
    assert all(c.grounded for c in claims)


def test_ordinary_prose_produces_no_claims():
    draft = "This project makes onboarding faster. It is written in Python and is easy to run.\n"

    assert extract_path_claims(draft, "onboarding") == []


def test_urls_are_not_treated_as_file_paths():
    draft = "See https://github.com/example/repo for details, or `docs/index.md`.\n"

    targets = {c.target for c in extract_path_claims(draft, "onboarding")}

    assert not any(t.startswith("http") for t in targets)


def test_path_spellings_are_normalised():
    """A model writes ./src/main.py, src\\main.py and src/main.py interchangeably."""
    evidence = Evidence()
    evidence.record_listing(["src/main.py"])

    assert evidence.knows_path("./src/main.py")
    assert evidence.knows_path("src\\main.py")
    assert evidence.knows_path("`src/main.py`")
    assert not evidence.knows_path("src/other.py")


def test_directories_count_as_known_when_files_live_under_them():
    evidence = make_evidence()

    assert evidence.directory_exists("src/sample")
    assert evidence.directory_exists("tests")
    assert not evidence.directory_exists("docs")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_report_counts_are_accurate():
    from repomind.models import CriticReport

    draft = "`src/sample/main.py` and `src/sample/ghost.py` and `README.md`\n"
    claims = verify_path_claims(extract_path_claims(draft, "onboarding"), make_evidence())
    report = CriticReport(claims=claims)

    assert report.grounded_count == 2
    assert report.hallucination_count == 1
    assert "2/3" in report.verdict
