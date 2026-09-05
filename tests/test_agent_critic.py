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


# --------------------------------------------------------------------------- #
# Regressions from the first live run against RepoMind's own repo
# --------------------------------------------------------------------------- #


def test_a_shell_command_is_not_mistaken_for_a_fabricated_file():
    """`python scripts/run_agent.py` is a command; verify the path inside it."""
    draft = "Run `python scripts/run_agent.py` to generate the docs.\n"
    evidence = Evidence()
    evidence.record_listing(["scripts/run_agent.py"])

    claims = verify_path_claims(extract_path_claims(draft, "onboarding"), evidence)

    assert [c.target for c in claims] == ["scripts/run_agent.py"]
    assert claims[0].grounded is True


def test_a_command_naming_a_fake_file_is_still_caught():
    draft = "Run `python scripts/deploy.py` to ship it.\n"

    claims = verify_path_claims(extract_path_claims(draft, "onboarding"), make_evidence())

    assert claims[0].target == "scripts/deploy.py"
    assert claims[0].grounded is False


def test_directories_are_distinguished_from_files():
    """Explorer nominated `src/repomind/agent` — a directory — as a file to read."""
    evidence = Evidence()
    evidence.record_listing(
        ["src/repomind/agent", "src/repomind/agent/graph.py"],
        files_only=["src/repomind/agent/graph.py"],
    )

    assert evidence.knows_path("src/repomind/agent")
    assert not evidence.is_file("src/repomind/agent"), "a directory is not readable by read_file"
    assert evidence.is_file("src/repomind/agent/graph.py")


def test_a_null_target_from_the_model_is_coerced():
    """Models return target: null for behavioural claims; rejecting it burns retries."""
    from repomind.models import Claim

    claim = Claim.model_validate(
        {"text": "It scales well.", "kind": "behaviour", "target": None, "grounded": False}
    )

    assert claim.target == ""


def test_verdict_separates_deterministic_findings_from_advisory_ones():
    from repomind.models import Claim, ClaimKind, CriticReport

    report = CriticReport(
        claims=[
            Claim(text="`README.md`", kind=ClaimKind.FILE_PATH, target="README.md", grounded=True),
            Claim(text="`ghost.py`", kind=ClaimKind.FILE_PATH, target="ghost.py", grounded=False),
            Claim(text="Uses Redis", kind=ClaimKind.BEHAVIOUR, target="", grounded=False),
        ]
    )

    assert report.hallucination_count == 1, "advisory opinion must not inflate the failure count"
    assert len(report.advisory_claims) == 1
    assert "1/2 file references verified" in report.verdict


def test_globs_and_elisions_are_not_treated_as_claims():
    """Shorthand the Synthesizer writes: `tests/test_*.py`, `tests/…`, `src/{a,b}.py`."""
    draft = (
        "Tests live in `tests/test_tools_*.py` and `tests/…`.\n"
        "Config is in `src/{dev,prod}.py` or `config/<env>.yaml`.\n"
    )

    claims = extract_path_claims(draft, "onboarding")

    assert claims == [], f"shorthand must not be reported as fabrication, got {claims}"


def test_a_real_path_beside_shorthand_is_still_checked():
    draft = "See `tests/test_*.py` and also `src/sample/ghost.py`.\n"

    claims = verify_path_claims(extract_path_claims(draft, "onboarding"), make_evidence())

    assert [c.target for c in claims] == ["src/sample/ghost.py"]
    assert claims[0].grounded is False


# --------------------------------------------------------------------------- #
# Abbreviated paths: documents shorten, and shortening is not lying
# --------------------------------------------------------------------------- #


def test_an_abbreviated_path_is_recognised():
    """`run_agent.py` names the file the tools recorded as `scripts/run_agent.py`."""
    evidence = Evidence()
    evidence.record_listing(["scripts/run_agent.py", "src/repomind/agent/graph.py"])

    assert evidence.knows_path("run_agent.py")
    assert evidence.knows_path("repomind/agent/graph.py")
    assert evidence.knows_path("agent/graph.py")


def test_suffix_matching_stays_anchored_to_path_segments():
    """The looser rule must not turn into 'endswith', which grounds anything."""
    evidence = Evidence()
    evidence.record_listing(["src/my_agent.py"])

    assert evidence.knows_path("src/my_agent.py")
    assert not evidence.knows_path("agent.py"), "must not match mid-segment"
    assert not evidence.knows_path("other/my_agent.py"), "wrong parent is still wrong"


def test_a_genuinely_invented_file_is_still_caught():
    evidence = Evidence()
    evidence.record_listing(["scripts/run_agent.py"])

    assert not evidence.knows_path("scripts/deploy.py")
    assert not evidence.knows_path("database.py")


def test_technology_names_are_not_treated_as_files():
    """Found by reading a generated guide: `Node.js` was flagged unverified."""
    draft = "Works in Node.js and the browser, built with Vue.js and Next.js.\n"

    assert extract_path_claims(draft, "onboarding") == []


def test_lowercase_bare_filenames_are_still_checked():
    """The fix must not blind the Critic to real bare filenames."""
    evidence = Evidence()
    evidence.record_listing(["source/index.js"])

    claims = verify_path_claims(extract_path_claims("See index.js for details.", "d"), evidence)

    assert [c.target for c in claims] == ["index.js"]
    assert claims[0].grounded is True


def test_a_capitalised_path_with_a_separator_is_still_a_path():
    """Only *bare* tokens get the lowercase rule; `docs/README.md` is a real path."""
    targets = {c.target for c in extract_path_claims("See `docs/README.md`.", "d")}

    assert "docs/README.md" in targets
