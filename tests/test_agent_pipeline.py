"""End-to-end pipeline with a scripted LLM: no network, no API keys, no cost.

The fake router lets the tests script a *misbehaving* model — one that invents
files — which is the only way to prove the Critic actually protects the output.
"""

from __future__ import annotations

import json

from repomind.agent.graph import run_pipeline, write_documents
from repomind.agent.providers import LLMResponse
from repomind.tools import RepoContext


class FakeRouter:
    """Returns scripted JSON based on which node is asking."""

    def __init__(self, *, synthesizer_payload: dict, critic_claims: list | None = None):
        self.synthesizer_payload = synthesizer_payload
        self.critic_claims = critic_claims if critic_claims is not None else []
        self.calls: list[str] = []

    def complete(self, messages, **kwargs) -> LLMResponse:
        system = messages[0]["content"]

        if "Explorer stage" in system:
            self.calls.append("explorer")
            payload = {
                "summary": "A sample Python application.",
                "primary_language": "Python",
                "entry_points": ["src/sample/main.py"],
                # 'src/sample/ghost.py' does not exist: the deterministic guard
                # in the Explorer must drop it before Deep-Dive tries to open it.
                "key_files": ["src/sample/core.py", "src/sample/ghost.py"],
                "open_questions": [],
            }
        elif "Deep-Dive stage" in system:
            self.calls.append("deep_dive")
            payload = {
                "path": "unknown",
                "purpose": "Does something specific and real.",
                "key_symbols": ["Engine"],
                "depends_on": [],
            }
        elif "Synthesizer stage" in system:
            self.calls.append("synthesizer")
            payload = self.synthesizer_payload
        else:
            self.calls.append("critic")
            payload = {"claims": self.critic_claims}

        return LLMResponse(
            text=json.dumps(payload),
            provider="fake",
            model="fake-1",
            prompt_tokens=100,
            completion_tokens=50,
        )


HONEST_DRAFT = {
    "onboarding_md": (
        "# Sample\n\nStart at `src/sample/main.py`, then read `src/sample/core.py`.\n"
    ),
    "architecture_md": "# Architecture\n\n```mermaid\nflowchart TD\n  A --> B\n```\n",
}

LYING_DRAFT = {
    "onboarding_md": (
        "# Sample\n\n"
        "Start at `src/sample/main.py`.\n\n"
        "## Structure\n\n"
        "- `src/sample/main.py` — entry point\n"
        "- `src/sample/database.py` — persistence layer\n"
        "- `src/sample/api/routes.py` — HTTP handlers\n"
    ),
    "architecture_md": "# Architecture\n\nState is stored via `src/sample/database.py`.\n",
}


def run(repo: RepoContext, payload: dict, claims=None):
    router = FakeRouter(synthesizer_payload=payload, critic_claims=claims)
    state = run_pipeline(repo, router, use_llm_critic=True, use_langgraph=False)  # type: ignore[arg-type]
    return router, state


# --------------------------------------------------------------------------- #


def test_pipeline_runs_all_four_nodes(sample_repo: RepoContext):
    router, state = run(sample_repo, HONEST_DRAFT)

    assert router.calls[0] == "explorer"
    assert "deep_dive" in router.calls
    assert "synthesizer" in router.calls
    assert router.calls[-1] == "critic"
    assert state["errors"] == []


def test_explorer_drops_files_it_invented(sample_repo: RepoContext):
    """The Explorer named a file that does not exist; it must never reach Deep-Dive."""
    _, state = run(sample_repo, HONEST_DRAFT)

    assert "src/sample/ghost.py" not in state["repo_map"].key_files
    assert "src/sample/core.py" in state["repo_map"].key_files
    assert not any("ghost" in p for p in state["evidence"].read_paths)


def test_deep_dive_reads_real_files_and_records_them(sample_repo: RepoContext):
    _, state = run(sample_repo, HONEST_DRAFT)

    assert state["evidence"].has_read("src/sample/main.py")
    assert [n.path for n in state["file_notes"]], "notes should be attributed to real paths"


def test_end_to_end_hallucinations_are_stripped(sample_repo: RepoContext):
    """The whole point: a lying model must not produce a lying document."""
    _, state = run(sample_repo, LYING_DRAFT)

    output = state["verified"].onboarding_md
    report = state["critic_report"]

    assert "database.py" not in output
    assert "api/routes.py" not in output
    assert "src/sample/main.py" in output, "real content survives"
    assert report.hallucination_count >= 2
    assert len(report.removed_lines) >= 2


def test_honest_draft_passes_through_unchanged(sample_repo: RepoContext):
    _, state = run(sample_repo, HONEST_DRAFT)
    report = state["critic_report"]

    assert state["verified"].onboarding_md == HONEST_DRAFT["onboarding_md"]
    assert report.hallucination_count == 0
    assert report.removed_lines == []


def test_usage_is_recorded_for_the_benchmark_table(sample_repo: RepoContext):
    _, state = run(sample_repo, HONEST_DRAFT)
    usage = state["usage"]

    assert usage.total_tokens > 0
    assert usage.provider_calls and set(usage.provider_calls) == {"fake"}
    assert usage.wall_clock_s >= 0


def test_documents_are_written_to_disk(sample_repo: RepoContext, tmp_path):
    _, state = run(sample_repo, HONEST_DRAFT)

    written = write_documents(state, tmp_path / "out")

    assert {p.name for p in written} == {"ONBOARDING.md", "ARCHITECTURE.md"}
    assert (tmp_path / "out" / "ONBOARDING.md").read_text(encoding="utf-8").startswith("# Sample")


def test_llm_critic_findings_are_added_to_the_report(sample_repo: RepoContext):
    """The regex cannot catch invented behaviour; the LLM pass is what does."""
    extra = [
        {
            "text": "Runs on Kubernetes with a Redis cache.",
            "kind": "dependency",
            "target": "redis",
            "grounded": False,
            "reason": "no manifest mentions redis",
        }
    ]

    _, state = run(sample_repo, HONEST_DRAFT, claims=extra)

    assert any(c.target == "redis" for c in state["critic_report"].claims)


def test_a_null_document_from_the_model_does_not_crash_the_run(sample_repo: RepoContext):
    """A benchmark repo died on 'NoneType has no attribute splitlines'."""
    payload = {"onboarding_md": None, "architecture_md": "# Arch\n"}

    _, state = run(sample_repo, payload)

    assert state["verified"] is not None, "the other document should still be produced"
    assert state["verified"].onboarding_md == ""
    assert "# Arch" in state["verified"].architecture_md
