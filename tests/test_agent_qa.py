"""Q&A: targeted retrieval, and citations that are checked rather than trusted."""

from __future__ import annotations

import json

import pytest

from repomind.agent.providers import LLMResponse
from repomind.agent.qa import answer_question
from repomind.tools import RepoContext, RepoError


class QARouter:
    """Scripts the planner and the answerer separately."""

    def __init__(self, *, plan: dict, answer: dict, fail_plan: bool = False):
        self.plan = plan
        self.answer = answer
        self.fail_plan = fail_plan
        self.calls: list[str] = []

    def complete(self, messages, **kwargs) -> LLMResponse:
        system = messages[0]["content"]
        if "planning how to answer" in system:
            self.calls.append("plan")
            payload = "not json" if self.fail_plan else json.dumps(self.plan)
        else:
            self.calls.append("answer")
            payload = json.dumps(self.answer)
        return LLMResponse(
            text=payload, provider="fake", model="m", prompt_tokens=10, completion_tokens=5
        )


GOOD_PLAN = {"queries": ["class Engine"], "files": ["src/sample/core.py"], "reasoning": "…"}
GOOD_ANSWER = {
    "answer": "The Engine class in src/sample/core.py does the work; run() returns 0.",
    "citations": [{"path": "src/sample/core.py", "line_number": 1, "excerpt": "class Engine:"}],
    "confident": True,
}


def test_answers_a_question_from_retrieved_code(sample_repo: RepoContext):
    router = QARouter(plan=GOOD_PLAN, answer=GOOD_ANSWER)

    result = answer_question(sample_repo, router, "What does the Engine class do?")  # type: ignore[arg-type]

    assert "Engine" in result.answer
    assert result.confident is True
    assert "src/sample/core.py" in result.files_consulted
    assert router.calls == ["plan", "answer"], "exactly two LLM calls: plan, then answer"


def test_citations_to_files_that_were_read_are_verified(sample_repo: RepoContext):
    router = QARouter(plan=GOOD_PLAN, answer=GOOD_ANSWER)

    result = answer_question(sample_repo, router, "What does Engine do?")  # type: ignore[arg-type]

    assert result.citations[0].verified is True
    assert result.is_grounded is True


def test_a_citation_to_an_unread_file_is_marked_unverified(sample_repo: RepoContext):
    """An invented source is worse than none: it looks like diligence."""
    router = QARouter(
        plan=GOOD_PLAN,
        answer={
            "answer": "Configuration is loaded in src/sample/settings.py.",
            "citations": [{"path": "src/sample/settings.py", "line_number": 4, "excerpt": "…"}],
            "confident": True,
        },
    )

    result = answer_question(sample_repo, router, "Where is config loaded?")  # type: ignore[arg-type]

    assert result.citations[0].verified is False
    assert result.is_grounded is False
    assert result.confident is False, "unverifiable sources must lower confidence"


def test_qa_reads_far_fewer_files_than_the_full_pipeline(sample_repo: RepoContext):
    """The whole point of a separate entry point: don't walk the repo for one question."""
    router = QARouter(plan=GOOD_PLAN, answer=GOOD_ANSWER)

    result = answer_question(sample_repo, router, "What does Engine do?")  # type: ignore[arg-type]

    assert 0 < len(result.files_consulted) <= 4


def test_no_matches_produces_an_honest_non_answer(sample_repo: RepoContext):
    router = QARouter(
        plan={"queries": ["ZZZNotInThisRepo"], "files": [], "reasoning": "…"},
        answer=GOOD_ANSWER,
    )

    result = answer_question(sample_repo, router, "How does the Kubernetes operator work?")  # type: ignore[arg-type]

    assert result.confident is False
    assert "could not find" in result.answer.lower()
    assert router.calls == ["plan"], "no point asking the model to answer with no evidence"


def test_planner_failure_falls_back_to_keywords(sample_repo: RepoContext):
    """One bad LLM reply should degrade the search, not end the run."""
    router = QARouter(plan=GOOD_PLAN, answer=GOOD_ANSWER, fail_plan=True)

    result = answer_question(sample_repo, router, "What does the Engine class do here?")  # type: ignore[arg-type]

    assert result.searches_run, "fell back to keywords from the question"
    assert result.answer


def test_empty_question_is_rejected(sample_repo: RepoContext):
    router = QARouter(plan=GOOD_PLAN, answer=GOOD_ANSWER)

    with pytest.raises(RepoError, match="empty"):
        answer_question(sample_repo, router, "   ")  # type: ignore[arg-type]


def test_secrets_are_never_retrieved_as_context(sample_repo: RepoContext):
    """The credential guard has to hold on this path too, not just in the docs pipeline."""
    (sample_repo.root / ".env").write_text("GROQ_API_KEY=gsk_secret\n", encoding="utf-8")
    router = QARouter(
        plan={"queries": ["GROQ_API_KEY"], "files": [".env"], "reasoning": "…"},
        answer=GOOD_ANSWER,
    )

    result = answer_question(sample_repo, router, "What is the API key?")  # type: ignore[arg-type]

    assert ".env" not in result.files_consulted
    assert "gsk_secret" not in result.answer
