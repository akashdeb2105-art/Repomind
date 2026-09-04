"""Q&A: answer a question about a codebase with targeted retrieval.

A separate entry point from the documentation pipeline, and deliberately so.
Generating docs means understanding a repository broadly; answering "where is
rate limiting handled?" means finding three files and ignoring the rest. Running
the full Explorer -> Deep-Dive walk for a single question would be slow, would
burn free-tier quota, and would bury the relevant code in irrelevant context.

The shape is plan -> retrieve -> answer -> verify:

1. The model proposes search terms from the question and the directory listing.
   It has read no code yet, so this is cheap and often wrong in useful ways.
2. Tools run those searches and open the files that matched. Deterministic.
3. The model answers using only what was retrieved.
4. Citations are checked against the evidence ledger, exactly as the Critic
   checks the generated documents. An answer citing a file nobody opened is
   marked unverified rather than presented as fact.
"""

from __future__ import annotations

import logging

from repomind.agent.llm import StructuredOutputError, structured_call
from repomind.agent.providers import LLMRouter
from repomind.agent.state import Evidence
from repomind.models import QAAnswer, SearchPlan
from repomind.tools import RepoContext, RepoError, list_directory, read_file, search_code

logger = logging.getLogger("repomind.qa")

MAX_QUERIES = 4
MAX_FILES_TO_READ = 4
MAX_MATCHES_PER_QUERY = 12
FILE_EXCERPT_LINES = 200

PLANNER_SYSTEM = """\
You are planning how to answer a question about a codebase. You have NOT read \
any code yet — you have only a directory listing.

Produce search terms and candidate files. Good search terms are things that \
literally appear in source code: function names, class names, decorators, \
config keys, error message fragments. Bad search terms are conceptual phrases \
that no programmer would type, like "handles authentication logic".

Prefer several narrow terms over one broad one. Name at most 4 queries and 4 \
files, and only files that appear in the listing.

Reply with JSON only: {"queries": [...], "files": [...], "reasoning": "..."}.
"""

ANSWER_SYSTEM = """\
You are answering a question about a codebase using ONLY the code excerpts \
provided. You cannot see anything else.

Rules:
- Every specific claim must be traceable to an excerpt you were shown.
- Cite the file (and line, where you can tell) for each claim.
- If the excerpts do not answer the question, say so plainly and set
  confident=false. Name what you would need to look at instead. A truthful
  "I could not find this" is worth more than a confident guess — the guess is
  what makes tools like this untrustworthy.
- Be concrete. Name the functions and classes that do the work.

Reply with JSON only:
{"answer": "...", "citations": [{"path": "...", "line_number": 1,
"excerpt": "..."}], "confident": true}
"""


def answer_question(
    repo: RepoContext,
    router: LLMRouter,
    question: str,
    *,
    usage_sink: list | None = None,
) -> QAAnswer:
    """Answer `question` about `repo`, with verified citations."""
    if not question.strip():
        raise RepoError("question is empty")

    evidence = Evidence()
    listing = list_directory(repo, ".", depth=5)
    evidence.record_listing(
        [e.path for e in listing.entries],
        files_only=[e.path for e in listing.entries if e.type.value == "file"],
    )

    plan = _plan(router, question, sorted(evidence.file_paths), usage_sink)
    excerpts, searches = _retrieve(repo, evidence, plan)

    if not excerpts:
        return QAAnswer(
            question=question,
            answer=(
                "I could not find anything in this repository that answers that. "
                f"Searched for: {', '.join(searches) or 'nothing'}."
            ),
            confident=False,
            searches_run=searches,
        )

    context = "\n\n".join(f"### {path}\n```\n{text}\n```" for path, text in excerpts.items())
    try:
        answer = structured_call(
            router,
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nCode excerpts:\n\n{context}",
                },
            ],
            QAAnswer,
            max_tokens=2048,
            usage_sink=usage_sink,
        )
    except StructuredOutputError as exc:
        return QAAnswer(
            question=question,
            answer=f"The model could not produce a usable answer: {exc}",
            confident=False,
            searches_run=searches,
            files_consulted=sorted(excerpts),
        )

    answer.question = question
    answer.searches_run = searches
    answer.files_consulted = sorted(excerpts)
    _verify_citations(answer, evidence)
    return answer


def _plan(
    router: LLMRouter, question: str, files: list[str], usage_sink: list | None
) -> SearchPlan:
    try:
        plan = structured_call(
            router,
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Files in the repository:\n" + "\n".join(files[:300])
                    ),
                },
            ],
            SearchPlan,
            max_tokens=1024,
            usage_sink=usage_sink,
        )
    except StructuredOutputError as exc:
        # Falling back to the question's own words is a poor search, but a poor
        # search beats no answer — and it keeps one bad LLM reply from ending the run.
        logger.warning("planning failed (%s); falling back to keywords from the question", exc)
        keywords = [w for w in question.split() if len(w) > 4][:MAX_QUERIES]
        return SearchPlan(queries=keywords, reasoning="fallback: keywords from the question")

    plan.queries = plan.queries[:MAX_QUERIES]
    plan.files = plan.files[:MAX_FILES_TO_READ]
    return plan


def _retrieve(
    repo: RepoContext, evidence: Evidence, plan: SearchPlan
) -> tuple[dict[str, str], list[str]]:
    """Run the plan's searches and open what matched, most-hit files first."""
    hits: dict[str, int] = {}
    searches: list[str] = []

    for query in plan.queries:
        try:
            result = search_code(repo, query, max_matches=MAX_MATCHES_PER_QUERY)
        except RepoError as exc:
            logger.info("search %r failed: %s", query, exc)
            continue
        searches.append(query)
        for match in result.matches:
            hits[match.path] = hits.get(match.path, 0) + 1

    # Files the planner named come first: it saw the listing and made a
    # judgement, which is worth more than raw match counts.
    ordered = [p for p in plan.files if evidence.is_file(p)]
    ordered += [p for p, _ in sorted(hits.items(), key=lambda kv: -kv[1]) if p not in ordered]

    excerpts: dict[str, str] = {}
    for path in ordered[:MAX_FILES_TO_READ]:
        try:
            content = read_file(repo, path, max_lines=FILE_EXCERPT_LINES)
        except RepoError as exc:
            logger.info("could not read %s: %s", path, exc)
            continue
        evidence.record_read(content.path, content.content)
        excerpts[content.path] = content.content

    return excerpts, searches


def _verify_citations(answer: QAAnswer, evidence: Evidence) -> None:
    """Mark citations that point at files the tools actually opened.

    Same guarantee as the Critic, applied to answers: a citation to a file
    nobody read is an invented source, and an invented source is worse than no
    source at all because it looks like diligence.
    """
    for citation in answer.citations:
        citation.verified = evidence.has_read(citation.path)

    if answer.citations and not answer.is_grounded:
        answer.confident = False
