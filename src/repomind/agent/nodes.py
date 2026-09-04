"""The four agent nodes: Explorer, Deep-Dive, Synthesizer, Critic.

Each node is a plain function of state, so it can be tested without LangGraph
installed and without touching a network. `graph.py` only wires them together.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from repomind.agent.llm import StructuredOutputError, structured_call
from repomind.agent.prompts import (
    CRITIC_SYSTEM,
    DEEP_DIVE_SYSTEM,
    EXPLORER_SYSTEM,
    SYNTHESIZER_SYSTEM,
)
from repomind.agent.providers import LLMRouter
from repomind.agent.state import AgentState, Evidence, normalise_path
from repomind.models import (
    Claim,
    ClaimKind,
    CriticReport,
    DocumentDraft,
    FileNote,
    RepoMap,
)
from repomind.tools import (
    RepoContext,
    RepoError,
    get_dependencies,
    get_git_history,
    get_readme,
    list_directory,
    read_file,
)

logger = logging.getLogger("repomind.agent")

MAX_LISTED_PATHS_IN_PROMPT = 300
MAX_DEEP_DIVE_FILES = 6
MAX_FILE_CHARS_IN_PROMPT = 8_000

Node = Callable[[AgentState], AgentState]

# Paths inside backticks, plus bare filenames with a known code extension.
_BACKTICKED = re.compile(r"`([^`\n]{2,120})`")
_BARE_PATH = re.compile(
    r"\b((?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|rb|java|toml|cfg|ini|"
    r"json|ya?ml|md|txt|sh|lock))\b"
)
_CODE_EXTENSIONS = (
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".rb",
    ".java",
    ".toml",
    ".cfg",
    ".ini",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".sh",
)


# --------------------------------------------------------------------------- #
# Explorer
# --------------------------------------------------------------------------- #


def make_explorer(repo: RepoContext, router: LLMRouter) -> Node:
    """Build a structural map from cheap, broad tool calls. Reads no code."""

    def explorer(state: AgentState) -> AgentState:
        evidence: Evidence = state["evidence"]

        listing = list_directory(repo, ".", depth=3)
        evidence.record_listing([e.path for e in listing.entries])

        report = get_dependencies(repo)
        dependency_names = [d.name for m in report.manifests for d in m.dependencies]
        evidence.record_dependencies(dependency_names)
        evidence.record_listing([m.path for m in report.manifests])

        readme = get_readme(repo, max_lines=120)
        if readme.found:
            evidence.readme_text = readme.content
            evidence.record_listing([readme.path or "README.md"])

        try:
            history = get_git_history(repo, limit=10)
            evidence.commit_subjects = [c.subject for c in history.commits]
        except RepoError:
            pass  # not a git repo, or git unavailable: not fatal

        paths = sorted(evidence.listed_paths)[:MAX_LISTED_PATHS_IN_PROMPT]
        deps = ", ".join(sorted(evidence.dependencies)) or "none"
        user = (
            f"Directory listing ({len(paths)} paths):\n" + "\n".join(paths) + "\n\n"
            f"Dependencies parsed from manifests: {deps}\n\n"
            f"README (may be absent or stale):\n{evidence.readme_text[:3000] or '(no README)'}\n\n"
            f"Recent commit subjects:\n"
            + ("\n".join(evidence.commit_subjects[:10]) or "(none)")
            + "\n\nReturn JSON with keys: summary, primary_language, entry_points, "
            "key_files, open_questions."
        )

        try:
            repo_map = structured_call(
                router,
                [{"role": "system", "content": EXPLORER_SYSTEM}, {"role": "user", "content": user}],
                RepoMap,
                usage_sink=state.get("llm_responses"),
            )
        except StructuredOutputError as exc:
            return {"errors": [*state.get("errors", []), f"explorer: {exc}"], "repo_map": None}

        # Deterministic guard: the Explorer is told not to invent paths, but
        # telling is not enforcing. Drop anything the tools never saw, so
        # Deep-Dive is never sent hunting for a file that does not exist.
        repo_map.entry_points = [p for p in repo_map.entry_points if evidence.knows_path(p)]
        repo_map.key_files = [p for p in repo_map.key_files if evidence.knows_path(p)]

        return {"repo_map": repo_map, "evidence": evidence}

    return explorer


# --------------------------------------------------------------------------- #
# Deep-Dive
# --------------------------------------------------------------------------- #


def make_deep_dive(repo: RepoContext, router: LLMRouter) -> Node:
    """Actually open the files Explorer flagged. Filenames lie; contents do not."""

    def deep_dive(state: AgentState) -> AgentState:
        evidence: Evidence = state["evidence"]
        repo_map: RepoMap | None = state.get("repo_map")
        if repo_map is None:
            return {"file_notes": []}

        targets: list[str] = []
        for path in [*repo_map.entry_points, *repo_map.key_files]:
            if path not in targets:
                targets.append(path)
        targets = targets[:MAX_DEEP_DIVE_FILES]

        notes: list[FileNote] = []
        errors = list(state.get("errors", []))

        for path in targets:
            try:
                content = read_file(repo, path, max_lines=300)
            except RepoError as exc:
                errors.append(f"deep_dive: could not read {path}: {exc}")
                continue

            evidence.record_read(content.path, content.content)

            user = (
                f"File: {content.path}\n"
                f"({content.total_lines} lines total"
                f"{', truncated' if content.truncated else ''})\n\n"
                f"```\n{content.content[:MAX_FILE_CHARS_IN_PROMPT]}\n```\n\n"
                "Return JSON with keys: path, purpose, key_symbols, depends_on."
            )
            try:
                note = structured_call(
                    router,
                    [
                        {"role": "system", "content": DEEP_DIVE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    FileNote,
                    usage_sink=state.get("llm_responses"),
                )
            except StructuredOutputError as exc:
                errors.append(f"deep_dive: {path}: {exc}")
                continue

            note.path = content.path  # trust the tool's path, not the model's
            notes.append(note)

        return {"file_notes": notes, "evidence": evidence, "errors": errors}

    return deep_dive


# --------------------------------------------------------------------------- #
# Synthesizer
# --------------------------------------------------------------------------- #


def make_synthesizer(router: LLMRouter) -> Node:
    """Draft ONBOARDING.md and ARCHITECTURE.md from what the earlier nodes found."""

    def synthesizer(state: AgentState) -> AgentState:
        evidence: Evidence = state["evidence"]
        repo_map: RepoMap | None = state.get("repo_map")
        notes: list[FileNote] = state.get("file_notes", [])

        if repo_map is None:
            return {"draft": None}

        note_block = "\n\n".join(
            f"### {n.path}\npurpose: {n.purpose}\n"
            f"symbols: {', '.join(n.key_symbols) or 'none listed'}\n"
            f"imports from this repo: {', '.join(n.depends_on) or 'none listed'}"
            for n in notes
        )

        user = (
            f"Project summary from the structural pass:\n{repo_map.summary}\n\n"
            f"Primary language: {repo_map.primary_language or 'unknown'}\n\n"
            f"Files that were READ (only these may have their behaviour described):\n"
            f"{note_block or '(none were read)'}\n\n"
            f"All paths observed in the repository:\n"
            f"{chr(10).join(sorted(evidence.listed_paths)[:MAX_LISTED_PATHS_IN_PROMPT])}\n\n"
            f"Dependencies from manifests: {', '.join(sorted(evidence.dependencies)) or 'none'}\n\n"
            f"Existing README (a signal, possibly stale — do not copy):\n"
            f"{evidence.readme_text[:2500] or '(none)'}\n\n"
            "Return JSON with exactly two keys: onboarding_md and architecture_md."
        )

        try:
            draft = structured_call(
                router,
                [
                    {"role": "system", "content": SYNTHESIZER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                DocumentDraft,
                max_tokens=6000,
                usage_sink=state.get("llm_responses"),
            )
        except StructuredOutputError as exc:
            return {"draft": None, "errors": [*state.get("errors", []), f"synthesizer: {exc}"]}

        return {"draft": draft}

    return synthesizer


# --------------------------------------------------------------------------- #
# Critic — the anti-hallucination guardrail
# --------------------------------------------------------------------------- #


def extract_path_claims(text: str, document: str) -> list[Claim]:
    """Pull every path-shaped token out of a document.

    Deliberately deterministic. Asking a model "did you make up any paths?" is
    asking the thing that hallucinated to notice it hallucinated. A regex plus a
    set lookup cannot be talked out of its answer.
    """
    candidates: list[str] = []
    for match in _BACKTICKED.finditer(text):
        token = match.group(1).strip()
        if token.endswith(_CODE_EXTENSIONS) or ("/" in token and " " not in token):
            candidates.append(token)
    candidates.extend(match.group(1) for match in _BARE_PATH.finditer(text))

    claims: list[Claim] = []
    seen: set[str] = set()
    for raw in candidates:
        target = normalise_path(raw)
        if not target or target in seen:
            continue
        if target.startswith(("http://", "https://", "www.")):
            continue
        seen.add(target)
        claims.append(Claim(text=raw, kind=ClaimKind.FILE_PATH, target=target, document=document))
    return claims


def verify_path_claims(claims: list[Claim], evidence: Evidence) -> list[Claim]:
    """Check each path against what the tools actually observed."""
    for claim in claims:
        if evidence.knows_path(claim.target) or evidence.directory_exists(claim.target):
            claim.grounded = True
            claim.reason = "observed in a tool result"
        else:
            claim.grounded = False
            claim.reason = "no tool call ever saw this path"
    return claims


def apply_corrections(markdown: str, claims: list[Claim]) -> tuple[str, list[str], list[str]]:
    """Strip or flag ungrounded claims, returning (corrected, removed, flagged).

    A whole line is removed when it exists only to talk about a path that does
    not exist — usually a bullet in a project-structure list. Otherwise the
    fabrication is marked inline, because deleting a sentence mid-paragraph
    reads worse than admitting it could not be verified.
    """
    bad = [c for c in claims if not c.grounded]
    if not bad:
        return markdown, [], []

    removed: list[str] = []
    flagged: list[str] = []
    output: list[str] = []

    for line in markdown.splitlines():
        hits = [c for c in bad if c.text in line or c.target in line]
        if not hits:
            output.append(line)
            continue

        stripped = line.strip()
        is_list_item = stripped.startswith(("-", "*", "+")) or bool(
            re.match(r"^\d+[.)]\s", stripped)
        )
        if is_list_item:
            removed.append(stripped)
            continue

        targets = ", ".join(sorted({c.target for c in hits}))
        output.append(f"{line}  <!-- unverified: {targets} -->")
        flagged.append(stripped)

    return "\n".join(output), removed, flagged


def make_critic(router: LLMRouter, use_llm: bool = True) -> Node:
    """Verify the draft against the evidence ledger, then correct it."""

    def critic(state: AgentState) -> AgentState:
        evidence: Evidence = state["evidence"]
        draft: DocumentDraft | None = state.get("draft")
        if draft is None:
            return {"verified": None, "critic_report": CriticReport()}

        claims = verify_path_claims(
            extract_path_claims(draft.onboarding_md, "onboarding")
            + extract_path_claims(draft.architecture_md, "architecture"),
            evidence,
        )

        onboarding, removed_a, flagged_a = apply_corrections(
            draft.onboarding_md, [c for c in claims if c.document == "onboarding"]
        )
        architecture, removed_b, flagged_b = apply_corrections(
            draft.architecture_md, [c for c in claims if c.document == "architecture"]
        )

        report = CriticReport(
            claims=claims,
            removed_lines=removed_a + removed_b,
            flagged=flagged_a + flagged_b,
        )

        # The LLM pass catches what a regex cannot: invented behaviour, install
        # commands, versions. It runs second, and only ever adds findings — the
        # deterministic verdict on paths is never overridden by a model.
        if use_llm:
            try:
                extra = _llm_review(router, state, onboarding)
                report.claims.extend(extra)
            except StructuredOutputError as exc:
                logger.warning("critic LLM pass failed, keeping deterministic result: %s", exc)

        return {
            "verified": DocumentDraft(onboarding_md=onboarding, architecture_md=architecture),
            "critic_report": report,
            "claims": report.claims,
        }

    return critic


class _ClaimList(BaseModel):
    """Wrapper so the model returns an object, not a bare array."""

    claims: list[Claim] = Field(default_factory=list)


def _llm_review(router: LLMRouter, state: AgentState, document: str) -> list[Claim]:
    evidence: Evidence = state["evidence"]
    user = (
        f"Paths ACTUALLY observed by tools:\n"
        f"{chr(10).join(sorted(evidence.listed_paths)[:200])}\n\n"
        f"Files ACTUALLY read:\n{chr(10).join(sorted(evidence.read_paths)) or '(none)'}\n\n"
        f"Dependencies parsed from real manifests:\n"
        f"{', '.join(sorted(evidence.dependencies)) or '(none)'}\n\n"
        f"Draft document to verify:\n{document[:12000]}\n\n"
        'Return JSON: {"claims": [{"text": ..., "kind": "behaviour"|"dependency"|"file_path", '
        '"target": ..., "grounded": true|false, "reason": ...}]}. '
        "Include only claims you judge UNGROUNDED."
    )
    result = structured_call(
        router,
        [{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": user}],
        _ClaimList,
        usage_sink=state.get("llm_responses"),
    )
    for claim in result.claims:
        claim.grounded = False
    return result.claims
