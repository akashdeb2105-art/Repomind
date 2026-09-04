"""Agent state, and the evidence ledger the Critic checks claims against.

The ledger is the load-bearing idea in RepoMind. Every tool call the agent
makes is recorded here — which paths were listed, which files were actually
opened, which dependencies were parsed out of a real manifest. The generated
documents are then checked against *this*, not against the model's memory.

That is what turns "did the model hallucinate?" from a judgement call into a
lookup: a path either appeared in a `list_directory` result or it did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from repomind.models import (
    Claim,
    CriticReport,
    DocumentDraft,
    FileNote,
    RepoMap,
    RunUsage,
)


@dataclass
class Evidence:
    """Everything the tools actually observed, accumulated across nodes."""

    listed_paths: set[str] = field(default_factory=set)
    read_paths: set[str] = field(default_factory=set)
    file_texts: dict[str, str] = field(default_factory=dict)
    dependencies: set[str] = field(default_factory=set)
    searches: list[str] = field(default_factory=list)
    commit_subjects: list[str] = field(default_factory=list)
    readme_text: str = ""

    # -- recording ---------------------------------------------------------- #

    def record_listing(self, paths: list[str]) -> None:
        self.listed_paths.update(paths)

    def record_read(self, path: str, text: str) -> None:
        self.read_paths.add(path)
        self.listed_paths.add(path)
        self.file_texts[path] = text

    def record_dependencies(self, names: list[str]) -> None:
        self.dependencies.update(n.lower() for n in names)

    # -- querying (this is what the Critic uses) ---------------------------- #

    def knows_path(self, path: str) -> bool:
        """True if a tool actually saw this path.

        Compares normalised forms so `./src/main.py`, `src\\main.py` and
        `src/main.py` are one path — a model will produce all three spellings,
        and rejecting a real file over a backslash would make the Critic
        useless through false positives.
        """
        needle = normalise_path(path)
        if not needle:
            return False
        return any(normalise_path(known) == needle for known in self.listed_paths)

    def has_read(self, path: str) -> bool:
        needle = normalise_path(path)
        return any(normalise_path(known) == needle for known in self.read_paths)

    def has_dependency(self, name: str) -> bool:
        return name.strip().lower() in self.dependencies

    def directory_exists(self, path: str) -> bool:
        """A directory counts as known if anything under it was seen."""
        prefix = normalise_path(path).rstrip("/")
        if not prefix:
            return False
        return any(normalise_path(k).startswith(prefix + "/") for k in self.listed_paths)

    def summary(self) -> str:
        """Compact evidence digest for prompting the Critic."""
        return (
            f"paths seen: {len(self.listed_paths)}\n"
            f"files read: {len(self.read_paths)}\n"
            f"dependencies parsed: {len(self.dependencies)}"
        )


def normalise_path(path: str) -> str:
    """Canonical form for comparison: forward slashes, no ./ prefix, no quotes."""
    cleaned = path.strip().strip("`\"'").replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.strip("/")


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    A TypedDict rather than a class because LangGraph merges partial dicts
    returned by each node — a node returns only the keys it changed.
    """

    repo_path: str
    evidence: Evidence
    repo_map: RepoMap | None
    file_notes: list[FileNote]
    draft: DocumentDraft | None
    verified: DocumentDraft | None
    critic_report: CriticReport | None
    claims: list[Claim]
    usage: RunUsage
    errors: list[str]
    llm_responses: list[Any]
