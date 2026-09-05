"""Pydantic schemas for every MCP tool response.

Two reasons everything is typed rather than free-form dicts:

1. MCP clients get a real schema, so Claude knows what a tool returns.
2. The Critic node in Phase 2 verifies claims against these objects. Structure
   is what makes "does this path exist?" a checkable question rather than a
   string search through prose.

Every list-shaped result carries a `truncated` flag. Silently dropping results
would let the agent conclude "there are no other matches" from a capped list —
a subtle way to manufacture a confident wrong answer.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ToolError(BaseModel):
    """A tool refused or failed. Returned instead of raising, so the agent can reason about it."""

    error: str
    detail: str = ""
    path: str | None = None


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #


# `str, Enum` rather than `StrEnum`: it serialises identically through Pydantic
# and keeps the models importable on interpreters older than 3.11.
class EntryType(str, Enum):  # noqa: UP042
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


class DirectoryEntry(BaseModel):
    path: str = Field(description="Path relative to the repository root")
    type: EntryType
    size_bytes: int | None = None
    depth: int = 0


class DirectoryListing(BaseModel):
    path: str
    entries: list[DirectoryEntry] = Field(default_factory=list)
    truncated: bool = False
    total_entries: int = 0


class FileContent(BaseModel):
    path: str
    content: str
    start_line: int = 1
    end_line: int = 0
    total_lines: int = 0
    truncated: bool = False
    size_bytes: int = 0


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class SearchMatch(BaseModel):
    path: str
    line_number: int
    line: str


class SearchResult(BaseModel):
    query: str
    matches: list[SearchMatch] = Field(default_factory=list)
    truncated: bool = False
    files_with_matches: int = 0
    engine: str = "ripgrep"


# --------------------------------------------------------------------------- #
# Git
# --------------------------------------------------------------------------- #


class Commit(BaseModel):
    sha: str
    short_sha: str
    author: str
    email: str
    date: str
    subject: str


class GitHistory(BaseModel):
    path: str | None = None
    commits: list[Commit] = Field(default_factory=list)
    truncated: bool = False


class BlameEntry(BaseModel):
    line_number: int
    sha: str
    author: str
    date: str
    line: str


class FileBlame(BaseModel):
    path: str
    entries: list[BlameEntry] = Field(default_factory=list)
    truncated: bool = False
    authors: dict[str, int] = Field(
        default_factory=dict, description="Author -> number of lines they last touched"
    )


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


class Dependency(BaseModel):
    name: str
    version_spec: str | None = None
    group: str = "main"


class Manifest(BaseModel):
    path: str
    ecosystem: str
    dependencies: list[Dependency] = Field(default_factory=list)
    parse_error: str | None = None


class DependencyReport(BaseModel):
    manifests: list[Manifest] = Field(default_factory=list)

    @property
    def ecosystems(self) -> list[str]:
        return sorted({m.ecosystem for m in self.manifests})


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestRunResult(BaseModel):
    command: str
    exit_code: int | None = None
    passed: bool = False
    timed_out: bool = False
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    detected_framework: str | None = None
    skipped_reason: str | None = None


# --------------------------------------------------------------------------- #
# Readme
# --------------------------------------------------------------------------- #


class ReadmeResult(BaseModel):
    path: str | None = None
    content: str = ""
    truncated: bool = False
    found: bool = False


# --------------------------------------------------------------------------- #
# Agent outputs (Phase 2)
# --------------------------------------------------------------------------- #


class RepoMap(BaseModel):
    """Explorer's structural read of a repository."""

    summary: str = Field(description="Two or three sentences: what this project is")
    primary_language: str = ""
    entry_points: list[str] = Field(
        default_factory=list, description="Paths a newcomer should read first"
    )
    key_files: list[str] = Field(
        default_factory=list, description="Files worth opening to understand the design"
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Things the structure alone cannot answer"
    )


class FileNote(BaseModel):
    """Deep-Dive's understanding of one file it actually read."""

    path: str
    purpose: str
    key_symbols: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(
        default_factory=list, description="Other modules in this repo that it imports"
    )


class DocumentDraft(BaseModel):
    onboarding_md: str = ""
    architecture_md: str = ""

    @field_validator("onboarding_md", "architecture_md", mode="before")
    @classmethod
    def _null_document_is_empty(cls, value: object) -> str:
        """A model that returns null for one document should not kill the run.

        Same shape as the Claim.target fix: the field is required in spirit, but
        rejecting null costs three retries and then the whole analysis, when
        writing the other document would have been useful.
        """
        return "" if value is None else str(value)


class ClaimKind(str, Enum):  # noqa: UP042
    FILE_PATH = "file_path"
    DEPENDENCY = "dependency"
    BEHAVIOUR = "behaviour"


class Claim(BaseModel):
    """One checkable assertion extracted from a draft document."""

    text: str
    kind: ClaimKind
    # Optional with a coercing validator: models return `null` here constantly
    # for behavioural claims that rest on no single path, and rejecting that
    # burned two retries per critic call on the first live run.
    target: str = Field(default="", description="The path or package name the claim rests on")
    grounded: bool = False
    reason: str = ""
    document: str = "onboarding"

    @field_validator("target", mode="before")
    @classmethod
    def _null_target_is_empty(cls, value: object) -> str:
        return "" if value is None else str(value)


class CriticReport(BaseModel):
    """What verification found, and what it did about it."""

    claims: list[Claim] = Field(default_factory=list)
    removed_lines: list[str] = Field(default_factory=list)
    flagged: list[str] = Field(default_factory=list)

    @property
    def path_claims(self) -> list[Claim]:
        """Deterministically verified. These are the findings that edit the document."""
        return [c for c in self.claims if c.kind is ClaimKind.FILE_PATH]

    @property
    def advisory_claims(self) -> list[Claim]:
        """Raised by the LLM pass: worth reading, never authoritative."""
        return [c for c in self.claims if c.kind is not ClaimKind.FILE_PATH]

    @property
    def grounded_count(self) -> int:
        return sum(1 for c in self.path_claims if c.grounded)

    @property
    def hallucination_count(self) -> int:
        return sum(1 for c in self.path_claims if not c.grounded)

    @property
    def verdict(self) -> str:
        paths = self.path_claims
        if not paths:
            return "no verifiable file references found"
        line = f"{self.grounded_count}/{len(paths)} file references verified against tool results"
        if self.advisory_claims:
            line += f" ({len(self.advisory_claims)} advisory flags from the LLM pass)"
        return line


class RunUsage(BaseModel):
    """Per-run telemetry. Feeds the benchmark table in Phase 5."""

    provider_calls: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# --------------------------------------------------------------------------- #
# Q&A (Phase 3)
# --------------------------------------------------------------------------- #


class SearchPlan(BaseModel):
    """How the Q&A node intends to find an answer before it reads anything."""

    queries: list[str] = Field(
        default_factory=list, description="Literal symbols or phrases to grep for"
    )
    files: list[str] = Field(
        default_factory=list, description="Paths worth opening based on the listing alone"
    )
    reasoning: str = ""


class Citation(BaseModel):
    """Where an answer came from. An answer without one is an opinion."""

    path: str
    line_number: int | None = None
    excerpt: str = ""
    verified: bool = False


class QAAnswer(BaseModel):
    question: str = ""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confident: bool = True
    files_consulted: list[str] = Field(default_factory=list)
    searches_run: list[str] = Field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True when at least one citation survived verification."""
        return any(c.verified for c in self.citations)
