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

from pydantic import BaseModel, Field


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
