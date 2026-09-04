"""Filesystem tools: read_file, list_directory, get_readme."""

from __future__ import annotations

from pathlib import Path

from repomind.models import (
    DirectoryEntry,
    DirectoryListing,
    EntryType,
    FileContent,
    ReadmeResult,
)
from repomind.tools.repo import MAX_FILE_BYTES, MAX_LINE_CHARS, RepoContext, RepoError

DEFAULT_MAX_LINES = 400
MAX_DIR_ENTRIES = 500
README_CANDIDATES = ("README.md", "README.rst", "README.txt", "README", "readme.md")


def read_file(
    repo: RepoContext,
    path: str,
    line_range: tuple[int, int] | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
) -> FileContent:
    """Read a text file, or a line range of one.

    Large files are read in a bounded window rather than refused outright — the
    agent usually wants the top of a file (imports, entry point, class defs),
    and returning the first N lines is far more useful than an error.
    """
    target = repo.resolve(path)

    if repo.is_secret(target):
        raise RepoError(f"refusing to read a credentials file: {path}")
    if not target.exists():
        raise RepoError(f"file does not exist: {path}")
    if target.is_dir():
        raise RepoError(f"path is a directory, use list_directory: {path}")

    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        raise RepoError(f"file is too large to read ({size:,} bytes > {MAX_FILE_BYTES:,}): {path}")
    if repo.is_probably_binary(target):
        raise RepoError(f"file appears to be binary: {path}")

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)

    if line_range:
        start, end = line_range
        start = max(1, start)
        end = min(total, end if end > 0 else total)
        if start > total:
            raise RepoError(f"start line {start} is past end of file ({total} lines): {path}")
        selected = lines[start - 1 : end]
        truncated = end < total
    else:
        start = 1
        selected = lines[:max_lines]
        end = len(selected)
        truncated = total > max_lines

    # Minified bundles are technically text but useless in a context window.
    selected = [
        ln if len(ln) <= MAX_LINE_CHARS else ln[:MAX_LINE_CHARS] + "  … [line truncated]"
        for ln in selected
    ]

    return FileContent(
        path=repo.relative(target),
        content="\n".join(selected),
        start_line=start,
        end_line=end,
        total_lines=total,
        truncated=truncated,
        size_bytes=size,
    )


def list_directory(repo: RepoContext, path: str = ".", depth: int = 2) -> DirectoryListing:
    """Walk a directory tree to a bounded depth, skipping noise directories."""
    root = repo.resolve(path)
    if not root.exists():
        raise RepoError(f"directory does not exist: {path}")
    if not root.is_dir():
        raise RepoError(f"not a directory: {path}")

    depth = max(1, min(depth, 6))
    collected: list[DirectoryEntry] = []
    total = 0
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        current, level = stack.pop(0)
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            continue

        for entry in entries:
            if repo.is_ignored(entry):
                continue
            total += 1
            if len(collected) >= MAX_DIR_ENTRIES:
                continue

            if entry.is_symlink():
                kind = EntryType.SYMLINK
                size = None
            elif entry.is_dir():
                kind = EntryType.DIRECTORY
                size = None
            else:
                kind = EntryType.FILE
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = None

            collected.append(
                DirectoryEntry(
                    path=repo.relative(entry), type=kind, size_bytes=size, depth=level + 1
                )
            )

            if kind is EntryType.DIRECTORY and level + 1 < depth:
                stack.append((entry, level + 1))

    return DirectoryListing(
        path=repo.relative(root),
        entries=collected,
        truncated=total > len(collected),
        total_entries=total,
    )


def get_readme(repo: RepoContext, max_lines: int = 300) -> ReadmeResult:
    """Fetch the repo's own README, if it has one.

    Used as a *signal*, never copied: an existing README states intent, which is
    useful context, but it is often stale or aspirational. Claims still have to
    be verified against the code.
    """
    for name in README_CANDIDATES:
        candidate = repo.root / name
        if candidate.is_file():
            content = read_file(repo, name, max_lines=max_lines)
            return ReadmeResult(
                path=content.path,
                content=content.content,
                truncated=content.truncated,
                found=True,
            )

    # Some projects keep it in docs/.
    for name in README_CANDIDATES:
        candidate = repo.root / "docs" / name
        if candidate.is_file():
            content = read_file(repo, f"docs/{name}", max_lines=max_lines)
            return ReadmeResult(
                path=content.path,
                content=content.content,
                truncated=content.truncated,
                found=True,
            )

    return ReadmeResult(found=False)
