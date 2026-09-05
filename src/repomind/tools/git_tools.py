"""Git tools: get_git_history and get_file_blame.

Uses the `git` CLI over a library binding. Two reasons: no extra dependency,
and `--pretty=format:` with a control-character separator gives output that
parses unambiguously — commit subjects contain every punctuation mark there is,
so splitting on anything printable eventually corrupts a field.
"""

from __future__ import annotations

import subprocess

from repomind.models import BlameEntry, Commit, FileBlame, GitHistory
from repomind.tools.repo import MAX_LINE_CHARS, RepoContext, RepoError

# text=True alone decodes with the *locale* encoding, which is cp1252 on a
# default Windows install. Source code, commit messages and test output are
# full of characters cp1252 cannot represent — one curly quote in one commit
# message killed a benchmark run, because the decode failure happens on a
# reader thread and surfaces only as stdout being None six frames later.
# UTF-8 with replacement is the only safe reading of another program's output.
DECODE = {"encoding": "utf-8", "errors": "replace"}

GIT_TIMEOUT_S = 20.0
SEP = "\x1f"  # ASCII unit separator: cannot appear in a commit message
DEFAULT_HISTORY_LIMIT = 20
MAX_BLAME_LINES = 400


def _run_git(repo: RepoContext, args: list[str]) -> str:
    if not repo.is_git_repo:
        raise RepoError("not a git repository (no .git directory)")
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo.root), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
            **DECODE,
        )
    except FileNotFoundError:
        raise RepoError("git is not installed or not on PATH") from None
    except subprocess.TimeoutExpired:
        raise RepoError(f"git command timed out after {GIT_TIMEOUT_S}s") from None

    if completed.returncode != 0:
        stderr = completed.stderr or ""
        raise RepoError(f"git {' '.join(args[:2])} failed: {stderr.strip()[:300]}")
    return completed.stdout or ""


def get_git_history(
    repo: RepoContext, path: str | None = None, limit: int = DEFAULT_HISTORY_LIMIT
) -> GitHistory:
    """Recent commits, optionally scoped to one path.

    Recency and authorship are a decent proxy for which parts of a codebase are
    alive — useful for telling a newcomer where to start reading.
    """
    limit = max(1, min(limit, 200))
    args = [
        "log",
        f"--max-count={limit + 1}",  # one extra, purely to detect truncation
        f"--pretty=format:%H{SEP}%h{SEP}%an{SEP}%ae{SEP}%aI{SEP}%s",
        "--no-merges",
    ]
    if path:
        repo.resolve(path)  # containment check before handing anything to git
        args += ["--", path]

    output = _run_git(repo, args)
    commits: list[Commit] = []

    for line in output.splitlines():
        fields = line.split(SEP)
        if len(fields) != 6:
            continue
        sha, short, author, email, date, subject = fields
        commits.append(
            Commit(
                sha=sha,
                short_sha=short,
                author=author,
                email=email,
                date=date,
                subject=subject[:MAX_LINE_CHARS],
            )
        )

    truncated = len(commits) > limit
    return GitHistory(path=path, commits=commits[:limit], truncated=truncated)


def get_file_blame(repo: RepoContext, path: str, max_lines: int = MAX_BLAME_LINES) -> FileBlame:
    """Who last touched each line, plus a per-author line count."""
    target = repo.resolve(path)
    if not target.is_file():
        raise RepoError(f"not a file: {path}")

    max_lines = max(1, min(max_lines, MAX_BLAME_LINES))
    output = _run_git(repo, ["blame", "--line-porcelain", "-L", f"1,{max_lines}", "--", path])

    entries: list[BlameEntry] = []
    authors: dict[str, int] = {}
    sha = author = date = ""
    line_number = 0

    for line in output.splitlines():
        if line.startswith("\t"):
            # A tab-prefixed line is the source itself, ending one blame record.
            entries.append(
                BlameEntry(
                    line_number=line_number,
                    sha=sha[:8],
                    author=author,
                    date=date,
                    line=line[1:][:MAX_LINE_CHARS],
                )
            )
            authors[author] = authors.get(author, 0) + 1
        elif line.startswith("author "):
            author = line[len("author ") :]
        elif line.startswith("author-time "):
            date = line[len("author-time ") :]
        elif len(line) >= 40 and " " in line and all(c in "0123456789abcdef" for c in line[:40]):
            parts = line.split()
            sha = parts[0]
            if len(parts) >= 3:
                line_number = int(parts[2])

    return FileBlame(
        path=repo.relative(target),
        entries=entries,
        truncated=len(entries) >= max_lines,
        authors=dict(sorted(authors.items(), key=lambda kv: -kv[1])),
    )
