"""search_code: ripgrep when available, pure-Python otherwise.

ripgrep is dramatically faster on large repos, but it is not guaranteed to be
installed on a user's machine — and RepoMind ships as a pip package that cannot
install a system binary. So the tool degrades to a Python scan rather than
failing, and reports which engine ran so the benchmark can account for it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

from repomind.models import SearchMatch, SearchResult
from repomind.tools.repo import IGNORED_DIRS, MAX_LINE_CHARS, RepoContext, RepoError

DEFAULT_MAX_MATCHES = 100
SEARCH_TIMEOUT_S = 20.0

# text=True alone decodes with the *locale* encoding, which is cp1252 on a
# default Windows install. Source code, commit messages and test output are
# full of characters cp1252 cannot represent — one curly quote in one commit
# message killed a benchmark run, because the decode failure happens on a
# reader thread and surfaces only as stdout being None six frames later.
# UTF-8 with replacement is the only safe reading of another program's output.
DECODE = {"encoding": "utf-8", "errors": "replace"}


def _ripgrep_available() -> bool:
    return shutil.which("rg") is not None


def search_code(
    repo: RepoContext,
    query: str,
    path_glob: str | None = None,
    max_matches: int = DEFAULT_MAX_MATCHES,
    regex: bool = False,
) -> SearchResult:
    """Search the repository for `query`.

    `regex=False` (the default) treats the query as a literal string. An agent
    composing search terms from natural language will otherwise produce things
    like ``def main(`` and get a regex syntax error rather than results.
    """
    if not query.strip():
        raise RepoError("search query is empty")

    max_matches = max(1, min(max_matches, 500))
    if _ripgrep_available():
        return _search_ripgrep(repo, query, path_glob, max_matches, regex)
    return _search_python(repo, query, path_glob, max_matches, regex)


def _search_ripgrep(
    repo: RepoContext, query: str, path_glob: str | None, max_matches: int, regex: bool
) -> SearchResult:
    command = [
        "rg",
        "--json",
        "--line-number",
        "--no-heading",
        "--max-count",
        str(max_matches + 1),
        "--max-filesize",
        "512K",
    ]
    if not regex:
        command.append("--fixed-strings")
    for ignored in sorted(IGNORED_DIRS):
        command += ["--glob", f"!**/{ignored}/**"]
    if path_glob:
        command += ["--glob", path_glob]
    command += ["--", query, str(repo.root)]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SEARCH_TIMEOUT_S,
            check=False,
            **DECODE,
        )
    except subprocess.TimeoutExpired:
        raise RepoError(f"search timed out after {SEARCH_TIMEOUT_S}s: {query}") from None

    # rg exits 1 for "no matches", which is a valid empty result, not an error.
    if completed.returncode not in (0, 1):
        raise RepoError(f"ripgrep failed: {(completed.stderr or '')[:300]}")

    matches: list[SearchMatch] = []
    files: set[str] = set()
    truncated = False

    for raw in (completed.stdout or "").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue

        data = event["data"]
        absolute = data["path"].get("text")
        if not absolute:
            continue
        try:
            relative = repo.relative(repo.root / absolute)
        except (ValueError, OSError):
            continue

        if len(matches) >= max_matches:
            truncated = True
            break

        files.add(relative)
        matches.append(
            SearchMatch(
                path=relative,
                line_number=data.get("line_number") or 0,
                line=(data["lines"].get("text") or "").rstrip("\n")[:MAX_LINE_CHARS],
            )
        )

    return SearchResult(
        query=query,
        matches=matches,
        truncated=truncated,
        files_with_matches=len(files),
        engine="ripgrep",
    )


def _search_python(
    repo: RepoContext, query: str, path_glob: str | None, max_matches: int, regex: bool
) -> SearchResult:
    pattern = re.compile(query if regex else re.escape(query))
    matches: list[SearchMatch] = []
    files: set[str] = set()
    truncated = False

    for path in repo.walk_files():
        relative = repo.relative(path)
        if path_glob and not path.match(path_glob):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for number, line in enumerate(text.splitlines(), start=1):
            if not pattern.search(line):
                continue
            if len(matches) >= max_matches:
                truncated = True
                break
            files.add(relative)
            matches.append(
                SearchMatch(path=relative, line_number=number, line=line[:MAX_LINE_CHARS])
            )
        if truncated:
            break

    return SearchResult(
        query=query,
        matches=matches,
        truncated=truncated,
        files_with_matches=len(files),
        engine="python",
    )
