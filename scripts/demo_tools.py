"""Phase 1 acceptance check: run every MCP tool against a real repository.

Defaults to RepoMind itself — dogfooding is the fastest way to notice that a
tool returns something technically correct but useless.

Run:  python scripts/demo_tools.py [path-to-repo]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repomind.tools import (  # noqa: E402
    RepoContext,
    RepoError,
    get_dependencies,
    get_file_blame,
    get_git_history,
    get_readme,
    list_directory,
    read_file,
    run_tests,
    search_code,
)

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"


def section(title: str) -> None:
    print(f"\n{BOLD}── {title} {'─' * max(0, 60 - len(title))}{RESET}")


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent)
    repo = RepoContext.create(target)
    print(f"{BOLD}RepoMind — Phase 1 tool check{RESET}\nrepo: {repo.root}")

    section("list_directory(depth=2)")
    listing = list_directory(repo, ".", depth=2)
    for entry in listing.entries[:18]:
        marker = "/" if entry.type.value == "directory" else ""
        print(f"  {'  ' * (entry.depth - 1)}{entry.path.split('/')[-1]}{marker}")
    print(f"  {DIM}{listing.total_entries} entries, truncated={listing.truncated}{RESET}")

    section("get_dependencies()")
    for manifest in get_dependencies(repo).manifests:
        names = ", ".join(d.name for d in manifest.dependencies[:8]) or "(none parsed)"
        print(f"  {manifest.path} [{manifest.ecosystem}]: {names}")

    section("get_readme()")
    readme = get_readme(repo)
    print(f"  found={readme.found} path={readme.path}")
    if readme.found:
        print(f"  {DIM}{readme.content.splitlines()[0][:70]}{RESET}")

    section("search_code('def ')")
    result = search_code(repo, "def ", max_matches=8)
    for match in result.matches:
        print(f"  {match.path}:{match.line_number}: {match.line.strip()[:60]}")
    print(f"  {DIM}engine={result.engine} files={result.files_with_matches}{RESET}")

    section("read_file(first hit, lines 1-12)")
    if result.matches:
        content = read_file(repo, result.matches[0].path, line_range=(1, 12))
        for i, line in enumerate(content.content.splitlines(), start=content.start_line):
            print(f"  {DIM}{i:>3}{RESET} {line[:70]}")
        print(f"  {DIM}{content.total_lines} lines total{RESET}")

    section("get_git_history(limit=5)")
    try:
        for commit in get_git_history(repo, limit=5).commits:
            print(f"  {commit.short_sha}  {commit.author:<20} {commit.subject[:45]}")
    except RepoError as exc:
        print(f"  {DIM}{exc}{RESET}")

    section("get_file_blame(README.md)")
    try:
        blame = get_file_blame(repo, "README.md", max_lines=40)
        for author, lines in blame.authors.items():
            print(f"  {author}: {lines} lines")
    except RepoError as exc:
        print(f"  {DIM}{exc}{RESET}")

    section("run_tests()")
    outcome = run_tests(repo, timeout_s=180)
    if outcome.skipped_reason:
        print(f"  {DIM}skipped: {outcome.skipped_reason}{RESET}")
    else:
        verdict = f"{GREEN}passed{RESET}" if outcome.passed else f"{RED}failed{RESET}"
        print(f"  {outcome.command}\n  {verdict} in {outcome.duration_s}s")
        print(f"  {DIM}{outcome.stdout_tail.strip().splitlines()[-1][:70]}{RESET}")

    print(f"\n{GREEN}All eight tools ran.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
