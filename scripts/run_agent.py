"""Phase 2 acceptance check: run the real agent against a real repository.

Run:  python scripts/run_agent.py [repo-path] [--out DIR] [--no-llm-critic]

Prints the verification report, because the interesting output is not the
document — it is how much of the document survived being checked.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repomind.agent.graph import run_pipeline, write_documents  # noqa: E402
from repomind.agent.providers import LLMRouter  # noqa: E402
from repomind.tools import RepoContext  # noqa: E402

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[0m",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate onboarding docs for a repository.")
    parser.add_argument("repo", nargs="?", default=".", help="path to the repository")
    parser.add_argument("--out", default="repomind-output", help="where to write the documents")
    parser.add_argument(
        "--no-llm-critic",
        action="store_true",
        help="run only the deterministic half of verification (free, instant)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo = RepoContext.create(args.repo)
    router = LLMRouter()

    if not router.available_providers:
        print(f"{RED}No providers configured.{RESET} Put API keys in .env — see .env.example.")
        return 1

    print(f"{BOLD}RepoMind{RESET}  analysing {repo.root}")
    print(f"{DIM}providers: {', '.join(router.available_providers)}{RESET}\n")

    state = run_pipeline(repo, router, use_llm_critic=not args.no_llm_critic)

    repo_map = state.get("repo_map")
    if repo_map:
        print(f"{BOLD}Explorer{RESET}  {repo_map.summary}")
        print(f"{DIM}  key files: {', '.join(repo_map.key_files) or 'none'}{RESET}")

    notes = state.get("file_notes", [])
    print(f"{BOLD}Deep-Dive{RESET}  read {len(notes)} files")
    for note in notes:
        print(f"{DIM}  {note.path}: {note.purpose[:70]}{RESET}")

    report = state.get("critic_report")
    if report:
        colour = GREEN if report.hallucination_count == 0 else YELLOW
        print(f"\n{BOLD}Critic{RESET}  {colour}{report.verdict}{RESET}")
        for claim in report.claims:
            if not claim.grounded:
                print(f"{RED}  ungrounded{RESET} {claim.target}  {DIM}({claim.reason}){RESET}")
        if report.removed_lines:
            print(f"{DIM}  removed {len(report.removed_lines)} fabricated line(s){RESET}")

    for error in state.get("errors", []):
        print(f"{YELLOW}  warning: {error}{RESET}")

    usage = state.get("usage")
    if usage:
        providers = ", ".join(sorted(set(usage.provider_calls))) or "none"
        print(
            f"\n{BOLD}Usage{RESET}  {len(usage.provider_calls)} LLM calls, "
            f"{usage.total_tokens:,} tokens, {usage.wall_clock_s}s  {DIM}via {providers}{RESET}"
        )

    written = write_documents(state, args.out)
    if not written:
        print(f"\n{RED}No documents produced.{RESET}")
        return 1

    print(f"\n{GREEN}Wrote:{RESET}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
