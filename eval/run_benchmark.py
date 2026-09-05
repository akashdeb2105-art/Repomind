"""Benchmark RepoMind across a fixed set of real repositories.

Not part of the shipped package. This clones each repo in `repo_list.yaml`,
runs the full documentation pipeline, and records what the README's benchmark
table reports: time, tokens, which providers answered, and how much of the
generated document survived verification.

Two things about the scoring are deliberate:

* **The LLM judge is clearly labelled as one.** It reads the repo's own README
  and the generated onboarding doc and rates accuracy and usefulness 1-5. That
  is a weak signal — a model grading a model — so it is reported as one column
  among several, never as "the score".
* **The hallucination count is not a judgement at all.** It comes from the
  Critic's deterministic path check: a path either appeared in a tool result or
  it did not. That column is the one worth trusting.

Run:
    python eval/run_benchmark.py                 # everything, resuming
    python eval/run_benchmark.py --limit 3       # a quick sample
    python eval/run_benchmark.py --no-judge      # skip the LLM scoring
    python eval/run_benchmark.py --fresh         # ignore cached results
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel, Field  # noqa: E402  (imports follow sys.path setup)

from repomind.agent.graph import run_pipeline  # noqa: E402
from repomind.agent.llm import StructuredOutputError, structured_call  # noqa: E402
from repomind.agent.providers import LLMRouter  # noqa: E402
from repomind.tools import RepoContext  # noqa: E402

COLUMNS = (
    "Repo",
    "Size",
    "Lang",
    "Files",
    "Read",
    "Time",
    "Tokens",
    "Provider",
    "Paths verified",
    "Fabricated",
    "Accuracy",
    "Usefulness",
)

WORKDIR = ROOT / "eval" / "_workdir"
RESULTS = ROOT / "eval" / "_results"
CLONE_DEPTH = 20
CLONE_TIMEOUT_S = 300

JUDGE_SYSTEM = """\
You are grading a generated onboarding document for a code repository. This is \
an automated quality score, one signal among several — be strict.

You are given the repository's own README (written by its maintainers) and a \
generated onboarding guide.

accuracy (1-5): does the guide agree with the README and describe the project
  correctly? 5 = nothing contradicts the README. 1 = describes a different
  project.
usefulness (1-5): would a developer who has never seen this repository be able
  to orient themselves? 5 = names the real entry points and explains why they
  matter. 1 = generic prose that would fit any project.

A document can be accurate and useless. Score those dimensions independently.

Reply with JSON only: {"accuracy": n, "usefulness": n, "comment": "one sentence"}
"""


class JudgeScore(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    comment: str = ""


@dataclass
class RepoResult:
    name: str
    url: str
    size: str
    language: str
    files_in_repo: int = 0
    files_read: int = 0
    wall_clock_s: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    providers: list[str] = field(default_factory=list)
    path_claims: int = 0
    grounded: int = 0
    hallucinations: int = 0
    lines_removed: int = 0
    advisory_flags: int = 0
    accuracy: int | None = None
    usefulness: int | None = None
    judge_comment: str = ""
    error: str = ""
    traceback: str = ""
    # Per-node warnings from the pipeline. Without these a repo that reads zero
    # files looks identical to one that read eight — the summary line cannot
    # distinguish "nothing to read" from "every read failed".
    pipeline_errors: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    key_files: list[str] = field(default_factory=list)

    @property
    def verified_pct(self) -> str:
        if not self.path_claims:
            return "n/a"
        return f"{round(100 * self.grounded / self.path_claims)}%"


def remove_tree(path: Path) -> None:
    """Delete a checkout, including files git marked read-only.

    Best effort only. On Windows the indexer and antivirus hold handles on
    files git has just written, so even after clearing the read-only bit the
    delete can fail — which is why nothing depends on this succeeding.
    """
    if not path.exists():
        return

    def force_delete(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    try:  # 3.12 renamed the hook; 3.14 no longer accepts the old name.
        shutil.rmtree(path, onexc=force_delete)
    except TypeError:  # pragma: no cover - older interpreters
        shutil.rmtree(path, onerror=force_delete)


def clone(url: str, name: str) -> Path:
    """Clone into a fresh unique directory and return the checkout path.

    Reusing a fixed path per repo made every run depend on the previous run's
    cleanup having worked. When it did not, `git clone` refused to write into a
    non-empty directory and exited 128 with nothing on stderr. Cloning
    somewhere new sidesteps it: cleanup becomes housekeeping rather than a
    precondition for the next run.
    """
    WORKDIR.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=WORKDIR)) / "repo"
    completed = subprocess.run(
        ["git", "clone", "--depth", str(CLONE_DEPTH), url, str(destination)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=CLONE_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(stderr or f"git exited {completed.returncode}")
    return destination


def judge(router: LLMRouter, readme: str, onboarding: str) -> JudgeScore | None:
    if not readme.strip():
        return None  # nothing to grade against; an ungrounded score is worse than none
    try:
        return structured_call(
            router,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Repository README:\n{readme[:6000]}\n\n"
                        f"Generated onboarding guide:\n{onboarding[:6000]}"
                    ),
                },
            ],
            JudgeScore,
            max_tokens=1024,
        )
    except StructuredOutputError:
        return None


def benchmark_one(entry: dict, router: LLMRouter, use_judge: bool) -> RepoResult:
    result = RepoResult(
        name=entry["name"], url=entry["url"], size=entry["size"], language=entry["language"]
    )
    try:
        checkout = clone(entry["url"], entry["name"])
    except (subprocess.TimeoutExpired, RuntimeError, OSError) as exc:
        result.error = f"clone failed: {str(exc).strip()[:200]}"
        return result

    try:
        repo = RepoContext.create(checkout)
        started = time.perf_counter()
        state = run_pipeline(repo, router, use_llm_critic=True)
        result.wall_clock_s = round(time.perf_counter() - started, 1)

        evidence = state["evidence"]
        result.files_in_repo = len(evidence.file_paths)
        result.files_read = len(evidence.read_paths)

        usage = state.get("usage")
        if usage:
            result.llm_calls = len(usage.provider_calls)
            result.tokens = usage.total_tokens
            result.providers = sorted(set(usage.provider_calls))

        result.pipeline_errors = [str(e)[:200] for e in state.get("errors", [])]
        repo_map = state.get("repo_map")
        if repo_map:
            result.entry_points = list(repo_map.entry_points)
            result.key_files = list(repo_map.key_files)

        report = state.get("critic_report")
        if report:
            result.path_claims = len(report.path_claims)
            result.grounded = report.grounded_count
            result.hallucinations = report.hallucination_count
            result.lines_removed = len(report.removed_lines)
            result.advisory_flags = len(report.advisory_claims)

        draft = state.get("verified") or state.get("draft")
        if draft is None:
            result.error = "; ".join(state.get("errors", [])) or "no document produced"
            return result

        out_dir = RESULTS / entry["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ONBOARDING.md").write_text(draft.onboarding_md, encoding="utf-8")
        (out_dir / "ARCHITECTURE.md").write_text(draft.architecture_md, encoding="utf-8")

        if use_judge:
            score = judge(router, evidence.readme_text, draft.onboarding_md)
            if score:
                result.accuracy = score.accuracy
                result.usefulness = score.usefulness
                result.judge_comment = score.comment

    except Exception as exc:  # noqa: BLE001 - one bad repo must not end the run
        result.error = f"{type(exc).__name__}: {exc}"[:200]
        # Keep the frames. A one-line message cannot explain a crash six frames
        # deep, and re-running a 25-minute benchmark to learn where it broke is
        # not a debugging strategy.
        result.traceback = traceback.format_exc()[-2000:]
        print(f"    traceback:\n{result.traceback}")
    finally:
        # A leftover checkout costs disk, not correctness: the next run clones
        # somewhere new regardless.
        with contextlib.suppress(OSError):
            remove_tree(checkout.parent)

    return result


def render_markdown(results: list[RepoResult]) -> str:
    ok = [r for r in results if not r.error]
    total_tokens = sum(r.tokens for r in ok)
    total_claims = sum(r.path_claims for r in ok)
    total_grounded = sum(r.grounded for r in ok)
    total_hallucinations = sum(r.hallucinations for r in ok)
    providers = Counter(p for r in ok for p in r.providers)
    provider_mix = ", ".join(f"{name} x{n}" for name, n in providers.most_common())

    lines = [
        "# Benchmarks",
        "",
        "Generated by `python eval/run_benchmark.py`. Every run used free-tier APIs only;",
        "total spend was $0.00.",
        "",
        "## Results",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for r in results:
        if r.error:
            lines.append(
                f"| {r.name} | {r.size} | {r.language} | — | — | — | — | — | — | — | — | "
                f"error: {r.error[:40]} |"
            )
            continue
        lines.append(
            f"| {r.name} | {r.size} | {r.language} | {r.files_in_repo} | {r.files_read} | "
            f"{r.wall_clock_s}s | {r.tokens:,} | {', '.join(r.providers) or '—'} | "
            f"{r.grounded}/{r.path_claims} ({r.verified_pct}) | {r.hallucinations} | "
            f"{r.accuracy or '—'}/5 | {r.usefulness or '—'}/5 |"
        )

    verified_pct = round(100 * total_grounded / total_claims) if total_claims else 0
    lines += [
        "",
        "## Totals",
        "",
        f"- Repositories analysed: **{len(ok)}** of {len(results)}",
        f"- File references checked: **{total_claims}**, "
        f"verified: **{total_grounded}** ({verified_pct}%)",
        f"- Fabricated paths caught and removed: **{total_hallucinations}**",
        f"- Tokens used: **{total_tokens:,}**",
        "- Cost: **$0.00** (free tiers only)",
        f"- Provider mix: {provider_mix or '—'}",
        "",
        "## How to read this",
        "",
        "**Paths verified** is the column that matters, and it is not a judgement: the",
        "Critic checks every file path in the generated document against the paths tools",
        "actually observed. A path either appeared in a `list_directory` or `read_file`",
        "result or it did not. **Fabricated** counts the ones that did not and were",
        "removed from the output before it was written.",
        "",
        "**Accuracy** and **Usefulness** come from an LLM-as-judge pass and are much",
        "weaker evidence — a model grading a model. They are here because they catch",
        "something the deterministic check cannot: a document can cite only real files",
        "and still be useless. Treat them as a smoke signal, not a measurement.",
        "",
        "Neither number replaces reading the output. See the manual spot-check below.",
        "",
        "## Manually verified",
        "",
        "<!-- Fill this in after reading the generated docs yourself. Name the repos you",
        "     checked, what you compared against, and anything the automated columns",
        "     missed. Reviewer initials and date. -->",
        "",
        "_Not yet completed._",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RepoMind across real repositories.")
    parser.add_argument("--limit", type=int, default=0, help="only the first N repos")
    parser.add_argument("--only", default="", help="comma-separated repo names")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    parser.add_argument("--fresh", action="store_true", help="ignore cached results")
    parser.add_argument(
        "--sleep",
        type=float,
        default=20.0,
        help="pause between repos so free-tier per-minute limits recover",
    )
    args = parser.parse_args()

    load_dotenv()
    router = LLMRouter()
    if not router.available_providers:
        print("No providers configured. Put API keys in .env.")
        return 1

    entries = yaml.safe_load((ROOT / "eval" / "repo_list.yaml").read_text())["repos"]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        entries = [e for e in entries if e["name"] in wanted]
    if args.limit:
        entries = entries[: args.limit]

    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_path = RESULTS / "results.json"
    cached: dict[str, dict] = {}
    if cache_path.exists() and not args.fresh:
        cached = {r["name"]: r for r in json.loads(cache_path.read_text())}

    results: list[RepoResult] = []
    for index, entry in enumerate(entries, start=1):
        if entry["name"] in cached:
            print(f"[{index}/{len(entries)}] {entry['name']}: cached")
            results.append(RepoResult(**cached[entry["name"]]))
            continue

        print(f"[{index}/{len(entries)}] {entry['name']}: cloning and analysing…", flush=True)
        result = benchmark_one(entry, router, use_judge=not args.no_judge)
        results.append(result)

        if result.error:
            print(f"    error: {result.error}")
        else:
            print(
                f"    {result.files_read} files read, {result.wall_clock_s}s, "
                f"{result.tokens:,} tokens, {result.grounded}/{result.path_claims} verified, "
                f"{result.hallucinations} fabricated"
            )
            if result.files_read == 0:
                # The most damaging outcome: documents written from a file
                # listing with no code read are where fabrication comes from.
                print(f"    !! read nothing. explorer chose: {result.key_files or 'nothing'}")
            for warning in result.pipeline_errors:
                print(f"    warning: {warning}")

        # Write after every repo: a long benchmark that loses everything to a
        # rate limit at repo 11 is a benchmark nobody runs twice.
        cache_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        if index < len(entries) and args.sleep:
            time.sleep(args.sleep)

    (ROOT / "BENCHMARKS.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"\nWrote {ROOT / 'BENCHMARKS.md'}")
    print(f"Generated documents are under {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
