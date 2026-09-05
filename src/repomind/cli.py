"""`repomind` — the command-line interface.

Four commands, matching the four things the project does:

    repomind analyze <repo>        generate ONBOARDING.md and ARCHITECTURE.md
    repomind ask "<question>"      answer a question about a codebase
    repomind check                 verify the free-tier providers work
    repomind tools <repo>          run the MCP tools directly, no LLM involved

`analyze` accepts a GitHub URL as well as a local path, cloning shallowly into a
temporary directory. `--depth 1` keeps a large repository's clone fast; the git
history tool degrades to what is there rather than failing.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from repomind import __version__
from repomind.agent.graph import run_pipeline, write_documents
from repomind.agent.providers import PROVIDER_CONFIGS, LLMRouter, Provider
from repomind.agent.qa import answer_question
from repomind.tools import RepoContext, get_dependencies, list_directory

app = typer.Typer(
    name="repomind",
    help="Make any repository self-explaining: onboarding docs, architecture, and answers.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
# stdout is the MCP transport when serving, so status messages need their own
# stream. Rich has no per-call file argument — it is a property of the Console.
err_console = Console(stderr=True)

CLONE_TIMEOUT_S = 300.0


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"repomind {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    load_dotenv()


def _open_repo(target: str, clone_depth: int) -> tuple[RepoContext, Path | None]:
    """Resolve a local path or clone a URL. Returns (repo, temp_dir_to_clean)."""
    if not target.startswith(("http://", "https://", "git@")):
        return RepoContext.create(target), None

    temp_dir = Path(tempfile.mkdtemp(prefix="repomind-"))
    console.print(f"[dim]cloning {target}…[/dim]")
    try:
        subprocess.run(
            ["git", "clone", "--depth", str(clone_depth), "--quiet", target, str(temp_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_S,
        )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise typer.BadParameter(f"clone failed: {exc.stderr.strip()[:200]}") from exc
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise typer.BadParameter(f"clone timed out after {CLONE_TIMEOUT_S}s") from exc

    return RepoContext.create(temp_dir), temp_dir


def _router_or_exit() -> LLMRouter:
    router = LLMRouter()
    if not router.available_providers:
        console.print(
            "[red]No LLM providers configured.[/red] Copy .env.example to .env and add a "
            "free API key from console.groq.com, aistudio.google.com or openrouter.ai."
        )
        raise typer.Exit(1)
    return router


# --------------------------------------------------------------------------- #


@app.command()
def analyze(
    repo: Annotated[str, typer.Argument(help="Local path or GitHub URL")] = ".",
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the documents")] = Path(
        "repomind-output"
    ),
    depth: Annotated[int, typer.Option(help="Clone depth for URLs")] = 20,
    no_llm_critic: Annotated[
        bool, typer.Option("--no-llm-critic", help="Deterministic verification only (free)")
    ] = False,
) -> None:
    """Generate ONBOARDING.md and ARCHITECTURE.md for a repository."""
    context, temp_dir = _open_repo(repo, depth)
    router = _router_or_exit()

    try:
        console.print(f"[bold]Analysing[/bold] {context.root}")
        state = run_pipeline(context, router, use_llm_critic=not no_llm_critic)

        notes = state.get("file_notes", [])
        console.print(f"[dim]read {len(notes)} files[/dim]")

        report = state.get("critic_report")
        if report:
            colour = "green" if report.hallucination_count == 0 else "yellow"
            console.print(f"[{colour}]Critic:[/{colour}] {report.verdict}")
            for claim in report.path_claims:
                if not claim.grounded:
                    console.print(f"  [red]fabricated:[/red] {claim.target}")
            for claim in report.advisory_claims:
                console.print(f"  [yellow]advisory:[/yellow] {claim.text[:70]}")

        for error in state.get("errors", []):
            console.print(f"  [yellow]warning:[/yellow] {error}")

        usage = state.get("usage")
        if usage:
            console.print(
                f"[dim]{len(usage.provider_calls)} calls, {usage.total_tokens:,} tokens, "
                f"{usage.wall_clock_s}s, $0.00 (free tier)[/dim]"
            )

        written = write_documents(state, out)
        if not written:
            console.print("[red]No documents were produced.[/red]")
            raise typer.Exit(1)
        for path in written:
            console.print(f"[green]wrote[/green] {path}")
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="What do you want to know?")],
    repo: Annotated[str, typer.Option("--repo", "-r", help="Local path or GitHub URL")] = ".",
    depth: Annotated[int, typer.Option(help="Clone depth for URLs")] = 20,
) -> None:
    """Answer a question about a codebase, with verified citations."""
    context, temp_dir = _open_repo(repo, depth)
    router = _router_or_exit()

    try:
        with console.status("searching…"):
            result = answer_question(context, router, question)

        console.print(Panel(result.answer, title=question, border_style="cyan"))

        if result.citations:
            table = Table("source", "line", "verified", box=None, pad_edge=False)
            for citation in result.citations:
                table.add_row(
                    citation.path,
                    str(citation.line_number or "—"),
                    "[green]yes[/green]" if citation.verified else "[red]NO[/red]",
                )
            console.print(table)

        if not result.confident:
            console.print(
                "[yellow]Low confidence — treat this answer as a starting point.[/yellow]"
            )
        console.print(
            f"[dim]searched: {', '.join(result.searches_run) or 'nothing'} | "
            f"read: {', '.join(result.files_consulted) or 'nothing'}[/dim]"
        )
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.command()
def check() -> None:
    """Check that the free-tier providers answer, and that fallback works."""
    table = Table("provider", "model", "status", "latency", box=None)
    working: list[str] = []

    for name, config in PROVIDER_CONFIGS.items():
        provider = Provider(config)
        if not provider.available:
            table.add_row(name, provider.model, "[yellow]no key[/yellow]", "—")
            continue
        try:
            reply = provider.complete(
                [{"role": "user", "content": "Reply with the single word OK."}], max_tokens=512
            )
        except Exception as exc:  # noqa: BLE001 - report every failure shape the same way
            table.add_row(name, provider.model, f"[red]{str(exc)[:40]}[/red]", "—")
        else:
            status = "[green]ok[/green]" if reply.text else "[red]empty reply[/red]"
            table.add_row(name, provider.model, status, f"{reply.latency_s}s")
            if reply.text:
                working.append(name)

    console.print(table)
    if len(working) < 2:
        console.print(
            "[yellow]Fewer than two working providers — fallback cannot be tested.[/yellow]"
        )
        raise typer.Exit(1)

    router = LLMRouter(order=working, force_fail={working[0]})
    reply = router.complete([{"role": "user", "content": "Reply with the single word OK."}])
    console.print(
        f"[green]fallback ok:[/green] forced {working[0]} to fail, answered by {reply.provider}"
    )


@app.command()
def serve(
    repo: Annotated[str, typer.Argument(help="Repository to serve tools over")] = ".",
    transport: Annotated[
        str,
        typer.Option(
            "--transport",
            "-t",
            help="stdio (a client launches the process) or http (a client connects to a URL)",
        ),
    ] = "stdio",
    port: Annotated[int, typer.Option(help="Port for --transport http")] = 8765,
) -> None:
    """Run the MCP server, over stdio or streamable HTTP.

    stdio is what Claude Desktop's config file launches. http suits clients that
    take a URL instead of a command, and is how a remotely hosted MCP server
    would be reached.
    """
    from repomind.mcp_server import build_server

    if transport not in ("stdio", "http"):
        raise typer.BadParameter("transport must be 'stdio' or 'http'")

    context = RepoContext.create(repo)
    server = build_server(context)

    if transport == "http":
        url = f"http://127.0.0.1:{port}/mcp"
        console.print(f"[bold]repomind mcp server[/bold] — {context.root}")
        console.print(f"[green]listening:[/green] {url}")
        console.print("[dim]add that URL as a custom connector; Ctrl+C to stop[/dim]")
        # host/port are transport kwargs in SDK 2.x, not server settings.
        try:
            server.run(transport="streamable-http", host="127.0.0.1", port=port)
        except TypeError:
            # SDK 1.x kept them on .settings instead.
            server.settings.host = "127.0.0.1"  # type: ignore[attr-defined]
            server.settings.port = port  # type: ignore[attr-defined]
            server.run(transport="streamable-http")
        return

    # stdio: anything written to stdout corrupts the JSON-RPC stream the client
    # is reading, so every human-readable byte goes to stderr instead.
    err_console.print(f"[dim]repomind mcp server — {context.root}[/dim]")
    server.run(transport="stdio")


@app.command()
def tools(
    repo: Annotated[str, typer.Argument(help="Local path")] = ".",
) -> None:
    """Run the MCP tools directly. No LLM, no API key, no cost."""
    context = RepoContext.create(repo)
    listing = list_directory(context, ".", depth=2)

    console.print(f"[bold]{context.root}[/bold]")
    console.print(f"[dim]{listing.total_entries} entries, git={context.is_git_repo}[/dim]\n")

    table = Table("manifest", "ecosystem", "dependencies", box=None)
    for manifest in get_dependencies(context).manifests:
        table.add_row(
            manifest.path,
            manifest.ecosystem,
            ", ".join(d.name for d in manifest.dependencies[:6]) or "—",
        )
    console.print(table)


if __name__ == "__main__":
    app()
