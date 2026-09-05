"""MCP server: expose RepoMind's tools to Claude Desktop and Claude Code.

This is the deliverable the project is named for. Everything else — the agent,
the CLI, the eval harness — is built on these tools; MCP is what lets someone
else's Claude use them directly, with no RepoMind code in their loop.

Design notes:

* **The repository root is fixed at startup**, from argv or REPOMIND_REPO. Tools
  take repo-relative paths only. A tool that accepted an arbitrary root would
  let any prompt reachable by the model read anything on the user's disk, and
  the model's input includes the contents of the repository being analysed.
* **Errors are returned, not raised.** A tool that raises gives the model a
  stack trace; a tool that returns "path is a directory, use list_directory"
  gives it something it can act on. Recoverable failures should teach.
* **Every response is a Pydantic model**, so the client receives a real schema
  rather than a wall of text it has to parse back out of prose.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Annotated, Any

from pydantic import Field

from repomind import __version__
from repomind.models import (
    DependencyReport,
    DirectoryListing,
    FileBlame,
    FileContent,
    GitHistory,
    ReadmeResult,
    SearchResult,
    TestRunResult,
    ToolError,
)
from repomind.tools import (
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

logger = logging.getLogger("repomind.mcp")

INSTRUCTIONS = """\
RepoMind exposes read-only tools over one local repository.

Start with `list_directory` and `get_dependencies` to learn the shape of the
project, then `search_code` to locate specific symbols, then `read_file` to read
what you found. Prefer searching for names that literally appear in source code.

Every path you pass must be relative to the repository root. Paths outside it
are refused. Credential files (.env, id_rsa, *.pem) are never readable.
"""


def _repo_from_environment() -> RepoContext:
    """Resolve the repository root once, at startup."""
    target = sys.argv[1] if len(sys.argv) > 1 else os.getenv("REPOMIND_REPO", ".")
    try:
        return RepoContext.create(target)
    except RepoError as exc:
        print(f"repomind: cannot open repository {target!r}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def build_server(repo: RepoContext | None = None):
    """Construct the MCP server. Imported lazily so the `mcp` extra stays optional."""
    # The SDK renamed FastMCP to MCPServer in 2.x. Support both rather than
    # pinning: a user installing repomind alongside other MCP packages should
    # not have their SDK version dictated by us.
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # SDK >= 2.0
    except ModuleNotFoundError:
        try:
            from mcp.server.fastmcp import FastMCP as _Server  # SDK 1.x
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "The MCP SDK is not installed. Install the mcp extra: pip install 'repomind[mcp]'"
            ) from exc

    context = repo or _repo_from_environment()
    server = _Server("repomind", instructions=INSTRUCTIONS)

    def guard(func):
        """Turn RepoError into a structured result the model can act on.

        functools.wraps matters here beyond tidiness: the SDK builds each tool's
        JSON schema by introspecting the callable, and an unwrapped wrapper
        publishes a signature of (*args, **kwargs) — so every tool would appear
        to take two mystery arguments and none of the real ones.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            try:
                return func(*args, **kwargs)
            except RepoError as exc:
                logger.info("tool refused: %s", exc)
                return ToolError(error=type(exc).__name__, detail=str(exc))
            except Exception as exc:  # noqa: BLE001 - never kill the server for one bad call
                logger.exception("tool failed")
                return ToolError(error="unexpected_error", detail=str(exc)[:300])

        return wrapper

    # -- tools ------------------------------------------------------------- #

    @server.tool()
    @guard
    def list_repo_directory(
        path: Annotated[str, Field(description="Repo-relative directory, '.' for the root")] = ".",
        depth: Annotated[int, Field(description="How many levels to walk, 1-6", ge=1, le=6)] = 2,
    ) -> DirectoryListing | ToolError:
        """List a directory tree, skipping vendor, build and cache directories."""
        return list_directory(context, path, depth=depth)

    @server.tool()
    @guard
    def read_repo_file(
        path: Annotated[str, Field(description="Repo-relative file path")],
        start_line: Annotated[int, Field(description="First line, 1-based; 0 for the top")] = 0,
        end_line: Annotated[int, Field(description="Last line; 0 for as much as fits")] = 0,
    ) -> FileContent | ToolError:
        """Read a text file, or a line range of one. Large files are truncated, not refused."""
        line_range = (start_line, end_line) if start_line > 0 else None
        return read_file(context, path, line_range=line_range)

    @server.tool()
    @guard
    def search_repo_code(
        query: Annotated[
            str, Field(description="Literal text to find: a symbol, decorator, or string")
        ],
        path_glob: Annotated[str, Field(description="Optional filter, e.g. '*.py'")] = "",
        regex: Annotated[
            bool, Field(description="Treat the query as a regular expression")
        ] = False,
        max_matches: Annotated[int, Field(description="Cap on results", ge=1, le=500)] = 100,
    ) -> SearchResult | ToolError:
        """Search the repository. Literal by default — 'def main(' works as typed."""
        return search_code(
            context, query, path_glob=path_glob or None, regex=regex, max_matches=max_matches
        )

    @server.tool()
    @guard
    def get_repo_dependencies() -> DependencyReport | ToolError:
        """Parse every dependency manifest: Python, JavaScript, Go, Rust, PHP."""
        return get_dependencies(context)

    @server.tool()
    @guard
    def get_repo_readme() -> ReadmeResult | ToolError:
        """Fetch the repository's own README. A signal of intent, not a source of truth."""
        return get_readme(context)

    @server.tool()
    @guard
    def get_repo_git_history(
        path: Annotated[str, Field(description="Optional repo-relative path to scope to")] = "",
        limit: Annotated[int, Field(description="How many commits", ge=1, le=200)] = 20,
    ) -> GitHistory | ToolError:
        """Recent commits. Recency and authorship show which parts of a codebase are alive."""
        return get_git_history(context, path=path or None, limit=limit)

    @server.tool()
    @guard
    def get_repo_file_blame(
        path: Annotated[str, Field(description="Repo-relative file path")],
    ) -> FileBlame | ToolError:
        """Per-line authorship, plus how many lines each author last touched."""
        return get_file_blame(context, path)

    @server.tool()
    @guard
    def run_repo_tests(
        timeout_s: Annotated[float, Field(description="Hard timeout", ge=5, le=600)] = 120.0,
    ) -> TestRunResult | ToolError:
        """Run the repository's own test suite.

        Disabled unless REPOMIND_ALLOW_TEST_EXECUTION=1 is set, because this
        executes code from the repository being analysed.
        """
        return run_tests(context, timeout_s=timeout_s)

    logger.info("repomind %s serving %s", __version__, context.root)
    return server


def main() -> None:
    """Entry point for `repomind-mcp` and for Claude Desktop's config."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
