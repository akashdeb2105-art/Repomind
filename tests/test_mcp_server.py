"""MCP server: the tools are registered, scoped, and fail without raising."""

from __future__ import annotations

import pytest

from repomind.mcp_server import build_server
from repomind.models import ToolError
from repomind.tools import RepoContext

pytest.importorskip("mcp", reason="the mcp extra is optional")

EXPECTED_TOOLS = {
    "list_repo_directory",
    "read_repo_file",
    "search_repo_code",
    "get_repo_dependencies",
    "get_repo_readme",
    "get_repo_git_history",
    "get_repo_file_blame",
    "run_repo_tests",
}


@pytest.fixture
def server(sample_repo: RepoContext):
    return build_server(sample_repo)


@pytest.mark.anyio
async def test_every_tool_is_registered(server):
    names = {tool.name for tool in await server.list_tools()}

    assert names == EXPECTED_TOOLS


@pytest.mark.anyio
async def test_tools_describe_themselves(server):
    """A tool with no description is a tool the model will misuse."""
    for tool in await server.list_tools():
        assert tool.description, f"{tool.name} has no description"
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
        assert schema is not None, f"{tool.name} publishes no input schema"
        properties = schema.get("properties", {})
        assert "args" not in properties and "kwargs" not in properties, (
            f"{tool.name} leaked its wrapper's signature — parameters are unusable"
        )


def test_the_repository_root_is_fixed_at_construction(sample_repo: RepoContext):
    """No tool takes a root, so no prompt can redirect the server elsewhere."""
    build_server(sample_repo)  # must not raise

    for name in EXPECTED_TOOLS:
        assert "repo_path" not in name


@pytest.mark.anyio
async def test_a_bad_path_returns_an_error_object_not_an_exception(server):
    """A stack trace teaches the model nothing; a refusal message teaches it something."""
    result = await server.call_tool("read_repo_file", {"path": "../../../etc/passwd"})

    assert "escapes the repository root" in str(result)


@pytest.mark.anyio
async def test_reading_a_real_file_works_end_to_end(server):
    result = await server.call_tool("read_repo_file", {"path": "src/sample/main.py"})

    assert "def main" in str(result)


@pytest.mark.anyio
async def test_credential_files_are_refused_through_mcp(sample_repo: RepoContext):
    """The security boundary has to hold on this path too, not only in-process."""
    (sample_repo.root / ".env").write_text("GROQ_API_KEY=gsk_secret\n", encoding="utf-8")
    server = build_server(sample_repo)

    result = await server.call_tool("read_repo_file", {"path": ".env"})

    assert "gsk_secret" not in str(result)


def test_guard_converts_repo_errors_into_tool_errors(sample_repo: RepoContext):
    from repomind.tools import RepoError

    build_server(sample_repo)
    error = ToolError(error="RepoError", detail=str(RepoError("nope")))

    assert error.error == "RepoError"
