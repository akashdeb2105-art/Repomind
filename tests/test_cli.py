"""CLI surface: commands exist, arguments parse, no-key paths fail cleanly."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from repomind.cli import app

runner = CliRunner()


@pytest.fixture
def no_keys(monkeypatch):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(var, "")


def test_version_prints_and_exits_zero():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "repomind" in result.stdout


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("analyze", "ask", "check", "tools"):
        assert command in result.stdout


def test_tools_command_needs_no_api_key(sample_repo, no_keys):
    """The deterministic half of the product must work with zero configuration."""
    result = runner.invoke(app, ["tools", str(sample_repo.root)])

    assert result.exit_code == 0
    assert "requirements.txt" in result.stdout


def test_analyze_without_keys_explains_itself(sample_repo, no_keys):
    result = runner.invoke(app, ["analyze", str(sample_repo.root)])

    assert result.exit_code == 1
    assert "No LLM providers configured" in result.stdout
    assert ".env" in result.stdout, "the error should say how to fix it"


def test_ask_without_keys_explains_itself(sample_repo, no_keys):
    result = runner.invoke(app, ["ask", "what is this?", "--repo", str(sample_repo.root)])

    assert result.exit_code == 1
    assert "No LLM providers configured" in result.stdout


def test_analyze_rejects_a_path_that_does_not_exist(no_keys):
    result = runner.invoke(app, ["analyze", "/nope/not/here"])

    assert result.exit_code != 0


def test_ask_requires_a_question():
    result = runner.invoke(app, ["ask"])

    assert result.exit_code != 0


def test_serve_is_registered():
    result = runner.invoke(app, ["--help"])

    assert "serve" in result.stdout


def test_serve_writes_nothing_to_stdout(sample_repo, monkeypatch):
    """stdout is the MCP transport. A single stray byte corrupts the protocol."""
    started: dict[str, object] = {}

    class FakeServer:
        def run(self, transport: str) -> None:
            started["transport"] = transport

    monkeypatch.setattr("repomind.mcp_server.build_server", lambda ctx: FakeServer())

    result = runner.invoke(app, ["serve", str(sample_repo.root)])

    assert result.exit_code == 0
    assert started["transport"] == "stdio"
    assert result.stdout == "", f"stdout must stay clean, got {result.stdout!r}"


def test_serve_rejects_an_unknown_transport(sample_repo):
    result = runner.invoke(app, ["serve", str(sample_repo.root), "--transport", "carrier-pigeon"])

    assert result.exit_code != 0


def test_serve_http_passes_host_and_port_to_the_transport(sample_repo, monkeypatch):
    """host/port are transport kwargs in SDK 2.x, not attributes on the server."""
    captured: dict[str, object] = {}

    class FakeServer:
        def run(self, transport: str, **kwargs: object) -> None:
            captured["transport"] = transport
            captured.update(kwargs)

    monkeypatch.setattr("repomind.mcp_server.build_server", lambda ctx: FakeServer())

    result = runner.invoke(
        app, ["serve", str(sample_repo.root), "--transport", "http", "--port", "9999"]
    )

    assert result.exit_code == 0
    assert captured["transport"] == "streamable-http"
    assert captured["port"] == 9999
    assert "127.0.0.1" in str(captured["host"]), "must bind to loopback, not 0.0.0.0"
