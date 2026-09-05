"""Shared fixtures: a small, realistic repository built on disk per test."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repomind.tools.repo import RepoContext

MAIN_PY = '''\
"""Entry point for the sample app."""

import os

from sample.core import Engine


def main() -> int:
    engine = Engine(os.getenv("MODE", "fast"))
    return engine.run()


if __name__ == "__main__":
    raise SystemExit(main())
'''

CORE_PY = '''\
class Engine:
    """Does the actual work."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def run(self) -> int:
        return 0
'''


@pytest.fixture
def anyio_backend():
    """MCP's server API is async; asyncio only, no trio dependency."""
    return "asyncio"


@pytest.fixture
def sample_repo(tmp_path: Path) -> RepoContext:
    """A repo with source, tests, manifests, noise directories and a binary."""
    (tmp_path / "src" / "sample").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
    (tmp_path / ".git-decoy").mkdir()

    (tmp_path / "src" / "sample" / "main.py").write_text(MAIN_PY, encoding="utf-8")
    (tmp_path / "src" / "sample" / "core.py").write_text(CORE_PY, encoding="utf-8")
    (tmp_path / "src" / "sample" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_core.py").write_text(
        "def test_engine():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Sample\n\nA sample project.\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(
        "# comment\nhttpx>=0.27\npydantic==2.7.0\nuvicorn[standard]>=0.30\n-r other.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "package.json").write_text(
        '{"name":"sample","dependencies":{"react":"^18.0.0"},'
        '"devDependencies":{"vitest":"^1.0.0"},"scripts":{"test":"vitest run"}}',
        encoding="utf-8",
    )
    (tmp_path / "go.mod").write_text(
        "module example.com/sample\n\ngo 1.22\n\nrequire (\n"
        "\tgithub.com/spf13/cobra v1.8.0 // indirect\n\tgolang.org/x/sync v0.7.0\n)\n",
        encoding="utf-8",
    )

    # Noise that must never surface in results.
    (tmp_path / "node_modules" / "left-pad" / "index.js").write_text(
        "module.exports = Engine;\n", encoding="utf-8"
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}', encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00" * 4)
    (tmp_path / "data.bin").write_bytes(b"binary\x00content\x00here")

    return RepoContext.create(tmp_path)


@pytest.fixture
def git_repo(sample_repo: RepoContext) -> RepoContext:
    """The sample repo with two real commits."""
    root = str(sample_repo.root)
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", root, *args],
            check=True,
            capture_output=True,
            env={**env, "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "dev@example.com")
    git("config", "user.name", "Sample Dev")
    git("add", "-A")
    git("commit", "-q", "-m", "Initial commit")

    (sample_repo.root / "src" / "sample" / "core.py").write_text(
        CORE_PY + "\n\nVERSION = '1.0'\n", encoding="utf-8"
    )
    git("add", "-A")
    git("commit", "-q", "-m", "Add version constant")

    return sample_repo
