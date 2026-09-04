"""Dependency manifest parsing across ecosystems."""

from __future__ import annotations

import sys

import pytest

from repomind.tools import RepoContext, get_dependencies


def _manifest(report, filename: str):
    return next((m for m in report.manifests if m.path.endswith(filename)), None)


def test_finds_every_ecosystem_present(sample_repo: RepoContext):
    report = get_dependencies(sample_repo)

    assert set(report.ecosystems) >= {"python", "javascript", "go"}


def test_parses_requirements_txt(sample_repo: RepoContext):
    manifest = _manifest(get_dependencies(sample_repo), "requirements.txt")
    names = {d.name for d in manifest.dependencies}

    assert names == {"httpx", "pydantic", "uvicorn"}, "comments and -r includes are skipped"
    versions = {d.name: d.version_spec for d in manifest.dependencies}
    assert versions["pydantic"] == "==2.7.0"


def test_parses_package_json_with_groups(sample_repo: RepoContext):
    manifest = _manifest(get_dependencies(sample_repo), "package.json")
    groups = {d.name: d.group for d in manifest.dependencies}

    assert groups["react"] == "main"
    assert groups["vitest"] == "dev"


def test_parses_go_mod_require_block(sample_repo: RepoContext):
    manifest = _manifest(get_dependencies(sample_repo), "go.mod")
    names = {d.name for d in manifest.dependencies}

    assert "github.com/spf13/cobra" in names
    assert "golang.org/x/sync" in names


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires Python 3.11+")
def test_parses_pyproject(sample_repo: RepoContext):
    (sample_repo.root / "pyproject.toml").write_text(
        '[project]\nname = "sample"\ndependencies = ["httpx>=0.27", "typer"]\n\n'
        '[project.optional-dependencies]\ndev = ["pytest>=8"]\n',
        encoding="utf-8",
    )

    manifest = _manifest(get_dependencies(sample_repo), "pyproject.toml")
    groups = {d.name: d.group for d in manifest.dependencies}

    assert groups["httpx"] == "main"
    assert groups["pytest"] == "dev"


def test_a_broken_manifest_does_not_kill_the_others(sample_repo: RepoContext):
    (sample_repo.root / "package.json").write_text("{not valid json", encoding="utf-8")

    report = get_dependencies(sample_repo)
    broken = _manifest(report, "package.json")

    assert broken.parse_error, "the failure is recorded..."
    assert _manifest(report, "requirements.txt").dependencies, "...and other manifests still parse"


def test_no_manifests_is_an_empty_report(tmp_path):
    empty = RepoContext.create(tmp_path)

    assert get_dependencies(empty).manifests == []
