"""Filesystem tools, and the containment boundary they all depend on."""

from __future__ import annotations

import pytest

from repomind.models import EntryType
from repomind.tools import RepoContext, RepoError, get_readme, list_directory, read_file

# --------------------------------------------------------------------------- #
# Containment — the security boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "attack",
    [
        "../etc/passwd",
        "../../.ssh/id_rsa",
        "src/../../outside.txt",
        "/etc/passwd",
        "src/sample/../../../escape.py",
    ],
)
def test_paths_cannot_escape_the_repository(sample_repo: RepoContext, attack: str):
    with pytest.raises(RepoError):
        sample_repo.resolve(attack)


def test_read_file_refuses_to_escape(sample_repo: RepoContext):
    with pytest.raises(RepoError):
        read_file(sample_repo, "../../../etc/passwd")


def test_paths_inside_the_repo_are_fine(sample_repo: RepoContext):
    assert sample_repo.resolve("src/sample/main.py").is_file()
    assert sample_repo.resolve("src/../src/sample/core.py").is_file()


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


def test_read_file_returns_content_and_metadata(sample_repo: RepoContext):
    result = read_file(sample_repo, "src/sample/main.py")

    assert "def main() -> int:" in result.content
    assert result.path == "src/sample/main.py"
    assert result.total_lines > 5
    assert result.truncated is False


def test_read_file_honours_a_line_range(sample_repo: RepoContext):
    result = read_file(sample_repo, "src/sample/main.py", line_range=(1, 3))

    assert result.start_line == 1
    assert result.end_line == 3
    assert len(result.content.splitlines()) == 3
    assert result.truncated is True


def test_read_file_truncates_long_files_and_says_so(sample_repo: RepoContext):
    (sample_repo.root / "big.py").write_text("\n".join(f"line {i}" for i in range(1000)))

    result = read_file(sample_repo, "big.py", max_lines=50)

    assert len(result.content.splitlines()) == 50
    assert result.truncated is True
    assert result.total_lines == 1000


def test_read_file_refuses_binaries(sample_repo: RepoContext):
    with pytest.raises(RepoError, match="binary"):
        read_file(sample_repo, "data.bin")


def test_read_file_refuses_oversized_files(sample_repo: RepoContext):
    (sample_repo.root / "huge.txt").write_text("x" * 600_000)

    with pytest.raises(RepoError, match="too large"):
        read_file(sample_repo, "huge.txt")


def test_read_file_on_a_directory_points_at_the_right_tool(sample_repo: RepoContext):
    with pytest.raises(RepoError, match="list_directory"):
        read_file(sample_repo, "src")


def test_missing_file_is_a_clear_error(sample_repo: RepoContext):
    with pytest.raises(RepoError, match="does not exist"):
        read_file(sample_repo, "src/nope.py")


# --------------------------------------------------------------------------- #
# list_directory
# --------------------------------------------------------------------------- #


def test_list_directory_skips_noise(sample_repo: RepoContext):
    listing = list_directory(sample_repo, ".", depth=3)
    paths = {e.path for e in listing.entries}

    assert "src" in paths
    assert "README.md" in paths
    assert not any(p.startswith("node_modules") for p in paths), "node_modules must be pruned"
    assert "package-lock.json" not in paths, "lockfiles carry no design signal"


def test_list_directory_respects_depth(sample_repo: RepoContext):
    shallow = list_directory(sample_repo, ".", depth=1)

    assert all(e.depth <= 1 for e in shallow.entries)
    assert "src/sample/main.py" not in {e.path for e in shallow.entries}


def test_list_directory_reports_types_and_sizes(sample_repo: RepoContext):
    listing = list_directory(sample_repo, "src/sample", depth=1)
    by_path = {e.path: e for e in listing.entries}

    assert by_path["src/sample/main.py"].type is EntryType.FILE
    assert by_path["src/sample/main.py"].size_bytes > 0


def test_list_directory_rejects_a_file(sample_repo: RepoContext):
    with pytest.raises(RepoError, match="not a directory"):
        list_directory(sample_repo, "README.md")


# --------------------------------------------------------------------------- #
# get_readme
# --------------------------------------------------------------------------- #


def test_get_readme_finds_it(sample_repo: RepoContext):
    readme = get_readme(sample_repo)

    assert readme.found is True
    assert readme.path == "README.md"
    assert "A sample project." in readme.content


def test_get_readme_reports_absence_rather_than_raising(sample_repo: RepoContext):
    (sample_repo.root / "README.md").unlink()

    readme = get_readme(sample_repo)

    assert readme.found is False
    assert readme.content == ""


# --------------------------------------------------------------------------- #
# Secrets — found by dogfooding the tools on RepoMind's own repo, where
# list_directory cheerfully listed the .env holding three live API keys.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "secret",
    [".env", ".env.local", ".netrc", "credentials.json", "id_rsa", "server.pem", "app.key"],
)
def test_read_file_refuses_credential_files(sample_repo: RepoContext, secret: str):
    (sample_repo.root / secret).write_text("API_KEY=sk-live-do-not-leak\n", encoding="utf-8")

    with pytest.raises(RepoError, match="credentials"):
        read_file(sample_repo, secret)


def test_credential_files_are_not_even_listed(sample_repo: RepoContext):
    """An agent cannot ask to read a file it was never told exists."""
    (sample_repo.root / ".env").write_text("GROQ_API_KEY=secret\n", encoding="utf-8")

    listing = list_directory(sample_repo, ".", depth=2)

    assert ".env" not in {e.path for e in listing.entries}


def test_secrets_never_appear_in_search_results(sample_repo: RepoContext):
    from repomind.tools import search_code

    (sample_repo.root / ".env").write_text("GROQ_API_KEY=gsk_supersecret\n", encoding="utf-8")

    result = search_code(sample_repo, "GROQ_API_KEY")

    assert result.matches == [], "a search must not surface credential file contents"


# The credential filter matched .env.example on its ".env." prefix and hid the
# repository's own setup instructions, so every mention of it read as a fabrication.
@pytest.mark.parametrize("template", [".env.example", ".env.sample", ".env.template"])
def test_env_templates_are_readable(sample_repo: RepoContext, template: str):
    (sample_repo.root / template).write_text("GROQ_API_KEY=\n", encoding="utf-8")

    result = read_file(sample_repo, template)

    assert "GROQ_API_KEY=" in result.content
    assert template in {e.path for e in list_directory(sample_repo, ".", depth=1).entries}


def test_the_real_env_file_is_still_refused(sample_repo: RepoContext):
    (sample_repo.root / ".env").write_text("GROQ_API_KEY=gsk_real\n", encoding="utf-8")
    (sample_repo.root / ".env.local").write_text("GROQ_API_KEY=gsk_real\n", encoding="utf-8")

    for secret in (".env", ".env.local"):
        with pytest.raises(RepoError, match="credentials"):
            read_file(sample_repo, secret)
