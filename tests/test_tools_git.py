"""git history and blame, against a real repository built in a temp dir."""

from __future__ import annotations

import pytest

from repomind.tools import RepoContext, RepoError, get_file_blame, get_git_history


def test_history_returns_commits_newest_first(git_repo: RepoContext):
    history = get_git_history(git_repo, limit=10)

    assert len(history.commits) == 2
    assert history.commits[0].subject == "Add version constant"
    assert history.commits[1].subject == "Initial commit"
    assert history.commits[0].author == "Sample Dev"
    assert len(history.commits[0].sha) == 40


def test_history_can_be_scoped_to_one_path(git_repo: RepoContext):
    history = get_git_history(git_repo, path="src/sample/core.py")

    assert [c.subject for c in history.commits] == ["Add version constant", "Initial commit"]


def test_history_limit_sets_the_truncated_flag(git_repo: RepoContext):
    history = get_git_history(git_repo, limit=1)

    assert len(history.commits) == 1
    assert history.truncated is True


def test_history_refuses_paths_outside_the_repo(git_repo: RepoContext):
    with pytest.raises(RepoError):
        get_git_history(git_repo, path="../../../etc")


def test_history_on_a_non_git_directory_is_a_clear_error(sample_repo: RepoContext):
    with pytest.raises(RepoError, match="not a git repository"):
        get_git_history(sample_repo)


def test_blame_attributes_lines_to_authors(git_repo: RepoContext):
    blame = get_file_blame(git_repo, "src/sample/core.py")

    assert blame.entries
    assert blame.authors.get("Sample Dev", 0) > 0
    assert all(e.line_number > 0 for e in blame.entries)
    assert all(len(e.sha) == 8 for e in blame.entries)


def test_non_ascii_commit_messages_do_not_break_history(git_repo: RepoContext):
    """A curly quote in one commit message killed a whole benchmark run.

    subprocess with text=True decodes using the locale encoding — cp1252 on a
    default Windows install — and the failure surfaces as stdout being None,
    far from its cause.
    """
    import subprocess

    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo.root),
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "fix: don’t drop “smart quotes” — café naïve 中文",
        ],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "GIT_CONFIG_GLOBAL": "/dev/null"},
    )

    history = get_git_history(git_repo, limit=5)

    assert history.commits, "history must survive non-ASCII output"
    assert "smart quotes" in history.commits[0].subject


def test_git_output_is_never_none(git_repo: RepoContext):
    """Even if a decode fails, downstream code gets a string, not None."""
    from repomind.tools.git_tools import _run_git

    assert isinstance(_run_git(git_repo, ["log", "--oneline", "-1"]), str)
