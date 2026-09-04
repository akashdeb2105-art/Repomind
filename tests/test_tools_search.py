"""search_code, across both engines."""

from __future__ import annotations

import pytest

from repomind.tools import RepoContext, RepoError, search_code
from repomind.tools.search import _search_python


def test_search_finds_a_symbol(sample_repo: RepoContext):
    result = search_code(sample_repo, "class Engine")

    assert result.matches, "should find the Engine class"
    assert any(m.path == "src/sample/core.py" for m in result.matches)
    assert all(m.line_number > 0 for m in result.matches)


def test_search_ignores_noise_directories(sample_repo: RepoContext):
    result = search_code(sample_repo, "Engine")

    assert not any(m.path.startswith("node_modules") for m in result.matches)


def test_no_matches_is_an_empty_result_not_an_error(sample_repo: RepoContext):
    result = search_code(sample_repo, "ThisSymbolDoesNotExistAnywhere")

    assert result.matches == []
    assert result.truncated is False


def test_query_is_literal_by_default(sample_repo: RepoContext):
    """An agent writing 'def main(' must not hit a regex syntax error."""
    result = search_code(sample_repo, "def main(")

    assert any("def main(" in m.line for m in result.matches)


def test_regex_mode_is_opt_in(sample_repo: RepoContext):
    result = search_code(sample_repo, r"def \w+\(", regex=True)

    assert len(result.matches) >= 2


def test_max_matches_is_enforced_and_flagged(sample_repo: RepoContext):
    (sample_repo.root / "many.py").write_text("\n".join(["needle"] * 50), encoding="utf-8")

    result = search_code(sample_repo, "needle", max_matches=10)

    assert len(result.matches) == 10
    assert result.truncated is True


def test_empty_query_is_rejected(sample_repo: RepoContext):
    with pytest.raises(RepoError, match="empty"):
        search_code(sample_repo, "   ")


def test_python_fallback_agrees_with_the_default_engine(sample_repo: RepoContext):
    """The pure-Python path must work when ripgrep is unavailable."""
    fallback = _search_python(sample_repo, "class Engine", None, 100, False)

    assert fallback.engine == "python"
    assert any(m.path == "src/sample/core.py" for m in fallback.matches)
