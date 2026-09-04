"""Repository context: the safety boundary every tool goes through.

`RepoContext` owns three concerns no individual tool should reimplement:

* **Containment.** Every path is resolved and checked against the repo root, so
  a traversal like ``../../.ssh/id_rsa`` is refused. The agent is driven by an
  LLM reading text from repositories nobody here wrote, so this is enforced in
  code rather than asked for in a prompt.
* **Noise.** Repositories are full of things nobody wants read: ``.git``,
  ``node_modules``, virtualenvs, build output, lockfiles, minified bundles.
* **Size.** Binaries and giant vendored files are detected and refused before
  their contents ever reach an LLM's context window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Directories that are never worth walking into.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "target",
        "vendor",
        ".next",
        ".nuxt",
        ".cache",
        "coverage",
        "htmlcov",
        ".gradle",
        ".terraform",
        "site-packages",
        ".eggs",
    }
)

# Extensions whose contents are meaningless as text.
BINARY_EXTENSIONS = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".dylib",
        ".a",
        ".o",
        ".obj",
        ".exe",
        ".bin",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".jar",
        ".war",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".svg",
        ".tiff",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".flac",
        ".webm",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        ".h5",
        ".pt",
    }
)

# Generated or lock files: real text, but no signal about how the code works.
NOISE_FILENAMES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
        "go.sum",
        "uv.lock",
    }
)

MAX_FILE_BYTES = 512_000  # ~500 KB; anything larger is vendored or generated
MAX_LINE_CHARS = 2_000  # a longer "line" means minified


class RepoError(Exception):
    """A request that cannot be served safely."""


@dataclass(frozen=True)
class RepoContext:
    """A resolved repository root that every tool call is scoped to."""

    root: Path

    @classmethod
    def create(cls, root: str | Path) -> RepoContext:
        resolved = Path(root).expanduser().resolve()
        if not resolved.exists():
            raise RepoError(f"path does not exist: {resolved}")
        if not resolved.is_dir():
            raise RepoError(f"not a directory: {resolved}")
        return cls(root=resolved)

    # -- containment -------------------------------------------------------- #

    def resolve(self, relative: str | Path = ".") -> Path:
        """Resolve a repo-relative path, refusing anything outside the root.

        Absolute paths, ``..`` traversal, and symlinks pointing outside the
        repository are all rejected here rather than in each tool.
        """
        candidate = Path(relative)
        if candidate.is_absolute():
            raise RepoError(f"absolute paths are not allowed: {relative}")

        target = (self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise RepoError(f"path escapes the repository root: {relative}")
        return target

    def relative(self, path: Path) -> str:
        """Repo-relative POSIX path, so output is identical on Windows and Linux."""
        return path.resolve().relative_to(self.root).as_posix()

    @property
    def is_git_repo(self) -> bool:
        return (self.root / ".git").exists()

    # -- filtering ---------------------------------------------------------- #

    def is_ignored(self, path: Path) -> bool:
        """True for paths inside noise directories, or that are noise themselves."""
        try:
            parts = path.resolve().relative_to(self.root).parts
        except ValueError:
            return True
        if any(part in IGNORED_DIRS for part in parts):
            return True
        return path.name in NOISE_FILENAMES

    @staticmethod
    def is_probably_binary(path: Path) -> bool:
        """Extension check first, then sniff for NUL bytes in the first 8 KB.

        The sniff matters because extensions lie — plenty of binaries have no
        extension at all.
        """
        if path.suffix.lower() in BINARY_EXTENSIONS:
            return True
        try:
            with path.open("rb") as handle:
                chunk = handle.read(8192)
        except OSError:
            return True
        return b"\x00" in chunk

    def readable_text_file(self, path: Path) -> bool:
        """A file worth showing an LLM: exists, not ignored, not binary, not huge."""
        if not path.is_file() or self.is_ignored(path):
            return False
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                return False
        except OSError:
            return False
        return not self.is_probably_binary(path)

    def walk_files(self, start: Path | None = None):
        """Yield every readable text file under `start`, skipping ignored trees.

        Prunes ignored directories rather than filtering afterwards — walking
        into node_modules and discarding the results wastes real seconds.
        """
        root = start or self.root
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(current.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_dir():
                    if entry.name not in IGNORED_DIRS and not entry.is_symlink():
                        stack.append(entry)
                elif self.readable_text_file(entry):
                    yield entry
