"""get_dependencies: parse whatever manifests a repository happens to have.

Answers "what is this built with", which is one of the first questions a
newcomer asks. Each manifest is parsed independently and a broken one records a
`parse_error` instead of failing the whole call — half an answer beats none.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from repomind.models import Dependency, DependencyReport, Manifest
from repomind.tools.repo import RepoContext

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on older interpreters
    tomllib = None  # type: ignore[assignment]

MANIFESTS = {
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "Pipfile": "python",
    "setup.py": "python",
    "package.json": "javascript",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "Gemfile": "ruby",
    "pom.xml": "java",
    "build.gradle": "java",
    "composer.json": "php",
}

# name, optional extras, optional version spec — e.g. "uvicorn[standard]>=0.30"
_REQ_LINE = re.compile(r"^\s*(?P<name>[A-Za-z0-9._-]+)\s*(?P<extras>\[[^\]]*\])?\s*(?P<spec>.*)$")


def get_dependencies(repo: RepoContext, max_depth: int = 2) -> DependencyReport:
    """Find and parse dependency manifests near the top of the repository."""
    manifests: list[Manifest] = []

    for name, ecosystem in MANIFESTS.items():
        for path in _find(repo, name, max_depth):
            relative = repo.relative(path)
            try:
                deps = _parse(path, name)
                manifests.append(Manifest(path=relative, ecosystem=ecosystem, dependencies=deps))
            except Exception as exc:  # noqa: BLE001 - one bad manifest must not kill the rest
                manifests.append(
                    Manifest(path=relative, ecosystem=ecosystem, parse_error=str(exc)[:200])
                )

    return DependencyReport(manifests=manifests)


def _find(repo: RepoContext, filename: str, max_depth: int) -> list[Path]:
    """Locate a manifest at the root or shallowly nested (monorepo packages)."""
    found: list[Path] = []
    root_level = repo.root / filename
    if root_level.is_file():
        found.append(root_level)

    for depth in range(1, max(1, max_depth) + 1):
        for path in repo.root.glob("/".join(["*"] * depth) + f"/{filename}"):
            if path.is_file() and not repo.is_ignored(path):
                found.append(path)
    return found


def _parse(path: Path, filename: str) -> list[Dependency]:
    if filename == "requirements.txt":
        return _parse_requirements(path)
    if filename == "pyproject.toml":
        return _parse_pyproject(path)
    if filename in ("package.json", "composer.json"):
        return _parse_package_json(path)
    if filename == "go.mod":
        return _parse_go_mod(path)
    if filename == "Cargo.toml":
        return _parse_cargo(path)
    return []


def _parse_requirements(path: Path) -> list[Dependency]:
    deps: list[Dependency] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):  # -r includes, -e editable, flags
            continue
        match = _REQ_LINE.match(line)
        if match:
            deps.append(
                Dependency(
                    name=match.group("name"),
                    version_spec=(match.group("spec") or "").strip() or None,
                )
            )
    return deps


def _parse_pyproject(path: Path) -> list[Dependency]:
    if tomllib is None:
        raise RuntimeError("TOML parsing requires Python 3.11+")
    data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
    deps: list[Dependency] = []

    for spec in data.get("project", {}).get("dependencies", []) or []:
        deps.append(_dep_from_spec(spec, "main"))
    for group, specs in (data.get("project", {}).get("optional-dependencies", {}) or {}).items():
        for spec in specs or []:
            deps.append(_dep_from_spec(spec, group))

    # Poetry keeps its own table.
    poetry = data.get("tool", {}).get("poetry", {})
    for name, spec in (poetry.get("dependencies", {}) or {}).items():
        if name.lower() != "python":
            deps.append(Dependency(name=name, version_spec=str(spec) if spec else None))

    return deps


def _dep_from_spec(spec: str, group: str) -> Dependency:
    match = _REQ_LINE.match(spec)
    if not match:
        return Dependency(name=spec, group=group)
    return Dependency(
        name=match.group("name"),
        version_spec=(match.group("spec") or "").strip() or None,
        group=group,
    )


def _parse_package_json(path: Path) -> list[Dependency]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    deps: list[Dependency] = []
    for key, group in (
        ("dependencies", "main"),
        ("devDependencies", "dev"),
        ("peerDependencies", "peer"),
        ("require", "main"),
        ("require-dev", "dev"),
    ):
        for name, spec in (data.get(key) or {}).items():
            deps.append(Dependency(name=name, version_spec=str(spec), group=group))
    return deps


def _parse_go_mod(path: Path) -> list[Dependency]:
    deps: list[Dependency] = []
    in_block = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("//", 1)[0].strip()
        if line.startswith("require ("):
            in_block = True
            continue
        if in_block and line == ")":
            in_block = False
            continue
        if in_block or line.startswith("require "):
            parts = line.replace("require ", "", 1).split()
            if len(parts) >= 2:
                deps.append(Dependency(name=parts[0], version_spec=parts[1]))
    return deps


def _parse_cargo(path: Path) -> list[Dependency]:
    if tomllib is None:
        raise RuntimeError("TOML parsing requires Python 3.11+")
    data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
    deps: list[Dependency] = []
    for key, group in (("dependencies", "main"), ("dev-dependencies", "dev")):
        for name, spec in (data.get(key) or {}).items():
            version = spec if isinstance(spec, str) else (spec or {}).get("version")
            deps.append(Dependency(name=name, version_spec=version, group=group))
    return deps
