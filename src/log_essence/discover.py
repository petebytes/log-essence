"""Discover analyzable log sources for the current code project.

Project-scoped by design — a developer in a repo wants *their* project's logs,
not the machine's. Scans:
- Log files within the project tree (recursively; vendored/VCS/build dirs pruned)
- Docker containers belonging to this project's Compose stack
- A Docker Compose file in the project root

It deliberately does NOT surface system logs (/var/log, ~/Library/Logs),
other projects' containers, or the systemd journal — those aren't project-specific.
Dev-server stdout is covered by piping: `pnpm dev 2>&1 | log-essence -`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Directories never worth walking in a code project (vendored deps, VCS, build output).
# Hidden dirs (".git", ".venv", ".next", ".turbo", …) are pruned separately.
_PRUNE_DIRS = {
    "node_modules",
    "dist",
    "build",
    "out",
    "coverage",
    "target",
    "vendor",
    "__pycache__",
}

# Compressed/rotated logs: read_text() can't decode these, so don't suggest them.
_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip", ".zst", ".7z", ".lz4"}

_COMPOSE_FILENAMES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)

# Lockfile → package manager (most code projects log to stdout, so the *command* is
# the real log source — not a file).
_LOCKFILE_PM = {
    "pnpm-lock.yaml": "pnpm",
    "yarn.lock": "yarn",
    "bun.lockb": "bun",
    "package-lock.json": "npm",
}

# package.json scripts that run a long-lived, log-producing process (not build/test/lint).
_RUN_SCRIPT_KEYS = ("dev", "start", "serve")

# Standard coding-agent / assistant instruction files. These document how to run,
# build, and test a project (and often where its logs go), so they're a prime place
# to look for the command whose stdout you'd pipe into log-essence.
_AGENT_DOC_FILES = (
    "CLAUDE.md",
    "AGENTS.md",
    "GEMINI.md",
    ".cursorrules",
    ".cursor/rules",
    ".windsurfrules",
    ".clinerules",
    ".github/copilot-instructions.md",
)


def _count_lines(path: Path) -> int:
    """Count lines in a file, returning 0 on error."""
    try:
        return sum(1 for _ in path.open(errors="replace"))
    except OSError:
        return 0


def _looks_like_text(path: Path, sample_size: int = 8192) -> bool:
    """Heuristic text test: a NUL byte in the first chunk means binary.

    Filters out files that merely *end* in ``.log`` but are binary — e.g. LevelDB
    write-ahead logs (``000003.log``) in Chromium/Playwright profiles, SQLite WAL, etc.
    """
    try:
        with path.open("rb") as fh:
            chunk = fh.read(sample_size)
    except OSError:
        return False
    return bool(chunk) and b"\x00" not in chunk


def _compose_files(root: Path) -> list[Path]:
    """Return the Docker Compose files present in the project root."""
    return [root / name for name in _COMPOSE_FILENAMES if (root / name).is_file()]


def _compose_project_name(root: Path) -> str:
    """Derive the Docker Compose project name the way Compose does (dir basename).

    `COMPOSE_PROJECT_NAME` overrides, matching Compose's own precedence.
    """
    env = os.environ.get("COMPOSE_PROJECT_NAME")
    if env:
        return env
    name = re.sub(r"[^a-z0-9_-]", "", root.name.lower())
    return name.lstrip("_-") or "default"


def _find_log_files(root: Path | None = None) -> list[dict]:
    """Find log files within the project tree, pruning vendored/VCS/build dirs."""
    root = root or Path.cwd()
    try:
        root = root.resolve()
    except OSError:
        return []
    if not root.is_dir():
        return []

    sources: list[dict] = []
    seen: set[Path] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place so os.walk never descends into them.
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS and not d.startswith(".")]

        current = Path(dirpath)
        in_log_dir = current.name in ("logs", "log")

        for fn in filenames:
            f = current / fn
            suffix = f.suffix.lower()
            if suffix in _COMPRESSED_SUFFIXES:
                continue
            # A ".log" anywhere, or any plain log-ish file inside a logs/ or log/ dir.
            is_log = suffix == ".log" or (in_log_dir and suffix in ("", ".log", ".txt"))
            if not is_log:
                continue

            try:
                resolved = f.resolve()
            except OSError:
                continue
            if resolved in seen or not f.is_file():
                continue
            seen.add(resolved)

            if not _looks_like_text(f):
                continue

            lines = _count_lines(f)
            if lines > 0:
                sources.append(
                    {
                        "type": "file",
                        "name": str(f),
                        "lines": lines,
                        "command": f"log-essence {f}",
                    }
                )

    sources.sort(key=lambda s: s["name"])
    return sources


def _find_docker_containers(root: Path | None = None) -> list[dict]:
    """Find running containers belonging to *this project's* Compose stack.

    Project-scoped: without a Compose file in the project there is no reliable
    project association, so nothing is returned (and docker is not invoked).
    """
    if not shutil.which("docker"):
        return []

    root = root or Path.cwd()
    if not _compose_files(root):
        return []

    project = _compose_project_name(root)
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    sources: list[dict] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, image = parts[1], parts[2]
        sources.append(
            {
                "type": "docker",
                "name": f"{name} ({image})",
                "lines": "?",
                "command": f"docker logs {name} 2>&1 | log-essence -",
            }
        )

    return sources


def _find_compose_projects(root: Path | None = None) -> list[dict]:
    """Report the Docker Compose file in the project root, if any."""
    root = root or Path.cwd()
    files = _compose_files(root)
    if not files:
        return []
    cf = files[0]
    return [
        {
            "type": "compose",
            "name": cf.name,
            "lines": "?",
            "command": "docker compose logs 2>&1 | log-essence -",
        }
    ]


def _detect_package_manager(root: Path) -> str | None:
    """Detect the JS package manager from its lockfile (npm if only package.json)."""
    for lockfile, pm in _LOCKFILE_PM.items():
        if (root / lockfile).is_file():
            return pm
    if (root / "package.json").is_file():
        return "npm"
    return None


def _find_run_commands(root: Path | None = None) -> list[dict]:
    """Surface the commands that run the app — its logs are stdout, not files.

    Reads package.json scripts (dev/start/serve) and emits a ready-to-pipe command.
    """
    root = root or Path.cwd()
    pkg = root / "package.json"
    if not pkg.is_file():
        return []
    try:
        scripts = json.loads(pkg.read_text(errors="replace")).get("scripts", {})
    except (OSError, ValueError):
        return []
    if not isinstance(scripts, dict):
        return []

    pm = _detect_package_manager(root) or "npm"
    sources: list[dict] = []
    for key in _RUN_SCRIPT_KEYS:
        if key in scripts:
            sources.append(
                {
                    "type": "command",
                    "name": f"{key} (package.json)",
                    "lines": "stream",
                    "command": f"{pm} run {key} 2>&1 | log-essence -",
                }
            )
    return sources


def _find_agent_files(root: Path | None = None) -> list[dict]:
    """Surface standard coding-agent instruction files (CLAUDE.md, AGENTS.md, …).

    They document how to run/build/test the project — read them to pick the
    command whose stdout to pipe into log-essence.
    """
    root = root or Path.cwd()
    sources: list[dict] = []
    for rel in _AGENT_DOC_FILES:
        if (root / rel).exists():
            sources.append(
                {
                    "type": "agent",
                    "name": rel,
                    "lines": "-",
                    "command": "documents how to run/build/test — read for log-producing commands",
                }
            )
    return sources


def discover_sources(root: Path | None = None) -> list[dict]:
    """Discover project-specific log sources under ``root`` (default: cwd).

    Returns:
        List of source dicts with keys: type, name, lines, command.
    """
    root = root or Path.cwd()
    sources: list[dict] = []
    sources.extend(_find_log_files(root))
    sources.extend(_find_run_commands(root))
    sources.extend(_find_agent_files(root))
    sources.extend(_find_docker_containers(root))
    sources.extend(_find_compose_projects(root))
    return sources


def format_discovery_table(sources: list[dict]) -> str:
    """Format discovered sources as a table."""
    if not sources:
        return "No log sources found in this project."

    # Calculate column widths
    type_w = max(len(s["type"]) for s in sources)
    name_w = max(len(str(s["name"])) for s in sources)
    lines_w = max(len(str(s["lines"])) for s in sources)

    type_w = max(type_w, 4)
    name_w = min(max(name_w, 4), 50)
    lines_w = max(lines_w, 5)

    header = f"{'Type':<{type_w}}  {'Source':<{name_w}}  {'Lines':>{lines_w}}  Command"
    sep = f"{'-' * type_w}  {'-' * name_w}  {'-' * lines_w}  {'-' * 30}"

    rows = [header, sep]
    for s in sources:
        name = str(s["name"])
        if len(name) > name_w:
            name = "..." + name[-(name_w - 3) :]
        lines_str = str(s["lines"])
        rows.append(
            f"{s['type']:<{type_w}}  {name:<{name_w}}  {lines_str:>{lines_w}}  {s['command']}"
        )

    return "\n".join(rows)


def run_discover_command(root: str | None = None) -> int:
    """Execute the discover subcommand.

    Args:
        root: Project directory to scan (default: current working directory).

    Returns:
        Exit code (0 for success).
    """
    target = Path(root).expanduser() if root else Path.cwd()
    print(f"Scanning {target} for log sources...", file=sys.stderr)
    sources = discover_sources(target)
    print(format_discovery_table(sources))
    print(f"\nFound {len(sources)} source(s).", file=sys.stderr)
    return 0
