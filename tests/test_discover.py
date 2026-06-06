"""Tests for the discover module."""

from pathlib import Path

from log_essence.discover import (
    _compose_project_name,
    _count_lines,
    _detect_package_manager,
    _find_agent_files,
    _find_docker_containers,
    _find_log_files,
    _find_run_commands,
    discover_sources,
    format_discovery_table,
)


def test_count_lines(tmp_path: Path) -> None:
    f = tmp_path / "test.log"
    f.write_text("line1\nline2\nline3\n")
    assert _count_lines(f) == 3


def test_count_lines_nonexistent(tmp_path: Path) -> None:
    assert _count_lines(tmp_path / "nonexistent") == 0


def test_find_log_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.log").write_text("line1\nline2\n")
    (tmp_path / "error.log").write_text("error1\n")

    sources = _find_log_files()
    log_names = [s["name"] for s in sources]
    assert any("app.log" in name for name in log_names)
    assert any("error.log" in name for name in log_names)
    assert all(s["type"] == "file" for s in sources)


def test_find_log_files_empty_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sources = _find_log_files()
    # Empty project dir → no sources (system logs are NOT scanned).
    assert sources == []


def test_find_log_files_is_project_scoped(tmp_path: Path) -> None:
    """Only files under the given project root are returned — never /var/log."""
    (tmp_path / "app.log").write_text("hi\n")
    sources = _find_log_files(root=tmp_path)
    assert sources, "expected the project's own log to be found"
    root = str(tmp_path.resolve())
    assert all(str(Path(s["name"]).resolve()).startswith(root) for s in sources)
    assert not any("/var/log" in s["name"] for s in sources)


def test_find_log_files_recurses_but_prunes_vendored_dirs(tmp_path: Path) -> None:
    """Monorepo subdir logs are found; node_modules/.git/dist logs are pruned."""
    (tmp_path / "apps" / "api" / "logs").mkdir(parents=True)
    (tmp_path / "apps" / "api" / "logs" / "server.log").write_text("up\n")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "debug.log").write_text("noise\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "gc.log").write_text("noise\n")

    names = [s["name"] for s in _find_log_files(root=tmp_path)]
    assert any("server.log" in n for n in names)
    assert not any("node_modules" in n for n in names)
    assert not any(".git" in n for n in names)


def test_find_log_files_skips_binary(tmp_path: Path) -> None:
    """A .log extension isn't enough — binary files (LevelDB, SQLite WAL) aren't text logs."""
    (tmp_path / "app.log").write_text("real log line\n")
    (tmp_path / "000003.log").write_bytes(b"\x00\x01\x02leveldb\x00record\x00")

    names = [s["name"] for s in _find_log_files(root=tmp_path)]
    assert any(n.endswith("app.log") for n in names)
    assert not any(n.endswith("000003.log") for n in names)


def test_find_log_files_excludes_compressed(tmp_path: Path) -> None:
    """Rotated/compressed logs are skipped — read_text can't decode them."""
    (tmp_path / "app.log").write_text("real\n")
    (tmp_path / "app.log.1.gz").write_bytes(b"\x1f\x8b\x08\x00rubbish")
    (tmp_path / "old.log.bz2").write_bytes(b"BZh91rubbish")

    names = [s["name"] for s in _find_log_files(root=tmp_path)]
    assert any(n.endswith("app.log") for n in names)
    assert not any(n.endswith(".gz") or n.endswith(".bz2") for n in names)


def test_find_docker_containers_requires_compose(tmp_path: Path, monkeypatch) -> None:
    """Docker is project-scoped: no compose file in the project → no containers,
    and docker is never even invoked."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")

    def _fail(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("docker should not be called without a compose file")

    monkeypatch.setattr("subprocess.run", _fail)
    assert _find_docker_containers(root=tmp_path) == []


def test_detect_package_manager(tmp_path: Path) -> None:
    assert _detect_package_manager(tmp_path) is None
    (tmp_path / "package.json").write_text("{}")
    assert _detect_package_manager(tmp_path) == "npm"
    (tmp_path / "pnpm-lock.yaml").write_text("")
    assert _detect_package_manager(tmp_path) == "pnpm"


def test_find_run_commands_from_package_json(tmp_path: Path) -> None:
    """Next.js/Node apps log to stdout — surface the run command, not a file."""
    (tmp_path / "pnpm-lock.yaml").write_text("")
    (tmp_path / "package.json").write_text(
        '{"scripts": {"dev": "next dev", "build": "next build", "test": "vitest"}}'
    )
    sources = _find_run_commands(root=tmp_path)
    cmds = [s["command"] for s in sources]
    assert any("pnpm run dev" in c and "log-essence -" in c for c in cmds)
    # build/test aren't long-running log sources — not surfaced
    assert not any("build" in c for c in cmds)
    assert all(s["type"] == "command" for s in sources)


def test_find_run_commands_no_package_json(tmp_path: Path) -> None:
    assert _find_run_commands(root=tmp_path) == []


def test_find_agent_files(tmp_path: Path) -> None:
    """Standard coding-agent instruction files are surfaced — they document run/test cmds."""
    (tmp_path / "CLAUDE.md").write_text("# run with pnpm dev\n")
    (tmp_path / "AGENTS.md").write_text("# agents\n")
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "copilot-instructions.md").write_text("# copilot\n")

    sources = _find_agent_files(root=tmp_path)
    names = [s["name"] for s in sources]
    assert "CLAUDE.md" in names
    assert "AGENTS.md" in names
    assert ".github/copilot-instructions.md" in names
    assert all(s["type"] == "agent" for s in sources)


def test_discover_sources_includes_commands_and_agent_files(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"scripts": {"dev": "next dev"}}')
    (tmp_path / "CLAUDE.md").write_text("# hi\n")
    types = {s["type"] for s in discover_sources(tmp_path)}
    assert "command" in types
    assert "agent" in types


def test_compose_project_name(monkeypatch) -> None:
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    assert _compose_project_name(Path("/x/Acme-Web")) == "acme-web"
    assert _compose_project_name(Path("/x/My App!")) == "myapp"
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "override")
    assert _compose_project_name(Path("/x/whatever")) == "override"


def test_format_discovery_table_empty() -> None:
    result = format_discovery_table([])
    assert "No log sources found" in result


def test_format_discovery_table_with_sources() -> None:
    sources = [
        {
            "type": "file",
            "name": "/var/log/app.log",
            "lines": 1000,
            "command": "log-essence /var/log/app.log",
        },
        {
            "type": "docker",
            "name": "web (nginx:latest)",
            "lines": "?",
            "command": "docker logs web | log-essence -",
        },
    ]
    result = format_discovery_table(sources)
    assert "file" in result
    assert "docker" in result
    assert "app.log" in result
    assert "web" in result


def test_discover_sources_returns_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sources = discover_sources()
    assert isinstance(sources, list)
    for s in sources:
        assert "type" in s
        assert "name" in s
        assert "command" in s
