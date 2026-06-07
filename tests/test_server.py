"""Tests for the log-essence server."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from log_essence.server import (
    LogEntry,
    _format_sources_for_agent,
    analyze_log_lines,
    detect_log_format,
    discover_log_sources,
    extract_exception_type,
    extract_severity,
    extract_templates,
    extract_timestamp,
    filter_by_time,
    find_error_chain,
    get_container_logs,
    get_error_chain,
    get_journald_logs,
    get_logs,
    is_error_line,
    list_containers,
    parse_duration,
    parse_log_entries,
    parse_since,
    parse_stack_frame,
    search_logs,
)


def test_detect_json_format() -> None:
    lines = [
        '{"timestamp": "2025-01-01", "level": "INFO", "message": "test"}',
        '{"timestamp": "2025-01-01", "level": "ERROR", "message": "error"}',
    ]
    assert detect_log_format(lines) == "json"


def test_detect_docker_format() -> None:
    lines = [
        "2025-12-20T10:00:00.123Z INFO Starting server",
        "2025-12-20T10:00:01.234Z DEBUG Loading config",
    ]
    assert detect_log_format(lines) == "docker"


def test_detect_syslog_format() -> None:
    lines = [
        "Dec 20 10:00:00 hostname process[1234]: message",
        "Dec 20 10:00:01 hostname process[1234]: another message",
    ]
    assert detect_log_format(lines) == "syslog"


def test_extract_severity() -> None:
    assert extract_severity("2025-01-01 ERROR something failed", "docker") == "ERROR"
    assert extract_severity("2025-01-01 WARNING low disk", "docker") == "WARNING"
    assert extract_severity("2025-01-01 INFO started", "docker") == "INFO"
    assert extract_severity("2025-01-01 DEBUG verbose", "docker") == "DEBUG"
    assert extract_severity("2025-01-01 CRITICAL panic", "docker") == "CRITICAL"


def test_extract_templates() -> None:
    lines = [
        "2025-01-01 INFO User 123 logged in",
        "2025-01-01 INFO User 456 logged in",
        "2025-01-01 INFO User 789 logged in",
    ]
    templates = extract_templates(lines, "docker")
    assert len(templates) == 1
    assert templates[0].count == 3
    assert "<*>" in templates[0].template


def test_normalize_severity_aliases() -> None:
    from log_essence.server import _normalize_severity

    assert _normalize_severity("fatal") == "CRITICAL"
    assert _normalize_severity("WARN") == "WARNING"
    assert _normalize_severity("err") == "ERROR"
    assert _normalize_severity("trace") == "DEBUG"
    assert _normalize_severity("notice") == "INFO"
    assert _normalize_severity("INFO") == "INFO"
    assert _normalize_severity(None) is None
    assert _normalize_severity("") is None
    assert _normalize_severity("weirdlevel") == "WEIRDLEVEL"


def test_severity_rank_order() -> None:
    from log_essence.server import _severity_rank

    assert _severity_rank("CRITICAL") > _severity_rank("ERROR")
    assert _severity_rank("ERROR") > _severity_rank("WARNING")
    assert _severity_rank("WARNING") > _severity_rank("INFO")
    assert _severity_rank("INFO") > _severity_rank("DEBUG")
    assert _severity_rank(None) == 0
    assert _severity_rank("UNKNOWN") == 0


def test_extract_severity_json_normalized() -> None:
    assert extract_severity('{"level":"fatal","message":"x"}', "json") == "CRITICAL"
    assert extract_severity('{"level":"warn","message":"x"}', "json") == "WARNING"
    assert extract_severity('{"severity":"ERR","message":"x"}', "json") == "ERROR"
    assert extract_severity('{"level":"trace","message":"x"}', "json") == "DEBUG"


@pytest.mark.parametrize(
    "log_level,filter_level",
    [("warn", "warning"), ("err", "error"), ("fatal", "critical"), ("trace", "debug")],
)
def test_analyze_alias_filter_normalized(log_level: str, filter_level: str) -> None:
    lines = ['{"level":"info","message":"a"}'] * 3 + [
        f'{{"level":"{log_level}","message":"b"}}'
    ] * 2
    result = analyze_log_lines(
        lines, token_budget=8000, num_clusters=10, severity_filter=[filter_level], redact=False
    )
    assert "No log patterns found" not in result.markdown


@pytest.mark.parametrize(
    "log_level,filter_level",
    [("warn", "warning"), ("err", "error"), ("fatal", "critical"), ("trace", "debug")],
)
def test_search_logs_alias_filter_normalized(
    tmp_path: Path, log_level: str, filter_level: str
) -> None:
    log_file = tmp_path / "s.log"
    log_file.write_text(
        '{"level":"info","message":"connection alpha"}\n'
        f'{{"level":"{log_level}","message":"connection beta"}}\n'
    )
    result = search_logs(path=str(log_file), query="connection", severity_filter=[filter_level])
    assert "Search Results" in result


def test_extract_templates_tracks_severity_over_all_lines() -> None:
    lines = ['{"level":"info","message":"request handled"}'] * 100
    lines.append('{"level":"error","message":"request handled"}')

    templates = extract_templates(lines, "json")

    assert len(templates) == 1
    t = templates[0]
    assert t.count == 101
    assert t.severity_counts == {"INFO": 100, "ERROR": 1}
    assert t.severity == "ERROR"  # highest present, not modal
    assert any(extract_severity(ex, "json") == "ERROR" for ex in t.examples)


def test_severity_distribution_accurate_for_collapsed_json() -> None:
    lines = ['{"level":"info","message":"x"}'] * 100 + ['{"level":"error","message":"x"}']
    result = analyze_log_lines(lines, token_budget=8000, num_clusters=10, redact=False)
    assert result.severity_distribution == {"INFO": 100, "ERROR": 1}


def test_severity_filter_matches_non_highest_level() -> None:
    lines = ['{"level":"info","message":"x"}'] * 50 + ['{"level":"error","message":"x"}']
    result = analyze_log_lines(
        lines, token_budget=8000, num_clusters=10, severity_filter=["INFO"], redact=False
    )
    # Template's highest severity is ERROR but it has 50 INFO lines -> must be kept
    assert "No log patterns found" not in result.markdown


def test_cluster_severity_rank() -> None:
    from log_essence.server import (
        LogTemplate,
        SemanticCluster,
        _cluster_severity_rank,
        _severity_rank,
    )

    mixed = SemanticCluster(
        templates=[
            LogTemplate("a", 1, 100, severity="INFO", severity_counts={"INFO": 100}),
            LogTemplate("b", 2, 1, severity="ERROR", severity_counts={"ERROR": 1}),
        ],
        centroid_idx=0,
        total_count=101,
        summary="a",
    )
    assert _cluster_severity_rank(mixed) == _severity_rank("ERROR")

    unlabeled = SemanticCluster(
        templates=[LogTemplate("x", 3, 5)], centroid_idx=0, total_count=5, summary="x"
    )
    assert _cluster_severity_rank(unlabeled) == 0


def test_error_cluster_ordered_before_info_cluster() -> None:
    info_lines = [f"2025-01-01 INFO heartbeat ping {i}" for i in range(500)]
    error_lines = ["2025-01-01 ERROR payment gateway timeout"]
    result = analyze_log_lines(
        info_lines + error_lines, token_budget=8000, num_clusters=10, redact=False
    )
    md = result.markdown
    assert "payment gateway timeout" in md
    assert "heartbeat ping" in md
    assert md.index("payment gateway timeout") < md.index("heartbeat ping")


def test_clusters_data_ordered_by_severity() -> None:
    info_lines = [f"2025-01-01 INFO heartbeat ping {i}" for i in range(500)]
    error_lines = ["2025-01-01 ERROR payment gateway timeout"]
    result = analyze_log_lines(
        info_lines + error_lines, token_budget=8000, num_clusters=10, redact=False
    )
    assert result.clusters_data is not None
    first = result.clusters_data[0]  # what CLI --format json / UI "Save JSON" emit first
    assert first.id == 1
    assert any(
        "payment gateway timeout" in t.template or t.severity == "ERROR" for t in first.templates
    )


def test_log_patterns_heading_is_severity() -> None:
    info_lines = [f"2025-01-01 INFO ping {i}" for i in range(5)]
    result = analyze_log_lines(info_lines, token_budget=8000, num_clusters=3, redact=False)
    assert "## Log Patterns by Severity" in result.markdown
    assert "## Log Patterns by Frequency" not in result.markdown


def test_get_logs_file_not_found() -> None:
    result = get_logs(path="/nonexistent/path.log")
    assert "Error: Path does not exist" in result


def test_get_logs_with_sample(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO Server started
2025-01-01T10:00:01Z INFO Request received from 192.168.1.1
2025-01-01T10:00:02Z INFO Request received from 192.168.1.2
2025-01-01T10:00:03Z ERROR Database connection failed
2025-01-01T10:00:04Z WARNING Low disk space
"""
    )

    result = get_logs(path=str(log_file), token_budget=2000, num_clusters=3)
    assert "# Log Analysis Summary" in result
    assert "Total lines:" in result
    assert "ERROR" in result or "WARNING" in result


def test_get_logs_severity_filter(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO Server started
2025-01-01T10:00:01Z ERROR Database connection failed for host 1
2025-01-01T10:00:02Z ERROR Database connection failed for host 2
2025-01-01T10:00:03Z ERROR Database connection failed for host 3
2025-01-01T10:00:04Z WARNING Low memory
2025-01-01T10:00:05Z DEBUG Verbose output
"""
    )

    result = get_logs(
        path=str(log_file),
        token_budget=2000,
        num_clusters=3,
        severity_filter=["ERROR"],
    )
    # Should only include ERROR patterns
    assert "ERROR" in result


# Phase 2: Time filtering tests


def test_parse_duration() -> None:
    assert parse_duration("1h") == timedelta(hours=1)
    assert parse_duration("30m") == timedelta(minutes=30)
    assert parse_duration("2d") == timedelta(days=2)
    assert parse_duration("1w") == timedelta(weeks=1)
    assert parse_duration("60s") == timedelta(seconds=60)
    assert parse_duration("invalid") is None
    assert parse_duration("") is None


def test_parse_since_duration() -> None:
    result = parse_since("1h")
    assert result is not None
    # Should be about 1 hour ago (within a few seconds)
    expected = datetime.now(UTC) - timedelta(hours=1)
    assert abs((result - expected).total_seconds()) < 5


def test_parse_since_datetime() -> None:
    result = parse_since("2025-01-15")
    assert result is not None
    assert result.year == 2025
    assert result.month == 1
    assert result.day == 15


def test_parse_since_full_datetime() -> None:
    result = parse_since("2025-01-15T10:30:00")
    assert result is not None
    assert result.year == 2025
    assert result.hour == 10
    assert result.minute == 30


def test_parse_since_invalid() -> None:
    assert parse_since("not-a-date") is None
    assert parse_since("yesterday") is None


def test_extract_timestamp_docker_format() -> None:
    line = "2025-12-20T10:00:00.123Z INFO Server started"
    ts = extract_timestamp(line, "docker")
    assert ts is not None
    assert ts.year == 2025
    assert ts.month == 12
    assert ts.day == 20
    assert ts.hour == 10


def test_extract_timestamp_json_format() -> None:
    line = '{"timestamp": "2025-01-15T14:30:00Z", "level": "INFO", "message": "test"}'
    ts = extract_timestamp(line, "json")
    assert ts is not None
    assert ts.year == 2025
    assert ts.month == 1
    assert ts.hour == 14


def test_filter_by_time() -> None:
    lines = [
        "2025-01-01T10:00:00Z INFO Old message",
        "2025-12-15T10:00:00Z INFO New message",
        "2025-12-20T10:00:00Z INFO Latest message",
    ]
    since = datetime(2025, 12, 1, tzinfo=UTC)
    filtered = filter_by_time(lines, since, "docker")
    assert len(filtered) == 2
    assert "Old message" not in " ".join(filtered)
    assert "New message" in " ".join(filtered)
    assert "Latest message" in " ".join(filtered)


def test_get_logs_with_since(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text(
        """2020-01-01T10:00:00Z INFO Very old message
2025-12-19T10:00:00Z INFO Recent message 1
2025-12-20T10:00:00Z INFO Recent message 2
"""
    )

    # Test with since filter that should exclude very old message
    result = get_logs(path=str(log_file), token_budget=2000, since="2025-12-01")
    assert "# Log Analysis Summary" in result


def test_get_logs_invalid_since() -> None:
    result = get_logs(path="/some/path", since="invalid-time-format")
    assert "Error: Invalid time format" in result


# Phase 2: Glob pattern tests


def test_get_logs_with_glob_pattern(tmp_path: Path) -> None:
    # Create multiple log files
    (tmp_path / "app1.log").write_text("2025-01-01T10:00:00Z INFO Message from app1\n")
    (tmp_path / "app2.log").write_text("2025-01-01T10:00:00Z ERROR Error from app2\n")
    (tmp_path / "other.txt").write_text("Some other content\n")

    # Test glob pattern
    result = get_logs(path=str(tmp_path / "*.log"), token_budget=2000)
    assert "# Log Analysis Summary" in result


# Phase 2: Docker container tests


def test_get_container_logs_docker_not_available() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        result = get_container_logs(container="test-container")
        assert "Docker not found" in result


def test_list_containers_docker_not_available() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        result = list_containers()
        assert "Docker not found" in result


# Phase 2: Journald tests


def test_get_journald_logs_not_available() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        result = get_journald_logs(unit="nginx")
        assert "journalctl not found" in result


@pytest.mark.skipif(
    not Path("/run/systemd/system").exists(),
    reason="Not a systemd system",
)
def test_get_journald_logs_integration() -> None:
    # Only runs on systems with journald
    result = get_journald_logs(lines_limit=10, token_budget=2000)
    # Should return either logs or an empty message, not an error
    assert "Error:" not in result or "No logs found" in result


# Phase 3: Stack trace extraction tests


def test_parse_stack_frame_python() -> None:
    line = '  File "/app/main.py", line 42, in process_request'
    frame = parse_stack_frame(line)
    assert frame is not None
    assert frame.file == "/app/main.py"
    assert frame.line == 42
    assert frame.function == "process_request"
    assert frame.language == "python"


def test_parse_stack_frame_java() -> None:
    line = "    at com.example.MyClass.myMethod(MyClass.java:123)"
    frame = parse_stack_frame(line)
    assert frame is not None
    assert frame.file == "MyClass.java"
    assert frame.line == 123
    assert frame.function == "com.example.MyClass.myMethod"
    assert frame.language == "java"


def test_parse_stack_frame_javascript() -> None:
    line = "    at processRequest (/app/server.js:456:12)"
    frame = parse_stack_frame(line)
    assert frame is not None
    assert frame.file == "/app/server.js"
    assert frame.line == 456
    assert frame.function == "processRequest"
    assert frame.language == "javascript"


def test_parse_stack_frame_not_a_frame() -> None:
    line = "2025-01-01T10:00:00Z INFO Normal log message"
    frame = parse_stack_frame(line)
    assert frame is None


def test_extract_exception_type() -> None:
    assert extract_exception_type("ValueError: invalid input") == "ValueError"
    assert extract_exception_type("ConnectionError: timeout") == "ConnectionError"
    assert extract_exception_type("Caused by: DatabaseError: connection refused") is not None
    assert extract_exception_type("Normal log message") is None


def test_is_error_line() -> None:
    assert is_error_line("Connection failed", "ERROR") is True
    assert is_error_line("Connection failed", "INFO") is True  # has "failed" keyword
    assert is_error_line("Server started", "INFO") is False
    assert is_error_line("CRITICAL: system panic", None) is True


def test_parse_log_entries() -> None:
    lines = [
        "2025-01-01T10:00:00Z INFO Server started",
        "2025-01-01T10:00:01Z ERROR Database connection failed",
        '  File "/app/db.py", line 42, in connect',
        "2025-01-01T10:00:02Z INFO Retry successful",
    ]
    entries = parse_log_entries(lines, "docker")
    assert len(entries) == 3
    assert entries[0].severity == "INFO"
    assert entries[1].is_error is True
    assert len(entries[1].stack_frames) == 1
    assert entries[1].stack_frames[0].file == "/app/db.py"


def test_find_error_chain() -> None:
    entries = [
        LogEntry(
            line_number=1,
            raw_line="2025-01-01T10:00:00Z INFO Server started",
            timestamp=datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC),
            severity="INFO",
            message="Server started",
            is_error=False,
        ),
        LogEntry(
            line_number=2,
            raw_line="2025-01-01T10:00:30Z WARNING Low memory",
            timestamp=datetime(2025, 1, 1, 10, 0, 30, tzinfo=UTC),
            severity="WARNING",
            message="Low memory",
            is_error=False,
        ),
        LogEntry(
            line_number=3,
            raw_line="2025-01-01T10:00:45Z ERROR Out of memory",
            timestamp=datetime(2025, 1, 1, 10, 0, 45, tzinfo=UTC),
            severity="ERROR",
            message="Out of memory",
            is_error=True,
        ),
    ]

    chain = find_error_chain(entries, entries[2], time_window_seconds=60)
    assert chain.root_cause == entries[2]  # The error itself is root cause
    assert len(chain.related_entries) == 2  # Both previous entries are within 60s


def test_get_error_chain_tool(tmp_path: Path) -> None:
    log_file = tmp_path / "errors.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO Application starting
2025-01-01T10:00:10Z WARNING Database slow
2025-01-01T10:00:15Z ERROR Connection timeout
2025-01-01T10:00:20Z INFO Retry attempt
2025-01-01T10:00:25Z ERROR Connection failed again
"""
    )

    result = get_error_chain(path=str(log_file), time_window=30)
    assert "Error Chain Analysis" in result
    assert "Root Cause" in result


def test_get_error_chain_with_pattern(tmp_path: Path) -> None:
    log_file = tmp_path / "errors.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z ERROR Database error
2025-01-01T10:00:10Z ERROR Network error
2025-01-01T10:00:20Z ERROR Database timeout
"""
    )

    result = get_error_chain(path=str(log_file), error_pattern="database")
    assert "Error Chain Analysis" in result
    # Should only find database-related errors
    assert "Database" in result


def test_get_error_chain_no_errors(tmp_path: Path) -> None:
    log_file = tmp_path / "clean.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO All good
2025-01-01T10:00:10Z INFO Still good
"""
    )

    result = get_error_chain(path=str(log_file))
    assert "No errors found" in result


# Phase 3: Semantic search tests


def test_search_logs_tool(tmp_path: Path) -> None:
    log_file = tmp_path / "search.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO User logged in successfully
2025-01-01T10:00:01Z INFO Payment processed for order 12345
2025-01-01T10:00:02Z ERROR Database connection timeout
2025-01-01T10:00:03Z INFO User session created
2025-01-01T10:00:04Z WARNING Memory usage high
"""
    )

    result = search_logs(path=str(log_file), query="database timeout error", top_k=3)
    assert "Search Results" in result
    assert "similarity:" in result


def test_search_logs_with_severity_filter(tmp_path: Path) -> None:
    log_file = tmp_path / "search.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO Connection established
2025-01-01T10:00:01Z ERROR Connection failed
2025-01-01T10:00:02Z INFO Retrying connection
"""
    )

    result = search_logs(path=str(log_file), query="connection", severity_filter=["ERROR"])
    assert "Search Results" in result


def test_search_logs_no_matches(tmp_path: Path) -> None:
    log_file = tmp_path / "empty.log"
    log_file.write_text("")

    result = search_logs(path=str(log_file), query="anything")
    assert "No log content found" in result or "No matching" in result


# --- Source discovery (MCP) ---


def test_format_sources_for_agent_maps_types_to_mcp_tools() -> None:
    """Discovered sources are rendered with the MCP tool an agent should call."""
    sources = [
        {
            "type": "file",
            "name": "/var/log/app.log",
            "lines": 100,
            "command": "log-essence /var/log/app.log",
        },
        {
            "type": "docker",
            "name": "web (nginx:latest)",
            "lines": "?",
            "command": "docker logs web 2>&1 | log-essence -",
        },
        {
            "type": "journald",
            "name": "system journal",
            "lines": "?",
            "command": "journalctl | log-essence -",
        },
    ]
    result = _format_sources_for_agent(sources)
    assert "/var/log/app.log" in result
    assert "get_logs" in result  # file -> get_logs
    assert "get_container_logs" in result  # docker -> get_container_logs
    assert 'container="web"' in result  # image suffix stripped to bare name
    assert "get_journald_logs" in result  # journald -> get_journald_logs


def test_format_sources_for_agent_empty() -> None:
    result = _format_sources_for_agent([])
    assert "No log sources" in result


def test_format_sources_for_agent_renders_commands_and_agent_docs() -> None:
    """Run-commands (stdout capture) and agent-instruction files get their own guidance."""
    sources = [
        {
            "type": "command",
            "name": "dev (package.json)",
            "lines": "stream",
            "command": "pnpm run dev 2>&1 | log-essence -",
        },
        {
            "type": "agent",
            "name": "CLAUDE.md",
            "lines": "-",
            "command": "documents how to run/build/test",
        },
    ]
    result = _format_sources_for_agent(sources)
    assert "pnpm run dev 2>&1 | log-essence -" in result
    assert "CLAUDE.md" in result


def test_discover_log_sources_lists_local_files(tmp_path: Path, monkeypatch) -> None:
    """The MCP tool surfaces local log files agents otherwise can't see."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.log").write_text("INFO line one\nERROR boom\n")
    result = discover_log_sources()
    assert isinstance(result, str)
    assert "app.log" in result


def test_discover_log_sources_accepts_path(tmp_path: Path) -> None:
    """An agent can scope discovery to a specific project directory."""
    (tmp_path / "svc.log").write_text("a\nb\n")
    result = discover_log_sources(path=str(tmp_path))
    assert "svc.log" in result
