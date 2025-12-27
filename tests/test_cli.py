"""Tests for the CLI module."""

import json
from pathlib import Path

from log_essence import __version__
from log_essence.cli import create_parser, run_analysis
from log_essence.config import (
    Config,
    ConfigDefaults,
    ConfigProfile,
    load_config,
    merge_config_with_args,
)


def test_parser_defaults() -> None:
    """CLI args default to None - config provides actual defaults."""
    parser = create_parser()
    args = parser.parse_args(["/path/to/logs"])
    assert args.path == "/path/to/logs"
    # CLI args default to None; config module provides defaults
    assert args.token_budget is None
    assert args.clusters is None
    assert args.redact is None
    assert args.output is None
    assert args.serve is False
    assert args.config is None
    assert args.profile is None


def test_parser_all_options() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "/var/log/app.log",
            "--token-budget",
            "4000",
            "--clusters",
            "5",
            "--severity",
            "ERROR",
            "WARNING",
            "--since",
            "1h",
            "--redact",
            "strict",
        ]
    )
    assert args.path == "/var/log/app.log"
    assert args.token_budget == 4000
    assert args.clusters == 5
    assert args.severity == ["ERROR", "WARNING"]
    assert args.since == "1h"
    assert args.redact == "strict"


def test_parser_serve_mode() -> None:
    parser = create_parser()
    args = parser.parse_args(["--serve"])
    assert args.serve is True
    assert args.path is None


def test_parser_no_redact() -> None:
    parser = create_parser()
    args = parser.parse_args(["/path", "--no-redact"])
    assert args.no_redact is True


def test_run_analysis_missing_path() -> None:
    parser = create_parser()
    args = parser.parse_args([])
    result = run_analysis(args)
    assert result == 1  # Error exit code


def test_run_analysis_nonexistent_path() -> None:
    parser = create_parser()
    args = parser.parse_args(["/nonexistent/path/to/logs"])
    result = run_analysis(args)
    assert result == 1  # Error exit code


def test_run_analysis_invalid_since() -> None:
    parser = create_parser()
    args = parser.parse_args(["/some/path", "--since", "invalid"])
    result = run_analysis(args)
    assert result == 1  # Error exit code


def test_run_analysis_success(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO Server started
2025-01-01T10:00:01Z INFO Request received
2025-01-01T10:00:02Z ERROR Connection failed
"""
    )

    parser = create_parser()
    args = parser.parse_args([str(log_file)])
    result = run_analysis(args)
    assert result == 0  # Success


def test_run_analysis_with_severity_filter(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO Server started
2025-01-01T10:00:01Z ERROR Connection failed
2025-01-01T10:00:02Z WARNING Low memory
"""
    )

    parser = create_parser()
    args = parser.parse_args([str(log_file), "--severity", "ERROR"])
    result = run_analysis(args)
    assert result == 0


def test_run_analysis_with_redaction(tmp_path: Path, capsys) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text("2025-01-01T10:00:00Z INFO user@example.com logged in\n")

    parser = create_parser()
    args = parser.parse_args([str(log_file)])
    result = run_analysis(args)

    captured = capsys.readouterr()
    assert result == 0
    assert "user@example.com" not in captured.out
    assert "[EMAIL:" in captured.out


def test_run_analysis_no_redact(tmp_path: Path, capsys) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text("2025-01-01T10:00:00Z INFO user@example.com logged in\n")

    parser = create_parser()
    args = parser.parse_args([str(log_file), "--no-redact"])
    result = run_analysis(args)

    captured = capsys.readouterr()
    assert result == 0
    assert "user@example.com" in captured.out


def test_parser_output_format_default() -> None:
    """CLI output arg defaults to None - config provides 'markdown' default."""
    parser = create_parser()
    args = parser.parse_args(["/path/to/logs"])
    # CLI args default to None; config module provides "markdown" default
    assert args.output is None


def test_parser_output_format_json() -> None:
    parser = create_parser()
    args = parser.parse_args(["/path/to/logs", "-o", "json"])
    assert args.output == "json"


def test_parser_output_format_long_flag() -> None:
    parser = create_parser()
    args = parser.parse_args(["/path/to/logs", "--output", "json"])
    assert args.output == "json"


def test_run_analysis_json_output(tmp_path: Path, capsys) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text(
        """2025-01-01T10:00:00Z INFO Server started
2025-01-01T10:00:01Z INFO Request received
2025-01-01T10:00:02Z ERROR Connection failed
"""
    )

    parser = create_parser()
    args = parser.parse_args([str(log_file), "-o", "json"])
    result = run_analysis(args)

    captured = capsys.readouterr()
    assert result == 0

    # Verify it's valid JSON
    output = json.loads(captured.out)

    # Verify structure
    assert "metadata" in output
    assert "stats" in output
    assert "severity_distribution" in output
    assert "clusters" in output

    # Verify metadata fields
    assert output["metadata"]["source"] == str(log_file)
    assert output["metadata"]["lines_processed"] == 3
    assert output["metadata"]["log_format"] == "docker"
    assert "timestamp" in output["metadata"]
    assert output["metadata"]["version"] == __version__

    # Verify stats fields
    assert "processing_time_ms" in output["stats"]
    assert "original_tokens" in output["stats"]
    assert "output_tokens" in output["stats"]

    # Verify clusters is a list
    assert isinstance(output["clusters"], list)
    if output["clusters"]:
        cluster = output["clusters"][0]
        assert "id" in cluster
        assert "summary" in cluster
        assert "total_count" in cluster
        assert "templates" in cluster


def test_run_analysis_json_output_with_redaction(tmp_path: Path, capsys) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text("2025-01-01T10:00:00Z INFO user@example.com logged in\n")

    parser = create_parser()
    args = parser.parse_args([str(log_file), "-o", "json"])
    result = run_analysis(args)

    captured = capsys.readouterr()
    assert result == 0

    output = json.loads(captured.out)
    assert output["stats"]["redaction_count"] > 0

    # Email should be redacted in the output
    output_str = json.dumps(output)
    assert "user@example.com" not in output_str
    assert "[EMAIL:" in output_str


# Config tests


def test_load_config_no_file() -> None:
    """When no config file exists, returns defaults."""
    config = load_config(Path("/nonexistent/config.yaml"))
    assert config.defaults.token_budget == 8000
    assert config.defaults.clusters == 10
    assert config.defaults.redaction == "moderate"
    assert config.defaults.output == "markdown"
    assert config.profiles == {}


def test_load_config_from_file(tmp_path: Path) -> None:
    """Load config from YAML file."""
    config_file = tmp_path / ".log-essence.yaml"
    config_file.write_text("""
defaults:
  token_budget: 16000
  clusters: 20
  redaction: strict
  output: json
  severity:
    - ERROR
    - WARNING

profiles:
  docker:
    clusters: 30
    since: 1h
""")

    config = load_config(config_file)
    assert config.defaults.token_budget == 16000
    assert config.defaults.clusters == 20
    assert config.defaults.redaction == "strict"
    assert config.defaults.output == "json"
    assert config.defaults.severity == ["ERROR", "WARNING"]
    assert "docker" in config.profiles
    assert config.profiles["docker"].clusters == 30
    assert config.profiles["docker"].since == "1h"


def test_merge_config_cli_overrides() -> None:
    """CLI args override config defaults."""
    config = Config(
        defaults=ConfigDefaults(
            token_budget=16000,
            clusters=20,
            redaction="strict",
        )
    )

    # CLI provides explicit values
    merged = merge_config_with_args(
        config,
        token_budget=4000,
        clusters=5,
    )

    assert merged["token_budget"] == 4000  # CLI override
    assert merged["clusters"] == 5  # CLI override
    assert merged["redaction"] == "strict"  # From config (CLI didn't override)


def test_merge_config_profile_applies() -> None:
    """Profile values apply when profile specified."""
    config = Config(
        defaults=ConfigDefaults(
            token_budget=8000,
            clusters=10,
        ),
        profiles={
            "docker": ConfigProfile(
                clusters=30,
                since="2h",
            )
        },
    )

    merged = merge_config_with_args(config, profile_name="docker")

    assert merged["token_budget"] == 8000  # From defaults
    assert merged["clusters"] == 30  # From profile
    assert merged["since"] == "2h"  # From profile


def test_merge_config_cli_overrides_profile() -> None:
    """CLI args override profile values."""
    config = Config(
        profiles={
            "docker": ConfigProfile(
                clusters=30,
            )
        },
    )

    merged = merge_config_with_args(
        config,
        profile_name="docker",
        clusters=5,  # CLI override
    )

    assert merged["clusters"] == 5  # CLI wins over profile


def test_run_analysis_with_config_file(tmp_path: Path, capsys) -> None:
    """Config file values are used when no CLI args provided."""
    # Create config file
    config_file = tmp_path / ".log-essence.yaml"
    config_file.write_text("""
defaults:
  token_budget: 2000
  clusters: 3
  output: json
""")

    # Create log file
    log_file = tmp_path / "test.log"
    log_file.write_text("2025-01-01T10:00:00Z INFO Server started\n")

    parser = create_parser()
    args = parser.parse_args([str(log_file), "--config", str(config_file)])
    result = run_analysis(args)

    captured = capsys.readouterr()
    assert result == 0

    # Should output JSON (from config)
    output = json.loads(captured.out)
    assert "metadata" in output


def test_run_analysis_with_profile(tmp_path: Path, capsys) -> None:
    """Profile values are applied when --profile specified."""
    # Create config file with profile
    config_file = tmp_path / ".log-essence.yaml"
    config_file.write_text("""
defaults:
  output: markdown

profiles:
  json-mode:
    output: json
""")

    # Create log file
    log_file = tmp_path / "test.log"
    log_file.write_text("2025-01-01T10:00:00Z INFO Server started\n")

    parser = create_parser()
    args = parser.parse_args(
        [
            str(log_file),
            "--config",
            str(config_file),
            "--profile",
            "json-mode",
        ]
    )
    result = run_analysis(args)

    captured = capsys.readouterr()
    assert result == 0

    # Should output JSON (from profile)
    output = json.loads(captured.out)
    assert "metadata" in output


# Watch mode tests


def test_parser_watch_mode() -> None:
    """Watch mode flags are parsed correctly."""
    parser = create_parser()
    args = parser.parse_args(["/path/to/logs", "-w"])
    assert args.watch is True
    assert args.interval == 3.0  # Default interval


def test_parser_watch_mode_with_interval() -> None:
    """Custom interval is parsed correctly."""
    parser = create_parser()
    args = parser.parse_args(["/path/to/logs", "--watch", "--interval", "5.0"])
    assert args.watch is True
    assert args.interval == 5.0


def test_watch_mode_requires_single_file(tmp_path: Path, capsys) -> None:
    """Watch mode should fail when given multiple files via glob."""
    # Create multiple log files
    (tmp_path / "app1.log").write_text("2025-01-01T10:00:00Z INFO Log 1\n")
    (tmp_path / "app2.log").write_text("2025-01-01T10:00:00Z INFO Log 2\n")

    parser = create_parser()
    args = parser.parse_args([str(tmp_path), "--watch"])
    result = run_analysis(args)

    captured = capsys.readouterr()
    assert result == 1
    assert "Watch mode only supports a single file" in captured.err
