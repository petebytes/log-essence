"""Command-line interface for log-essence.

Provides standalone log analysis without running as an MCP server.

Usage:
    log-essence /path/to/logs              # Analyze logs (default subcommand)
    log-essence analyze /path/to/logs      # Explicit analyze
    log-essence serve                      # Run as MCP server
    log-essence stats                      # Show cumulative analytics
    log-essence init                       # Auto-configure AI tools
    log-essence discover                   # Find log sources
    log-essence ui                         # Launch paste-and-copy web UI
    log-essence demo generate script.yaml  # Generate demo video
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from log_essence import __version__
from log_essence.config import load_config, merge_config_with_args
from log_essence.server import (
    analyze_log_lines,
    detect_log_format,
    filter_by_time,
    parse_since,
    resolve_glob_pattern,
)
from log_essence.ui.models import JSONOutput, MetadataOutput

# Global flag for graceful shutdown in watch mode
_watch_running = True

# Built-in defaults (used if no config file exists)
DEFAULT_TOKEN_BUDGET = 8000
DEFAULT_CLUSTERS = 10
DEFAULT_REDACTION = "moderate"
DEFAULT_OUTPUT = "markdown"


def _running_under_uvx() -> bool:
    """Detect uv-managed tool environment.

    Covers both ephemeral (~/.cache/uv/archive-v*) and persistent
    (~/.local/share/uv/tools/<name>) install paths used by uvx / uv tool run.
    """
    prefix = sys.prefix
    return "/uv/archive-v" in prefix or "/uv/tools/" in prefix


def _handle_ui_missing(err: ImportError, args: argparse.Namespace) -> int:
    """Auto-relaunch under uvx with [ui] extras, else print install hint."""
    if os.environ.get("LOG_ESSENCE_UI_BOOTSTRAP") == "1":
        print(f"Error: UI bootstrap failed: {err}", file=sys.stderr)
        return 1

    if _running_under_uvx() and (uvx := shutil.which("uvx")):
        print(
            "UI deps missing. Relaunching via 'uvx --from log-essence[ui]'...",
            file=sys.stderr,
        )
        cmd = [uvx, "--from", "log-essence[ui]", "log-essence", "ui"]
        if getattr(args, "no_browser", False):
            cmd.append("--no-browser")
        if getattr(args, "port", None):
            cmd.extend(["--port", str(args.port)])
        env = {**os.environ, "LOG_ESSENCE_UI_BOOTSTRAP": "1"}
        return subprocess.call(cmd, env=env)

    print(
        "Error: UI dependencies not installed.\n"
        "  uvx:  uvx --from 'log-essence[ui]' log-essence ui\n"
        "  pip:  pip install 'log-essence[ui]'\n"
        "  uv:   uv tool install 'log-essence[ui]'",
        file=sys.stderr,
    )
    print(f"Details: {err}", file=sys.stderr)
    return 1


def _add_analysis_args(parser: argparse.ArgumentParser) -> None:
    """Add common analysis arguments to a parser or subparser."""
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to log file, directory, or glob pattern",
    )

    # Config file options
    parser.add_argument(
        "--config",
        type=Path,
        metavar="FILE",
        help="Path to config file (default: auto-detect)",
    )

    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="Use named configuration profile",
    )

    # Analysis options
    parser.add_argument(
        "--token-budget",
        type=int,
        metavar="N",
        help=f"Maximum tokens in output (default: {DEFAULT_TOKEN_BUDGET})",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        metavar="N",
        help=f"Number of semantic clusters (default: {DEFAULT_CLUSTERS})",
    )

    parser.add_argument(
        "--severity",
        nargs="+",
        metavar="LEVEL",
        help="Filter by severity levels (e.g., ERROR WARNING)",
    )

    parser.add_argument(
        "--since",
        metavar="TIME",
        help="Only logs since TIME (e.g., 1h, 30m, 2d, 2025-01-01)",
    )

    parser.add_argument(
        "--redact",
        choices=["strict", "moderate", "minimal", "disabled"],
        help=f"Redaction mode for secrets/PII (default: {DEFAULT_REDACTION})",
    )

    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable redaction (alias for --redact disabled)",
    )

    parser.add_argument(
        "-o",
        "--output",
        choices=["markdown", "json"],
        help=f"Output format (default: {DEFAULT_OUTPUT})",
    )

    parser.add_argument(
        "-c",
        "--compact",
        action="store_true",
        help="Compact output mode (reduced tokens for AI agent consumption)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress stats footer on stderr",
    )

    # Watch mode options
    parser.add_argument(
        "-w",
        "--watch",
        action="store_true",
        help="Watch log file for changes and continuously update analysis",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Update interval in seconds for watch mode (default: 3.0)",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="log-essence",
        description="Log consolidator for LLM analysis. "
        "Analyzes logs using template extraction and semantic clustering.",
        epilog="Examples:\n"
        "  log-essence /var/log/app.log\n"
        "  log-essence /var/log/*.log --severity ERROR WARNING\n"
        "  log-essence serve\n"
        "  log-essence stats\n"
        "  log-essence init --tool claude-desktop\n"
        "  log-essence discover",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- analyze (default when path given) ---
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze log files (default command)",
    )
    _add_analysis_args(analyze_parser)

    # --- serve ---
    subparsers.add_parser(
        "serve",
        help="Run as MCP server",
    )

    # --- stats ---
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show cumulative analytics dashboard",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        dest="stats_json",
        help="Output stats as JSON",
    )
    stats_parser.add_argument(
        "--since",
        metavar="TIME",
        help="Only show stats since TIME (e.g., 7d, 1h)",
    )
    stats_parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear all analytics data",
    )

    # --- init ---
    init_parser = subparsers.add_parser(
        "init",
        help="Auto-configure log-essence for AI coding tools",
    )
    init_parser.add_argument(
        "--tool",
        choices=["claude-desktop", "claude-code"],
        help="Target a specific AI tool (default: auto-detect)",
    )
    init_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )
    init_parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove log-essence configuration",
    )

    # --- discover ---
    subparsers.add_parser(
        "discover",
        help="Scan for analyzable log sources",
    )

    # --- ui ---
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Streamlit web UI",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the web UI (default: 8501)",
    )
    ui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    # --- demo ---
    subparsers.add_parser(
        "demo",
        help="Generate demo video",
        add_help=False,  # Demo has its own arg handling
    )

    # --- Backward compat: --serve flag (hidden, use 'serve' subcommand) ---
    parser.add_argument(
        "--serve",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser


# Known subcommands for backward-compat detection
_KNOWN_COMMANDS = {"analyze", "serve", "stats", "init", "discover", "ui", "demo"}


def run_analysis(args: argparse.Namespace) -> int:
    """Run log analysis and print results."""
    if not args.path:
        print("Error: path is required for analysis mode", file=sys.stderr)
        print("Use 'log-essence serve' to run as MCP server, or provide a path", file=sys.stderr)
        return 1

    # Load config and merge with CLI args
    config = load_config(args.config)
    merged = merge_config_with_args(
        config,
        profile_name=args.profile,
        token_budget=args.token_budget,
        clusters=args.clusters,
        redaction=args.redact,
        severity=args.severity,
        since=args.since,
        output=args.output,
    )

    # Extract merged values
    token_budget = int(merged["token_budget"])  # type: ignore[arg-type]
    num_clusters = int(merged["clusters"])  # type: ignore[arg-type]
    redaction_mode = str(merged["redaction"])
    severity_filter = merged["severity"]  # type: ignore[assignment]
    since_value = merged["since"]
    output_format = str(merged["output"])

    # Parse since filter
    since_dt = None
    if since_value:
        since_dt = parse_since(str(since_value))
        if since_dt is None:
            print(
                f"Error: Invalid time format for --since: {since_value}",
                file=sys.stderr,
            )
            print("Use formats like: 1h, 30m, 2d, 2025-01-01", file=sys.stderr)
            return 1

    # Resolve path
    log_files = resolve_glob_pattern(args.path)

    if not log_files:
        log_path = Path(args.path).expanduser().resolve()

        if not log_path.exists():
            print(f"Error: Path does not exist: {args.path}", file=sys.stderr)
            return 1

        if log_path.is_file():
            log_files = [log_path]
        else:
            log_files = list(log_path.glob("**/*.log")) + list(log_path.glob("**/*.txt"))
            if not log_files:
                print(f"Error: No log files found in {args.path}", file=sys.stderr)
                return 1

    # Read all lines
    all_lines: list[str] = []
    for log_file in log_files:
        try:
            content = log_file.read_text(errors="replace")
            all_lines.extend(content.splitlines())
        except Exception as e:
            print(f"Warning: Error reading {log_file}: {e}", file=sys.stderr)

    if not all_lines:
        print("Error: No log content found", file=sys.stderr)
        return 1

    # Apply time filter
    if since_dt:
        log_format = detect_log_format(all_lines)
        all_lines = filter_by_time(all_lines, since_dt, log_format)
        if not all_lines:
            print("No logs found matching the time filter", file=sys.stderr)
            return 1

    # Determine redaction mode
    if args.no_redact:
        redact: bool | str = False
    elif redaction_mode == "disabled":
        redact = False
    else:
        redact = redaction_mode

    # Check compact mode
    compact = getattr(args, "compact", False)

    # Watch mode
    if args.watch:
        if len(log_files) != 1:
            print("Error: Watch mode only supports a single file", file=sys.stderr)
            return 1
        return run_watch_mode(
            log_files[0],
            token_budget=token_budget,
            num_clusters=num_clusters,
            severity_filter=severity_filter,
            redact=redact,
            interval=args.interval,
        )

    # Run analysis
    result = analyze_log_lines(
        all_lines,
        token_budget=token_budget,
        num_clusters=num_clusters,
        severity_filter=severity_filter,
        redact=redact,
        compact=compact,
    )

    # Output in requested format
    if output_format == "json":
        json_output = JSONOutput(
            metadata=MetadataOutput(
                source=args.path,
                lines_processed=result.lines_processed,
                log_format=result.log_format,
                timestamp=datetime.now(UTC),
            ),
            stats=result.stats,
            severity_distribution=result.severity_distribution,
            clusters=result.clusters_data or [],
        )
        print(json_output.model_dump_json(indent=2))
    else:
        print(result.markdown)

    # Print stats footer to stderr (unless quiet)
    quiet = getattr(args, "quiet", False)
    if not quiet:
        from log_essence.analytics import format_stats_footer

        footer = format_stats_footer(
            lines_in=result.lines_processed,
            tokens_out=result.stats.output_tokens,
            tokens_in=result.stats.original_tokens,
            redactions=result.stats.redaction_count,
            duration_ms=result.stats.processing_time_ms,
        )
        print(footer, file=sys.stderr)

    # Record analytics
    from log_essence.analytics import record_analysis

    record_analysis(
        source=args.path,
        lines_in=result.lines_processed,
        tokens_in=result.stats.original_tokens,
        tokens_out=result.stats.output_tokens,
        redactions=result.stats.redaction_count,
        duration_ms=result.stats.processing_time_ms,
        log_format=result.log_format,
    )

    return 0


def _signal_handler(signum: int, frame: object) -> None:
    """Handle interrupt signals gracefully."""
    global _watch_running
    _watch_running = False


def _clear_screen() -> None:
    """Clear terminal screen using ANSI escape sequence."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def run_watch_mode(
    log_path: Path,
    *,
    token_budget: int,
    num_clusters: int,
    severity_filter: list[str] | None,
    redact: bool | str,
    interval: float,
) -> int:
    """Run continuous watch mode on a log file.

    Monitors the file for changes and re-analyzes periodically.

    Args:
        log_path: Path to the log file to watch.
        token_budget: Maximum tokens in output.
        num_clusters: Number of semantic clusters.
        severity_filter: Severity levels to include.
        redact: Redaction mode.
        interval: Update interval in seconds.

    Returns:
        Exit code (0 for success).
    """
    global _watch_running
    _watch_running = True

    # Set up signal handlers for graceful exit
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    last_size = 0
    last_mtime = 0.0
    update_count = 0

    print(f"Watching {log_path} (Ctrl+C to stop)...", file=sys.stderr)
    print(f"Update interval: {interval}s", file=sys.stderr)
    print("", file=sys.stderr)

    try:
        while _watch_running:
            # Check if file has changed
            try:
                stat = log_path.stat()
                current_size = stat.st_size
                current_mtime = stat.st_mtime
            except (FileNotFoundError, PermissionError) as e:
                print(f"Error accessing file: {e}", file=sys.stderr)
                time.sleep(interval)
                continue

            # Only re-analyze if file changed
            if current_size != last_size or current_mtime != last_mtime:
                last_size = current_size
                last_mtime = current_mtime
                update_count += 1

                # Read file content
                try:
                    content = log_path.read_text(errors="replace")
                    all_lines = content.splitlines()
                except Exception as e:
                    print(f"Error reading file: {e}", file=sys.stderr)
                    time.sleep(interval)
                    continue

                if not all_lines:
                    time.sleep(interval)
                    continue

                # Run analysis
                result = analyze_log_lines(
                    all_lines,
                    token_budget=token_budget,
                    num_clusters=num_clusters,
                    severity_filter=severity_filter,
                    redact=redact,
                )

                # Clear screen and display
                _clear_screen()
                print(f"=== log-essence watch mode | Update #{update_count} ===")
                print(f"File: {log_path}")
                print(f"Lines: {len(all_lines):,} | Format: {result.log_format}")
                print(f"Processing time: {result.stats.processing_time_ms:.0f}ms")
                print(f"Tokens: {result.stats.original_tokens:,} → {result.stats.output_tokens:,}")
                print(f"Compression: {result.stats.savings_percent:.1f}%")
                if result.stats.redaction_count > 0:
                    print(f"Redactions: {result.stats.redaction_count}")
                print("=" * 50)
                print()
                print(result.markdown)

            time.sleep(interval)

    except KeyboardInterrupt:
        pass  # Handled by signal handler

    # Final summary
    print("\n" + "=" * 50, file=sys.stderr)
    print(f"Watch mode ended. Total updates: {update_count}", file=sys.stderr)

    return 0


def _preprocess_args(argv: list[str]) -> list[str]:
    """Insert 'analyze' subcommand for backward compat when first arg is a path.

    Allows `log-essence /path/to/logs` to work the same as `log-essence analyze /path/to/logs`.
    """
    if not argv:
        return argv

    first = argv[0]

    # If first arg is a known subcommand or starts with -, don't modify
    if first in _KNOWN_COMMANDS or first.startswith("-"):
        return argv

    # First arg looks like a path — insert 'analyze' before it
    return ["analyze", *argv]


def main() -> int:
    """Main entry point for CLI."""
    # Check for demo subcommand first (it has its own arg handling)
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        try:
            from log_essence.demo.cli import main as demo_main

            return demo_main(sys.argv[2:])
        except ImportError as e:
            print(
                "Error: Demo dependencies not installed. "
                "Install with: pip install log-essence[demo]",
                file=sys.stderr,
            )
            print(f"Details: {e}", file=sys.stderr)
            return 1

    parser = create_parser()
    processed = _preprocess_args(sys.argv[1:])
    args = parser.parse_args(processed)

    # Route to subcommand
    if args.command == "serve" or getattr(args, "serve", False):
        from log_essence.server import mcp

        mcp.run()
        return 0

    if args.command == "stats":
        from log_essence.analytics import run_stats_command

        return run_stats_command(
            as_json=args.stats_json,
            since=args.since,
            reset=args.reset,
        )

    if args.command == "init":
        from log_essence.init import run_init_command

        return run_init_command(
            tool=args.tool,
            dry_run=args.dry_run,
            uninstall=args.uninstall,
        )

    if args.command == "discover":
        from log_essence.discover import run_discover_command

        return run_discover_command()

    if args.command == "ui":
        try:
            from log_essence.ui import launch_ui

            open_browser = not args.no_browser
            launch_ui(open_browser=open_browser, port=args.port)
            return 0
        except ImportError as e:
            return _handle_ui_missing(e, args)

    # Default: analyze (explicit subcommand or bare path)
    if args.command == "analyze":
        return run_analysis(args)

    # No subcommand — show help (--serve handled above)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
