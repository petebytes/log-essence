"""Command-line interface for log-essence.

Provides standalone log analysis without running as an MCP server.

Usage:
    log-essence /path/to/logs              # Analyze logs
    log-essence /var/log/*.log             # Glob patterns supported
    log-essence --serve                    # Run as MCP server
    log-essence /path/to/logs --strict     # Strict redaction mode
    log-essence /path/to/logs -o json      # JSON output format
    log-essence /path/to/logs --profile docker  # Use named profile
    log-essence demo generate script.yaml  # Generate demo video
    log-essence ui                         # Launch paste-and-copy web UI
"""

from __future__ import annotations

import argparse
import os
import signal
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


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="log-essence",
        description="Log consolidator for LLM analysis. "
        "Analyzes logs using template extraction and semantic clustering.",
        epilog="Examples:\n"
        "  log-essence /var/log/app.log\n"
        "  log-essence /var/log/*.log --severity ERROR WARNING\n"
        "  log-essence /var/log/app.log --since 1h --redact strict\n"
        "  log-essence /var/log/app.log --profile docker\n"
        "  log-essence --serve  # Run as MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="Path to log file, directory, or glob pattern",
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as MCP server instead of CLI mode",
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

    # Analysis options (defaults applied from config)
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

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def run_analysis(args: argparse.Namespace) -> int:
    """Run log analysis and print results."""
    if not args.path:
        print("Error: path is required for analysis mode", file=sys.stderr)
        print("Use --serve to run as MCP server, or provide a path", file=sys.stderr)
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

    return 0


def _signal_handler(signum: int, frame: object) -> None:
    """Handle interrupt signals gracefully."""
    global _watch_running
    _watch_running = False


def _clear_screen() -> None:
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


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


def main() -> int:
    """Main entry point for CLI."""
    # Check for demo subcommand first
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

    # Check for ui subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        try:
            from log_essence.ui import launch_ui

            # Parse optional flags
            open_browser = "--no-browser" not in sys.argv
            port = 8501
            for arg in sys.argv:
                if arg.startswith("--port="):
                    port = int(arg.split("=")[1])

            launch_ui(open_browser=open_browser, port=port)
            return 0
        except ImportError as e:
            print(
                "Error: UI dependencies not installed. Install with: pip install log-essence[ui]",
                file=sys.stderr,
            )
            print(f"Details: {e}", file=sys.stderr)
            return 1

    parser = create_parser()
    args = parser.parse_args()

    if args.serve:
        # Run as MCP server
        from log_essence.server import mcp

        mcp.run()
        return 0

    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
