"""MCP server for log consolidation and analysis."""

from __future__ import annotations

import glob as glob_module
import hashlib
import json
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from fastembed import TextEmbedding
from fastmcp import FastMCP

from log_essence.redaction import RedactionMode, redact_lines
from log_essence.ui.models import (
    AnalysisResult,
    AnalysisStats,
    ClusterOutput,
    TemplateOutput,
)

mcp = FastMCP("log-essence")

# Pre-compiled regex patterns for log format detection
LOG_PATTERNS = {
    "json": re.compile(r'^\s*\{.*"(?:message|msg|level|timestamp|time)".*\}\s*$', re.IGNORECASE),
    "syslog": re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+(?:\[\d+\])?:"),
    "apache_combined": re.compile(
        r'^[\d.]+\s+-\s+\S+\s+\[.+\]\s+"[A-Z]+\s+\S+\s+HTTP/[\d.]+"\s+\d+\s+\d+'
    ),
    "apache_error": re.compile(r"^\[(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+[A-Z][a-z]{2}\s+\d{1,2}"),
    "nginx_error": re.compile(
        r"^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[(?:debug|info|notice|warn|error|crit|alert|emerg)\]"
    ),
    "docker": re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
    "kubernetes": re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+(?:stdout|stderr)"),
    "generic_timestamp": re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}"),
}

# JSON field mappings for different log formats
JSON_MESSAGE_FIELDS = ["message", "msg", "log", "text", "body", "content"]
JSON_LEVEL_FIELDS = ["level", "severity", "lvl", "loglevel", "log_level"]
JSON_TIME_FIELDS = ["timestamp", "time", "@timestamp", "ts", "datetime", "date"]

# Timestamp extraction patterns
TIMESTAMP_PATTERNS = [
    # ISO 8601 with optional milliseconds and timezone
    re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"),
    # Syslog format (requires current year injection)
    re.compile(r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"),
]

# Stack trace patterns for various languages
STACK_TRACE_PATTERNS = {
    # Python: "  File "/path/file.py", line 123, in func_name"
    "python": re.compile(r'^\s+File "([^"]+)", line (\d+), in (.+)$'),
    # Python exception: "ExceptionType: message"
    "python_exception": re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Warning)):\s*(.*)$"),
    # Java/Kotlin: "    at com.example.Class.method(File.java:123)"
    "java": re.compile(r"^\s+at\s+([\w.$]+)\(([\w.]+):(\d+)\)$"),
    # JavaScript/Node: "    at functionName (/path/file.js:123:45)"
    "javascript": re.compile(r"^\s+at\s+(?:(\S+)\s+)?\(([^:]+):(\d+):\d+\)$"),
    # Go: "goroutine 1 [running]:" or "/path/file.go:123"
    "go": re.compile(r"^\s*([^\s]+\.go):(\d+)"),
    # Rust: "   0: rust_begin_unwind" or "at /path/file.rs:123"
    "rust": re.compile(r"^\s+at\s+([^\s]+):(\d+)"),
    # Generic "Caused by:" pattern
    "caused_by": re.compile(r"^Caused by:\s*(.+)$", re.IGNORECASE),
}

# Error severity keywords for chain detection
ERROR_KEYWORDS = re.compile(
    r"\b(error|exception|fatal|critical|panic|fail(?:ed|ure)?|crash|abort)\b",
    re.IGNORECASE,
)


def parse_duration(duration_str: str) -> timedelta | None:
    """Parse a duration string like '1h', '30m', '2d' into a timedelta."""
    match = re.match(r"^(\d+)([smhdw])$", duration_str.lower())
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)

    unit_map = {
        "s": timedelta(seconds=value),
        "m": timedelta(minutes=value),
        "h": timedelta(hours=value),
        "d": timedelta(days=value),
        "w": timedelta(weeks=value),
    }
    return unit_map.get(unit)


def parse_since(since: str) -> datetime | None:
    """Parse a --since value into a datetime.

    Accepts:
    - Duration strings: "1h", "30m", "2d", "1w"
    - ISO datetime: "2025-01-01T10:00:00"
    - Date only: "2025-01-01"
    """
    # Try duration first
    duration = parse_duration(since)
    if duration:
        return datetime.now(UTC) - duration

    # Try ISO datetime formats
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(since, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


def extract_timestamp(line: str, log_format: str) -> datetime | None:
    """Extract timestamp from a log line."""
    # Check JSON format first
    if log_format == "json":
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                return None
            for field in JSON_TIME_FIELDS:
                if field in data:
                    ts_str = str(data[field])
                    # Try parsing the timestamp
                    for fmt in [
                        "%Y-%m-%dT%H:%M:%S.%fZ",
                        "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%dT%H:%M:%S.%f",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d %H:%M:%S",
                    ]:
                        try:
                            dt = datetime.strptime(ts_str, fmt)
                            return dt.replace(tzinfo=UTC)
                        except ValueError:
                            continue
                    # Try epoch timestamps
                    try:
                        epoch = float(ts_str)
                        # Handle milliseconds
                        if epoch > 1e12:
                            epoch /= 1000
                        return datetime.fromtimestamp(epoch, tz=UTC)
                    except ValueError:
                        pass
        except json.JSONDecodeError:
            pass

    # Pattern-based extraction
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(line)
        if match:
            ts_str = match.group(1)
            # Try parsing ISO format
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ]:
                try:
                    normalized_ts = ts_str.replace("+00:00", "Z").rstrip("Z")
                    dt = datetime.strptime(normalized_ts, fmt.rstrip("Z"))
                    return dt.replace(tzinfo=UTC)
                except ValueError:
                    continue

            # Handle syslog format (Mon DD HH:MM:SS)
            try:
                # Add current year for syslog timestamps
                current_year = datetime.now().year
                dt = datetime.strptime(f"{current_year} {ts_str}", "%Y %b %d %H:%M:%S")
                return dt.replace(tzinfo=UTC)
            except ValueError:
                pass

    return None


def filter_by_time(lines: list[str], since: datetime, log_format: str) -> list[str]:
    """Filter log lines to only include those after the given timestamp."""
    filtered: list[str] = []
    for line in lines:
        ts = extract_timestamp(line, log_format)
        if ts is None or ts >= since:
            # Keep lines without timestamps (might be continuations)
            # and lines that are newer than since
            filtered.append(line)
    return filtered


@dataclass
class LogTemplate:
    """A log template extracted by Drain3."""

    template: str
    cluster_id: int
    count: int
    examples: list[str] = field(default_factory=list)
    severity: str | None = None


@dataclass
class SemanticCluster:
    """A cluster of semantically similar log templates."""

    templates: list[LogTemplate]
    centroid_idx: int
    total_count: int
    summary: str


@dataclass
class StackFrame:
    """A single frame in a stack trace."""

    file: str
    line: int
    function: str | None = None
    language: str = "unknown"


@dataclass
class LogEntry:
    """A parsed log entry with metadata."""

    line_number: int
    raw_line: str
    timestamp: datetime | None = None
    severity: str | None = None
    message: str = ""
    is_error: bool = False
    stack_frames: list[StackFrame] = field(default_factory=list)
    exception_type: str | None = None


@dataclass
class ErrorChain:
    """A chain of related errors with root cause analysis."""

    root_cause: LogEntry
    related_entries: list[LogEntry]
    stack_trace: list[StackFrame]
    time_span_seconds: float
    summary: str


def parse_stack_frame(line: str) -> StackFrame | None:
    """Try to parse a line as a stack frame."""
    for lang, pattern in STACK_TRACE_PATTERNS.items():
        if lang in ("python_exception", "caused_by"):
            continue
        match = pattern.match(line)
        if match:
            groups = match.groups()
            if lang == "python":
                return StackFrame(
                    file=groups[0], line=int(groups[1]), function=groups[2], language="python"
                )
            elif lang == "java":
                return StackFrame(
                    file=groups[1], line=int(groups[2]), function=groups[0], language="java"
                )
            elif lang == "javascript":
                return StackFrame(
                    file=groups[1],
                    line=int(groups[2]),
                    function=groups[0] if groups[0] else None,
                    language="javascript",
                )
            elif lang in ("go", "rust"):
                return StackFrame(file=groups[0], line=int(groups[1]), language=lang)
    return None


def is_error_line(line: str, severity: str | None) -> bool:
    """Check if a line represents an error."""
    if severity and severity.upper() in ("ERROR", "CRITICAL", "FATAL"):
        return True
    return bool(ERROR_KEYWORDS.search(line))


def extract_exception_type(line: str) -> str | None:
    """Extract exception type from a line."""
    match = STACK_TRACE_PATTERNS["python_exception"].match(line)
    if match:
        return match.group(1)
    # Check for "Caused by:" pattern
    match = STACK_TRACE_PATTERNS["caused_by"].match(line)
    if match:
        return match.group(1).split(":")[0].strip()
    return None


def parse_log_entries(lines: list[str], log_format: str) -> list[LogEntry]:
    """Parse log lines into structured entries."""
    entries: list[LogEntry] = []
    current_entry: LogEntry | None = None
    stack_frames: list[StackFrame] = []

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        # Try to parse as stack frame first
        frame = parse_stack_frame(line)
        if frame:
            stack_frames.append(frame)
            continue

        # Check if this is a new log entry (has timestamp) or continuation
        timestamp = extract_timestamp(line, log_format)
        severity = extract_severity(line, log_format)

        if timestamp or (severity and current_entry is None):
            # Save previous entry with its stack frames
            if current_entry:
                current_entry.stack_frames = stack_frames
                entries.append(current_entry)
                stack_frames = []

            # Extract message content
            message = normalize_line(line, log_format)
            exception_type = extract_exception_type(line)

            current_entry = LogEntry(
                line_number=i + 1,
                raw_line=line,
                timestamp=timestamp,
                severity=severity,
                message=message,
                is_error=is_error_line(line, severity),
                exception_type=exception_type,
            )
        elif current_entry:
            # Continuation line - append to message
            current_entry.message += "\n" + line
            exception_type = extract_exception_type(line)
            if exception_type:
                current_entry.exception_type = exception_type

    # Don't forget the last entry
    if current_entry:
        current_entry.stack_frames = stack_frames
        entries.append(current_entry)

    return entries


def find_error_chain(
    entries: list[LogEntry],
    error_entry: LogEntry,
    time_window_seconds: float = 60.0,
) -> ErrorChain:
    """Find related log entries that may have caused or resulted from an error."""
    if not error_entry.timestamp:
        # Without timestamp, just return the error itself
        return ErrorChain(
            root_cause=error_entry,
            related_entries=[],
            stack_trace=error_entry.stack_frames,
            time_span_seconds=0,
            summary=error_entry.message[:200],
        )

    # Find entries within time window before the error
    related: list[LogEntry] = []
    earliest_time = error_entry.timestamp

    for entry in entries:
        if entry is error_entry:
            continue
        if not entry.timestamp:
            continue

        time_diff = (error_entry.timestamp - entry.timestamp).total_seconds()

        # Look for entries in the window before the error
        if 0 < time_diff <= time_window_seconds:
            related.append(entry)
            if entry.timestamp < earliest_time:
                earliest_time = entry.timestamp

    # Sort by timestamp (oldest first)
    related.sort(key=lambda e: e.timestamp or datetime.min.replace(tzinfo=UTC))

    # Find root cause - the earliest error in the chain
    root_cause = error_entry
    for entry in related:
        if (
            entry.is_error
            and entry.timestamp
            and entry.timestamp < (root_cause.timestamp or datetime.max.replace(tzinfo=UTC))
        ):
            root_cause = entry

    # Collect all stack frames
    all_frames = list(error_entry.stack_frames)
    for entry in related:
        all_frames.extend(entry.stack_frames)

    # Calculate time span
    time_span = (error_entry.timestamp - earliest_time).total_seconds()

    # Build summary
    if root_cause.exception_type:
        summary = f"{root_cause.exception_type}: {root_cause.message[:150]}"
    else:
        summary = root_cause.message[:200]

    return ErrorChain(
        root_cause=root_cause,
        related_entries=related,
        stack_trace=all_frames,
        time_span_seconds=time_span,
        summary=summary,
    )


def detect_log_format(lines: list[str]) -> str:
    """Detect the log format from a sample of lines."""
    if not lines:
        return "unknown"

    # Sample first 20 non-empty lines
    sample = [line for line in lines[:100] if line.strip()][:20]
    if not sample:
        return "unknown"

    format_scores: dict[str, int] = defaultdict(int)

    for line in sample:
        for fmt, pattern in LOG_PATTERNS.items():
            if pattern.match(line):
                format_scores[fmt] += 1
                break

    if not format_scores:
        return "plain"

    return max(format_scores, key=lambda k: format_scores[k])


def extract_json_message(line: str) -> str | None:
    """Extract the message content from a JSON log line."""
    try:
        data = json.loads(line)
        if not isinstance(data, dict):
            return None

        # Try to extract message
        for field in JSON_MESSAGE_FIELDS:
            if field in data and isinstance(data[field], str):
                return data[field]

        # Fall back to stringified JSON without common metadata
        stripped = {k: v for k, v in data.items() if k not in JSON_TIME_FIELDS + JSON_LEVEL_FIELDS}
        return json.dumps(stripped, separators=(",", ":")) if stripped else None
    except json.JSONDecodeError:
        return None


def extract_severity(line: str, log_format: str) -> str | None:
    """Extract severity/level from a log line."""
    # Check JSON format first
    if log_format == "json":
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                for field in JSON_LEVEL_FIELDS:
                    if field in data:
                        return str(data[field]).upper()
        except json.JSONDecodeError:
            pass

    # Pattern-based detection
    severity_patterns = [
        (r"\b(FATAL|CRITICAL)\b", "CRITICAL"),
        (r"\b(ERROR|ERR)\b", "ERROR"),
        (r"\b(WARN|WARNING)\b", "WARNING"),
        (r"\b(INFO)\b", "INFO"),
        (r"\b(DEBUG|TRACE)\b", "DEBUG"),
    ]

    for pattern, level in severity_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return level

    return None


def normalize_line(line: str, log_format: str) -> str:
    """Normalize a log line for template extraction."""
    if log_format == "json":
        extracted = extract_json_message(line)
        if extracted:
            return extracted

    # Remove timestamps and common prefixes
    normalized = line

    # Remove ISO timestamps
    iso_pattern = r"\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    normalized = re.sub(iso_pattern, "", normalized)

    # Remove syslog timestamps
    normalized = re.sub(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+", "", normalized)

    # Remove common log prefixes (hostname, process name, PID)
    normalized = re.sub(r"^\S+\s+\S+(?:\[\d+\])?:\s*", "", normalized)

    return normalized.strip()


def create_drain_miner() -> TemplateMiner:
    """Create a configured Drain3 template miner."""
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    config.drain_max_clusters = 1000
    config.drain_max_children = 100
    return TemplateMiner(config=config)


def extract_templates(lines: list[str], log_format: str) -> list[LogTemplate]:
    """Extract log templates using Drain3."""
    miner = create_drain_miner()
    line_to_cluster: dict[int, int] = {}

    for i, line in enumerate(lines):
        normalized = normalize_line(line, log_format)
        if not normalized:
            continue

        result = miner.add_log_message(normalized)
        line_to_cluster[i] = result["cluster_id"]

    # Build template objects
    templates: list[LogTemplate] = []
    for cluster in miner.drain.clusters:
        template_lines = [
            lines[i] for i, cid in line_to_cluster.items() if cid == cluster.cluster_id
        ]

        # Extract severity from examples
        severities = [extract_severity(line, log_format) for line in template_lines[:10]]
        severity = None
        if any(severities):
            severity = max(set(s for s in severities if s), key=severities.count, default=None)

        templates.append(
            LogTemplate(
                template=cluster.get_template(),
                cluster_id=cluster.cluster_id,
                count=cluster.size,
                examples=template_lines[:3],
                severity=severity,
            )
        )

    return templates


def cluster_templates_semantically(
    templates: list[LogTemplate], num_clusters: int = 10
) -> list[SemanticCluster]:
    """Cluster templates semantically using FastEmbed."""
    if not templates:
        return []

    if len(templates) <= num_clusters:
        # Each template is its own cluster
        return [
            SemanticCluster(
                templates=[t],
                centroid_idx=0,
                total_count=t.count,
                summary=t.template,
            )
            for t in templates
        ]

    # Generate embeddings
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    template_texts = [t.template for t in templates]
    embeddings = list(embedding_model.embed(template_texts))
    embedding_matrix = np.array(embeddings)

    # Simple k-means clustering
    clusters = kmeans_cluster(embedding_matrix, num_clusters)

    # Group templates by cluster
    cluster_groups: dict[int, list[tuple[int, LogTemplate]]] = defaultdict(list)
    for i, (cluster_id, template) in enumerate(zip(clusters, templates, strict=True)):
        cluster_groups[cluster_id].append((i, template))

    # Build semantic clusters
    result: list[SemanticCluster] = []
    for _cluster_id, group in cluster_groups.items():
        group_templates = [t for _, t in group]
        group_embeddings = np.array([embedding_matrix[i] for i, _ in group])

        # Find centroid
        centroid = group_embeddings.mean(axis=0)
        distances = np.linalg.norm(group_embeddings - centroid, axis=1)
        centroid_idx = int(np.argmin(distances))

        total_count = sum(t.count for t in group_templates)
        summary = group_templates[centroid_idx].template

        result.append(
            SemanticCluster(
                templates=group_templates,
                centroid_idx=centroid_idx,
                total_count=total_count,
                summary=summary,
            )
        )

    # Sort by total count descending
    result.sort(key=lambda c: c.total_count, reverse=True)
    return result


def kmeans_cluster(embeddings: np.ndarray, k: int, max_iters: int = 100) -> list[int]:
    """Simple k-means clustering."""
    n_samples = embeddings.shape[0]
    k = min(k, n_samples)

    # Initialize centroids randomly
    rng = np.random.default_rng(42)
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = embeddings[indices].copy()

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iters):
        # Assign to nearest centroid
        distances = np.linalg.norm(embeddings[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        # Update centroids
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                centroids[i] = embeddings[mask].mean(axis=0)

    return labels.tolist()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def format_as_markdown(
    clusters: list[SemanticCluster],
    log_format: str,
    total_lines: int,
    token_budget: int,
    compact: bool = False,
) -> str:
    """Format clusters as markdown, respecting token budget.

    Args:
        clusters: Semantic clusters to format.
        log_format: Detected log format name.
        total_lines: Total number of input lines.
        token_budget: Maximum tokens in output.
        compact: If True, use abbreviated format for AI agent consumption.
    """
    if compact:
        return _format_compact(clusters, log_format, total_lines, token_budget)

    sections: list[str] = []

    # Header
    header = f"""# Log Analysis Summary

**Format detected:** {log_format}
**Total lines:** {total_lines:,}
**Unique patterns:** {sum(len(c.templates) for c in clusters)}
**Semantic clusters:** {len(clusters)}

---

"""
    sections.append(header)

    # Severity summary
    severity_counts: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            if template.severity:
                severity_counts[template.severity] += template.count

    if severity_counts:
        severity_section = "## Severity Distribution\n\n"
        for level in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]:
            if level in severity_counts:
                severity_section += f"- **{level}:** {severity_counts[level]:,}\n"
        severity_section += "\n---\n\n"
        sections.append(severity_section)

    # Clusters
    sections.append("## Log Patterns by Frequency\n\n")

    current_tokens = count_tokens("".join(sections))

    for i, cluster in enumerate(clusters, 1):
        summary_text = cluster.summary[:80]
        ellipsis = "..." if len(cluster.summary) > 80 else ""
        cluster_section = f"### Cluster {i}: {summary_text}{ellipsis}\n\n"
        cluster_section += f"**Occurrences:** {cluster.total_count:,} | "
        cluster_section += f"**Patterns:** {len(cluster.templates)}\n\n"

        # Add top templates
        top_templates = sorted(cluster.templates, key=lambda t: t.count, reverse=True)[:5]
        for template in top_templates:
            severity_badge = f"[{template.severity}] " if template.severity else ""
            cluster_section += f"- {severity_badge}`{template.template}` ({template.count:,}x)\n"

        # Add example
        if cluster.templates[0].examples:
            example_text = cluster.templates[0].examples[0][:500]
            cluster_section += f"\n**Example:**\n```\n{example_text}\n```\n\n"

        cluster_section += "---\n\n"

        # Check token budget
        section_tokens = count_tokens(cluster_section)
        if current_tokens + section_tokens > token_budget:
            remaining = len(clusters) - i + 1
            sections.append(f"\n*... {remaining} more clusters omitted (token budget reached)*\n")
            break

        sections.append(cluster_section)
        current_tokens += section_tokens

    return "".join(sections)


def _format_compact(
    clusters: list[SemanticCluster],
    log_format: str,
    total_lines: int,
    token_budget: int,
) -> str:
    """Format clusters in compact mode for AI agent consumption.

    Removes markdown decorations, uses abbreviated labels, and reduces whitespace.
    """
    parts: list[str] = []
    patterns = sum(len(c.templates) for c in clusters)
    parts.append(
        f"fmt={log_format} lines={total_lines:,} patterns={patterns} clusters={len(clusters)}"
    )

    # Severity counts on one line
    severity_counts: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            if template.severity:
                severity_counts[template.severity] += template.count
    if severity_counts:
        sev_parts = [f"{k}:{v}" for k, v in severity_counts.items()]
        parts.append("sev: " + " ".join(sev_parts))

    parts.append("")
    current_tokens = count_tokens("\n".join(parts))

    for i, cluster in enumerate(clusters, 1):
        summary_text = cluster.summary[:60]
        cluster_lines = [f"[{i}] {summary_text} (n={cluster.total_count:,})"]

        top_templates = sorted(cluster.templates, key=lambda t: t.count, reverse=True)[:3]
        for template in top_templates:
            sev = f"{template.severity} " if template.severity else ""
            cluster_lines.append(f"  {sev}{template.template} ({template.count}x)")

        # Only first example, truncated
        if cluster.templates[0].examples:
            example_text = cluster.templates[0].examples[0][:200]
            cluster_lines.append(f"  ex: {example_text}")

        section = "\n".join(cluster_lines) + "\n"
        section_tokens = count_tokens(section)
        if current_tokens + section_tokens > token_budget:
            remaining = len(clusters) - i + 1
            parts.append(f"...{remaining} more clusters omitted")
            break

        parts.append(section)
        current_tokens += section_tokens

    return "\n".join(parts)


def analyze_log_lines(
    all_lines: list[str],
    token_budget: int = 8000,
    num_clusters: int = 10,
    severity_filter: list[str] | None = None,
    redact: bool | str = True,
    compact: bool = False,
) -> AnalysisResult:
    """Core analysis function for log lines.

    Args:
        all_lines: List of log lines to analyze
        token_budget: Maximum tokens in output
        num_clusters: Number of semantic clusters
        severity_filter: Only include these severity levels
        redact: Redaction mode - True/"moderate" (default), "strict", "minimal", or False
        compact: If True, use abbreviated format for AI agent consumption

    Returns:
        AnalysisResult with markdown output and statistics
    """
    start_time = time.perf_counter()

    # Count original tokens
    original_text = "\n".join(all_lines)
    original_tokens = count_tokens(original_text)

    if not all_lines:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return AnalysisResult(
            markdown="Error: No log content found",
            stats=AnalysisStats(
                processing_time_ms=elapsed_ms,
                redaction_count=0,
                original_tokens=original_tokens,
                output_tokens=0,
            ),
        )

    # Apply redaction before analysis
    redaction_count = 0
    if redact is not False:
        if redact is True or redact == "moderate":
            mode = RedactionMode.MODERATE
        elif redact == "strict":
            mode = RedactionMode.STRICT
        elif redact == "minimal":
            mode = RedactionMode.MINIMAL
        else:
            mode = RedactionMode.MODERATE
        all_lines, redaction_count = redact_lines(all_lines, mode)

    # Detect format
    log_format = detect_log_format(all_lines)

    # Extract templates
    templates = extract_templates(all_lines, log_format)

    # Apply severity filter
    if severity_filter:
        severity_set = {s.upper() for s in severity_filter}
        templates = [t for t in templates if t.severity in severity_set]

    if not templates:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        markdown = "No log patterns found matching the criteria"
        return AnalysisResult(
            markdown=markdown,
            stats=AnalysisStats(
                processing_time_ms=elapsed_ms,
                redaction_count=redaction_count,
                original_tokens=original_tokens,
                output_tokens=count_tokens(markdown),
            ),
        )

    # Cluster semantically
    clusters = cluster_templates_semantically(templates, num_clusters)

    # Format output
    markdown = format_as_markdown(
        clusters, log_format, len(all_lines), token_budget, compact=compact
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    output_tokens = count_tokens(markdown)

    # Compute severity distribution
    severity_distribution: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            if template.severity:
                severity_distribution[template.severity] += template.count

    # Convert clusters to output format
    clusters_data = [
        ClusterOutput(
            id=i,
            summary=cluster.summary,
            total_count=cluster.total_count,
            templates=[
                TemplateOutput(
                    template=t.template,
                    count=t.count,
                    severity=t.severity,
                    examples=t.examples[:3],  # Limit examples
                )
                for t in sorted(cluster.templates, key=lambda x: x.count, reverse=True)[:10]
            ],
        )
        for i, cluster in enumerate(clusters, 1)
    ]

    return AnalysisResult(
        markdown=markdown,
        stats=AnalysisStats(
            processing_time_ms=elapsed_ms,
            redaction_count=redaction_count,
            original_tokens=original_tokens,
            output_tokens=output_tokens,
        ),
        log_format=log_format,
        lines_processed=len(all_lines),
        severity_distribution=dict(severity_distribution),
        clusters_data=clusters_data,
    )


def resolve_glob_pattern(pattern: str) -> list[Path]:
    """Resolve a glob pattern to a list of files."""
    # Check if it looks like a glob pattern
    if any(c in pattern for c in ["*", "?", "[", "]"]):
        # Use glob module for pattern matching
        matches = glob_module.glob(pattern, recursive=True)
        return [Path(m) for m in matches if Path(m).is_file()]
    return []


@mcp.tool()
def get_logs(
    path: str,
    token_budget: int = 8000,
    num_clusters: int = 10,
    severity_filter: list[str] | None = None,
    since: str | None = None,
    redact: bool | str = True,
) -> str:
    """Analyze and consolidate log files for LLM consumption.

    Args:
        path: Path to log file, directory, or glob pattern (e.g., "/var/log/*.log")
        token_budget: Maximum tokens in output (default: 8000)
        num_clusters: Number of semantic clusters to create (default: 10)
        severity_filter: Only include these severity levels (e.g., ["ERROR", "WARNING"])
        since: Only logs since this time (e.g., "1h", "30m", "2d", "2025-01-01")
        redact: Redact secrets/PII before analysis. Options:
            - True or "moderate" (default): Emails, IPs, credit cards, API keys, etc.
            - "strict": All moderate patterns + high-entropy strings
            - "minimal": Only obvious secrets (bearer tokens, API keys)
            - False: No redaction (use with caution for internal logs only)

    Returns:
        Markdown-formatted log analysis with patterns grouped by semantic similarity
    """
    # Parse since filter if provided
    since_dt: datetime | None = None
    if since:
        since_dt = parse_since(since)
        if since_dt is None:
            return (
                f"Error: Invalid time format for 'since': {since}. "
                "Use '1h', '30m', '2d', or 'YYYY-MM-DD'"
            )

    # Try glob pattern first
    log_files = resolve_glob_pattern(path)

    if not log_files:
        # Fall back to path-based resolution
        log_path = Path(path).expanduser().resolve()

        if not log_path.exists():
            return f"Error: Path does not exist: {path}"

        # Collect log files
        if log_path.is_file():
            log_files = [log_path]
        else:
            log_files = list(log_path.glob("**/*.log")) + list(log_path.glob("**/*.txt"))
            if not log_files:
                return f"Error: No log files found in {path}"

    # Read all lines
    all_lines: list[str] = []
    for log_file in log_files:
        try:
            content = log_file.read_text(errors="replace")
            all_lines.extend(content.splitlines())
        except Exception as e:
            all_lines.append(f"[Error reading {log_file}: {e}]")

    # Apply time filter if provided
    if since_dt and all_lines:
        log_format = detect_log_format(all_lines)
        all_lines = filter_by_time(all_lines, since_dt, log_format)

    result = analyze_log_lines(all_lines, token_budget, num_clusters, severity_filter, redact)

    # Store in tee cache for later retrieval via get_raw_logs
    analysis_id = tee_store(all_lines, path)

    return result.markdown + f"\n\n_analysis_id: {analysis_id}_"


def discover_compose_file(path: str | None = None) -> Path | None:
    """Find docker-compose file in path or current directory."""
    search_dir = Path(path).expanduser().resolve() if path else Path.cwd()

    # Common compose file names in order of preference
    compose_names = [
        "docker-compose.yml",
        "docker-compose.yaml",
        "docker-compose.dev.yml",
        "docker-compose.dev.yaml",
        "compose.yml",
        "compose.yaml",
    ]

    for name in compose_names:
        compose_file = search_dir / name
        if compose_file.exists():
            return compose_file

    return None


def get_compose_services(compose_file: Path) -> list[dict[str, str]]:
    """Get list of services from a running compose project."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "ps", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return []

    services = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            services.append(
                {
                    "name": data.get("Service", ""),
                    "state": data.get("State", ""),
                    "status": data.get("Status", ""),
                }
            )
        except json.JSONDecodeError:
            continue

    return services


def fetch_docker_logs(
    compose_file: Path,
    services: list[str] | None = None,
    tail: int = 1000,
    since: str | None = None,
) -> str:
    """Fetch logs from docker compose services."""
    cmd = ["docker", "compose", "-f", str(compose_file), "logs", "--no-color"]

    if tail:
        cmd.extend(["--tail", str(tail)])

    if since:
        cmd.extend(["--since", since])

    if services:
        cmd.extend(services)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    return result.stdout


@mcp.tool()
def get_docker_logs(
    path: str | None = None,
    services: list[str] | None = None,
    tail: int = 1000,
    since: str | None = None,
    token_budget: int = 8000,
    num_clusters: int = 10,
    severity_filter: list[str] | None = None,
) -> str:
    """Analyze logs from Docker Compose services.

    Auto-discovers docker-compose.yml in the specified path or current directory.

    Args:
        path: Directory containing docker-compose.yml (defaults to current directory)
        services: List of service names to include (defaults to all services)
        tail: Number of recent log lines per service (default: 1000)
        since: Only logs since this time (e.g., "1h", "30m", "2024-01-01")
        token_budget: Maximum tokens in output (default: 8000)
        num_clusters: Number of semantic clusters (default: 10)
        severity_filter: Only include these severity levels (e.g., ["ERROR", "WARNING"])

    Returns:
        Markdown-formatted log analysis with patterns grouped by semantic similarity
    """
    compose_file = discover_compose_file(path)

    if not compose_file:
        search_path = path or "current directory"
        return f"Error: No docker-compose.yml found in {search_path}"

    # Get running services
    running_services = get_compose_services(compose_file)
    if not running_services:
        return f"Error: No running services found for {compose_file}"

    # Build service info header
    service_names = [s["name"] for s in running_services]
    if services:
        # Validate requested services exist
        invalid = set(services) - set(service_names)
        if invalid:
            return f"Error: Unknown services: {invalid}. Available: {service_names}"
        target_services = services
    else:
        target_services = service_names

    # Fetch logs
    try:
        raw_logs = fetch_docker_logs(compose_file, target_services, tail, since)
    except subprocess.TimeoutExpired:
        return "Error: Timeout fetching docker logs"
    except Exception as e:
        return f"Error fetching docker logs: {e}"

    if not raw_logs.strip():
        return "No logs found for the specified services"

    all_lines = raw_logs.splitlines()

    # Add compose project info to the analysis
    analysis = analyze_log_lines(all_lines, token_budget, num_clusters, severity_filter)

    # Prepend docker-specific header
    header = f"""**Docker Compose Project:** {compose_file.parent.name}
**Compose File:** {compose_file.name}
**Services:** {", ".join(target_services)}
**Tail:** {tail} lines per service

"""
    return header + analysis.markdown


@mcp.tool()
def list_docker_services(path: str | None = None) -> str:
    """List available Docker Compose services in a project.

    Args:
        path: Directory containing docker-compose.yml (defaults to current directory)

    Returns:
        List of services with their current status
    """
    compose_file = discover_compose_file(path)

    if not compose_file:
        search_path = path or "current directory"
        return f"No docker-compose.yml found in {search_path}"

    services = get_compose_services(compose_file)

    if not services:
        return f"No running services found for {compose_file}"

    lines = [
        "# Docker Compose Services",
        f"**Project:** {compose_file.parent.name}",
        f"**File:** {compose_file}",
        "",
        "| Service | State | Status |",
        "|---------|-------|--------|",
    ]

    for svc in services:
        lines.append(f"| {svc['name']} | {svc['state']} | {svc['status']} |")

    return "\n".join(lines)


def get_docker_containers() -> list[dict[str, str]]:
    """Get list of running Docker containers."""
    result = subprocess.run(
        ["docker", "ps", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return []

    containers = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            containers.append(
                {
                    "id": data.get("ID", ""),
                    "name": data.get("Names", ""),
                    "image": data.get("Image", ""),
                    "status": data.get("Status", ""),
                    "state": data.get("State", ""),
                }
            )
        except json.JSONDecodeError:
            continue

    return containers


def fetch_container_logs(
    container: str,
    tail: int = 1000,
    since: str | None = None,
) -> str:
    """Fetch logs from a Docker container."""
    cmd = ["docker", "logs", "--timestamps"]

    if tail:
        cmd.extend(["--tail", str(tail)])

    if since:
        cmd.extend(["--since", since])

    cmd.append(container)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Docker logs outputs to stderr for container stderr
    return result.stdout + result.stderr


@mcp.tool()
def get_container_logs(
    container: str,
    tail: int = 1000,
    since: str | None = None,
    token_budget: int = 8000,
    num_clusters: int = 10,
    severity_filter: list[str] | None = None,
) -> str:
    """Analyze logs from a standalone Docker container.

    Args:
        container: Container name or ID
        tail: Number of recent log lines (default: 1000)
        since: Only logs since this time (e.g., "1h", "30m", "2024-01-01")
        token_budget: Maximum tokens in output (default: 8000)
        num_clusters: Number of semantic clusters (default: 10)
        severity_filter: Only include these severity levels (e.g., ["ERROR", "WARNING"])

    Returns:
        Markdown-formatted log analysis with patterns grouped by semantic similarity
    """
    # Fetch logs
    try:
        raw_logs = fetch_container_logs(container, tail, since)
    except subprocess.TimeoutExpired:
        return "Error: Timeout fetching container logs"
    except FileNotFoundError:
        return "Error: Docker not found. Is Docker installed and in PATH?"
    except Exception as e:
        return f"Error fetching container logs: {e}"

    if not raw_logs.strip():
        return f"No logs found for container: {container}"

    all_lines = raw_logs.splitlines()

    # Analyze
    analysis = analyze_log_lines(all_lines, token_budget, num_clusters, severity_filter)

    # Prepend container header
    header = f"""**Docker Container:** {container}
**Tail:** {tail} lines

"""
    return header + analysis.markdown


@mcp.tool()
def list_containers() -> str:
    """List running Docker containers.

    Returns:
        List of containers with their current status
    """
    try:
        containers = get_docker_containers()
    except FileNotFoundError:
        return "Error: Docker not found. Is Docker installed and in PATH?"
    except subprocess.TimeoutExpired:
        return "Error: Timeout listing containers"
    except Exception as e:
        return f"Error listing containers: {e}"

    if not containers:
        return "No running containers found"

    lines = [
        "# Docker Containers",
        "",
        "| Name | Image | Status |",
        "|------|-------|--------|",
    ]

    for c in containers:
        lines.append(f"| {c['name']} | {c['image']} | {c['status']} |")

    return "\n".join(lines)


def fetch_journald_logs(
    unit: str | None = None,
    priority: str | None = None,
    since: str | None = None,
    lines_limit: int = 1000,
) -> str:
    """Fetch logs from journald/systemd."""
    cmd = ["journalctl", "--no-pager", "-o", "short-iso"]

    if unit:
        cmd.extend(["-u", unit])

    if priority:
        cmd.extend(["-p", priority])

    if since:
        cmd.extend(["--since", since])

    cmd.extend(["-n", str(lines_limit)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    return result.stdout


@mcp.tool()
def get_journald_logs(
    unit: str | None = None,
    priority: str | None = None,
    since: str | None = None,
    lines_limit: int = 1000,
    token_budget: int = 8000,
    num_clusters: int = 10,
    severity_filter: list[str] | None = None,
) -> str:
    """Analyze logs from journald/systemd.

    Args:
        unit: Filter by systemd unit (e.g., "nginx", "docker", "sshd")
        priority: Filter by priority (e.g., "err", "warning", "info")
        since: Only logs since this time (e.g., "1 hour ago", "today", "2025-01-01")
        lines_limit: Maximum number of lines to fetch (default: 1000)
        token_budget: Maximum tokens in output (default: 8000)
        num_clusters: Number of semantic clusters (default: 10)
        severity_filter: Only include these severity levels (e.g., ["ERROR", "WARNING"])

    Returns:
        Markdown-formatted log analysis with patterns grouped by semantic similarity
    """
    try:
        raw_logs = fetch_journald_logs(unit, priority, since, lines_limit)
    except FileNotFoundError:
        return "Error: journalctl not found. Is this a systemd-based system?"
    except subprocess.TimeoutExpired:
        return "Error: Timeout fetching journald logs"
    except Exception as e:
        return f"Error fetching journald logs: {e}"

    if not raw_logs.strip():
        filters = []
        if unit:
            filters.append(f"unit={unit}")
        if priority:
            filters.append(f"priority={priority}")
        if since:
            filters.append(f"since={since}")
        filter_str = ", ".join(filters) if filters else "none"
        return f"No logs found (filters: {filter_str})"

    all_lines = raw_logs.splitlines()

    # Analyze
    analysis = analyze_log_lines(all_lines, token_budget, num_clusters, severity_filter)

    # Prepend journald header
    header_parts = ["**Source:** journald"]
    if unit:
        header_parts.append(f"**Unit:** {unit}")
    if priority:
        header_parts.append(f"**Priority:** {priority}")
    if since:
        header_parts.append(f"**Since:** {since}")
    header_parts.append(f"**Lines:** {lines_limit}")
    header = "\n".join(header_parts) + "\n\n"

    return header + analysis.markdown


def read_log_source(path: str) -> tuple[list[str], str]:
    """Read logs from a file path and return lines with source description."""
    log_files = resolve_glob_pattern(path)

    if not log_files:
        log_path = Path(path).expanduser().resolve()
        if not log_path.exists():
            return [], f"Path does not exist: {path}"
        if log_path.is_file():
            log_files = [log_path]
        else:
            log_files = list(log_path.glob("**/*.log")) + list(log_path.glob("**/*.txt"))

    if not log_files:
        return [], f"No log files found in {path}"

    all_lines: list[str] = []
    for log_file in log_files:
        try:
            content = log_file.read_text(errors="replace")
            all_lines.extend(content.splitlines())
        except Exception as e:
            all_lines.append(f"[Error reading {log_file}: {e}]")

    return all_lines, ""


def format_error_chain(chain: ErrorChain, include_context: bool = True) -> str:
    """Format an error chain as markdown."""
    sections: list[str] = []

    # Header
    sections.append("# Error Chain Analysis\n")

    # Root cause
    sections.append("## Root Cause\n")
    if chain.root_cause.exception_type:
        sections.append(f"**Exception:** `{chain.root_cause.exception_type}`\n")
    sections.append(f"**Line:** {chain.root_cause.line_number}\n")
    if chain.root_cause.timestamp:
        sections.append(f"**Time:** {chain.root_cause.timestamp.isoformat()}\n")
    if chain.root_cause.severity:
        sections.append(f"**Severity:** {chain.root_cause.severity}\n")
    sections.append(f"\n```\n{chain.root_cause.raw_line[:500]}\n```\n")

    # Stack trace
    if chain.stack_trace:
        sections.append("\n## Stack Trace\n")
        sections.append("```\n")
        for frame in chain.stack_trace[:20]:  # Limit to 20 frames
            if frame.function:
                sections.append(f"  {frame.file}:{frame.line} in {frame.function}\n")
            else:
                sections.append(f"  {frame.file}:{frame.line}\n")
        if len(chain.stack_trace) > 20:
            sections.append(f"  ... and {len(chain.stack_trace) - 20} more frames\n")
        sections.append("```\n")

    # Related entries
    if chain.related_entries and include_context:
        sections.append("\n## Related Log Entries\n")
        sections.append(
            f"Found {len(chain.related_entries)} entries within "
            f"{chain.time_span_seconds:.1f}s before the error:\n\n"
        )

        # Show errors first, then warnings, then others
        errors = [e for e in chain.related_entries if e.is_error]
        warnings = [e for e in chain.related_entries if e.severity == "WARNING" and not e.is_error]
        others = [e for e in chain.related_entries if e not in errors and e not in warnings]

        for label, entries in [("Errors", errors), ("Warnings", warnings), ("Context", others)]:
            if entries:
                sections.append(f"### {label}\n")
                for entry in entries[:5]:  # Limit each category
                    ts = entry.timestamp.isoformat() if entry.timestamp else "?"
                    sections.append(f"- `{ts}` {entry.message[:100]}\n")
                if len(entries) > 5:
                    sections.append(f"- ... and {len(entries) - 5} more\n")
                sections.append("\n")

    # Summary
    sections.append("\n## Summary\n")
    sections.append(f"{chain.summary}\n")

    return "".join(sections)


@mcp.tool()
def get_error_chain(
    path: str,
    error_pattern: str | None = None,
    time_window: int = 60,
    max_chains: int = 5,
) -> str:
    """Analyze error chains and find root causes in log files.

    Traces errors back through time to find related log entries that may have
    caused or contributed to the error. Identifies the root cause by finding
    the earliest error in each chain.

    Args:
        path: Path to log file, directory, or glob pattern
        error_pattern: Optional regex to filter specific errors (e.g., "database|connection")
        time_window: Seconds to look back for related entries (default: 60)
        max_chains: Maximum number of error chains to return (default: 5)

    Returns:
        Markdown-formatted error chain analysis with root causes and stack traces
    """
    all_lines, error = read_log_source(path)
    if error:
        return f"Error: {error}"

    if not all_lines:
        return "No log content found"

    # Parse logs
    log_format = detect_log_format(all_lines)
    entries = parse_log_entries(all_lines, log_format)

    # Filter to error entries
    error_entries = [e for e in entries if e.is_error]

    # Apply pattern filter if provided
    if error_pattern:
        try:
            pattern = re.compile(error_pattern, re.IGNORECASE)
            error_entries = [e for e in error_entries if pattern.search(e.message)]
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

    if not error_entries:
        return "No errors found matching the criteria"

    # Build error chains for the most recent errors
    error_entries.sort(key=lambda e: e.timestamp or datetime.min.replace(tzinfo=UTC), reverse=True)

    chains: list[ErrorChain] = []
    seen_root_causes: set[int] = set()

    for error_entry in error_entries[: max_chains * 2]:  # Check more to find unique chains
        chain = find_error_chain(entries, error_entry, float(time_window))

        # Skip if we've already seen this root cause
        if chain.root_cause.line_number in seen_root_causes:
            continue

        seen_root_causes.add(chain.root_cause.line_number)
        chains.append(chain)

        if len(chains) >= max_chains:
            break

    if not chains:
        return "No error chains found"

    # Format output
    sections = [f"# Error Chain Analysis\n\n**Found {len(chains)} error chain(s)**\n\n---\n"]

    for i, chain in enumerate(chains, 1):
        sections.append(f"\n## Chain {i}\n")
        sections.append(format_error_chain(chain, include_context=True))
        sections.append("\n---\n")

    return "".join(sections)


def semantic_search_logs(
    lines: list[str],
    query: str,
    log_format: str,
    top_k: int = 10,
) -> list[tuple[LogEntry, float]]:
    """Search logs semantically using embeddings."""
    entries = parse_log_entries(lines, log_format)

    if not entries:
        return []

    # Generate embeddings
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Get query embedding
    query_embedding = np.array(next(iter(embedding_model.embed([query]))))

    # Get embeddings for log messages
    messages = [e.message for e in entries]
    message_embeddings = np.array(list(embedding_model.embed(messages)))

    # Calculate cosine similarities
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    message_norms = message_embeddings / np.linalg.norm(message_embeddings, axis=1, keepdims=True)
    similarities = np.dot(message_norms, query_norm)

    # Get top-k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results: list[tuple[LogEntry, float]] = []
    for idx in top_indices:
        results.append((entries[idx], float(similarities[idx])))

    return results


@mcp.tool()
def search_logs(
    path: str,
    query: str,
    top_k: int = 10,
    since: str | None = None,
    severity_filter: list[str] | None = None,
) -> str:
    """Semantic search through logs using natural language queries.

    Uses AI embeddings to find log entries that are semantically similar to your
    query, even if they don't contain the exact keywords.

    Args:
        path: Path to log file, directory, or glob pattern
        query: Natural language search query (e.g., "database connection timeout")
        top_k: Number of results to return (default: 10)
        since: Only search logs since this time (e.g., "1h", "2025-01-01")
        severity_filter: Only search these severity levels (e.g., ["ERROR", "WARNING"])

    Returns:
        Markdown-formatted search results ranked by relevance
    """
    all_lines, error = read_log_source(path)
    if error:
        return f"Error: {error}"

    if not all_lines:
        return "No log content found"

    # Apply time filter if provided
    log_format = detect_log_format(all_lines)
    if since:
        since_dt = parse_since(since)
        if since_dt:
            all_lines = filter_by_time(all_lines, since_dt, log_format)

    # Apply severity filter
    if severity_filter:
        severity_set = {s.upper() for s in severity_filter}
        filtered_lines: list[str] = []
        for line in all_lines:
            severity = extract_severity(line, log_format)
            if severity and severity in severity_set:
                filtered_lines.append(line)
        all_lines = filtered_lines

    if not all_lines:
        return "No logs found matching the filters"

    # Perform semantic search
    try:
        results = semantic_search_logs(all_lines, query, log_format, top_k)
    except Exception as e:
        return f"Error performing semantic search: {e}"

    if not results:
        return "No matching log entries found"

    # Format results
    sections = [f"# Search Results\n\n**Query:** {query}\n**Results:** {len(results)}\n\n---\n"]

    for i, (entry, score) in enumerate(results, 1):
        sections.append(f"\n## Result {i} (similarity: {score:.3f})\n")
        if entry.timestamp:
            sections.append(f"**Time:** {entry.timestamp.isoformat()}\n")
        if entry.severity:
            sections.append(f"**Severity:** {entry.severity}\n")
        sections.append(f"**Line:** {entry.line_number}\n")
        sections.append(f"\n```\n{entry.raw_line[:500]}\n```\n")

        if entry.stack_frames:
            sections.append("\n**Stack frames:**\n")
            for frame in entry.stack_frames[:3]:
                sections.append(f"- `{frame.file}:{frame.line}`\n")

        sections.append("\n---\n")

    return "".join(sections)


# --- Tee-style output preservation ---
# Cache raw (redacted) log content so the AI agent can request full context
# after seeing the summary. Auto-cleanup after TTL.

_tee_cache: dict[str, dict[str, Any]] = {}
_tee_lock = threading.Lock()
_TEE_TTL_SECONDS = 3600  # 1 hour


def _tee_cleanup() -> None:
    """Remove expired entries from the tee cache."""
    now = datetime.now(UTC)
    with _tee_lock:
        expired = [
            k
            for k, v in _tee_cache.items()
            if (now - v["timestamp"]).total_seconds() > _TEE_TTL_SECONDS
        ]
        for k in expired:
            del _tee_cache[k]


def tee_store(lines: list[str], source: str) -> str:
    """Store raw log lines in the tee cache.

    Args:
        lines: Redacted log lines to cache.
        source: Source identifier for the logs.

    Returns:
        Analysis ID that can be used to retrieve the lines.
    """
    _tee_cleanup()
    analysis_id = hashlib.sha256(
        f"{source}:{datetime.now(UTC).isoformat()}:{len(lines)}".encode()
    ).hexdigest()[:12]

    with _tee_lock:
        _tee_cache[analysis_id] = {
            "lines": lines,
            "source": source,
            "line_count": len(lines),
            "timestamp": datetime.now(UTC),
        }

    return analysis_id


@mcp.tool()
def get_raw_logs(
    analysis_id: str,
    start_line: int = 0,
    max_lines: int = 500,
) -> str:
    """Retrieve raw (redacted) log lines from a previous analysis.

    After reviewing a log analysis summary, use this tool to get the full
    log content for deeper investigation. Lines are already redacted.

    Args:
        analysis_id: The analysis ID returned from a previous get_logs call.
        start_line: Line offset to start from (default: 0).
        max_lines: Maximum lines to return (default: 500).

    Returns:
        Raw log lines from the cached analysis.
    """
    with _tee_lock:
        entry = _tee_cache.get(analysis_id)

    if entry is None:
        return (
            f"Error: Analysis '{analysis_id}' not found or expired. "
            "Re-run the analysis to generate a new cache."
        )

    lines = entry["lines"][start_line : start_line + max_lines]
    total = entry["line_count"]
    end = start_line + len(lines)
    header = f"Source: {entry['source']} | Lines {start_line + 1}-{end} of {total}\n\n"
    return header + "\n".join(lines)


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
