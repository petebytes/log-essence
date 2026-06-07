"""Tests for the raw-log retrieval subsystem (tee cache + get_raw_logs).

The retrieval path (get_logs -> analysis_id -> get_raw_logs) had no test
coverage. The headline guarantee is that retrieved lines are redacted: the
summary is redacted, so the on-demand full context must be too.
"""

from pathlib import Path

from log_essence.server import get_logs, get_raw_logs


def _analysis_id(get_logs_output: str) -> str:
    """Pull the analysis_id out of the get_logs trailer '_analysis_id: <id>_'."""
    return get_logs_output.rsplit("_analysis_id: ", 1)[1].strip().rstrip("_")


def test_get_raw_logs_redacts_cached_lines(tmp_path: Path) -> None:
    """Raw retrieval must not leak secrets the summary redacted (redaction-bypass)."""
    log_file = tmp_path / "app.log"
    lines = [f"2025-01-01T10:00:00Z INFO user@acme.com action {i}" for i in range(15)]
    lines.append("2025-01-01T10:00:01Z ERROR payment failed for card 4111111111111111")
    log_file.write_text("\n".join(lines) + "\n")

    summary = get_logs(path=str(log_file), redact=True)
    assert "user@acme.com" not in summary  # sanity: summary is redacted

    raw = get_raw_logs(analysis_id=_analysis_id(summary))
    assert "user@acme.com" not in raw
    assert "4111111111111111" not in raw
    assert "[EMAIL:" in raw  # redaction actually applied, not merely absent


def test_get_raw_logs_pagination_round_trip(tmp_path: Path) -> None:
    """start_line/max_lines return the requested window with an accurate header."""
    log_file = tmp_path / "app.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(10)) + "\n")

    summary = get_logs(path=str(log_file), redact=False)
    raw = get_raw_logs(analysis_id=_analysis_id(summary), start_line=2, max_lines=3)

    assert "Lines 3-5 of 10" in raw
    assert "line 2" in raw and "line 4" in raw
    assert "line 1" not in raw and "line 5" not in raw


def test_get_raw_logs_unknown_id_returns_error() -> None:
    """An unknown/expired id yields an actionable error, not a crash."""
    raw = get_raw_logs(analysis_id="deadbeefcafe")
    assert "not found or expired" in raw


def test_get_raw_logs_redact_false_keeps_raw(tmp_path: Path) -> None:
    """redact=False is an explicit opt-out: retrieval returns the raw lines."""
    log_file = tmp_path / "app.log"
    log_file.write_text("2025-01-01T10:00:00Z INFO user@acme.com logged in\n")

    summary = get_logs(path=str(log_file), redact=False)
    raw = get_raw_logs(analysis_id=_analysis_id(summary))
    assert "user@acme.com" in raw


def test_get_raw_logs_redacted_when_no_templates(tmp_path: Path) -> None:
    """A filter that leaves zero templates still caches redacted lines for retrieval."""
    log_file = tmp_path / "app.log"
    log_file.write_text(
        "\n".join(f"2025-01-01T10:00:00Z INFO user@acme.com hit {i}" for i in range(5)) + "\n"
    )

    # severity_filter matches nothing here -> "no patterns" early-return path
    summary = get_logs(path=str(log_file), redact=True, severity_filter=["CRITICAL"])
    assert "No log patterns found" in summary

    raw = get_raw_logs(analysis_id=_analysis_id(summary))
    assert "user@acme.com" not in raw
    assert "[EMAIL:" in raw


def test_get_raw_logs_clamps_out_of_range_start(tmp_path: Path) -> None:
    """Negative/past-the-end start_line is clamped, not a tail slice or garbled header."""
    log_file = tmp_path / "app.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(5)) + "\n")
    aid = _analysis_id(get_logs(path=str(log_file), redact=False))

    # Negative start must read from the top, never negative-index into the tail.
    neg = get_raw_logs(analysis_id=aid, start_line=-1, max_lines=2)
    assert "line 0" in neg
    assert "Lines 1-2 of 5" in neg

    # Past-the-end start yields a clear empty-range message, not "Lines 100-99 of 5".
    past = get_raw_logs(analysis_id=aid, start_line=99)
    assert "no lines in range" in past.lower()
    assert "Lines 100-99" not in past


def test_get_raw_logs_redacts_under_strict_mode(tmp_path: Path) -> None:
    """Strict mode's redaction is reflected in retrieved lines, not just the summary."""
    log_file = tmp_path / "app.log"
    log_file.write_text("2025-01-01T10:00:00Z INFO user@acme.com from 10.0.0.5\n")

    summary = get_logs(path=str(log_file), redact="strict")
    raw = get_raw_logs(analysis_id=_analysis_id(summary))
    assert "user@acme.com" not in raw
    assert "10.0.0.5" not in raw


def test_analyze_log_lines_cluster_line_indices_match_clusters() -> None:
    """cluster_line_indices is keyed by the same 1-based id as clusters_data, and
    each id's index set has the cluster's 'Occurrences' size."""
    from log_essence.server import analyze_log_lines

    lines = ["ERROR boom"] * 3 + ["INFO heartbeat ok"] * 5
    result = analyze_log_lines(lines, redact=False)

    assert result.clusters_data is not None
    cluster_ids = {c.id for c in result.clusters_data}
    assert set(result.cluster_line_indices) == cluster_ids

    for c in result.clusters_data:
        idxs = result.cluster_line_indices[c.id]
        assert len(idxs) == c.total_count
        assert idxs == sorted(idxs)  # original file order


def test_extract_templates_records_member_indices() -> None:
    """Each template knows which input line indices belong to it (1 line -> 1 template)."""
    from log_essence.server import detect_log_format, extract_templates

    lines = ["ERROR boom alpha", "INFO ok", "ERROR boom beta", "INFO ok"]
    fmt = detect_log_format(lines)
    templates = extract_templates(lines, fmt)

    # Every line index is covered exactly once across all templates.
    covered = sorted(i for t in templates for i in t.member_indices)
    assert covered == [0, 1, 2, 3]
    # member_indices size matches the template's own count.
    for t in templates:
        assert len(t.member_indices) == t.count
