# F1 — Severity-weighted budget allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When formatted output is truncated to a token budget, high-severity log clusters survive and are visible — across markdown, CLI JSON, and UI alike — instead of being crowded out by high-volume INFO noise.

**Architecture:** Normalize severity to one canonical vocabulary at extraction; track per-template severity over *all* lines (a distribution, `severity` = highest present); order clusters once in `analyze_log_lines` by `(highest severity, frequency)` and feed that single ordered list to both the formatter and the JSON output; select within-cluster templates/examples by the same key. No persisted-schema or MCP-signature changes.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `ruff`, Drain3, FastEmbed/k-means, Pydantic. Spec: `docs/superpowers/specs/2026-06-06-f1-severity-weighted-budget-design.md`.

---

## File Structure

- `src/log_essence/server.py` — all logic: severity constants + 5 helpers, `extract_severity`, `LogTemplate`, `extract_templates`, the two formatters, three distribution sites, the two severity-filter consumers (`analyze_log_lines` + `search_logs`), the one-time cluster ordering, and the `ClusterOutput` build.
- `tests/test_server.py` — all new tests (unit + end-to-end). Append to the existing file.
- Docs touch-up: `README.md` / `ROADMAP.md` wording asserting frequency ordering (Task 6).

**Each commit is internally correct** (no regressed intermediate states): filter-input normalization ships in Task 1; the highest-present severity change ships *with* its distribution + filter-matching fixes in Task 2. Run analysis tests with `redact=False` so redaction doesn't rewrite assertion substrings. Commit after each task; pre-commit (ruff + ruff-format + gitleaks) must pass. Do **not** edit `REPOMAP.md` (git hook regenerates it).

---

## Task 1: Canonical severity vocabulary + normalize every filter input

Fixes review findings #2 and #6 (alias normalization at extraction *and* at both filter consumers), with no alias-filter regression left for a later task.

**Files:**
- Modify: `src/log_essence/server.py` (constants `~:51`; helpers before `extract_severity` `:483`; JSON branch `:492`; `analyze_log_lines` filter input `:888`; `search_logs` filter input `:1790`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Add the import and write the failing tests**

In `tests/test_server.py`, add `analyze_log_lines` to the `from log_essence.server import (...)` block (it is **not** currently imported; later tasks reuse it). Then add:

```python
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
    result = search_logs(
        path=str(log_file), query="connection", severity_filter=[filter_level]
    )
    assert "Search Results" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "normalize_severity or severity_rank or json_normalized or alias_filter" -v`
Expected: FAIL — helper imports fail; `'FATAL' != 'CRITICAL'`; the alias-filter tests fail because the raw `{s.upper()}` filter input (`{"WARNING"}`) doesn't match the raw extracted level (`"WARN"`) and vice versa.

- [ ] **Step 3: Add constants**

In `src/log_essence/server.py`, just after `JSON_TIME_FIELDS` (around `:52`):

```python
SEVERITY_RANK = {"CRITICAL": 5, "ERROR": 4, "WARNING": 3, "INFO": 2, "DEBUG": 1}

SEVERITY_ALIASES = {
    "FATAL": "CRITICAL", "CRITICAL": "CRITICAL", "CRIT": "CRITICAL",
    "EMERG": "CRITICAL", "EMERGENCY": "CRITICAL", "ALERT": "CRITICAL",
    "PANIC": "CRITICAL", "SEVERE": "CRITICAL",
    "ERROR": "ERROR", "ERR": "ERROR",
    "WARNING": "WARNING", "WARN": "WARNING",
    "INFO": "INFO", "INFORMATION": "INFO", "NOTICE": "INFO",
    "DEBUG": "DEBUG", "TRACE": "DEBUG", "VERBOSE": "DEBUG",
    "FINE": "DEBUG", "FINER": "DEBUG", "FINEST": "DEBUG",
}
```

- [ ] **Step 4: Add helper functions**

Immediately before `def extract_severity(` (`:483`):

```python
def _normalize_severity(raw: str | None) -> str | None:
    """Map a raw level string to a canonical severity (CRITICAL/ERROR/WARNING/INFO/DEBUG).

    Known aliases (FATAL, WARN, ERR, TRACE, NOTICE, ...) map to canonical names.
    Unknown non-empty levels pass through uppercased (they rank 0). Empty -> None.
    """
    if not raw:
        return None
    key = raw.strip().upper()
    if not key:
        return None
    return SEVERITY_ALIASES.get(key, key)


def _severity_rank(severity: str | None) -> int:
    """Rank a severity for ordering (higher = more severe; unknown/None = 0)."""
    return SEVERITY_RANK.get(severity, 0)
```

- [ ] **Step 5: Normalize the JSON branch of `extract_severity`**

In `extract_severity` (`:492`), change the JSON return to:

```python
                for field in JSON_LEVEL_FIELDS:
                    if field in data:
                        return _normalize_severity(str(data[field]))
```

- [ ] **Step 6: Normalize the filter input in BOTH consumers**

In `analyze_log_lines` (`:887–889`):

```python
    # Apply severity filter
    if severity_filter:
        severity_set = {_normalize_severity(s) for s in severity_filter}
        severity_set.discard(None)
        templates = [t for t in templates if t.severity in severity_set]
```

In `search_logs` (`:1789–1790`):

```python
    if severity_filter:
        severity_set = {_normalize_severity(s) for s in severity_filter}
        severity_set.discard(None)
```

(Leave the rest of each filter body unchanged in this task; `analyze_log_lines` matching moves to `severity_counts` in Task 2.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -k "normalize_severity or severity_rank or json_normalized or alias_filter" -v`
Expected: PASS.
Run: `uv run pytest tests/test_server.py::test_extract_severity tests/test_server.py::test_get_logs_severity_filter tests/test_server.py::test_search_logs_with_severity_filter -v`
Expected: PASS (no regression).

- [ ] **Step 8: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: normalize severity aliases at extraction and both filter inputs"
```

---

## Task 2: Per-template severity over all lines (+ distribution & filter-matching)

Fixes review findings #1/#4 (track severity over all lines, highest-present) and ships the dependent corrections (#3/#7) in the same commit so the distribution and filter are never wrong: distribution sums `severity_counts`, and the `analyze_log_lines` filter matches if *any* line has the level.

**Files:**
- Modify: `src/log_essence/server.py` (`LogTemplate` `:212`; `extract_templates` `:545–581`; distribution `:720–724`, `:789–793`, `:919–923`; `analyze_log_lines` filter `:887–889`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "tracks_severity_over_all_lines or distribution_accurate or filter_matches_non_highest" -v`
Expected: FAIL — `LogTemplate` has no `severity_counts`; distribution is single-key; the `["INFO"]` filter drops the template (its `severity` will be `ERROR`).

- [ ] **Step 3: Add the `severity_counts` field to `LogTemplate`**

```python
@dataclass
class LogTemplate:
    """A log template extracted by Drain3."""

    template: str
    cluster_id: int
    count: int
    examples: list[str] = field(default_factory=list)
    severity: str | None = None
    severity_counts: dict[str, int] = field(default_factory=dict)
```

- [ ] **Step 4: Rewrite `extract_templates`**

Replace the whole function body (`:545–581`) with:

```python
def extract_templates(lines: list[str], log_format: str) -> list[LogTemplate]:
    """Extract log templates using Drain3."""
    miner = create_drain_miner()
    line_to_cluster: dict[int, int] = {}
    line_severities: dict[int, str | None] = {}

    for i, line in enumerate(lines):
        normalized = normalize_line(line, log_format)
        if not normalized:
            continue

        result = miner.add_log_message(normalized)
        line_to_cluster[i] = result["cluster_id"]
        line_severities[i] = extract_severity(line, log_format)

    # Build template objects
    templates: list[LogTemplate] = []
    for cluster in miner.drain.clusters:
        member_idxs = [i for i, cid in line_to_cluster.items() if cid == cluster.cluster_id]
        template_lines = [lines[i] for i in member_idxs]

        # Severity distribution across ALL member lines (normalized at extraction)
        severity_counts: dict[str, int] = defaultdict(int)
        for i in member_idxs:
            sev = line_severities.get(i)
            if sev:
                severity_counts[sev] += 1

        # Template severity = highest present (None if unlabeled)
        severity = max(severity_counts, key=_severity_rank, default=None)

        # Lead examples with a line of the highest severity so it stays visible
        examples = template_lines[:3]
        if severity is not None:
            top_idx = next(
                (i for i in member_idxs if line_severities.get(i) == severity), None
            )
            if top_idx is not None:
                top_example = lines[top_idx]
                examples = [top_example] + [ln for ln in template_lines[:3] if ln != top_example]
                examples = examples[:3]

        templates.append(
            LogTemplate(
                template=cluster.get_template(),
                cluster_id=cluster.cluster_id,
                count=cluster.size,
                examples=examples,
                severity=severity,
                severity_counts=dict(severity_counts),
            )
        )

    return templates
```

- [ ] **Step 5: Fix the three distribution sites**

In `format_as_markdown` (`:720–724`):

```python
    # Severity summary (counted from per-template distributions)
    severity_counts: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            for sev, n in template.severity_counts.items():
                severity_counts[sev] += n
```

In `_format_compact` (`:789–793`):

```python
    # Severity counts on one line (from per-template distributions)
    severity_counts: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            for sev, n in template.severity_counts.items():
                severity_counts[sev] += n
```

In `analyze_log_lines` (`:919–923`):

```python
    # Compute severity distribution (from per-template distributions)
    severity_distribution: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            for sev, n in template.severity_counts.items():
                severity_distribution[sev] += n
```

- [ ] **Step 6: Switch the `analyze_log_lines` filter to match on `severity_counts`**

In `analyze_log_lines` (`:887–889`, as left by Task 1):

```python
    # Apply severity filter (match if ANY line in the template has the level)
    if severity_filter:
        severity_set = {_normalize_severity(s) for s in severity_filter}
        severity_set.discard(None)
        templates = [t for t in templates if severity_set & set(t.severity_counts)]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -k "tracks_severity_over_all_lines or distribution_accurate or filter_matches_non_highest" -v`
Expected: PASS.
Run: `uv run pytest tests/test_server.py::test_extract_templates tests/test_server.py::test_get_logs_severity_filter tests/test_server.py::test_get_logs_with_sample -v`
Expected: PASS (no regression).

- [ ] **Step 8: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: track per-template severity over all lines; fix distribution and filter"
```

---

## Task 3: Order clusters once (markdown + JSON agree) + heading

Fixes review finding #5: order in `analyze_log_lines` so `clusters_data` (CLI/UI JSON) matches the markdown; `format_as_markdown` orders idempotently for direct callers.

**Files:**
- Modify: `src/log_essence/server.py` (helpers before `format_as_markdown` `:685`; `format_as_markdown` top `<:701`; heading `:735`; `analyze_log_lines` after `cluster_templates_semantically` `~:906`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

```python
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
        "payment gateway timeout" in t.template or t.severity == "ERROR"
        for t in first.templates
    )


def test_log_patterns_heading_is_severity() -> None:
    info_lines = [f"2025-01-01 INFO ping {i}" for i in range(5)]
    result = analyze_log_lines(info_lines, token_budget=8000, num_clusters=3, redact=False)
    assert "## Log Patterns by Severity" in result.markdown
    assert "## Log Patterns by Frequency" not in result.markdown
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "cluster_severity_rank or ordered_before_info or clusters_data_ordered or heading_is_severity" -v`
Expected: FAIL — helper import errors; INFO (count 500) precedes ERROR in both markdown and `clusters_data`; old heading text.

- [ ] **Step 3: Add `_cluster_severity_rank` and `_order_clusters`**

Immediately before `def format_as_markdown(` (`:685`):

```python
def _cluster_severity_rank(cluster: SemanticCluster) -> int:
    """Highest severity rank among a cluster's templates (0 if all unlabeled)."""
    return max((_severity_rank(t.severity) for t in cluster.templates), default=0)


def _order_clusters(clusters: list[SemanticCluster]) -> list[SemanticCluster]:
    """Order clusters by severity (highest first), then frequency. Deterministic + stable."""
    return sorted(
        clusters,
        key=lambda c: (_cluster_severity_rank(c), c.total_count),
        reverse=True,
    )
```

- [ ] **Step 4: Order at the top of `format_as_markdown`**

Between the docstring end (`:700`) and `if compact:` (`:701`):

```python
    # Order so high-severity content survives truncation (idempotent if already ordered)
    clusters = _order_clusters(clusters)

    if compact:
        return _format_compact(clusters, log_format, total_lines, token_budget)
```

- [ ] **Step 5: Order once in `analyze_log_lines` (the single source for all outputs)**

In `analyze_log_lines`, immediately after `clusters = cluster_templates_semantically(templates, num_clusters)` (`~:906`):

```python
    # Order once here so markdown AND clusters_data (CLI/UI JSON) agree.
    clusters = _order_clusters(clusters)
```

- [ ] **Step 6: Rename the heading**

In `format_as_markdown` (`:735`):

```python
    sections.append("## Log Patterns by Severity\n\n")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -k "cluster_severity_rank or ordered_before_info or clusters_data_ordered or heading_is_severity" -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: order clusters by severity once so all outputs agree"
```

---

## Task 4: Surface the highest-severity template + example within a cluster

Fixes review finding #3 (display): a cluster that floats on severity must render its high-severity template and example.

**Files:**
- Modify: `src/log_essence/server.py` (`_templates_by_priority` before `format_as_markdown`; `format_as_markdown` `:747`, `:752–755`; `_format_compact` `:805`, `:811–813`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_templates_by_priority_orders_severity_then_count() -> None:
    from log_essence.server import LogTemplate, _templates_by_priority

    ts = [
        LogTemplate("a", 1, 100, severity="INFO", severity_counts={"INFO": 100}),
        LogTemplate("b", 2, 1, severity="ERROR", severity_counts={"ERROR": 1}),
        LogTemplate("c", 3, 50, severity="INFO", severity_counts={"INFO": 50}),
    ]
    assert [t.template for t in _templates_by_priority(ts)] == ["b", "a", "c"]


def test_mixed_cluster_surfaces_error_template_and_example() -> None:
    from log_essence.server import LogTemplate, SemanticCluster, format_as_markdown

    templates = [
        LogTemplate(
            f"info event {i}", i, 100,
            examples=[f"2025-01-01 INFO info event {i}"],
            severity="INFO", severity_counts={"INFO": 100},
        )
        for i in range(6)
    ]
    templates.append(
        LogTemplate(
            "disk corruption detected", 99, 1,
            examples=["2025-01-01 ERROR disk corruption detected"],
            severity="ERROR", severity_counts={"ERROR": 1},
        )
    )
    cluster = SemanticCluster(
        templates=templates, centroid_idx=0, total_count=601, summary="info event 0"
    )

    md = format_as_markdown([cluster], "docker", 601, token_budget=8000)
    assert "disk corruption detected" in md

    compact = format_as_markdown([cluster], "docker", 601, token_budget=8000, compact=True)
    assert "disk corruption detected" in compact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "templates_by_priority or mixed_cluster_surfaces" -v`
Expected: FAIL — `_templates_by_priority` import error; the count-sorted top-5/top-3 excludes the rare ERROR template.

- [ ] **Step 3: Add `_templates_by_priority`**

Next to `_order_clusters` (before `format_as_markdown`):

```python
def _templates_by_priority(templates: list[LogTemplate]) -> list[LogTemplate]:
    """Sort templates by severity (highest first), then frequency."""
    return sorted(
        templates, key=lambda t: (_severity_rank(t.severity), t.count), reverse=True
    )
```

- [ ] **Step 4: Use it in `format_as_markdown`**

Top templates (`:747`):

```python
        # Add top templates (highest severity first, then frequency)
        top_templates = _templates_by_priority(cluster.templates)[:5]
```

Example block (`:752–755`):

```python
        # Add example from the highest-severity template
        if top_templates and top_templates[0].examples:
            example_text = top_templates[0].examples[0][:500]
            cluster_section += f"\n**Example:**\n```\n{example_text}\n```\n\n"
```

- [ ] **Step 5: Use it in `_format_compact`**

Top templates (`:805`):

```python
        top_templates = _templates_by_priority(cluster.templates)[:3]
```

Example block (`:811–813`):

```python
        # Only first example (from the highest-severity template), truncated
        if top_templates and top_templates[0].examples:
            example_text = top_templates[0].examples[0][:200]
            cluster_lines.append(f"  ex: {example_text}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -k "templates_by_priority or mixed_cluster_surfaces" -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: surface highest-severity template and example within clusters"
```

---

## Task 5: Prioritize high-severity templates in structured cluster output

**Files:**
- Modify: `src/log_essence/server.py` (`ClusterOutput` template selection `:931`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cluster_output_includes_high_severity_template() -> None:
    msgs = [
        "user login succeeded", "cache warmed up", "config reloaded from disk",
        "worker pool resized", "scheduled job completed", "metrics flushed to collector",
        "session token refreshed", "feature flag toggled", "background sync finished",
        "health probe responded", "queue drained empty", "snapshot persisted",
    ]
    lines: list[str] = []
    for m in msgs:
        lines += [f"2025-01-01 INFO {m}"] * 10
    lines.append("2025-01-01 ERROR critical subsystem failure")

    result = analyze_log_lines(lines, token_budget=8000, num_clusters=1, redact=False)

    assert result.clusters_data is not None
    assert max(len(c.templates) for c in result.clusters_data) >= 10  # output caps at 10
    out_templates = [t.template for c in result.clusters_data for t in c.templates]
    assert any("critical subsystem failure" in t for t in out_templates)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::test_cluster_output_includes_high_severity_template -v`
Expected: FAIL — the count-sorted `[:10]` drops the rare ERROR (count 1). (If the `>= 10` sanity assertion fails instead, Drain merged messages — make the message list more structurally varied until the single cluster holds ≥11 templates, then re-run.)

- [ ] **Step 3: Make the `ClusterOutput` selection severity-aware**

In `analyze_log_lines`, the `clusters_data` comprehension selects templates via `sorted(cluster.templates, key=lambda x: x.count, reverse=True)[:10]` (`:931`). Replace the inner loop with:

```python
                for t in _templates_by_priority(cluster.templates)[:10]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py::test_cluster_output_includes_high_severity_template -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: prioritize high-severity templates in structured cluster output"
```

---

## Task 6: End-to-end acceptance + full suite + docs

**Files:**
- Test: `tests/test_server.py`
- Modify (docs): `README.md`, `ROADMAP.md` (only if they assert frequency ordering)

- [ ] **Step 1: Write the end-to-end acceptance tests**

```python
def test_e2e_json_error_survives_small_budget() -> None:
    info: list[str] = []
    for n in range(14):
        info += [f'{{"level":"info","message":"served route {n}"}}'] * 40
    err = ['{"level":"error","message":"upstream connection refused"}']
    result = analyze_log_lines(info + err, token_budget=400, num_clusters=10, redact=False)
    md = result.markdown
    assert "upstream connection refused" in md  # ERROR survived
    assert "omitted" in md  # truncation actually happened


def test_e2e_compact_error_survives_small_budget() -> None:
    info: list[str] = []
    for n in range(14):
        info += [f"2025-01-01 INFO route {n} served"] * 40
    err = ["2025-01-01 ERROR upstream connection refused"]
    result = analyze_log_lines(
        info + err, token_budget=400, num_clusters=10, redact=False, compact=True
    )
    assert "upstream connection refused" in result.markdown
    assert "omitted" in result.markdown


def test_e2e_json_fatal_alias_ranks_first() -> None:
    info = ['{"level":"info","message":"steady state"}'] * 300
    fatal = ['{"level":"fatal","message":"kernel oops"}']
    result = analyze_log_lines(info + fatal, token_budget=8000, num_clusters=10, redact=False)
    md = result.markdown
    assert md.index("kernel oops") < md.index("steady state")
    assert result.severity_distribution.get("CRITICAL") == 1  # fatal -> CRITICAL


def test_e2e_json_collapsed_error_is_visible() -> None:
    lines = ['{"level":"info","message":"request handled"}'] * 200
    lines.append('{"level":"error","message":"request handled"}')
    result = analyze_log_lines(lines, token_budget=8000, num_clusters=10, redact=False)
    assert "[ERROR]" in result.markdown  # badge reflects the highest present
    assert result.severity_distribution == {"INFO": 200, "ERROR": 1}
```

- [ ] **Step 2: Run the acceptance tests**

Run: `uv run pytest tests/test_server.py -k "e2e_" -v`
Expected: PASS (4 passed). If `test_e2e_json_error_survives_small_budget` shows no `"omitted"`, raise `range(14)` → `range(20)` or lower `token_budget` until truncation occurs (ERROR must still survive). If `[ERROR]` is absent, assert on the exact badge produced by the formatter (`f"[{template.severity}] "`).

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: baseline (208 passed / 1 skipped) plus the new tests pass. Fix any regression before continuing.

- [ ] **Step 4: Lint and format**

Run: `uv run ruff check src/ tests/` → no errors.
Run: `uv run ruff format --check src/ tests/` → no changes (run `uv run ruff format src/ tests/` then re-check if it reports reformatting).

- [ ] **Step 5: Update docs if they assert frequency ordering**

Run: `grep -rn "by Frequency\|by frequency\|sorted by frequency\|frequency.*order" README.md ROADMAP.md`
For any hit describing output ordering as frequency-based, reword to severity-first (e.g. "patterns are listed highest-severity first, then by frequency"). If none, skip. Do **not** edit `REPOMAP.md`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_server.py README.md ROADMAP.md
git commit -m "test: end-to-end severity-budget acceptance; docs for ordering"
```

(Drop README.md/ROADMAP.md from `git add` if unchanged.)

---

## Self-Review (performed against spec v3)

**1. Spec coverage:**
- Findings #2/#6 (alias normalization at extraction + both filter inputs) → Task 1. ✓
- Findings #1/#4 (per-template severity over all lines, highest-present, high-severity example) + #3/#7 dependent distribution/filter-matching → Task 2 (same commit). ✓
- Finding #5 (order once, markdown + clusters_data agree) + heading → Task 3. ✓
- Finding #3 display (within-cluster severity-aware) → Task 4. ✓
- Severity-aware `ClusterOutput[:10]` → Task 5. ✓
- End-to-end (JSON same-shape, aliases, plaintext, compact) → Task 6. ✓
- `None`/unknown rank 0 → `_severity_rank` (Task 1) + `_cluster_severity_rank` test (Task 3). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every test step shows the body + exact command + expected outcome. ✓

**3. Type/name consistency:** `_normalize_severity`, `_severity_rank` (Task 1) → `extract_templates` (Task 2), `_cluster_severity_rank`/`_order_clusters` (Task 3), `_templates_by_priority` (Task 4/5). `severity_counts: dict[str,int]` (Task 2) → distribution/filter (Task 2), used by ordering helpers. `_order_clusters` applied in both `format_as_markdown` and `analyze_log_lines` (Task 3). `AnalysisResult.severity_distribution` / `.clusters_data`, `ClusterOutput.id/.templates`, `TemplateOutput.template/.severity` (Pydantic, `ui/models.py`) match real field names. ✓

**4. Commit correctness:** Task 1 leaves no alias-filter regression; Task 2 fixes distribution + filter in the same commit as the severity-semantics change. No intermediate commit is regressed (safe under `AGENTS.md` push-at-session-end). ✓

**Out-of-scope (per spec):** `severity_counts` in `TemplateOutput` (F2); per-cluster retrieval (F2); `cluster_templates_semantically` internal order untouched.
