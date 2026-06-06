# F1 — Severity-weighted budget allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When formatted output is truncated to a token budget, high-severity log clusters survive and their high-severity content is visible — instead of being crowded out by high-volume INFO noise.

**Architecture:** Preserve and normalize severity through extraction (one canonical vocabulary; per-template severity tracked over *all* lines as a distribution), then order clusters by `(highest severity, frequency)` at format time and select within-cluster templates/examples by the same key. No persisted-schema or MCP-signature changes.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `ruff`, Drain3 (template mining), FastEmbed/k-means (semantic clustering), Pydantic (output models). Spec: `docs/superpowers/specs/2026-06-06-f1-severity-weighted-budget-design.md`.

---

## File Structure

- `src/log_essence/server.py` — all logic changes: severity constants + helpers, `extract_severity` normalization, `LogTemplate` field, `extract_templates`, the two formatters, the three severity-distribution computations, the `severity_filter`, and the `ClusterOutput` build. (Single large module; follow the existing pattern rather than splitting.)
- `tests/test_server.py` — all new tests (unit + end-to-end through `analyze_log_lines`). Append to the existing file.
- `docs/superpowers/plans/2026-06-06-f1-severity-weighted-budget.md` — this plan.
- Docs touch-up: `README.md` / `ROADMAP.md` wording asserting frequency ordering (Task 7).

**Conventions:** `defaultdict` is already imported in `server.py`. Tests use plain `pytest` functions. Run `redact=False` in analysis tests so redaction doesn't rewrite assertion substrings. Commit after each task. Pre-commit (ruff + ruff-format + gitleaks) must pass on every commit.

---

## Task 1: Severity vocabulary — constants, normalization, rank

**Files:**
- Modify: `src/log_essence/server.py` (constants block near `:51`; new helpers before `extract_severity` at `:483`; JSON branch of `extract_severity` at `:492`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py` (ensure `extract_severity` is imported — it already is at the top of the file):

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
    assert _normalize_severity("weirdlevel") == "WEIRDLEVEL"  # unknown passes through


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py::test_normalize_severity_aliases tests/test_server.py::test_severity_rank_order tests/test_server.py::test_extract_severity_json_normalized -v`
Expected: FAIL — `ImportError: cannot import name '_normalize_severity'` (and the JSON test asserts `'FATAL' == 'CRITICAL'`).

- [ ] **Step 3: Add constants**

In `src/log_essence/server.py`, just after the `JSON_LEVEL_FIELDS`/`JSON_TIME_FIELDS` constants (around `:52`), add:

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

In `src/log_essence/server.py`, immediately before `def extract_severity(` (`:483`), add:

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

In `extract_severity` (`:483`), the JSON branch currently returns `str(data[field]).upper()` (`:492`). Change it to normalize:

```python
                for field in JSON_LEVEL_FIELDS:
                    if field in data:
                        return _normalize_severity(str(data[field]))
```

(The regex branch already returns canonical labels — leave it unchanged.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py::test_normalize_severity_aliases tests/test_server.py::test_severity_rank_order tests/test_server.py::test_extract_severity_json_normalized -v`
Expected: PASS (3 passed).

Also confirm no regression in the existing severity test:
Run: `uv run pytest tests/test_server.py::test_extract_severity -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: normalize log severity aliases to a canonical vocabulary"
```

---

## Task 2: Per-template severity over all lines (highest-present)

**Files:**
- Modify: `src/log_essence/server.py` (`LogTemplate` dataclass `:212`; `extract_templates` `:545–581`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
def test_extract_templates_tracks_severity_over_all_lines() -> None:
    # Same-shape JSON: 100 INFO + 1 ERROR collapse to one Drain template.
    lines = ['{"level":"info","message":"request handled"}'] * 100
    lines.append('{"level":"error","message":"request handled"}')

    templates = extract_templates(lines, "json")

    assert len(templates) == 1
    t = templates[0]
    assert t.count == 101
    # severity counted across ALL lines, not just the first 10
    assert t.severity_counts == {"INFO": 100, "ERROR": 1}
    # template severity is the HIGHEST present, not the modal
    assert t.severity == "ERROR"
    # an ERROR example is captured so the high-severity line is visible downstream
    assert any(extract_severity(ex, "json") == "ERROR" for ex in t.examples)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::test_extract_templates_tracks_severity_over_all_lines -v`
Expected: FAIL — `AttributeError: 'LogTemplate' object has no attribute 'severity_counts'` (and `severity` would be `'INFO'`).

- [ ] **Step 3: Add the `severity_counts` field to `LogTemplate`**

In `src/log_essence/server.py`, the `LogTemplate` dataclass (`:212`) becomes:

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

- [ ] **Step 4: Rewrite `extract_templates` severity handling**

Replace the body of `extract_templates` (`:545–581`) with the version below. It computes each line's severity once in the first pass, then aggregates per template over **all** member lines, sets `severity` to the highest present, and leads `examples` with a highest-severity line.

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

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py::test_extract_templates_tracks_severity_over_all_lines -v`
Expected: PASS.

Confirm the existing template test still passes:
Run: `uv run pytest tests/test_server.py::test_extract_templates -v`
Expected: PASS (3 INFO lines → one template, `count == 3`).

- [ ] **Step 6: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: track per-template severity over all lines (highest-present)"
```

---

## Task 3: Order clusters by severity, then frequency

**Files:**
- Modify: `src/log_essence/server.py` (new `_cluster_severity_rank` before `format_as_markdown` at `:685`; reorder inside `format_as_markdown` before the `if compact:` at `:701`; heading at `:735`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`:

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
    # 500 INFO heartbeats (one cluster) + 1 ERROR (its own cluster).
    info_lines = [f"2025-01-01 INFO heartbeat ping {i}" for i in range(500)]
    error_lines = ["2025-01-01 ERROR payment gateway timeout"]

    result = analyze_log_lines(
        info_lines + error_lines, token_budget=8000, num_clusters=10, redact=False
    )
    md = result.markdown

    assert "payment gateway timeout" in md
    assert "heartbeat ping" in md
    # ERROR cluster is rendered before the INFO cluster (severity-ordered)
    assert md.index("payment gateway timeout") < md.index("heartbeat ping")


def test_log_patterns_heading_is_severity() -> None:
    info_lines = [f"2025-01-01 INFO ping {i}" for i in range(5)]
    result = analyze_log_lines(info_lines, token_budget=8000, num_clusters=3, redact=False)
    assert "## Log Patterns by Severity" in result.markdown
    assert "## Log Patterns by Frequency" not in result.markdown
```

Before these tests will run, add `analyze_log_lines` to the `from log_essence.server import (...)` block at the top of `tests/test_server.py` — it is **not** currently imported. (Later tasks reuse it; add it once here.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py::test_cluster_severity_rank tests/test_server.py::test_error_cluster_ordered_before_info_cluster tests/test_server.py::test_log_patterns_heading_is_severity -v`
Expected: FAIL — `ImportError: cannot import name '_cluster_severity_rank'`; the ordering test fails because INFO (count 500) sorts before ERROR; the heading test fails on the old "by Frequency" text.

- [ ] **Step 3: Add `_cluster_severity_rank`**

In `src/log_essence/server.py`, immediately before `def format_as_markdown(` (`:685`), add:

```python
def _cluster_severity_rank(cluster: SemanticCluster) -> int:
    """Highest severity rank among a cluster's templates (0 if all unlabeled)."""
    return max((_severity_rank(t.severity) for t in cluster.templates), default=0)
```

- [ ] **Step 4: Reorder clusters at the top of `format_as_markdown`**

In `format_as_markdown`, between the end of the docstring (`:700`) and the `if compact:` line (`:701`), insert the reorder so both the verbose and compact paths use it:

```python
    # Order clusters so the highest-severity content survives truncation;
    # frequency breaks ties within a tier. reverse=True sorts both fields desc.
    clusters = sorted(
        clusters,
        key=lambda c: (_cluster_severity_rank(c), c.total_count),
        reverse=True,
    )

    if compact:
        return _format_compact(clusters, log_format, total_lines, token_budget)
```

- [ ] **Step 5: Rename the heading**

In `format_as_markdown` (`:735`), change:

```python
    sections.append("## Log Patterns by Severity\n\n")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py::test_cluster_severity_rank tests/test_server.py::test_error_cluster_ordered_before_info_cluster tests/test_server.py::test_log_patterns_heading_is_severity -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: order clusters by severity so high-severity survives truncation"
```

---

## Task 4: Surface the highest-severity template + example within a cluster

**Files:**
- Modify: `src/log_essence/server.py` (new `_templates_by_priority` before `format_as_markdown` at `:685`; `format_as_markdown` top-templates `:747` + example `:752–755`; `_format_compact` top-templates `:805` + example `:811–813`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`:

```python
def test_templates_by_priority_orders_severity_then_count() -> None:
    from log_essence.server import LogTemplate, _templates_by_priority

    ts = [
        LogTemplate("a", 1, 100, severity="INFO", severity_counts={"INFO": 100}),
        LogTemplate("b", 2, 1, severity="ERROR", severity_counts={"ERROR": 1}),
        LogTemplate("c", 3, 50, severity="INFO", severity_counts={"INFO": 50}),
    ]
    ordered = _templates_by_priority(ts)
    assert [t.template for t in ordered] == ["b", "a", "c"]  # ERROR first despite count 1


def test_mixed_cluster_surfaces_error_template_and_example() -> None:
    from log_essence.server import LogTemplate, SemanticCluster, format_as_markdown

    templates = [
        LogTemplate(
            f"info event {i}",
            i,
            100,
            examples=[f"2025-01-01 INFO info event {i}"],
            severity="INFO",
            severity_counts={"INFO": 100},
        )
        for i in range(6)
    ]
    templates.append(
        LogTemplate(
            "disk corruption detected",
            99,
            1,
            examples=["2025-01-01 ERROR disk corruption detected"],
            severity="ERROR",
            severity_counts={"ERROR": 1},
        )
    )
    cluster = SemanticCluster(
        templates=templates, centroid_idx=0, total_count=601, summary="info event 0"
    )

    md = format_as_markdown([cluster], "docker", 601, token_budget=8000)
    assert "disk corruption detected" in md  # ERROR template shown despite low count

    compact = format_as_markdown([cluster], "docker", 601, token_budget=8000, compact=True)
    assert "disk corruption detected" in compact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py::test_templates_by_priority_orders_severity_then_count tests/test_server.py::test_mixed_cluster_surfaces_error_template_and_example -v`
Expected: FAIL — `ImportError: cannot import name '_templates_by_priority'`; the mixed-cluster test fails because the count-sorted top-5/top-3 excludes the rare ERROR template.

- [ ] **Step 3: Add `_templates_by_priority`**

In `src/log_essence/server.py`, immediately before `def format_as_markdown(` (next to `_cluster_severity_rank`), add:

```python
def _templates_by_priority(templates: list[LogTemplate]) -> list[LogTemplate]:
    """Sort templates by severity (highest first), then frequency."""
    return sorted(
        templates, key=lambda t: (_severity_rank(t.severity), t.count), reverse=True
    )
```

- [ ] **Step 4: Use it in `format_as_markdown`**

In `format_as_markdown`, replace the top-templates selection (`:747`):

```python
        # Add top templates (highest severity first, then frequency)
        top_templates = _templates_by_priority(cluster.templates)[:5]
```

and replace the example block (`:752–755`) to use the lead (highest-priority) template:

```python
        # Add example from the highest-severity template
        if top_templates and top_templates[0].examples:
            example_text = top_templates[0].examples[0][:500]
            cluster_section += f"\n**Example:**\n```\n{example_text}\n```\n\n"
```

- [ ] **Step 5: Use it in `_format_compact`**

In `_format_compact`, replace the top-templates selection (`:805`):

```python
        top_templates = _templates_by_priority(cluster.templates)[:3]
```

and replace the example block (`:811–813`):

```python
        # Only first example (from the highest-severity template), truncated
        if top_templates and top_templates[0].examples:
            example_text = top_templates[0].examples[0][:200]
            cluster_lines.append(f"  ex: {example_text}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py::test_templates_by_priority_orders_severity_then_count tests/test_server.py::test_mixed_cluster_surfaces_error_template_and_example -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "feat: surface highest-severity template and example within clusters"
```

---

## Task 5: Accurate severity distribution + any-line severity filter

**Files:**
- Modify: `src/log_essence/server.py` (markdown distribution `:720–724`; compact distribution `:789–793`; stats distribution `:919–923`; `severity_filter` `:887–889`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`:

```python
def test_severity_distribution_accurate_for_collapsed_json() -> None:
    lines = ['{"level":"info","message":"x"}'] * 100 + ['{"level":"error","message":"x"}']
    result = analyze_log_lines(lines, token_budget=8000, num_clusters=10, redact=False)
    # Distribution reflects BOTH levels with real counts, not all-INFO or all-ERROR
    assert result.severity_distribution == {"INFO": 100, "ERROR": 1}


def test_severity_filter_matches_non_highest_level() -> None:
    # Collapsed template whose highest severity is ERROR but which has 50 INFO lines.
    lines = ['{"level":"info","message":"x"}'] * 50 + ['{"level":"error","message":"x"}']
    result = analyze_log_lines(
        lines, token_budget=8000, num_clusters=10, severity_filter=["INFO"], redact=False
    )
    # Must be kept: the template contains INFO lines even though severity == ERROR
    assert "No log patterns found" not in result.markdown


def test_severity_filter_input_alias_normalized() -> None:
    lines = ['{"level":"info","message":"a"}'] * 5 + ['{"level":"warning","message":"b"}'] * 3
    result = analyze_log_lines(
        lines, token_budget=8000, num_clusters=10, severity_filter=["warn"], redact=False
    )
    # "warn" filter normalizes to WARNING and matches the WARNING template
    assert "No log patterns found" not in result.markdown
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py::test_severity_distribution_accurate_for_collapsed_json tests/test_server.py::test_severity_filter_matches_non_highest_level tests/test_server.py::test_severity_filter_input_alias_normalized -v`
Expected: FAIL — distribution is `{"ERROR": 101}` (severity now == highest); the `["INFO"]` filter drops the template (its `severity` is `ERROR`); the `["warn"]` filter builds `{"WARN"}` which matches nothing.

- [ ] **Step 3: Fix the markdown distribution**

In `format_as_markdown` (`:720–724`), replace the distribution accumulation:

```python
    # Severity summary (counted from per-template distributions)
    severity_counts: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            for sev, n in template.severity_counts.items():
                severity_counts[sev] += n
```

- [ ] **Step 4: Fix the compact distribution**

In `_format_compact` (`:789–793`), replace:

```python
    # Severity counts on one line (from per-template distributions)
    severity_counts: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            for sev, n in template.severity_counts.items():
                severity_counts[sev] += n
```

- [ ] **Step 5: Fix the stats distribution**

In `analyze_log_lines` (`:919–923`), replace:

```python
    # Compute severity distribution (from per-template distributions)
    severity_distribution: dict[str, int] = defaultdict(int)
    for cluster in clusters:
        for template in cluster.templates:
            for sev, n in template.severity_counts.items():
                severity_distribution[sev] += n
```

- [ ] **Step 6: Fix the severity filter**

In `analyze_log_lines` (`:887–889`), replace:

```python
    # Apply severity filter (match if ANY line in the template has the level)
    if severity_filter:
        severity_set = {_normalize_severity(s) for s in severity_filter}
        severity_set.discard(None)
        templates = [t for t in templates if severity_set & set(t.severity_counts)]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py::test_severity_distribution_accurate_for_collapsed_json tests/test_server.py::test_severity_filter_matches_non_highest_level tests/test_server.py::test_severity_filter_input_alias_normalized -v`
Expected: PASS (3 passed).

Confirm the existing filter test still passes:
Run: `uv run pytest tests/test_server.py::test_get_logs_severity_filter -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/log_essence/server.py tests/test_server.py
git commit -m "fix: accurate severity distribution and any-line severity filter"
```

---

## Task 6: Prioritize high-severity templates in structured cluster output

**Files:**
- Modify: `src/log_essence/server.py` (`ClusterOutput` template selection `:931`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
def test_cluster_output_includes_high_severity_template() -> None:
    # 12 structurally-distinct INFO templates (count 10 each) + 1 rare ERROR (count 1),
    # forced into ONE semantic cluster (num_clusters=1). The output caps templates at 10.
    msgs = [
        "user login succeeded",
        "cache warmed up",
        "config reloaded from disk",
        "worker pool resized",
        "scheduled job completed",
        "metrics flushed to collector",
        "session token refreshed",
        "feature flag toggled",
        "background sync finished",
        "health probe responded",
        "queue drained empty",
        "snapshot persisted",
    ]
    lines: list[str] = []
    for m in msgs:
        lines += [f"2025-01-01 INFO {m}"] * 10
    lines.append("2025-01-01 ERROR critical subsystem failure")

    result = analyze_log_lines(lines, token_budget=8000, num_clusters=1, redact=False)

    assert result.clusters_data is not None
    # Sanity: the single cluster holds more templates than the output cap (10),
    # so plain count-sorting would drop the rare ERROR.
    assert max(len(c.templates) for c in result.clusters_data) >= 10
    out_templates = [t.template for c in result.clusters_data for t in c.templates]
    assert any("critical subsystem failure" in t for t in out_templates)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::test_cluster_output_includes_high_severity_template -v`
Expected: FAIL — the ERROR template (count 1) is excluded by the count-sorted `[:10]`, so the `any(...)` assertion fails.

(If instead the sanity assertion `>= 10` fails, Drain merged some messages — make the message list more structurally varied until the single cluster holds ≥11 templates, then re-run to see the real failure.)

- [ ] **Step 3: Make the `ClusterOutput` selection severity-aware**

In `analyze_log_lines`, the `clusters_data` comprehension selects templates via
`sorted(cluster.templates, key=lambda x: x.count, reverse=True)[:10]` (`:931`). Replace that inner loop with the shared priority sort:

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

## Task 7: End-to-end acceptance + full suite + docs

**Files:**
- Test: `tests/test_server.py`
- Modify (docs): `README.md`, `ROADMAP.md` (only if they assert frequency ordering)

- [ ] **Step 1: Write the end-to-end acceptance tests**

Add to `tests/test_server.py`:

```python
def test_e2e_json_error_survives_small_budget() -> None:
    # Many INFO message types (own clusters) + one low-volume ERROR message.
    info: list[str] = []
    for n in range(14):
        info += [f'{{"level":"info","message":"served route {n}"}}'] * 40
    err = ['{"level":"error","message":"upstream connection refused"}']

    result = analyze_log_lines(
        info + err, token_budget=400, num_clusters=10, redact=False
    )
    md = result.markdown
    assert "upstream connection refused" in md  # ERROR survived truncation
    assert "omitted" in md  # truncation actually happened


def test_e2e_json_fatal_alias_ranks_first() -> None:
    info = ['{"level":"info","message":"steady state"}'] * 300
    fatal = ['{"level":"fatal","message":"kernel oops"}']
    result = analyze_log_lines(
        info + fatal, token_budget=8000, num_clusters=10, redact=False
    )
    md = result.markdown
    assert "kernel oops" in md
    assert md.index("kernel oops") < md.index("steady state")  # FATAL not sunk below INFO
    assert result.severity_distribution.get("CRITICAL") == 1  # fatal -> CRITICAL


def test_e2e_json_collapsed_error_is_visible() -> None:
    # Same-shape collapse: the one ERROR must still surface in the rendered output.
    lines = ['{"level":"info","message":"request handled"}'] * 200
    lines.append('{"level":"error","message":"request handled"}')
    result = analyze_log_lines(lines, token_budget=8000, num_clusters=10, redact=False)
    md = result.markdown
    assert "[ERROR]" in md  # severity badge reflects the highest present
    assert result.severity_distribution == {"INFO": 200, "ERROR": 1}


def test_e2e_compact_error_survives_small_budget() -> None:
    info: list[str] = []
    for n in range(14):
        info += [f"2025-01-01 INFO route {n} served"] * 40
    err = ["2025-01-01 ERROR upstream connection refused"]
    result = analyze_log_lines(
        info + err, token_budget=400, num_clusters=10, redact=False, compact=True
    )
    md = result.markdown
    assert "upstream connection refused" in md
    assert "omitted" in md
```

- [ ] **Step 2: Run the acceptance tests**

Run: `uv run pytest tests/test_server.py -k "e2e_" -v`
Expected: PASS (4 passed). These exercise the full `analyze_log_lines` pipeline (extraction → clustering → ordered, severity-aware formatting). If `test_e2e_json_error_survives_small_budget` does not show `"omitted"`, raise the message-type count (`range(14)` → `range(20)`) or lower `token_budget` until truncation occurs, then confirm the ERROR still survives. If `[ERROR]` is absent in `test_e2e_json_collapsed_error_is_visible`, the badge format differs — assert on the exact badge produced by the formatter (`f"[{template.severity}] "`).

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all prior tests (208 passed / 1 skipped baseline) plus the ~18 new tests pass. Investigate and fix any regression before continuing.

- [ ] **Step 4: Lint and format**

Run: `uv run ruff check src/ tests/`
Expected: no errors.
Run: `uv run ruff format --check src/ tests/`
Expected: no changes needed (run `uv run ruff format src/ tests/` if it reports reformatting, then re-run the check).

- [ ] **Step 5: Update docs if they assert frequency ordering**

Run: `grep -rn "by Frequency\|by frequency\|sorted by frequency\|frequency.*order" README.md ROADMAP.md`
For any hit that describes output ordering as frequency-based, update the wording to reflect severity-first ordering (e.g. "patterns are listed highest-severity first, then by frequency"). If there are no such hits, skip. Do **not** edit `REPOMAP.md` (regenerated by a git hook).

- [ ] **Step 6: Commit**

```bash
git add tests/test_server.py README.md ROADMAP.md
git commit -m "test: end-to-end severity-budget acceptance; docs for ordering"
```

(If no docs changed, drop them from the `git add`.)

---

## Self-Review (performed against the spec)

**1. Spec coverage:**
- Finding 2 (alias normalization) → Task 1. ✓
- Finding 1 (per-template severity over all lines, highest-present, high-severity example) → Task 2. ✓
- Tiered ordering + heading → Task 3. ✓
- Finding 3 (within-cluster severity-aware display + example) → Task 4. ✓
- Three accurate distribution sites + `severity_filter` via counts → Task 5. ✓
- Severity-aware `ClusterOutput[:10]` → Task 6. ✓
- Finding 4 (end-to-end tests: JSON same-shape, JSON aliases, plaintext, compact) → Task 7. ✓
- `None`/unknown rank 0 → covered by `_severity_rank` (Task 1) + `_cluster_severity_rank` test (Task 3). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every test step shows the test body and the exact command + expected outcome. ✓

**3. Type/name consistency:** `_normalize_severity`, `_severity_rank` (Task 1) → used by `extract_templates` (Task 2), `_cluster_severity_rank` (Task 3), `_templates_by_priority` (Task 4). `severity_counts: dict[str,int]` defined in Task 2 → consumed in Tasks 3/5/6. `_templates_by_priority` defined in Task 4 → reused in Task 6. `AnalysisResult.severity_distribution` / `.clusters_data` (Pydantic, `ui/models.py`) used in Tasks 5/6/7 match the real field names. ✓

**Out-of-scope (per spec, not in this plan):** exposing `severity_counts` in `TemplateOutput` (F2), per-cluster retrieval (F2). Canonical `cluster_templates_semantically` ordering untouched.
