# F1 — Severity-weighted budget allocation (design)

- **Status:** approved (design v2), pending implementation
- **Date:** 2026-06-06
- **Roadmap ID:** F1 (`P0`) — see `ROADMAP.md`
- **Sizing:** Churn **M** · Sites many · Files 2 (`server.py`, `tests/`) · Horizon medium ·
  Verify test (unit + end-to-end) · slightly **messy** (Drain/severity edge cases)

> **v2 supersedes the formatter-only v1.** A code review (verified empirically, below)
> showed the severity signal is degraded **before** the formatter runs, so reordering
> clusters alone does not satisfy the acceptance criterion on real logs — JSON logs in
> particular. The fix reaches into severity extraction, which raises the size from S to M.

## Problem

When formatted output is truncated to fit `token_budget`, both formatters iterate
clusters in arrival order and `break` once the budget is hit (`format_as_markdown`
loop `server.py:739`/budget `:761`; `_format_compact` loop `:801`/budget `:817`).
Clusters arrive sorted by `total_count` descending (`server.py:641`), so a high-volume
INFO heartbeat leads and a small but critical ERROR/FATAL cluster sits at the tail —
the first thing dropped. **Signal gets truncated, noise survives.**

But reordering is necessary, not sufficient. Severity is reduced too early and too
lossily, so by the time clusters reach the formatter the high-severity signal may
already be gone or mislabeled:

- **JSON level-stripping.** `extract_json_message` (`server.py:464`) returns only the
  message (or re-serializes with `JSON_LEVEL_FIELDS` removed), so same-message JSON
  lines of different levels collapse into one Drain template.
- **Modal-of-10 sampling.** `extract_templates` (`server.py:566`) samples severity from
  only `template_lines[:10]` and takes the mode, so a rare high-severity line in a large
  template is invisible.
- **Unnormalized JSON aliases.** `extract_severity` (`server.py:492`) returns the raw
  level uppercased for JSON (`FATAL`, `WARN`, `ERR`, `TRACE`), bypassing the alias
  mapping the regex path applies.
- **Count-only display.** Within a cluster the top templates and example are chosen by
  frequency (`server.py:747`, `:805`, `:753`), so a cluster can float on severity yet
  render with no visible high-severity template or example.

## Verification — code review findings (all confirmed empirically)

Reproduced against the real functions (`tmp/f1_repro.py`, gitignored):

| # | Sev | Claim | Evidence |
|---|-----|-------|----------|
| 1 | CRITICAL | Same-shape JSON `100 INFO + 1 ERROR` collapses to one template | `extract_templates` → **1 template, count=101, severity=INFO**; e2e markdown mentions "ERROR" → **False** |
| 2 | HIGH | JSON severity returns raw aliases below the rank table | `extract_severity` JSON → `FATAL`/`WARN`/`ERR`/`TRACE` (rank 0); regex path correctly maps the same to CRITICAL/WARNING/ERROR/DEBUG |
| 3 | HIGH | Mixed cluster floats by max severity but hides the ERROR | 6 INFO templates (count 100) + 1 ERROR template (count 1) → ERROR template **not** in rendered body (top-5 by count); example is INFO |
| 4 | MEDIUM | Manual-cluster unit test passes while e2e fails | Proposed v1 test builds `SemanticCluster`s directly, bypassing `extract_templates`/JSON extraction where #1/#2 live |

Clean checks confirmed: no schema/migration (analytics SQLite is stats-only),
`_format_compact` is single-caller (`server.py:702`), no persisted-state hazard.

## Goal / acceptance criteria

A high-severity cluster **survives truncation** at a small `token_budget`, **and the
rendered output makes the high-severity content visible** (badge + example). Holds at
any volume ratio and across formats:

1. Plaintext/Docker: INFO-dominated log with one ERROR cluster → ERROR survives.
2. **JSON same-shape:** `100 {"level":"info"} + 1 {"level":"error"}` (identical message)
   → the collapsed cluster ranks high and the output shows ERROR (badge/example), not
   just INFO.
3. **JSON aliases:** a `{"level":"fatal"}` cluster ranks at the top, never below DEBUG.
4. Both `compact=False` and `compact=True`.

## Decision — tiered ordering + upstream severity integrity

**Ordering (unchanged from v1):** sort clusters by `(highest severity rank in the
cluster, total_count)`, both descending — severity primary, frequency tiebreak. The
only option that *guarantees* the criterion at any ratio. (Rejected: weighted blend
`Σ weight×count` fails at `1M INFO : 10 ERROR`; dampened `Σ weight×log1p(count)` only
holds probabilistically. Both verified against the criterion.)

**Severity integrity (new in v2):** preserve and normalize severity through extraction
so the ordering has a correct signal to act on, and surface it in the rendered body.

## Design details

### Constants (`server.py`, constants block near `:51`)

```python
SEVERITY_RANK = {"CRITICAL": 5, "ERROR": 4, "WARNING": 3, "INFO": 2, "DEBUG": 1}
# None / unrecognized -> 0 via SEVERITY_RANK.get(sev, 0)

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

Hardcoded by intent (YAGNI): no caller wants tunable weights; constants are testable and
trivially promoted later.

### Helpers (`server.py`, near the formatters)

```python
def _normalize_severity(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().upper()
    return SEVERITY_ALIASES.get(key, key or None)  # unknown levels pass through (rank 0)

def _severity_rank(sev: str | None) -> int:
    return SEVERITY_RANK.get(sev, 0)

def _cluster_severity_rank(cluster: SemanticCluster) -> int:
    return max((_severity_rank(t.severity) for t in cluster.templates), default=0)
```

### Finding 2 — normalize at extraction

`extract_severity` (`server.py:483`): route both paths through `_normalize_severity`.
JSON branch: `return _normalize_severity(str(data[field]))` (was `.upper()`). The regex
branch already emits canonical labels, which `_normalize_severity` returns unchanged.
Result: a single canonical vocabulary everywhere severity is read.

### Finding 1 — per-template severity over all lines

Add to `LogTemplate` (`server.py:212`):

```python
severity_counts: dict[str, int] = field(default_factory=dict)
```

`LogTemplate.severity` keeps type `str | None` but its **meaning changes**: it is now the
**highest** severity present in the template (derived from `severity_counts`), not the
modal-of-first-10. This is the right single value for a severity-prioritization tool.

`extract_templates` (`server.py:558–579`): build `severity_counts` over **all**
`template_lines` (each normalized via `_normalize_severity`), set `severity` to the
highest-rank key (None if empty). Build `examples` so at least one example of the highest
severity present leads the list (so the high-severity line is visible), then fill with
the first lines, deduped, capped at 3.

### Finding 3 — severity-aware within-cluster display

Both formatters select the cluster's lead templates and example by severity, then
frequency:

- `top_templates` (markdown `:747`, compact `:805`):
  `sorted(cluster.templates, key=lambda t: (_severity_rank(t.severity), t.count), reverse=True)[:N]`
  (N=5 markdown, 3 compact).
- Example (markdown `:753`, compact `:811`): take it from the highest-severity template,
  `max(cluster.templates, key=lambda t: (_severity_rank(t.severity), t.count))`, instead
  of `cluster.templates[0]`.
- The per-template severity badge already prints (`:749`, `:807`); with `severity`=highest
  it now reflects the most severe content.

### Cluster reorder (single site) + heading

At the top of `format_as_markdown`, **before** the `if compact:` branch (covers the
compact path via the `:702` caller):

```python
clusters = sorted(
    clusters, key=lambda c: (_cluster_severity_rank(c), c.total_count), reverse=True
)
```

`reverse=True` on the tuple sorts both fields descending. Python's sort is stable and the
incoming order is already `total_count` desc, so ties are deterministic. Rename the
heading `## Log Patterns by Frequency` → `## Log Patterns by Severity` (`server.py:735`).

### Accurate severity distribution (3 sites)

Replace `dist[t.severity] += t.count` with a per-key sum from `severity_counts`, at all
three computations — markdown (`:720–724`), compact (`:789–793`), and `analyze_log_lines`
stats (`:919–923`):

```python
for t in cluster.templates:
    for sev, n in t.severity_counts.items():
        dist[sev] += n
```

With aliases normalized, the canonical-only display loop (`:728`) now shows every level.

### severity_filter (blast-radius fix)

`analyze_log_lines` (`server.py:887–889`) filters by `t.severity`. With `severity`=highest
that would wrongly drop a template whose filtered level is present but not highest. Match
against `severity_counts` keys, with the filter normalized:

```python
severity_set = {_normalize_severity(s) for s in severity_filter}
templates = [t for t in templates if severity_set & set(t.severity_counts)]
```

This also fixes a latent miss: `--severity ERROR` previously failed to match JSON
`{"level":"err"}` lines.

### ClusterOutput serialization

The structured result selects `sorted(..., key=count)[:10]` (`server.py:931`). Make it
severity-aware too — `key=lambda x: (_severity_rank(x.severity), x.count)` — so the
programmatic `AnalysisResult.clusters` does not drop the ERROR template either.
`TemplateOutput` is left unchanged (no `severity_counts` field) — surfacing the
distribution in the MCP contract is F2 territory (YAGNI here).

### None / unknown severity

`severity=None` (no level detected) and unknown levels rank 0 (below DEBUG). A cluster
only sinks to rank 0 if *all* its templates are unlabeled; the Severity Distribution
section still reports their volumes — nothing hidden.

## Scope

**In scope:** the constants, helpers, `extract_severity` normalization, `LogTemplate`
`severity_counts` + highest-severity semantics, `extract_templates` rewrite, cluster
reorder + heading, severity-aware within-cluster display + example, three accurate
distribution sites, severity-aware `ClusterOutput` `[:10]`, `severity_filter` via counts,
and tests. Doc touch-ups (README/ROADMAP wording asserting frequency ordering).

**Out of scope (follow-ups):** exposing `severity_counts` in the MCP `TemplateOutput`
contract (F2); per-cluster retrieval handles (F2); adaptive K (F4). Canonical
`analysis.clusters` order and `cluster_id` are untouched — no `AnalysisResult` ordering
change; the markdown display index ("Cluster N") may differ from stored index, which F2
will address by exposing the real `cluster_id`.

## Blast radius

- `LogTemplate` gains `severity_counts` (internal dataclass, default keeps construction
  working; not persisted — no migration).
- **`LogTemplate.severity` meaning changes** (modal → highest present). Consumers updated:
  formatters (display), `severity_filter` (now via counts), `ClusterOutput.severity`
  (now highest — an improvement), stats `severity_distribution` (now via counts).
- **Output ordering + heading change** (user-facing) and **`severity_filter` behavior
  change** (now matches if *any* line in the template has the level; normalizes aliases).
- No MCP/CLI signatures change; `TemplateOutput` contract unchanged.

## Test plan (TDD — failing test first)

**Unit (`tests/test_server.py`):**
1. `_normalize_severity` — `fatal`/`warn`/`err`/`trace` → CRITICAL/WARNING/ERROR/DEBUG;
   unknown passes through; None/"" → None.
2. `_severity_rank` / `_cluster_severity_rank` — max over templates; all-None → 0.
3. `extract_templates` over all lines — collapsed-JSON case
   (`100 info + 1 error`) → one template with `severity_counts={"INFO":100,"ERROR":1}`,
   `severity=="ERROR"`, and an ERROR example present.
4. Formatter mixed-cluster — 6 INFO templates + 1 rare ERROR template → ERROR template
   and an ERROR example appear in both verbose and compact output (beyond the top-5/3
   count limit).
5. Within-tier ordering — two ERROR clusters, larger `total_count` first.

**End-to-end via `analyze_log_lines` (Finding 4):**
6. JSON same-shape `100 info + 1 error`, small `token_budget` → markdown surfaces ERROR
   (badge/example), ERROR content not truncated. (verbose + compact)
7. JSON aliases — a `fatal` cluster ranks first; not sunk below DEBUG.
8. Docker/plaintext INFO-dominated + one ERROR cluster, small budget → ERROR survives.
9. Existing tests stay green (`test_extract_templates`, `test_get_logs_with_sample`,
   `test_get_logs_severity_filter`).

Gates: `uv run pytest -q` · `uv run ruff check src/ tests/` ·
`uv run ruff format --check src/ tests/` · pre-commit passes.

## Implementation touch-points

| Change | Location |
|--------|----------|
| `SEVERITY_RANK`, `SEVERITY_ALIASES` | `server.py` constants (~`:51`) |
| `_normalize_severity`, `_severity_rank`, `_cluster_severity_rank` | `server.py` near formatters |
| Normalize JSON + regex severity | `extract_severity` (`:483`, JSON branch `:492`) |
| `severity_counts` field; `severity`=highest | `LogTemplate` (`:212`) |
| Build `severity_counts` over all lines; high-severity example | `extract_templates` (`:558–579`) |
| Reorder clusters; heading rename | `format_as_markdown` top (`<:701`), heading (`:735`) |
| Severity-aware top templates + example | markdown (`:747`, `:753`), compact (`:805`, `:811`) |
| Accurate distribution (3 sites) | `:720–724`, `:789–793`, `:919–923` |
| Severity-aware `[:10]`; filter via counts | `ClusterOutput` (`:931`), filter (`:887–889`) |
| Tests | `tests/test_server.py` |
