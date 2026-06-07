# F1 — Severity-weighted budget allocation (design)

- **Status:** approved (design v3), pending implementation
- **Date:** 2026-06-06
- **Roadmap ID:** F1 (`P0`) — see `ROADMAP.md`
- **Sizing:** Churn **M** · Sites many · Files 2 (`server.py`, `tests/`) · Horizon medium ·
  Verify test (unit + end-to-end) · slightly **messy** (Drain/severity edge cases)

> **v3 supersedes v2.** v2 corrected v1's formatter-only blind spot (severity is degraded
> *before* the formatter). v3 corrects v2's *ordering-location* blind spot: ordering only
> inside `format_as_markdown` left `clusters_data` (CLI `--format json`, UI "Save JSON") in
> frequency order, so outputs disagreed. v3 orders clusters **once** in `analyze_log_lines`
> so every output agrees, and normalizes the **second** severity-filter consumer
> (`search_logs`). Both were caught by an adversarial review of the plan (verified below).

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

**Round 1 — reproduced against real functions (`tmp/f1_repro.py`, gitignored):**

| # | Sev | Claim | Evidence |
|---|-----|-------|----------|
| 1 | CRITICAL | Same-shape JSON `100 INFO + 1 ERROR` collapses to one template | `extract_templates` → **1 template, count=101, severity=INFO**; e2e markdown mentions "ERROR" → **False** |
| 2 | HIGH | JSON severity returns raw aliases below the rank table | `extract_severity` JSON → `FATAL`/`WARN`/`ERR`/`TRACE` (rank 0); regex path maps the same to CRITICAL/WARNING/ERROR/DEBUG |
| 3 | HIGH | Mixed cluster floats by max severity but hides the ERROR | 6 INFO templates (count 100) + 1 ERROR template (count 1) → ERROR template **not** in rendered body; example is INFO |
| 4 | MEDIUM | Manual-cluster unit test passes while e2e fails | A unit test building `SemanticCluster`s directly bypasses `extract_templates` where #1/#2 live |

**Round 2 — review of the plan, verified against the code:**

| # | Sev | Claim | Evidence |
|---|-----|-------|----------|
| 5 | HIGH | Ordering only in `format_as_markdown` leaves `clusters_data` in frequency order | `format_as_markdown` has one caller (`server.py:908`); `clusters_data` built separately and surfaced by CLI (`cli.py:452`) + UI (`ui/app.py:169`) — markdown shows ERROR first, JSON shows INFO first |
| 6 | HIGH | `search_logs` severity filter still compares raw `{s.upper()}` | `server.py:1790` `{s.upper()}` vs normalized `extract_severity` → `severity_filter=["err"]` silently returns nothing after alias normalization |
| 7 | HIGH | "Commit per task" leaves regressed intermediate commits; repo pushes at session end | `AGENTS.md:17` requires `git push` to finish; Task 1 regresses alias filters and Task 2 breaks distribution until a later task |

Clean checks confirmed both rounds: analytics SQLite is stats-only (no migration);
`severity_counts` uses `default_factory=dict` (no constructor breakage); `tee_store`
hash inputs untouched; `_format_compact` is single-caller.

## Goal / acceptance criteria

A high-severity cluster **survives truncation** at a small `token_budget`, the rendered
output makes its high-severity content **visible** (badge + example), and **all outputs
agree** (markdown, CLI JSON, UI). Holds at any volume ratio and across formats:

1. Plaintext/Docker: INFO-dominated log with one ERROR cluster → ERROR survives.
2. **JSON same-shape:** `100 {"level":"info"} + 1 {"level":"error"}` (identical message)
   → the collapsed cluster ranks high and the output shows ERROR (badge/example).
3. **JSON aliases:** a `{"level":"fatal"}` cluster ranks at the top, never below DEBUG.
4. **Cross-output consistency:** `clusters_data[0]` (CLI/UI JSON) is the same
   high-severity cluster shown first in the markdown.
5. **Filter parity:** alias filters (`err`/`warn`/`fatal`/`trace`) work on JSON for both
   `analyze_log_lines` (and its MCP wrappers) and `search_logs`.
6. Both `compact=False` and `compact=True`.

## Decision — tiered ordering + upstream severity integrity + one ordering source

**Ordering:** sort clusters by `(highest severity rank in the cluster, total_count)`,
both descending — severity primary, frequency tiebreak. The only option that *guarantees*
the criterion at any ratio. (Rejected: weighted blend `Σ weight×count` fails at
`1M INFO : 10 ERROR`; dampened `Σ weight×log1p(count)` holds only probabilistically.)

**Severity integrity:** preserve and normalize severity through extraction so ordering
has a correct signal, and surface it in the rendered body.

**One ordering source:** apply the ordering **once** in `analyze_log_lines`, before both
`format_as_markdown` and `clusters_data`, so markdown / CLI JSON / UI never diverge.

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

### Helpers (`server.py`)

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

def _order_clusters(clusters: list[SemanticCluster]) -> list[SemanticCluster]:
    return sorted(clusters, key=lambda c: (_cluster_severity_rank(c), c.total_count), reverse=True)

def _templates_by_priority(templates: list[LogTemplate]) -> list[LogTemplate]:
    return sorted(templates, key=lambda t: (_severity_rank(t.severity), t.count), reverse=True)
```

### Finding 2/6 — normalize at extraction and at every filter input

`extract_severity` JSON branch: `return _normalize_severity(str(data[field]))`. The regex
branch already emits canonical labels. **Both** severity-filter consumers normalize their
*input* so alias filters keep working: `analyze_log_lines` (`server.py:888`) and
`search_logs` (`server.py:1790`) — `{_normalize_severity(s) for s in severity_filter}`
(drop `None`). The four MCP wrappers delegate to `analyze_log_lines`, so they inherit it.

### Finding 1/4 — per-template severity over all lines

Add `severity_counts: dict[str, int] = field(default_factory=dict)` to `LogTemplate`
(`server.py:212`). `LogTemplate.severity` keeps type `str | None` but its **meaning
changes**: it is the **highest** severity present (derived from `severity_counts`), not
the modal-of-first-10. `extract_templates` (`:558–579`) builds `severity_counts` over
**all** member lines (each normalized), sets `severity` to the highest-rank key, and leads
`examples` with a highest-severity line so it stays visible.

### Finding 3/7 — distribution + filter-matching ship with the semantics change

Because `severity` becomes highest-present, the three count-only distribution sites
(markdown `:720–724`, compact `:789–793`, stats `:919–923`) and the `analyze_log_lines`
filter (`:887–889`) become wrong **at that same commit** unless fixed together. So they
are fixed in the **same task** as the `severity_counts`/highest-present change:

- Distribution: sum `severity_counts` per key — `for t in cluster.templates: for sev, n
  in t.severity_counts.items(): dist[sev] += n`. With aliases normalized, the canonical
  display loop (`:728`) shows every level.
- `analyze_log_lines` filter matches if **any** line in a template has the level:
  `templates = [t for t in templates if severity_set & set(t.severity_counts)]` (also
  fixes a latent miss where `--severity ERROR` skipped JSON `err` lines).

### Finding 5 — order clusters once (one source of truth)

In `analyze_log_lines`, immediately after `cluster_templates_semantically(...)`:
`clusters = _order_clusters(clusters)`. This ordered list feeds both
`format_as_markdown(...)` and the `clusters_data` comprehension (whose `enumerate(...,1)`
now yields severity-ordered `ClusterOutput.id`). `format_as_markdown` also calls
`_order_clusters` at its top (idempotent on already-ordered input) so direct callers and
its unit tests still get ordering. The heading `## Log Patterns by Frequency` →
`## Log Patterns by Severity` (`:735`).

### Finding 3 (display) — severity-aware within-cluster selection

Both formatters and the `ClusterOutput` build use `_templates_by_priority`:
- markdown top templates (`:747`) `[:5]`, example from `top_templates[0]` (`:752–755`).
- compact top templates (`:805`) `[:3]`, example from `top_templates[0]` (`:811–813`).
- `ClusterOutput` templates (`:931`) `[:10]`.

### None / unknown severity

`severity=None` and unknown levels rank 0 (below DEBUG). A cluster only sinks to rank 0
if *all* its templates are unlabeled; the Severity Distribution still reports volumes.

## Scope

**In scope:** the constants + five helpers; `extract_severity` normalization; filter-input
normalization in `analyze_log_lines` **and** `search_logs`; `LogTemplate.severity_counts`
+ highest-severity semantics; `extract_templates` rewrite; three accurate distribution
sites; `analyze_log_lines` filter via `severity_counts`; `_order_clusters` applied in
`analyze_log_lines` (feeding markdown + `clusters_data`) and in `format_as_markdown`;
heading; severity-aware within-cluster display + example + `ClusterOutput[:10]`; tests.
Doc touch-ups (README/ROADMAP wording asserting frequency ordering).

**Out of scope (follow-ups):** exposing `severity_counts` in the MCP `TemplateOutput`
contract (F2); per-cluster retrieval handles (F2); adaptive K (F4). Canonical
`cluster_templates_semantically` ordering is untouched; ordering is applied downstream in
`analyze_log_lines`, so `ClusterOutput.id` is now severity-ordered consistently.

## Blast radius

- `LogTemplate` gains `severity_counts` (internal dataclass, default keeps construction
  working; not persisted — no migration).
- **`LogTemplate.severity` meaning changes** (modal → highest present). Consumers updated:
  formatters, both filters, `ClusterOutput.severity` (now highest — improvement), stats
  distribution.
- **Output ordering + heading change** across markdown, CLI JSON, and UI;
  `ClusterOutput.id` is now severity-ordered (was frequency).
- **Both severity filters change** (match if any line has the level; normalize aliases).
- No MCP/CLI signatures change; `TemplateOutput` contract unchanged.

## Test plan (TDD — failing test first)

**Unit (`tests/test_server.py`):** `_normalize_severity` aliases; `_severity_rank` /
`_cluster_severity_rank`; `extract_templates` collapsed-JSON → `severity_counts` +
`severity==ERROR` + ERROR example; `_templates_by_priority` ordering; formatter
mixed-cluster surfaces ERROR template + example (verbose + compact); within-tier by
frequency; distribution accuracy (`result.severity_distribution`); filter matches
non-highest level; filter input alias normalized (`analyze` + `search_logs`);
`clusters_data[0]` is the high-severity cluster; `ClusterOutput[:10]` includes a rare
high-severity template.

**End-to-end via `analyze_log_lines` / `search_logs`:** JSON same-shape ERROR visible +
survives small budget (verbose + compact); JSON `fatal` alias ranks first; plaintext
INFO-dominated ERROR survives; `search_logs(severity_filter=["err"|"warn"|"fatal"])` on
JSON returns matches. Existing tests stay green.

Gates: `uv run pytest -q` · `uv run ruff check src/ tests/` ·
`uv run ruff format --check src/ tests/` · pre-commit passes.

## Implementation touch-points

| Change | Location |
|--------|----------|
| `SEVERITY_RANK`, `SEVERITY_ALIASES` | `server.py` constants (~`:51`) |
| `_normalize_severity`, `_severity_rank`, `_cluster_severity_rank`, `_order_clusters`, `_templates_by_priority` | `server.py` |
| Normalize JSON severity | `extract_severity` (`:492`) |
| Normalize filter input (2 sites) | `analyze_log_lines` (`:888`), `search_logs` (`:1790`) |
| `severity_counts` field; `severity`=highest | `LogTemplate` (`:212`) |
| Build `severity_counts` over all lines; high-severity example | `extract_templates` (`:558–579`) |
| Accurate distribution (3 sites) | `:720–724`, `:789–793`, `:919–923` |
| `analyze_log_lines` filter via counts | `:887–889` |
| `_order_clusters` (once) | `analyze_log_lines` after `:906`; `format_as_markdown` top (`<:701`) |
| Heading rename | `:735` |
| Severity-aware top templates + example | markdown (`:747`, `:752–755`), compact (`:805`, `:811–813`) |
| Severity-aware `ClusterOutput[:10]` | `:931` |
| Tests | `tests/test_server.py` |
