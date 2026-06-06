# F1 — Severity-weighted budget allocation (design)

- **Status:** approved (design), pending implementation
- **Date:** 2026-06-06
- **Roadmap ID:** F1 (`P0`) — see `ROADMAP.md`
- **Sizing:** Churn S · Sites 1 (single reorder) · Files 1 (`server.py`) · Horizon short · Verify test

## Problem

When formatted output is truncated to fit `token_budget`, both formatters iterate
clusters in arrival order and `break` once the budget is hit:

- `format_as_markdown` — loop at `server.py:739`, budget check at `:761`.
- `_format_compact` — loop at `server.py:801`, budget check at `:817`.

Clusters arrive sorted by `total_count` descending (`cluster_templates_semantically`,
`server.py:641`). So a high-volume INFO heartbeat leads the list and a small but
critical ERROR/FATAL cluster sits at the tail — and is the first thing dropped when
the budget is tight. That is the exact failure mode for the tool's incident-debugging
use case: **the signal gets truncated, the noise survives.**

Severity is already extracted per template (`extract_severity`, `server.py:483`;
stored on `LogTemplate.severity`) but is only *summarized*, never used to *order*.

## Goal / acceptance criterion

Given a log dominated by INFO with one ERROR (or higher) cluster, the high-severity
cluster **survives truncation** at a small `token_budget`. Holds at any volume ratio.
A regression test encodes this.

## Decision — tiered ordering (severity primary, frequency tiebreak)

Order clusters by a tuple key: **(highest severity rank in the cluster, then
`total_count`), both descending.** Severity is categorical/primary; frequency only
ranks *within* a tier.

This is the only option that **guarantees** the acceptance criterion regardless of
volume ratio, and it is the least code.

### Rejected alternatives

- **Weighted blend** (`Σ weight(sev) × count`, headroom
  `transforms/log_compressor.py:299`): the ROADMAP's original wording, but it fails
  the acceptance test exactly when the tool is needed. At `1,000,000 INFO : 10 ERROR`,
  INFO scores `100,000` vs ERROR's `8` → ERROR still truncated. It only "works" when
  volumes are already close, i.e. when there was no problem.
- **Dampened blend** (`Σ weight(sev) × log1p(count)`): holds for moderate ratios but
  breaks at extreme ones (≈`1e9` INFO buries a 10-line ERROR) and depends entirely on
  hand-tuned weights. A probabilistic guard for a deterministic requirement.

Tiered's one notable behavior — a lone CRITICAL outranks a massive ERROR storm — is
correct for incident triage (FATAL is the top of the pyramid), and nothing is lost:
the storm is still in the list, just second, dropped only if the budget is genuinely
tiny.

## Design details

### Severity rank table

`extract_severity` (`server.py:497`) emits exactly six possibilities. Tiered ordering
only cares about *order*, so integer ranks beat fragile decimal weights:

| Severity | Rank |
|----------|------|
| CRITICAL | 5 |
| ERROR    | 4 |
| WARNING  | 3 |
| INFO     | 2 |
| DEBUG    | 1 |
| `None` / unrecognized | 0 |

Add a module-level constant near the other constants (`server.py:52` neighborhood):

```python
SEVERITY_RANK = {
    "CRITICAL": 5,
    "ERROR": 4,
    "WARNING": 3,
    "INFO": 2,
    "DEBUG": 1,
}  # None / unrecognized -> 0 via .get(level, 0)
```

Hardcoded by intent (YAGNI): no caller wants tunable weights, a constant is testable,
and it is trivially promoted to a parameter later. Keeps F1 at Churn S.

### Cluster tier = `max` over templates

A `SemanticCluster` holds many `LogTemplate`s of possibly mixed severity. The cluster's
tier is the **maximum** rank across its templates:

```python
def _cluster_severity_rank(cluster: SemanticCluster) -> int:
    return max((SEVERITY_RANK.get(t.severity, 0) for t in cluster.templates), default=0)
```

`max` (not weighted-dominant): a cluster that *contains* an error must float up even if
it is 99% INFO. Weighted-dominant would let the INFO majority bury the error inside its
own cluster — defeating the point.

### Single reorder site

`_format_compact` is reached **only** through `format_as_markdown` (`server.py:702`), so
one sort at the top of `format_as_markdown` — before the `if compact:` branch — covers
both outputs:

```python
clusters = sorted(
    clusters,
    key=lambda c: (_cluster_severity_rank(c), c.total_count),
    reverse=True,
)
```

Applying `reverse=True` to the tuple sorts both fields descending (higher rank first,
then higher count first).

### `severity=None` clusters

Rank 0 (below DEBUG). A cluster only lands here if *all* its templates are unlabeled
(tier = `max`). Detectable severity is more actionable; high-volume `None` clusters
still lead their own tier, and the **Severity Distribution** section still reports their
volume — nothing is hidden.

### Determinism

Python's sort is stable and the incoming order is already `total_count` descending
(`server.py:641`), so ties resolve deterministically. The k-means seed (`server.py:651`)
is unchanged. Same input → same order.

### Heading rename

`format_as_markdown` prints `## Log Patterns by Frequency` (`server.py:735`). Rename to
`## Log Patterns by Severity` — the order is no longer by frequency. `_format_compact`
has no such heading. The existing **Severity Distribution** section
(`server.py:726–732`) is retained unchanged (it reports per-level volumes, which is
complementary to ordering).

## Scope

**In scope**
- `SEVERITY_RANK` constant + `_cluster_severity_rank` helper.
- One reorder in `format_as_markdown` (covers compact path).
- Heading rename.
- Regression tests (below).
- Doc touch-up: README / ROADMAP wording that asserts frequency ordering, if any.

**Out of scope (noted as follow-ups)**
- **Within-cluster template display order** stays frequency-ordered (`server.py:747`,
  `:805`). The acceptance test is whole-cluster survival; reordering within a cluster so
  the reason it floated is visible is a separate, optional refinement.
- Canonical `analysis.clusters` order and `cluster_id` are **untouched** — no
  `AnalysisResult` contract change, no risk to future per-cluster retrieval (F2). The
  markdown's display index ("Cluster N") may now differ from stored index; F2 will
  expose the real `cluster_id` rather than relying on display number.

## Blast radius

- **Output ordering** changes (clusters reordered; truncation drops low-severity first).
- **One user-facing string** changes (the heading).
- No function signatures, no MCP/CLI parameters, no stored-order/`cluster_id` changes.
- No existing test asserts truncation order or the old heading (verified via grep), so
  test churn is additive.

## Test plan (TDD — failing test first)

In `tests/test_server.py`:

1. **`test_format_high_severity_survives_budget`** — construct one large INFO cluster
   (`total_count` ~1,000,000) and one small ERROR cluster (`total_count` ~10); call
   `format_as_markdown` with a `token_budget` that fits roughly one cluster body; assert
   the ERROR cluster's summary is present and the INFO cluster is the one marked omitted.
   Parametrize for `compact=False` and `compact=True`.
2. **`test_cluster_severity_rank`** (unit) — mixed-severity cluster ranks by its max;
   all-`None` cluster ranks 0; empty templates → 0.
3. **`test_format_within_tier_orders_by_frequency`** — two ERROR clusters; the larger
   `total_count` appears first (frequency still ranks within a tier).

Verify: `uv run pytest -q` (expect prior 208 still green + new tests), `uv run ruff
check src/ tests/`, `uv run ruff format --check src/ tests/`. Pre-commit must pass.

## Implementation touch-points (summary)

| Change | Location |
|--------|----------|
| `SEVERITY_RANK` constant | `server.py` (constants block, ~`:52`) |
| `_cluster_severity_rank` helper | `server.py` (near formatters, before `:685`) |
| Reorder `clusters` | `format_as_markdown` top, before `:701` |
| Heading rename | `server.py:735` |
| Tests | `tests/test_server.py` |
