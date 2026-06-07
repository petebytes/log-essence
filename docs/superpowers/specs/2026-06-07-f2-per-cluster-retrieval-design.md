# F2 — Per-cluster retrieval (design)

> Status: approved design (revised after adversarial review), pre-implementation.
> Scope: per-cluster retrieval for **all four** analysis tools — `get_logs`,
> `get_docker_logs`, `get_container_logs`, `get_journald_logs` — see Scope.
> Depends on PR #31 (redaction-bypass fix) — see Verification.
> Date: 2026-06-07.

## Problem

`get_logs` returns a lossy semantic summary (clusters of templates + a few
examples). An agent investigating an incident often needs the **raw lines
behind one specific cluster** — e.g. "show me every line in the ERROR cluster" —
not a global offset into the whole file.

Today `get_raw_logs(analysis_id, start_line, max_lines)` only does **global**
pagination over the flat cached line list. There is no way to say "give me
cluster 3's lines." F2 turns the lossy summary into **lossless-on-demand at
cluster granularity**: expand exactly the cluster under investigation, keep the
rest retrievable.

## Verification — code state (confirmed empirically)

The roadmap/handoff notes were stale; these are the verified facts (post PR #31):

- `get_raw_logs` exists at `server.py:~1973` (ROADMAP claimed `:1800`; STATUS
  claimed it didn't exist). It is global-offset only.
- `extract_templates` (`server.py:~613`) already computes each Drain template's
  member line indices (`member_idxs`) — then **discards** them. `LogTemplate`
  and `SemanticCluster` carry **no** line membership.
- The tee cache entry (`tee_store`, `server.py:~1946`) stores only the flat
  line list + source/count/timestamp — **no line→cluster map**.
- The cluster handle is **positional**: `format_as_markdown` and
  `analyze_log_lines` both label clusters via `enumerate(ordered_clusters, 1)`
  after `_order_clusters` (severity, then frequency). `ClusterOutput.id` is that
  same 1-based ordered index. There is one ordering source (F1), so markdown,
  JSON, and `ClusterOutput.id` already agree.
- **Dependency on PR #31:** `extract_templates` runs on the *redacted* lines,
  and PR #31 makes the cache store those same redacted lines
  (`AnalysisResult.analyzed_lines`). So member indices align 1:1 with the cached
  lines — per-cluster retrieval is only correct *because* the cache is redacted.

## Goal / acceptance criteria

1. `get_raw_logs(analysis_id, cluster_id=N)` returns **only cluster N's**
   redacted lines, in original file order, paginated via `start_line`/`max_lines`.
2. `cluster_id=None` (default) → **unchanged** global behavior.
3. The `cluster_id` an agent passes is the **same 1-based "Cluster N"** shown in
   the `get_logs` summary and in `ClusterOutput.id`; retrieval resolves it
   against the stored snapshot, so it is stable within an `analysis_id`.
4. The per-cluster line count **equals** the cluster's "Occurrences" in the
   summary. Each line maps to exactly one Drain template (`line_to_cluster` is
   1:1), so the per-cluster index sets partition the templated lines and each
   set's size equals that cluster's summed template counts.
5. Every analysis tool's output carries the `analysis_id` and names how to
   expand a cluster (discoverability).
6. An out-of-range `cluster_id` returns an actionable error naming the valid range.
7. **All four** tools that summarize via `analyze_log_lines` — `get_logs`,
   `get_docker_logs`, `get_container_logs`, `get_journald_logs` — produce a
   retrievable `analysis_id` and support `cluster_id` retrieval. (Today only
   `get_logs` emits an `analysis_id` at all.)

## Decision

- **Extend `get_raw_logs`** with an additive optional `cluster_id` param (chosen
  over a new `get_cluster_logs` tool — keeps the MCP surface small, per the C4
  audit). **Appended last** — `get_raw_logs(analysis_id, start_line=0,
  max_lines=500, cluster_id=None)` — not inserted mid-signature, so the existing
  positional order `(analysis_id, start_line, max_lines)` is preserved. (No
  positional callers exist today — verified — but this avoids a future footgun
  where `get_raw_logs(aid, 3)` is misread as a cluster selector.)
- **Handle = the 1-based ordered cluster index** already visible to the agent.
- **Store membership as indices, not ranges** (a cluster's lines are scattered).
  Carry the indices the code already computes; key the map by the same ordered
  index used everywhere else. (Approach A below; Approach B — a flat per-line
  label array — rejected: O(N) scan per call, couples to line order.)
- Reuse the **internal-field pattern** from PR #31 (`exclude=True`,
  `SkipValidation`) to pass the map from `analyze_log_lines` to the cache.

## Design details

### `LogTemplate.member_indices` (`server.py` dataclass + `extract_templates`)
Add `member_indices: list[int] = field(default_factory=list)`. In
`extract_templates`, the `member_idxs` list is already built per Drain template
(`server.py:~613`) — store it on the `LogTemplate` instead of discarding it.
These are indices into the (redacted) `all_lines` passed to `extract_templates`.

### Cluster membership (`analyze_log_lines`)
A `SemanticCluster`'s line indices = the sorted union of its templates'
`member_indices`. Each input line belongs to exactly one Drain template, so
within a semantic cluster the per-template index lists are disjoint — union is
concatenation; sort restores file order.

After `clusters = _order_clusters(clusters)` and during the existing
`enumerate(clusters, 1)` that builds `clusters_data`, build in the **same loop**:

```
cluster_line_indices: dict[int, list[int]] = {
    i: sorted(idx for t in cluster.templates for idx in t.member_indices)
    for i, cluster in enumerate(clusters, 1)
}
```

so `cluster_line_indices[N]` is exactly the lines of the cluster shown as
"Cluster N" / `ClusterOutput.id == N`.

### Carry it to the cache (same pattern as `analyzed_lines`)
- `AnalysisResult.cluster_line_indices: SkipValidation[dict[int, list[int]]]`,
  `Field(default_factory=dict, exclude=True, repr=False)` (internal; never
  serialized into CLI/UI JSON).
- Set it on the success-path `AnalysisResult(...)`. (Empty/no-template paths
  leave the default `{}` — no clusters to retrieve.)
- `tee_store(lines, source, cluster_line_indices=None)` stores it in the entry
  (default `None`/`{}` keeps the signature backward-compatible).

### Shared retrieval trailer (`_store_and_annotate`) — used by all four tools
One helper centralizes teeing + the trailer so the ordering is correct in
exactly one place (and the four tools don't duplicate it):

```
def _store_and_annotate(result: AnalysisResult, source: str) -> str:
    analysis_id = tee_store(result.analyzed_lines, source, result.cluster_line_indices)
    return (
        result.markdown
        + "\n\n_Expand one cluster: get_raw_logs(analysis_id, cluster_id=N)._"
        + f"\n\n_analysis_id: {analysis_id}_"
    )
```

- The discoverability hint goes **before** the `_analysis_id: {id}_` trailer, so
  the id stays the **final token** of the output.
- `get_logs` (success path) returns `_store_and_annotate(result, path)` — this
  replaces its current inline `tee_store` + trailer.
- `get_docker_logs`, `get_container_logs`, `get_journald_logs` (success paths)
  return `_store_and_annotate(analysis, source)` where `source` is that tool's
  input identifier (container / unit / compose service). These three emit **no**
  `analysis_id` today, so this is additive. Error/empty paths are unchanged.
- Independently, the id parser is hardened: `_analysis_id()` (tests) extracts the
  12-hex token by **regex** (`_analysis_id: ([0-9a-f]{12})_`), not by assuming it
  is the trailing token — so trailing helper text can never cause a cache miss.

### `get_raw_logs(analysis_id, start_line=0, max_lines=500, cluster_id=None)`
- `cluster_id is None` → current global path (unchanged, incl. the `start_line`
  clamp from PR #31).
- `cluster_id` set:
  - `index_map = entry.get("cluster_line_indices") or {}`.
  - If `cluster_id` not in `index_map` → error:
    `Error: cluster_id {cluster_id} not found; analysis has {len(index_map)} clusters (1-{len}).`
  - `idxs = index_map[cluster_id]`; `lines = [entry["lines"][i] for i in idxs]`;
    apply the same `[clamped_start : start+max_lines]` window.
  - Header: `Source: {source} | Cluster {cluster_id} | Lines X-Y of {len(idxs)}`.

## Scope

**In:** per-cluster retrieval via `get_raw_logs(cluster_id=N)`; the membership
plumbing in `analyze_log_lines`; the shared `_store_and_annotate` trailer wired
into **all four** analysis tools (`get_logs`, `get_docker_logs`,
`get_container_logs`, `get_journald_logs`) — closing the gap that three of them
emit no `analysis_id` at all today; README update; tests.

**Out (explicitly):**
- C5 compression eval/benchmark harness — pairs with F2 to guard fidelity but is
  its own chore; not bundled here.
- F3/F4/F5.
- `search_logs` (`server.py:1846`) and `get_error_chain` (`:1729`) — they do not
  summarize via `analyze_log_lines` (verified — not in its caller set), so the
  cluster handle does not apply to them.

## Blast radius

- `get_raw_logs` contract: **additive** `cluster_id` param (appended last;
  backward-compatible).
- Output change for all four analysis tools: each success path gains a hint line
  + `analysis_id` trailer via the shared helper. `get_logs` already had a trailer;
  `get_docker_logs`/`get_container_logs`/`get_journald_logs` gain one (additive —
  they returned bare markdown before). Their existing tests are thin and mostly
  hit error/mocked paths (verified), so low regression risk; TDD catches any
  exact-output assertions.
- `tee_store`: additive optional param.
- `AnalysisResult`, `LogTemplate`: additive internal fields.
- Cache is in-memory, process-local, TTL 1h — **no persistence/migration**.
- Document the new `cluster_id` usage in README.

## Edges

- Lines that normalize to empty are skipped by `extract_templates`
  (`server.py:~603`) → in no template → not in any cluster's indices. They remain
  in global retrieval but are not reachable per-cluster. (Acceptance #4 is
  "matches Occurrences", which is itself computed from templated lines.)
- `cluster_id <= 0` → out-of-range error (1-based).
- Empty/no-template analyses → `cluster_line_indices == {}`; any `cluster_id`
  errors with "analysis has 0 clusters".

## Test plan (TDD — failing test first)

In `tests/test_retrieval.py` (the file PR #31 introduced):

1. **Per-cluster isolation:** a log with two clearly distinct patterns (e.g. a
   noisy INFO heartbeat + a rare ERROR). `get_raw_logs(aid, cluster_id=<error
   cluster>)` returns the ERROR lines and **not** the INFO lines; redacted.
2. **Handle agreement:** the `cluster_id` that returns the ERROR lines is the
   same N shown as "Cluster N" in the summary / `ClusterOutput.id`.
3. **Count match:** number of lines returned for cluster N == that cluster's
   summed template counts (its "Occurrences").
4. **Within-cluster pagination:** `start_line`/`max_lines` window a cluster.
5. **Out-of-range `cluster_id`:** actionable error naming the cluster count.
6. **`cluster_id=None` unchanged:** global retrieval identical to today.
7. **Redaction holds per-cluster:** a secret in a cluster's lines comes back
   redacted (guards against a future regression of the PR #31 fix).
8. **Coverage invariant:** union of every cluster's retrieved lines ⊆ the global
   `get_raw_logs` output.
9. **Id extraction is position-independent:** the `analysis_id` is still
   extractable (and a round-trip `get_raw_logs` succeeds) when the `get_logs`
   output carries the new cluster hint — guards against the trailing-text /
   cache-miss failure mode.
10. **All four tools are retrievable:** `get_docker_logs` / `get_container_logs` /
    `get_journald_logs` (with their underlying fetch mocked to return sample
    lines) each emit an `analysis_id`, and a `cluster_id` round-trip returns that
    source's redacted cluster lines — same shared path as `get_logs`.

## Implementation touch-points

- `src/log_essence/server.py`
  - `LogTemplate` dataclass — add `member_indices`.
  - `extract_templates` — store `member_idxs` on the template.
  - `analyze_log_lines` — build `cluster_line_indices`; set on result.
  - `tee_store` — accept + store the map.
  - `_store_and_annotate` — **new** shared helper (tee + hint + id trailer).
  - `get_logs` — return via `_store_and_annotate` (replaces inline tee+trailer).
  - `get_docker_logs`, `get_container_logs`, `get_journald_logs` — route their
    success-path return through `_store_and_annotate`.
  - `get_raw_logs` — add `cluster_id` (last param); per-cluster branch + error.
- `src/log_essence/ui/models.py`
  - `AnalysisResult.cluster_line_indices` internal field.
- `tests/test_retrieval.py` — the 10 tests above; harden `_analysis_id()` to regex.
- `README.md` — document `cluster_id` retrieval.
