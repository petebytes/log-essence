# F2 — Per-cluster Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an agent pull the raw (redacted) lines behind ONE cluster of a log analysis, across all four analysis tools, via `get_raw_logs(analysis_id, cluster_id=N)`.

**Architecture:** `extract_templates` already computes each Drain template's member line indices but discards them. Keep them on `LogTemplate`, aggregate them per semantic cluster in `analyze_log_lines` into a `cluster_line_indices: dict[int, list[int]]` keyed by the same 1-based "Cluster N" index used everywhere, carry that map into the tee cache, and add a `cluster_id` branch to `get_raw_logs`. A single shared helper `_store_and_annotate` tees the result and appends the discoverability hint + `analysis_id` trailer for all four tools.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `ruff`, FastMCP, pydantic v2, Drain3.

**Depends on:** PR #31 (redaction-bypass fix) — this branch (`feat/per-cluster-retrieval`) is stacked on `fix/redaction-bypass-get-raw-logs`. After #31 merges to `main`, rebase this branch onto `main` before opening the F2 PR. Correctness depends on #31: member indices align with the cached lines only because the cache stores the redacted lines that `extract_templates` ran on.

**Spec:** `docs/superpowers/specs/2026-06-07-f2-per-cluster-retrieval-design.md`

---

## File Structure

- `src/log_essence/server.py` — all logic: `LogTemplate.member_indices`, `extract_templates`, `analyze_log_lines`, `tee_store`, new `_store_and_annotate`, `get_logs`, `get_docker_logs`, `get_container_logs`, `get_journald_logs`, `get_raw_logs`.
- `src/log_essence/ui/models.py` — `AnalysisResult.cluster_line_indices` internal field.
- `tests/test_retrieval.py` — F2 tests (extends the file PR #31 created); harden `_analysis_id()` to regex.
- `README.md` — document `cluster_id` retrieval.

---

## Task 1: Carry member line indices on `LogTemplate`

**Files:**
- Modify: `src/log_essence/server.py` (`LogTemplate` dataclass ~`:240`; `extract_templates` ~`:635`)
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieval.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval.py::test_extract_templates_records_member_indices -v`
Expected: FAIL — `AttributeError: 'LogTemplate' object has no attribute 'member_indices'`.

- [ ] **Step 3: Add the field**

In `LogTemplate` (the `@dataclass` near `:240`), add after `severity_counts`:

```python
    member_indices: list[int] = field(default_factory=list)
```

- [ ] **Step 4: Populate it in `extract_templates`**

In `extract_templates`, the `LogTemplate(...)` constructor (near `:636`) currently ends with `severity_counts=dict(severity_counts),`. Add the new field to that constructor:

```python
        templates.append(
            LogTemplate(
                template=cluster.get_template(),
                cluster_id=cluster.cluster_id,
                count=cluster.size,
                examples=examples,
                severity=severity,
                severity_counts=dict(severity_counts),
                member_indices=member_idxs,
            )
        )
```

(`member_idxs` is the list already computed a few lines above as `member_idxs = [i for i, cid in line_to_cluster.items() if cid == cluster.cluster_id]`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval.py::test_extract_templates_records_member_indices -v`
Expected: PASS.

- [ ] **Step 6: Run full suite + lint**

Run: `uv run pytest -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: all pass (241 passed + 1 new), ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/log_essence/server.py tests/test_retrieval.py
git commit -m "feat: record per-template member line indices on LogTemplate"
```

---

## Task 2: Build `cluster_line_indices` in `analyze_log_lines`

**Files:**
- Modify: `src/log_essence/ui/models.py` (`AnalysisResult`)
- Modify: `src/log_essence/server.py` (`analyze_log_lines` success return ~`:1014-1044`)
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieval.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval.py::test_analyze_log_lines_cluster_line_indices_match_clusters -v`
Expected: FAIL — `AttributeError: 'AnalysisResult' object has no attribute 'cluster_line_indices'`.

- [ ] **Step 3: Add the internal field to `AnalysisResult`**

In `src/log_essence/ui/models.py`, in `AnalysisResult`, after the `analyzed_lines` field:

```python
    # Internal: 1-based cluster id -> sorted line indices into analyzed_lines.
    # Keyed by the same ordered index as clusters_data / "Cluster N". Excluded
    # from serialization; SkipValidation avoids per-entry cost on big logs.
    cluster_line_indices: SkipValidation[dict[int, list[int]]] = Field(
        default_factory=dict, exclude=True, repr=False
    )
```

(`SkipValidation` is already imported from PR #31.)

- [ ] **Step 4: Build the map and set it on the success return**

In `analyze_log_lines` (`src/log_essence/server.py`), the success path builds `clusters_data` via `enumerate(clusters, 1)` then `return AnalysisResult(...)`. Immediately before that `return`, add:

```python
    cluster_line_indices = {
        i: sorted(idx for t in cluster.templates for idx in t.member_indices)
        for i, cluster in enumerate(clusters, 1)
    }
```

and add the field to the `AnalysisResult(...)` constructor (the success one, alongside `analyzed_lines=all_lines,`):

```python
        analyzed_lines=all_lines,
        cluster_line_indices=cluster_line_indices,
```

(The no-template / empty early returns leave the default `{}` — correct, they have no clusters.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval.py::test_analyze_log_lines_cluster_line_indices_match_clusters -v`
Expected: PASS.

- [ ] **Step 6: Run full suite + lint**

Run: `uv run pytest -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/log_essence/server.py src/log_essence/ui/models.py tests/test_retrieval.py
git commit -m "feat: aggregate per-cluster line indices in analyze_log_lines"
```

---

## Task 3: Persist the map in the tee cache

**Files:**
- Modify: `src/log_essence/server.py` (`tee_store` ~`:1946`)
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieval.py  (append)
def test_tee_store_persists_cluster_line_indices() -> None:
    """tee_store records the cluster->indices map on the cache entry."""
    from log_essence.server import _tee_cache, tee_store

    aid = tee_store(["a", "b", "c"], "src", {1: [0, 2], 2: [1]})
    assert _tee_cache[aid]["cluster_line_indices"] == {1: [0, 2], 2: [1]}


def test_tee_store_defaults_cluster_line_indices_to_empty() -> None:
    """Two-arg calls (no map) still produce a valid entry."""
    from log_essence.server import _tee_cache, tee_store

    aid = tee_store(["a"], "src")
    assert _tee_cache[aid]["cluster_line_indices"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval.py::test_tee_store_persists_cluster_line_indices -v`
Expected: FAIL — `TypeError: tee_store() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Extend `tee_store`**

Change the signature and the stored dict:

```python
def tee_store(
    lines: list[str],
    source: str,
    cluster_line_indices: dict[int, list[int]] | None = None,
) -> str:
```

and inside, in the `_tee_cache[analysis_id] = {...}` block, add:

```python
            "cluster_line_indices": cluster_line_indices or {},
```

(Update the docstring `Args:` to mention `cluster_line_indices: 1-based cluster id -> line indices, for per-cluster retrieval.`)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval.py::test_tee_store_persists_cluster_line_indices tests/test_retrieval.py::test_tee_store_defaults_cluster_line_indices_to_empty -v`
Expected: PASS.

- [ ] **Step 5: Run full suite + lint**

Run: `uv run pytest -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: all pass (`get_logs` still calls `tee_store(result.analyzed_lines, path)` — the new param defaults, so its entries carry `{}` until Task 5).

- [ ] **Step 6: Commit**

```bash
git add src/log_essence/server.py tests/test_retrieval.py
git commit -m "feat: persist cluster line-index map in the tee cache"
```

---

## Task 4: `cluster_id` branch in `get_raw_logs`

**Files:**
- Modify: `src/log_essence/server.py` (`get_raw_logs` ~`:1976`)
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval.py  (append)
def test_get_raw_logs_cluster_id_returns_only_that_cluster() -> None:
    from log_essence.server import get_raw_logs, tee_store

    aid = tee_store(
        ["err one", "info one", "err two", "info two"],
        "src",
        {1: [0, 2], 2: [1, 3]},
    )
    out = get_raw_logs(analysis_id=aid, cluster_id=1)
    assert "err one" in out and "err two" in out
    assert "info one" not in out and "info two" not in out
    assert "Cluster 1" in out
    assert "of 2" in out  # cluster 1 has 2 lines


def test_get_raw_logs_cluster_id_out_of_range() -> None:
    from log_essence.server import get_raw_logs, tee_store

    aid = tee_store(["a"], "src", {1: [0]})
    out = get_raw_logs(analysis_id=aid, cluster_id=9)
    assert "not found" in out.lower()
    assert "1 cluster" in out  # names the available count
    # cluster_id is 1-based: 0 and negatives are out of range too
    assert "not found" in get_raw_logs(analysis_id=aid, cluster_id=0).lower()
    # an analysis with no clusters reports zero (no "(1-0)" range noise)
    empty = tee_store(["x"], "src", {})
    zero = get_raw_logs(analysis_id=empty, cluster_id=1)
    assert "0 clusters" in zero and "(1-" not in zero


def test_get_raw_logs_cluster_id_within_cluster_pagination() -> None:
    from log_essence.server import get_raw_logs, tee_store

    aid = tee_store(
        ["e0", "e1", "e2", "e3", "skip"], "src", {1: [0, 1, 2, 3]}
    )
    out = get_raw_logs(analysis_id=aid, cluster_id=1, start_line=1, max_lines=2)
    assert "e1" in out and "e2" in out
    assert "e0" not in out and "e3" not in out
    assert "Lines 2-3 of 4" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval.py -k cluster_id -v`
Expected: FAIL — `TypeError: get_raw_logs() got an unexpected keyword argument 'cluster_id'`.

- [ ] **Step 3: Add `cluster_id` (last param) and the per-cluster branch**

Replace the `get_raw_logs` signature and body from the `with _tee_lock:` block onward:

```python
@mcp.tool()
def get_raw_logs(
    analysis_id: str,
    start_line: int = 0,
    max_lines: int = 500,
    cluster_id: int | None = None,
) -> str:
    """Retrieve raw (redacted) log lines from a previous analysis.

    After reviewing a log analysis summary, use this tool to get the full
    log content for deeper investigation. Lines carry the same redaction
    applied during analysis (redacted by default; raw only if that analysis
    was run with redact=False).

    Args:
        analysis_id: The analysis ID returned from a previous get_logs call.
        start_line: Line offset to start from (default: 0).
        max_lines: Maximum lines to return (default: 500).
        cluster_id: If set, return only the lines of that cluster (the 1-based
            "Cluster N" shown in the summary). Default None returns all lines.

    Returns:
        Raw log lines from the cached analysis (whole analysis, or one cluster).
    """
    with _tee_lock:
        entry = _tee_cache.get(analysis_id)

    if entry is None:
        return (
            f"Error: Analysis '{analysis_id}' not found or expired. "
            "Re-run the analysis to generate a new cache."
        )

    if cluster_id is not None:
        index_map = entry.get("cluster_line_indices") or {}
        if cluster_id not in index_map:
            n = len(index_map)
            rng = f" (1-{n})" if n else ""
            return (
                f"Error: cluster_id {cluster_id} not found; "
                f"analysis has {n} cluster{'s' if n != 1 else ''}{rng}."
            )
        source_lines = [entry["lines"][i] for i in index_map[cluster_id]]
        total = len(source_lines)
        label = f"Cluster {cluster_id} | "
    else:
        source_lines = entry["lines"]
        total = entry["line_count"]
        label = ""

    # Clamp into [0, total] so a negative offset never tail-slices and an
    # over-range offset never yields a backwards "Lines 100-99" header.
    start = max(0, min(start_line, total))
    lines = source_lines[start : start + max_lines]
    if not lines:
        return (
            f"Source: {entry['source']} | "
            f"No lines in range (start_line={start_line}, total={total})"
        )
    end = start + len(lines)
    header = f"Source: {entry['source']} | {label}Lines {start + 1}-{end} of {total}\n\n"
    return header + "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval.py -k cluster_id -v`
Expected: PASS.

- [ ] **Step 5: Run full suite + lint**

Run: `uv run pytest -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: all pass (the global-path tests from PR #31 still pass — `cluster_id` defaults to None).

- [ ] **Step 6: Commit**

```bash
git add src/log_essence/server.py tests/test_retrieval.py
git commit -m "feat: per-cluster retrieval via get_raw_logs(cluster_id=N)"
```

---

## Task 5: Shared `_store_and_annotate`, wire `get_logs`, harden id parser

**Files:**
- Modify: `src/log_essence/server.py` (new `_store_and_annotate`; `get_logs` return ~`:1128-1131`)
- Modify: `tests/test_retrieval.py` (`_analysis_id` helper → regex)
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Harden the `_analysis_id` test helper to a regex**

Replace the helper at the top of `tests/test_retrieval.py`:

```python
import re
from pathlib import Path

from log_essence.server import get_logs, get_raw_logs


def _analysis_id(get_logs_output: str) -> str:
    """Extract the 12-hex analysis_id by pattern (position-independent)."""
    m = re.search(r"_analysis_id: ([0-9a-f]{12})_", get_logs_output)
    assert m is not None, f"no analysis_id in output tail: {get_logs_output[-200:]!r}"
    return m.group(1)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_retrieval.py  (append)
def test_get_logs_emits_cluster_hint_and_id_is_last(tmp_path: Path) -> None:
    """The output carries the expand-a-cluster hint, and the id stays extractable."""
    log_file = tmp_path / "app.log"
    log_file.write_text("\n".join(["ERROR boom"] * 3 + ["INFO ok"] * 4) + "\n")

    out = get_logs(path=str(log_file), redact=False)
    assert "cluster_id=N" in out  # discoverability hint present
    # id is still extractable and round-trips despite the trailing hint text
    raw = get_raw_logs(analysis_id=_analysis_id(out))
    assert "Lines 1-" in raw


def test_get_logs_end_to_end_cluster_retrieval(tmp_path: Path) -> None:
    """get_logs -> get_raw_logs(cluster_id) returns that cluster's redacted lines."""
    log_file = tmp_path / "app.log"
    lines = ["ERROR db pool exhausted user@acme.com"] * 2 + ["INFO heartbeat ok"] * 6
    log_file.write_text("\n".join(lines) + "\n")

    out = get_logs(path=str(log_file), redact=True)
    aid = _analysis_id(out)

    # Find the cluster id whose retrieved lines contain the ERROR pattern.
    err_cluster = None
    for cid in (1, 2):
        body = get_raw_logs(analysis_id=aid, cluster_id=cid)
        if "db pool exhausted" in body:
            err_cluster = cid
            assert "heartbeat" not in body
            assert "user@acme.com" not in body  # redaction holds per-cluster
            assert "[EMAIL:" in body
    assert err_cluster is not None


def test_cluster_retrieval_subset_of_global(tmp_path: Path) -> None:
    """Coverage invariant: every cluster's lines are a subset of global retrieval."""
    log_file = tmp_path / "app.log"
    log_file.write_text(
        "\n".join(["ERROR boom"] * 3 + ["INFO ok"] * 4 + ["WARN slow"] * 2) + "\n"
    )
    aid = _analysis_id(get_logs(path=str(log_file), redact=False))

    def _lines(body: str) -> set[str]:
        return set(body.split("\n\n", 1)[1].splitlines())

    global_lines = _lines(get_raw_logs(analysis_id=aid, max_lines=10_000))

    union: set[str] = set()
    cid = 1
    while True:
        body = get_raw_logs(analysis_id=aid, cluster_id=cid, max_lines=10_000)
        if "not found" in body.lower():
            break
        union |= _lines(body)
        cid += 1

    assert union  # non-empty
    assert union <= global_lines
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval.py::test_get_logs_emits_cluster_hint_and_id_is_last -v`
Expected: FAIL — `assert "cluster_id=N" in out` fails (no hint yet).

- [ ] **Step 4: Add the shared helper**

In `src/log_essence/server.py`, add immediately above `get_logs` (near `:1057`):

```python
def _store_and_annotate(result: AnalysisResult, source: str) -> str:
    """Cache the analyzed lines + cluster membership and append the retrieval trailer.

    Shared by every tool that returns an analyze_log_lines summary, so the
    discoverability hint and the analysis_id trailer are emitted in one place —
    with the id as the FINAL token so it stays parseable.
    """
    analysis_id = tee_store(result.analyzed_lines, source, result.cluster_line_indices)
    return (
        result.markdown
        + "\n\n_Expand one cluster: get_raw_logs(analysis_id, cluster_id=N)._"
        + f"\n\n_analysis_id: {analysis_id}_"
    )
```

- [ ] **Step 5: Route `get_logs` through it**

Replace the tail of `get_logs` (the `result = analyze_log_lines(...)` is unchanged; replace the tee + return):

```python
    result = analyze_log_lines(all_lines, token_budget, num_clusters, severity_filter, redact)
    return _store_and_annotate(result, path)
```

(Delete the old `# Cache the lines as analyzed ...` comment, the `analysis_id = tee_store(...)` line, and the old `return result.markdown + f"\n\n_analysis_id: {analysis_id}_"`.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: PASS (all retrieval tests, including the two new end-to-end ones).

- [ ] **Step 7: Run full suite + lint**

Run: `uv run pytest -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: all pass. If any pre-existing test asserts the exact END of `get_logs` output, update it to allow the trailing hint line (the `_analysis_id:` substring is unchanged).

- [ ] **Step 8: Commit**

```bash
git add src/log_essence/server.py tests/test_retrieval.py
git commit -m "feat: shared retrieval trailer + cluster hint for get_logs"
```

---

## Task 6: Wire `get_docker_logs`, `get_container_logs`, `get_journald_logs`

**Files:**
- Modify: `src/log_essence/server.py` (the three tools' success returns)
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval.py  (append)
from unittest.mock import patch  # add with the other imports at top of file


def _error_cluster_body(aid: str, marker: str = "boom") -> str:
    """Body of the cluster whose lines contain `marker` (exercises cluster_id)."""
    for cid in (1, 2):
        body = get_raw_logs(analysis_id=aid, cluster_id=cid)
        if marker in body:
            return body
    raise AssertionError(f"no cluster contained {marker!r}")


def test_get_container_logs_supports_cluster_retrieval() -> None:
    from log_essence.server import get_container_logs

    sample = "\n".join(["ERROR boom user@acme.com"] * 2 + ["INFO heartbeat ok"] * 5)
    with patch("log_essence.server.fetch_container_logs", return_value=sample):
        out = get_container_logs(container="web")
    assert "cluster_id=N" in out
    body = _error_cluster_body(_analysis_id(out))  # real cluster_id round-trip
    assert "boom" in body and "heartbeat" not in body  # per-cluster isolation
    assert "user@acme.com" not in body and "[EMAIL:" in body  # redacted per-cluster


def test_get_journald_logs_supports_cluster_retrieval() -> None:
    from log_essence.server import get_journald_logs

    sample = "\n".join(["ERROR boom"] * 2 + ["INFO heartbeat ok"] * 5)
    with patch("log_essence.server.fetch_journald_logs", return_value=sample):
        out = get_journald_logs(unit="nginx")
    assert "_analysis_id:" in out and "cluster_id=N" in out
    body = _error_cluster_body(_analysis_id(out))
    assert "boom" in body and "heartbeat" not in body


def test_get_docker_logs_supports_cluster_retrieval(tmp_path: Path) -> None:
    from log_essence.server import get_docker_logs

    sample = "\n".join(["ERROR boom"] * 2 + ["INFO heartbeat ok"] * 5)
    with (
        patch("log_essence.server.discover_compose_file", return_value=tmp_path / "docker-compose.yml"),
        patch("log_essence.server.get_compose_services", return_value=[{"name": "web"}]),
        patch("log_essence.server.fetch_docker_logs", return_value=sample),
    ):
        out = get_docker_logs(path=str(tmp_path))
    assert "_analysis_id:" in out and "cluster_id=N" in out
    body = _error_cluster_body(_analysis_id(out))
    assert "boom" in body and "heartbeat" not in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval.py -k "container_logs_supports or journald_logs_supports or docker_logs_supports" -v`
Expected: FAIL — `assert "cluster_id=N" in out` (the three tools still return bare `header + analysis.markdown`).

- [ ] **Step 3: Route each tool's success return through the helper**

`get_docker_logs` — replace the final `return header + analysis.markdown`:

```python
    return header + _store_and_annotate(analysis, f"docker-compose: {compose_file.parent.name}")
```

`get_container_logs` — replace the final `return header + analysis.markdown`:

```python
    return header + _store_and_annotate(analysis, f"container: {container}")
```

`get_journald_logs` — replace the final `return header + analysis.markdown`:

```python
    return header + _store_and_annotate(analysis, f"journald: {unit}" if unit else "journald")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval.py -k "container_logs_supports or journald_logs_supports or docker_logs_supports" -v`
Expected: PASS.

- [ ] **Step 5: Run full suite + lint**

Run: `uv run pytest -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: all pass (the existing docker/container/journald tests hit error/mocked paths, unaffected by the success-path change).

- [ ] **Step 6: Commit**

```bash
git add src/log_essence/server.py tests/test_retrieval.py
git commit -m "feat: cluster retrieval for docker/container/journald log tools"
```

---

## Task 7: Document `cluster_id` in the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a retrieval subsection**

Find the section documenting `get_raw_logs` / retrieval (or the MCP tools list). Add:

```markdown
### Per-cluster retrieval

Every analysis tool (`get_logs`, `get_docker_logs`, `get_container_logs`,
`get_journald_logs`) ends its summary with an `analysis_id`. Pass it back to
`get_raw_logs` to pull the full, **redacted** lines on demand:

- `get_raw_logs(analysis_id)` — all lines (paginate with `start_line`/`max_lines`).
- `get_raw_logs(analysis_id, cluster_id=N)` — only the lines behind "Cluster N"
  from the summary, so an agent can expand exactly the cluster under
  investigation without pulling the whole log back.

Retrieved lines carry the same redaction as the summary (redacted unless that
analysis was run with `redact=False`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document per-cluster get_raw_logs retrieval"
```

---

## After all tasks

- [ ] Full green: `uv run pytest -q` (expect ~251 passed / 1 skipped), `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`.
- [ ] Adversarial Codex review of the full diff (`git diff main...HEAD`) — the checkpoint discipline that caught real gaps on F1/the redaction fix.
- [ ] When PR #31 merges to `main`: `git rebase main` (drop the redaction-fix commits, keep F2), re-run the suite, then open the F2 PR.
- [ ] PR body: name the new `cluster_id` contract + the additive `analysis_id` trailer on the three previously-untrailed tools (blast radius).
