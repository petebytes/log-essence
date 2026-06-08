# C5 — Compression Eval/Benchmark Harness — Design

> Status: approved (brainstorm). Roadmap item **C5** (`ROADMAP.md`). Guards features
> **F1** (severity-weighted budget), **F2** (per-cluster retrieval), and pre-positions
> **F4** (adaptive cluster count) / **F5** (compression-quality floor).
> Date: 2026-06-07.

## 1. Context & motivation

`pyproject.toml:4` advertises log-essence as compressing logs **"50%+"**, and the README
surfaces a "Savings: compression percentage" metric. Nothing measures or enforces that
claim, and nothing guards the fidelity of the summary against regression.

A verify-first scoping pass (running the current pipeline on the two in-repo fixtures)
found the claim is **size-dependent**:

| Fixture | Lines | original → output tokens | `savings_percent` |
|---|---|---|---|
| `tests/sample.log` | 31 | 749 → 983 | **−31.2%** |
| `demos/assets/sample.log` | 182 | 7585 → 1689 | **+77.7%** |

On a small log the markdown scaffolding (header, severity section, per-cluster examples)
exceeds the input, so "compression" is negative. The claim only holds once the log is
large and repetitive enough for template collapse to dominate the fixed overhead. The
fixture the roadmap names as the primary eval target (`tests/sample.log`) therefore
*fails* a naive `savings ≥ 50%` assertion — a real constraint this design must respect.

The pipeline is **deterministic**: `kmeans_cluster` seeds `np.random.default_rng(42)`
(`server.py:718`) and FastEmbed `bge-small-en-v1.5` embeddings are deterministic. So
per-machine metrics are reproducible and a regression harness is viable.

## 2. Goals / non-goals

**Goals**
- Measure compression + fidelity on a curated fixture corpus, reproducibly.
- Enforce the headline claim (≥50% on a sufficiently large log) and the F1 invariant
  (highest-severity content survives truncation) as a hard CI gate.
- Detect silent drift in fidelity metrics against a committed baseline.
- Run inside the existing pytest CI with no new workflow.
- Provide a standalone report (table / JSON) for eyeballing trends and updating the
  baseline.

**Non-goals (YAGNI)**
- No real-world public datasets (LogHub HDFS/BGL): large files, network fetch, licensing —
  heavier than this tool's focused scope.
- No new `eval.yml` GitHub Actions workflow. A scheduled weekly benchmark is a deferrable
  follow-up that pairs with C6/C7.
- Not building F4/F5. The harness leaves obvious hooks (cluster-count metric, ratio data)
  for them to plug into.
- Not modifying F1/F2 production code. The harness *guards* those features; it does not
  refactor them.

## 3. Definition of done / acceptance

1. `uv run pytest tests/` includes an eval gate that:
   - passes on current `main` (locks today's numbers as the green baseline), and
   - fails if the large fixture's savings drops below 50%, if the highest-severity
     cluster stops surviving a tight budget, or if any baselined metric drifts beyond
     tolerance.
2. `python -m benchmarks.run` prints a per-fixture × per-scenario metrics table.
3. `python -m benchmarks.run --update-baseline` regenerates `benchmarks/baseline.json`.
4. `ruff check` / `ruff format --check` clean; pre-commit passes.

## 4. Architecture — one engine, two thin consumers

```
benchmarks/
  __init__.py
  fixtures.py     # repo-relative paths to existing logs + deterministic in-memory
                  # generators for the large + JSON-lines fixtures
  metrics.py      # ENGINE (pure): run pipeline on (fixture, params) -> BenchmarkResult
  baseline.json   # committed golden metrics (generated via --update-baseline, not by hand)
  run.py          # CLI: table report; --check (compare to baseline); --update-baseline; --json
tests/
  test_metrics.py # unit tests for the engine against hand-built analyses (TDD target)
  test_eval.py    # GATE: imports benchmarks.metrics; absolute floors + baseline-in-tolerance
```

`metrics.py` is the single source of truth. `tests/test_eval.py` (the gate) and
`benchmarks/run.py` (the report) are both thin wrappers over it — the chosen
"metrics engine + pytest gate" shape.

**Import path.** `benchmarks/` lives at the repo root, not under `src/`, because it must
not ship in the installed wheel. One-line change in `pyproject.toml`
`[tool.pytest.ini_options]`: `pythonpath = ["src", "."]` so `tests/` can
`import benchmarks.metrics`. `python -m benchmarks.run` already resolves because `-m`
puts the CWD on `sys.path`.

**Repo-relative paths.** `fixtures.py` derives `REPO_ROOT = Path(__file__).resolve().parent.parent`
and builds fixture paths from it. No absolute `/Users/...` paths are committed.

## 5. Metric definitions

All metrics are computed from a single pipeline run per (fixture, budget, compact) tuple.
The engine reuses the shipped pipeline so it measures exactly what the product reports.

### 5.1 Compression
- `savings_percent = (1 − output_tokens / original_tokens) × 100`, the shipped
  `AnalysisStats` definition. `original_tokens` counts the raw input (pre-redaction,
  `server.py:941`); `output_tokens` counts the rendered, redacted, truncated summary
  (`server.py:1007`).
- **Cross-check:** a unit test asserts the engine's `savings_percent` equals
  `analyze_log_lines(...).stats.savings_percent` for identical inputs, so the eval can
  never silently diverge from the product's own number.

### 5.2 Template retention
- `total_templates` = total distinct Drain templates extracted across all clusters
  (the full set, before the top-10/top-5 rendering caps). Equivalent to the markdown
  header's "Unique patterns" = `sum(len(c.templates) for c in clusters)` over the full
  ordered semantic clusters.
- `rendered_templates` = distinct templates actually present in the budget-truncated
  rendered summary (i.e. within surviving clusters, after the per-cluster top-N cap).
- `template_retention = rendered_templates / total_templates` (1.0 when nothing is
  truncated and no cluster exceeds the per-cluster cap).

### 5.3 Severity preservation (the F1 guard)
- `max_severity` = highest severity rank present across all clusters' cluster-severities.
- `severity_preserved` (bool) = **at least one maximal-severity cluster appears among the
  rendered survivors.** Computed independently of the assumed ordering — by matching
  rendered clusters back to the analysis, not by trusting that `_order_clusters` put the
  high-severity cluster first. If F1's ordering regresses, a low-frequency ERROR cluster
  drops out of the survivors and this flips to `False`.

### 5.4 Survivor determination (robustness note)
Clusters are emitted in order until the first one that would exceed the budget, then the
loop breaks — so survivors are a prefix of the rendered set. The engine determines the
surviving clusters by matching rendered cluster headers/content back to the cluster
objects (robust to the omitted-count sentinel wording). A unit test pins survivor
detection against a known-truncation case so a format change is caught, not silently
mismeasured.

### 5.5 Auxiliary (recorded, not floored)
`original_tokens`, `output_tokens`, `clusters_total`, `clusters_rendered`,
`processing_time_ms`. Useful for the report and for F4/F5 later.

## 6. Scenarios — two budgets per fixture

Each fixture is run at two token budgets:
- **Generous** (default `8000`): no truncation → measures *pure compression* (the claim).
- **Tight** (~`600`): forces truncation → exercises *severity survival* (the F1 invariant
  only bites under truncation).

The default markdown format is the primary measured artifact (it is the headline /
default output). Compact mode may be added as an extra scenario but is not required for
the gate.

## 7. Fixtures (curated, deterministic, offline)

| Fixture | Source | Role | Claim floor? |
|---|---|---|---|
| `tests/sample.log` | existing, 31 ln | small-input case (≈ −31%) | **no** — records that tiny logs don't compress; that is correct behavior |
| `demos/assets/sample.log` | existing, 182 ln | real-ish mid case (≈ 78%) | baseline-tracked |
| **large INFO+rare-ERROR** | NEW, generated in-memory | the 50%+ demonstrator **and** the F1 scenario | **yes** — `savings ≥ 50` (generous) + severity survives (tight) |
| **JSON-lines** | NEW, generated in-memory | format diversity; pre-positions F7 | baseline-tracked |

**Generation.** `fixtures.py` exposes deterministic generators (no RNG, or a fixed seed)
rather than committing a large blob — keeps the repo lean and the input reviewable:
- `make_high_volume_log()` → a few thousand INFO lines across a handful of well-separated
  templates (heartbeats, request logs with varying numeric ids), some WARN, and a small
  number of distinctly-worded ERROR/CRITICAL lines forming **one rare high-severity
  template**. Sized and shaped to (a) clear 50% with comfortable margin (~80%) and
  (b) make the ERROR cluster low-frequency, so frequency-only ordering would drop it under
  the tight budget but severity ordering keeps it. Templates are well-separated so
  k-means assignment is stable across BLAS/Python versions.
- `make_json_lines_log()` → a small JSON-lines sample exercising the JSON format path.

## 8. Gate design

### 8.1 Absolute floors (hard fail, jitter-robust)
- Large fixture, generous budget: `savings_percent ≥ 50` (the literal claim).
- Large fixture, tight budget: `severity_preserved is True` (the F1 guarantee).
These are robust to cluster-assignment jitter: a well-separated ~80% log does not drop
below 50% on float noise, and severity survival is boolean.

### 8.2 Baseline drift (hard fail, generous tolerance + escape hatch)
- `benchmarks/baseline.json` records every (fixture, budget) metric, plus a provenance
  note (the git ref it was generated from). Tolerances live as constants in
  `test_eval.py` (version-controlled with the gate logic), not in the JSON.
- The gate fails if, vs baseline:
  - `savings_percent` drops by more than the tolerance (default ~5 percentage points),
  - `template_retention` regresses beyond tolerance, or
  - `severity_preserved` flips `True → False`.
- Intentional metric changes are accepted by running
  `python -m benchmarks.run --update-baseline` and committing the new `baseline.json` in
  the same PR.

### 8.3 Determinism / cross-version jitter
Per-machine the pipeline is deterministic. The only flake risk is k-means *cluster
assignment* differing across numpy/BLAS builds on the 3.11–3.13 CI matrix. Mitigations,
in order: (1) fixtures shaped for well-separated clustering; (2) generous baseline
tolerances; (3) if jitter still appears in CI, pin the eval gate to a single Python
version (3.12) via a marker — the floors and the F1 boolean remain robust regardless.

## 9. Data model

```python
@dataclass(frozen=True)
class BenchmarkResult:
    fixture: str            # stable label, e.g. "high_volume_info"
    budget: int
    compact: bool
    original_tokens: int
    output_tokens: int
    savings_percent: float
    total_templates: int
    rendered_templates: int
    template_retention: float
    clusters_total: int
    clusters_rendered: int
    max_severity: str | None
    severity_preserved: bool
    processing_time_ms: float
```

`baseline.json` shape:
```json
{
  "generated_from": "<git ref>",
  "results": {
    "high_volume_info|budget=8000": {
      "savings_percent": 80.1, "template_retention": 1.0,
      "severity_preserved": true, "clusters_total": 10, "clusters_rendered": 10
    }
  }
}
```

## 10. CLI (`benchmarks/run.py`)

- `python -m benchmarks.run` → human-readable table of all fixtures × budgets.
- `python -m benchmarks.run --json` → machine-readable results.
- `python -m benchmarks.run --check` → compare to `baseline.json`; exit non-zero on
  regression (the gate logic, runnable outside pytest).
- `python -m benchmarks.run --update-baseline` → rewrite `baseline.json` from the current
  run, stamping `generated_from`.

## 11. CI integration

No workflow change. `tests/test_eval.py` is collected by the existing `ci.yml`
(`uv run pytest tests/ -v` on 3.11/3.12/3.13). `benchmarks/`/`tests/` edits are not
md/docs, so the workflow's `paths-ignore` does not skip them. FastEmbed's model download
is already incurred by existing clustering tests, so the eval adds no new heavy CI cost;
generated fixtures are in-memory and small.

## 12. TDD plan sketch (detailed steps deferred to writing-plans)

1. Engine unit tests against hand-built `AnalysisResult`/cluster inputs (red) →
   implement `metrics.py` compute functions (green). Include the
   savings-vs-`AnalysisStats` cross-check and the survivor-detection pin.
2. Fixture generators + their determinism/shape tests (the ERROR template is rare; the
   log clears 50%).
3. Absolute-floor gate tests (red until fixtures+engine exist) → green on current `main`.
4. Generate the initial `baseline.json` from `main`; add the drift gate.
5. `run.py` CLI + its smoke test.
6. `pyproject.toml` `pythonpath` tweak.

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Cross-version k-means jitter flakes the gate | Well-separated fixtures + generous tolerance; fall back to single-Python pin (§8.3) |
| Engine drifts from product's real numbers | Savings cross-checked against `AnalysisStats.savings_percent` in a unit test (§5.1) |
| Survivor detection breaks on a format change | Pinned by a known-truncation unit test (§5.4) |
| Baseline rots / blocks legitimate changes | `--update-baseline` escape hatch, committed with the change that moved metrics |
| FastEmbed model download slows CI | Already incurred by existing tests; no new dep |

## 14. Flagged finding (out of this gate's scope)

The "compress 50%+" claim is size-dependent (negative on tiny logs). Recommend a one-line
qualifier in `README.md` / `pyproject.toml` description (e.g. "50%+ on real-world
repetitive logs") as an optional tiny follow-up. This design *surfaces and measures* the
nuance; whether to reword the marketing copy is a separate doc decision for the user.
