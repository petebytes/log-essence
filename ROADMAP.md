# log-essence — Prioritized Roadmap

Improvement plan synthesized from (a) a code-level comparison against
[`chopratejas/headroom`](https://github.com/chopratejas/headroom) — a Rust+Python
context-compression platform — and (b) web research on log-parsing, prompt/context
compression, PII redaction, and MCP tool design best practices.

> **Framing.** headroom is a broad-scope *platform*; log-essence is a focused *log tool*.
> Where they overlap (log compression), log-essence is already the more sophisticated
> (real Drain template mining + embedding/k-means semantic clustering vs. headroom's
> heuristic line-scoring). This roadmap borrows headroom's *menu of methods* and the
> industry best practices that fit log-essence's scope — it does **not** chase platform
> breadth (proxy, cross-agent memory, trained models) that would dilute the tool.

## Sizing legend

Per the five-axis model (most predictive first): **Churn** (LoC) · **Sites** (distinct
edit locations) · **Files** · **Horizon** (sequential checkpoints) · **Verify** (how we
know it worked). `messy` = ambiguous spec / thin tests / unfamiliar area.

## Done (this pass)

| ID | Item | Type | Verify |
|----|------|------|--------|
| C11 | Implement stdin `-` (`… \| log-essence -`) — the form `discover` suggests | Chore (bug) | ✅ test (`test_run_analysis_reads_stdin`, `…_stdin_rejects_watch`) |
| F11 | `discover_log_sources` MCP tool — agent-facing source discovery incl. local files | Feature | ✅ test (`test_discover_log_sources_*`, `test_format_sources_for_agent_*`) |
| C12 | **Project-scoped discovery** — rewrote `discover.py` so discovery returns *this code project's* logs, not the machine's | Chore (bug/redesign) | ✅ test (`test_find_log_files_*`, `test_find_docker_containers_requires_compose`, `test_compose_project_name`) |
| F12 | `discover [path]` (CLI) / `discover_log_sources(path=…)` (MCP) — scope to any project without `cd` | Feature | ✅ test (`test_parser_discover_with_path`, `test_discover_log_sources_accepts_path`) |
| F13 | **Run-command + agent-file detection** — surface `package.json` dev/start/serve as pipe-ready commands, and standard agent files (`CLAUDE.md`/`AGENTS.md`/`GEMINI.md`/…) that document how to run | Feature | ✅ test (`test_find_run_commands_*`, `test_find_agent_files`, `test_detect_package_manager`) |

> **C11** fixed a latent bug: `discover` printed `docker logs … \| log-essence -` and
> `journalctl … \| log-essence -` commands, but `-`/stdin was never handled — every such
> suggestion failed with `Path does not exist: -`. Now `run_analysis` reads stdin when
> `path == "-"` (watch mode rejects stdin, since it can't be re-read).
> **F11** closed the MCP discovery gap: agents could `list_containers`/`list_docker_services`
> but had no way to discover **local log files** or get a unified "what's here" view.
> `discover_log_sources()` wraps `discover.discover_sources()` and annotates each source with
> the MCP tool to call (`get_logs` / `get_container_logs` / `get_docker_logs`).
> **C12** fixed discovery conflating "this project" with "this machine." On a real repo it
> returned ~24 `/var/log` system files, every machine-wide `docker ps` container (incl. other
> projects'), and binary LevelDB `.log` files from Playwright profiles. Now it is project-scoped:
> recursively scans the project tree **pruning** vendored/VCS/build dirs (`node_modules`, `.git`,
> `dist`, `.next`, `.turbo`, …), **skips binary** files (NUL-byte test → no LevelDB/SQLite),
> **skips compressed** logs (`.gz`/`.bz2`/… — `read_text` can't decode them), scopes Docker to the
> project's **Compose project label**, and drops machine-wide system logs + journald. On a real
> monorepo this went from 29 noisy sources → 1 correct one (the project's compose stack).
> **F12/F13** made discovery useful for code projects that log to **stdout** (Next.js, etc.):
> `discover [path]` scopes to any repo without `cd`, and discovery now surfaces the **run
> command** (`package.json` dev/start/serve → `pnpm run dev 2>&1 | log-essence -`) plus the
> standard **agent-instruction files** that document how to run/test (`CLAUDE.md`, `AGENTS.md`,
> `GEMINI.md`, `.cursorrules`, `.clinerules`, `.github/copilot-instructions.md`). On a Next.js
> monorepo: 1 → 6 *relevant* sources (run command + 4 agent files + compose).

### Discovery — remaining enhancements (backlog)

Grounded in how mature log shippers (Fluent Bit, Vector, Filebeat, Promtail, Grafana Alloy,
OTel Collector) and loggers (pino/winston) work:

- **`.gitignore`-aware pruning** — cleaner than the binary heuristic for cache/profile noise
  (e.g. Playwright profile dirs are gitignored); also avoids walking ignored build output.
- **Rotated-log grouping + optional `.gz`/`.bz2` read** — present `app.log` + `app.log.1` +
  `app.log.2.gz` as one logical source and decompress on read, instead of excluding compressed.
- **Logger-config detection** — sniff `pino`/`winston` file destinations in the project and
  surface those concrete log paths.
- **Command extraction from agent files** — parse fenced run commands out of `CLAUDE.md`/etc.
  (currently their presence is surfaced; the commands inside are not yet extracted).
- **Makefile/Procfile/justfile run targets** — additional command sources beyond `package.json`.
- **Multiline / rotation awareness in the analyzer** — join stack traces; handle split records
  on rotation (the failure modes the shipper benchmarks call out).

## Priority summary

| ID | Item | Pri | Type | Churn | Verify |
|----|------|-----|------|-------|--------|
| F1 | Severity-weighted budget allocation | P0 | Feature | S | test |
| F2 | Per-cluster retrieval handles (extend existing `get_raw_logs`) | P0 | Feature | S | test + manual (MCP) |
| F3 | Hybrid BM25 + embedding search | P1 | Feature | M | test |
| F4 | Adaptive cluster count (elbow/Kneedle) | P1 | Feature | S | test |
| F5 | Compression-quality floor + ratio guard | P1 | Feature | S | test |
| F6 | NER-assisted PII redaction (names/orgs/locations) | P2 | Feature | M | test (messy) |
| F7 | Within-log content routing (JSON vs free-text) | P2 | Feature | M | test |
| F8 | Anomaly/novelty surfacing (rare + first-seen) | P2 | Feature | M | test |
| F9 | Optional LLM template labeling | P3 | Feature | M | manual (messy) |
| F10 | Consolidated `get_log_context` tool | P3 | Feature | S | manual (MCP) |
| C1 | Delete stray `@AGENTS.md` duplicate | P0 | Chore | XS | manual |
| C2 | Add `SECURITY.md` (redaction-bypass reporting) | P0 | Chore | XS | manual |
| C3 | Self-log hygiene audit (no raw lines in logs/UI) | P0 | Chore | S | test + grep |
| C4 | MCP tool-surface audit (errors, defaults, descriptions) | P1 | Chore | S | manual (MCP) |
| C5 | Compression eval/benchmark harness | P1 | Chore | M | external (fixtures) |
| C6 | Release automation (release-please) | P1 | Chore | S | external (CI) |
| C7 | Coverage gate (`codecov.yml`) | P2 | Chore | XS | external (CI) |
| C8 | Secret-detection FP reduction (entropy + validation) | P2 | Chore | S | test |
| C9 | `llms.txt` machine-readable index | P2 | Chore | XS | manual |
| C10 | Consistent compact/verbose modes across all tools | P2 | Chore | S | test |

---

## Features

### F1 — Severity-weighted budget allocation `P0`
**What.** When formatting under `token_budget`, order/keep clusters by a
severity-weighted score (e.g. ERROR=1.0, WARN=0.5, INFO=0.1, DEBUG=0.05) blended
with frequency — not by raw `total_count` alone.
**Why.** Today `format_as_markdown`/`_format_compact` (`server.py:685`, `:772`) emit
clusters sorted by frequency and stop at the budget. A 1M-line INFO heartbeat
crowds out a single FATAL — the exact failure mode for the tool's incident-debugging
use case. Severity is already extracted per template; it's filtered but not weighted.
**Sources.** headroom `transforms/log_compressor.py:299` (ERROR=1.0…DEBUG=0.05 scoring);
LLMLingua / "make every token count" — allocate budget to high-information content.
**Sizing.** Churn S · Sites 2 (both formatters) · Files 1 · Horizon short · Verify test.
**Acceptance.** Given a log dominated by INFO with one ERROR cluster, the ERROR cluster
survives truncation at a small `token_budget`. Add a regression test.

### F2 — Per-cluster retrieval handles `P0`
> **Rescoped.** The reversible-retrieve capability already exists — `get_raw_logs`
> (`server.py:1800`) returns paginated redacted lines from a cached analysis keyed by
> `analysis_id` (`start_line`/`max_lines`). So this is *not* a build-from-scratch item.
> The remaining gap is granularity + discoverability, below.

**What.** (1) Make `get_logs` surface the `analysis_id` handle prominently in its summary
(today an agent may not know to pass it back). (2) Add **per-cluster** retrieval so an agent
can pull "the raw lines behind cluster 3" — not just a global line offset into the whole
analysis. Either extend `get_raw_logs` with a `cluster_id` filter or carry per-cluster line
ranges in the cache entry.
**Why.** Turns the lossy summary into precise *lossless-on-demand*: expand exactly the
cluster under investigation. Matches Anthropic's guidance — return relevant lines, keep the
rest retrievable — which `get_raw_logs` already half-implements at the analysis level.
**Sources.** Existing `get_raw_logs` (`server.py:1800`) + `_tee_cache`; headroom CCR
(`ccr/tool_injection.py`, `ccr/context_tracker.py` — originals keyed by hash); Anthropic
*Writing Effective Tools for Agents*.
**Sizing.** Churn S · Sites few · Files 1 (`server.py`) · Horizon short · Verify test +
manual MCP round-trip. **Blast radius:** changes the `get_raw_logs` contract / `get_logs`
output — document in README.
**Acceptance.** `get_logs` output names the `analysis_id`; `get_raw_logs(analysis_id,
cluster_id=3)` returns only that cluster's redacted source lines.

### F3 — Hybrid BM25 + embedding search `P1`
**What.** Fuse a lightweight BM25 lane with the existing embedding cosine in
`semantic_search_logs` (`server.py:1634`); adaptive weighting that favors BM25 when the
query looks like an exact identifier (UUID, request-id, error code).
**Why.** Search is embeddings-only today — pure vectors miss *exact* tokens, which is
precisely what you grep logs for ("find request `a7f2-…`"). headroom uses the *same*
`bge-small-en-v1.5` model, so adding a BM25 lane is cheap and high-value.
**Sources.** headroom `relevance/hybrid.py` (adaptive α; UUID-as-token, +0.3 long-token
boost), `relevance/bm25.py` (pure-Python, no deps).
**Sizing.** Churn M · Sites 1–2 · Files 1–2 · Horizon short · Verify test.
**Acceptance.** A query for a literal id ranks the line containing it #1, where
embedding-only currently buries it.

### F4 — Adaptive cluster count `P1`
**What.** Replace the hardcoded `num_clusters=10` default with an auto-selected K from an
information-saturation curve (elbow / Kneedle on cumulative unique-template or
unique-bigram gain), bounded by min/max. Keep manual override.
**Why.** Different log files have very different natural K; a magic 10 over- or
under-clusters most inputs. `cluster_templates_semantically` (`server.py:584`) +
`kmeans_cluster` (`:645`) already compute the inputs needed.
**Sources.** headroom `transforms/adaptive_sizer.py` (Kneedle on unique-bigram curve +
SimHash dedup + zlib sanity check); LLMLingua (dynamically choose compression ratio).
**Sizing.** Churn S · Sites 1–2 · Files 1 · Horizon short · Verify test.
**Acceptance.** On a homogeneous log, K is small; on a heterogeneous one, K grows —
both without user tuning. Deterministic (seed retained).

### F5 — Compression-quality floor + ratio guard `P1`
**What.** Add guardrails to the analysis stats: warn (and optionally fall back to a
larger budget / fewer drops) when output retains <X% of distinct templates or when
compression exceeds a configurable ceiling (e.g. >90%). Surface "N templates omitted"
prominently.
**Why.** Best-practice across LLMLingua/Selective-Context: cap compression (don't exceed
~80–90%) and verify quality at each step, because over-compression silently drops signal.
log-essence already computes original→output tokens; this adds the *quality* dimension.
**Sources.** PromptHub/LLMLingua ("set a compression threshold… iterative testing");
my own no-silent-caps principle (log what was dropped).
**Sizing.** Churn S · Sites 2 · Files 1 · Horizon short · Verify test.
**Acceptance.** Analysis on a high-cardinality log emits an explicit
"compressed N% — M of P templates shown" line; ratio-ceiling triggers a warning.

### F6 — NER-assisted PII redaction `P2`
**What.** Optional dependency (`[ner]` extra) adding Named-Entity Recognition to catch
PII regex can't: person names, organizations, locations. Feeds the existing
correlation-preserving redactor so `[PERSON:hash4]` stays stable like `[EMAIL:hash4]`.
**Why.** log-essence's redaction is a genuine differentiator (headroom does *no* content
PII redaction — only its own log hygiene). Regex covers structured PII (emails, IPs,
cards, SSNs) but misses freeform names/orgs/places — the dominant PII in app logs.
**Sources.** Elastic *PII NER + regex* guide (NER complements pattern matching; assess
then redact; field-level security on un-redacted data — pairs with F2's store).
**Sizing.** Churn M · Sites few · Files 1–2 (`redaction.py` + extra) · Horizon medium ·
Verify test · **messy** (NER models have FP/FN; needs eval). Keep off by default; opt-in.
**Acceptance.** With `[ner]` installed, a log line "user Sarah Chen from Acme" redacts the
name and org with stable hashes; without it, behavior is unchanged (no hard dep).

### F7 — Within-log content routing `P2`
**What.** Detect per-line/per-block content type and route: structured JSON log lines →
key-aware/columnar compaction (schema + values), free-text lines → Drain as today.
**Why.** Mixed streams (JSON access logs + plaintext app logs, k8s structured logging)
are common; running free-text Drain over JSON strings under-compresses and produces noisy
templates. `detect_log_format` (`server.py:440`) already distinguishes formats — extend to
per-block routing.
**Sources.** headroom `transforms/content_router.py`, `transforms/smart_crusher.py`
(JSON arrays → CSV+schema when ≥30% bytes saved, else drop-middle with retrieval marker —
pairs with F2).
**Sizing.** Churn M · Sites few · Files 1–2 · Horizon medium · Verify test.
**Acceptance.** A JSON-lines log compresses better and yields cleaner field-level
templates than the current text-Drain path; plaintext path unchanged.

### F8 — Anomaly / novelty surfacing `P2`
**What.** Explicitly flag *rare-but-significant* templates: first-seen patterns,
low-frequency + high-severity, and sudden frequency spikes (in `--watch`). Surface a
short "Anomalies" section above the frequency clusters.
**Why.** Frequency-ranking buries the needle. The observability literature converges on
LLMs being good at *contextual* anomaly/RCA framing once the rare signals are isolated.
This complements F1 (severity weighting) by adding a *novelty* axis.
**Sources.** LogLLM (arXiv:2411.08561), "AIOps for log anomaly detection in the era of
LLMs" survey (ScienceDirect S2667305325001346).
**Sizing.** Churn M · Sites few · Files 1 · Horizon medium · Verify test.
**Acceptance.** A log with one novel ERROR template among millions of INFO lines lists
that template in an Anomalies section regardless of token budget.

### F9 — Optional LLM template labeling `P3`
**What.** Optional pass that asks an LLM to assign human-readable cluster labels and merge
Drain templates it mis-split (hybrid Drain→LLM). Strictly opt-in (adds API cost/dep).
**Why.** Drain occasionally over-/under-splits; LogParser-LLM shows LLM-augmented parsing
hits ~90% grouping F1 with few calls. Labels make summaries more legible.
**Sources.** LogParser-LLM (arXiv:2408.13727); *System Log Parsing with LLMs: A Review*
(arXiv:2504.04877).
**Sizing.** Churn M · Files 1–2 · Horizon medium · Verify manual · **messy** (nondeterministic;
needs eval + cost guardrail). Defer until F1–F5 land.
**Acceptance.** With a key configured, clusters get concise labels; disabled by default,
core path stays deterministic and offline.

### F10 — Consolidated `get_log_context` tool `P3`
**What.** A single high-level MCP tool that compiles error chain + top clusters + a
targeted search for a given symptom, in one call.
**Why.** Anthropic's guide favors workflow-shaped tools (`get_customer_context`) over many
low-level ones, to save agent round-trips and context. Evaluate against the risk of
overlapping with existing tools.
**Sources.** Anthropic *Writing Effective Tools for Agents* (consolidate chained tools).
**Sizing.** Churn S · Files 1 · Horizon short · Verify manual (MCP). Validate with an eval
task before committing — may not beat the existing toolset.

---

## Chores

### C1 — Delete stray `@AGENTS.md` `P0`
Byte-identical duplicate of `AGENTS.md` in the working tree (both untracked) — a botched
paste. **Verify:** `diff @AGENTS.md AGENTS.md` then `rm '@AGENTS.md'`.

### C2 — Add `SECURITY.md` `P0`
For a tool whose headline feature is secret/PII redaction, there's no documented channel
to report a *redaction bypass*. Add `SECURITY.md` with a private reporting path and scope.
**Source.** MCP-in-production guidance; headroom ships `SECURITY.md`. **Sizing.** XS.

### C3 — Self-log hygiene audit `P0`
Ensure raw, **un-redacted** log lines never reach the tool's own `logging`/`print`,
exception messages, or Streamlit `session_state`. Redaction happens before analysis, but
read/parse/error paths handle raw content first.
**Why.** This is the one redaction discipline headroom *does* enforce (`_redact_*_log`
helpers keep secrets out of its debug logs/cache previews) and a core secret-handling rule.
**Sources.** headroom `cache/compression_store.py:65`, `proxy/request_logger.py`.
**Sizing.** Churn S · Verify grep + test. **Check:** grep for raw `line`/content reaching
`logging`, `print`, f-string error messages, and UI state; redact or hash before emit.

### C4 — MCP tool-surface audit `P1`
Apply Anthropic + production-MCP best practices across all tools: actionable error
messages with machine-readable codes; smart parameter defaults; descriptions that steer
agents (when to use which tool, token-cost hints); consistent namespacing; avoid
one-tool-per-operation sprawl.
**Sources.** Anthropic *Writing Effective Tools for Agents*; The New Stack *15 Best
Practices for Building MCP Servers in Production*; Block *MCP Servers Playbook*.
**Sizing.** Churn S · Sites many (each `@mcp.tool`) · Files 1 · Verify manual (MCP client).

### C5 — Compression eval/benchmark harness `P1`
A small `benchmarks/` that measures token savings + fidelity (templates retained,
severity preserved) on `tests/sample.log` and fixtures, with a regression threshold. Backs
the README/`pyproject` "compress 50%+" claim and guards F1/F4/F5 from regressing.
**Sources.** headroom `benchmarks/` + weekly `eval.yml`; Anthropic eval-driven tool dev.
**Sizing.** Churn M · Files 2–3 · Horizon medium · Verify external (fixtures + CI step).

### C6 — Release automation `P1`
Add `release-please` + conventional-commit enforcement so merge→tagged release + changelog
+ PyPI publish is one step. Already using `hatch-vcs` (tag-driven) and conventional
commits — this closes the loop.
**Source.** headroom `.release-please-config.json`, `.commitlintrc.json`. **Sizing.** S ·
Verify external (CI dry-run).

### C7 — Coverage gate `P2`
Add `codecov.yml` + a coverage step in CI. 2,072 lines of tests already exist (vs 4,308
src) — just not measured. Patch-coverage gate + badge. **Sizing.** XS · Verify external.

### C8 — Secret-detection false-positive reduction `P2`
Combine regex with entropy thresholds and format validation (extend the Luhn-style
validation already used for credit cards to other secret types) to cut false positives.
Measure FP/FN against a labeled fixture.
**Sources.** Cycode/Soteri secrets-detection guides; Reaves et al. (regex+entropy FP study).
**Sizing.** Churn S · Files 1 (`redaction.py`) · Verify test.

### C9 — `llms.txt` index `P2`
Ship a machine-readable doc index so agents can self-orient — on-brand for an LLM/MCP tool.
**Source.** headroom `llms.txt`. **Sizing.** XS.

### C10 — Consistent compact/verbose modes `P2`
`get_logs` has a `compact` flag; extend a detailed-vs-concise mode consistently to
`search_logs` and `get_error_chain` so agents can dial token cost per call.
**Source.** Anthropic ("support both detailed and concise response modes"). **Sizing.** S ·
Sites few · Verify test.

---

## Suggested execution order

1. **Quick wins first:** C1, C2, C9 (XS chores), then F1 (highest value/effort ratio).
2. **Core capability:** F2 (reversible retrieve) + C5 (eval harness to prove it doesn't
   regress fidelity) together.
3. **Search & sizing:** F3, F4, F5.
4. **Hygiene/infra:** C3, C4, C6, C7.
5. **Depth:** F6, F7, F8, C8, C10.
6. **Exploratory (gate on evals):** F9, F10.

## Sources

**Code comparison:** `chopratejas/headroom` @ `26f325f` (cloned to `./tmp/headroom`) —
`transforms/log_compressor.py`, `transforms/smart_crusher.py`, `transforms/content_router.py`,
`transforms/adaptive_sizer.py`, `relevance/{bm25,embedding,hybrid}.py`,
`ccr/{tool_injection,context_tracker}.py`, `transforms/kompress_compressor.py`.

**Web research:**
- Anthropic — *Writing Effective Tools for Agents* (modelcontextprotocol.info/docs/tutorials/writing-effective-tools)
- The New Stack — *15 Best Practices for Building MCP Servers in Production*
- Block Engineering — *Playbook for Designing MCP Servers*
- Microsoft Research / PromptHub — *LLMLingua* & *Selective-Context* prompt compression
- Elastic — *Using NLP/NER and Pattern Matching to Detect, Assess, and Redact PII in Logs*
- LogParser-LLM (arXiv:2408.13727); *System Log Parsing with LLMs: A Review* (arXiv:2504.04877)
- LogLLM (arXiv:2411.08561); *AIOps for log anomaly detection in the era of LLMs* (ScienceDirect)
- Cycode / Soteri — secrets-detection technique guides; Reaves et al. — regex+entropy FP study
- logpai/Drain3; XDrain / LogERT (evolving-tree Drain improvements)
