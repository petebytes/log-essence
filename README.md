# log-essence

Extract the essence of your logs for LLM analysis.

Analyzes log files using template extraction (Drain3) and semantic clustering (FastEmbed) to produce token-efficient summaries for LLM consumption. Includes automatic secret/PII redaction for safe external analysis.

![Demo Video](demos/output/full-demo.mp4)

<p align="center">
  <em>Raw logs → Token-efficient summary with automatic secret redaction</em>
</p>

## Features

- **Auto-detection**: JSON, syslog, Apache, nginx, Docker, Kubernetes log formats
- **Template extraction**: Drain3 algorithm identifies log patterns and groups similar messages
- **Semantic clustering**: Groups related patterns using FastEmbed embeddings
- **Token budget**: Respects LLM context limits with intelligent summarization
- **Secret redaction**: Correlation-preserving redaction of emails, IPs, API keys, credit cards
- **Error chain analysis**: Traces root causes through related log entries
- **Time filtering**: Filter logs by duration (1h, 30m, 2d) or datetime
- **Multi-source**: Files, directories, glob patterns, Docker containers, journald
- **Web UI**: Paste-and-copy interface with real-time processing metrics

## Installation

```bash
# Using uv (recommended)
uvx log-essence

# Using pip
pip install log-essence
```

## CLI Usage

```bash
# Analyze a log file
log-essence /var/log/app.log

# Analyze with glob pattern
log-essence "/var/log/*.log"

# Filter by severity
log-essence /var/log/app.log --severity ERROR WARNING

# Filter by time
log-essence /var/log/app.log --since 1h

# Strict redaction mode
log-essence /var/log/app.log --redact strict

# Disable redaction (for internal logs only)
log-essence /var/log/app.log --no-redact

# JSON output for programmatic use
log-essence /var/log/app.log -o json

# Watch mode for live log monitoring
log-essence /var/log/app.log --watch --interval 5

# Run as MCP server
log-essence --serve
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--token-budget N` | Maximum tokens in output (default: 8000) |
| `--clusters N` | Number of semantic clusters (default: 10) |
| `--severity LEVEL...` | Filter by severity (ERROR, WARNING, INFO, DEBUG) |
| `--since TIME` | Only logs since TIME (1h, 30m, 2d, 2025-01-01) |
| `--redact MODE` | Redaction: strict, moderate (default), minimal, disabled |
| `--no-redact` | Disable redaction |
| `-o, --output FORMAT` | Output format: markdown (default) or json |
| `-w, --watch` | Watch log file for changes (live updates) |
| `--interval SECONDS` | Update interval for watch mode (default: 3.0) |
| `--config FILE` | Path to config file |
| `--profile NAME` | Use named configuration profile |
| `--serve` | Run as MCP server |
| `--version` | Show version number |

## Web UI

A browser-based interface for quick log analysis without command-line setup.

```bash
# Using uvx (recommended - no installation needed)
uvx --with log-essence[ui] log-essence ui

# Or specify a custom port
uvx --with log-essence[ui] log-essence ui --port 8080

# Using pip (if you prefer to install)
pip install log-essence[ui]
log-essence ui
```

### Features

- **Browser-based analysis**: Paste logs, get LLM-ready output with real-time metrics
- **Configurable settings**: Token budget, cluster count, redaction mode, severity filter
- **Processing metrics**: Real-time stats displayed after analysis
  - **Time**: Processing duration
  - **Redactions**: Number of secrets/PII items redacted
  - **Tokens**: Original → Output token count
  - **Savings**: Compression percentage achieved
- **Download**: Export analysis as markdown file

### UI Options

| Option | Description |
|--------|-------------|
| `--port N` | Port to run on (default: 8501) |
| `--no-browser` | Don't auto-open browser |

## MCP Server Usage

log-essence can run as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server, allowing Claude Desktop, Claude Code, Cursor, and other MCP-compatible clients to directly analyze logs from your system. This enables natural language interactions like:

> "Check the logs from the last hour for any database errors"
> "What's causing the slow response times in my API?"
> "Analyze the docker logs and find the root cause of the crash"

### Setup

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "log-essence": {
      "command": "uvx",
      "args": ["log-essence", "--serve"]
    }
  }
}
```

Restart Claude Desktop. You'll see log-essence listed in the MCP servers (🔌 icon).

### How It Works

1. **You ask Claude** about logs in natural language
2. **Claude calls log-essence** with the appropriate tool and parameters
3. **log-essence analyzes** the logs, redacts secrets, and returns a compressed summary
4. **Claude interprets** the results and provides actionable insights

The logs never leave your machine unredacted - log-essence strips sensitive data before Claude sees it, and the analysis runs locally using Drain3 and FastEmbed.

### Available Tools

#### `get_logs`

Analyze and consolidate log files.

```python
get_logs(
    path="/var/log/app.log",
    token_budget=8000,
    num_clusters=10,
    severity_filter=["ERROR", "WARNING"],
    since="1h",
    redact=True  # or "strict", "minimal", False
)
```

#### `get_container_logs`

Analyze Docker container logs.

```python
get_container_logs(
    container="my-app",
    since="1h",
    token_budget=8000
)
```

#### `get_docker_logs`

Analyze logs from Docker Compose services.

```python
get_docker_logs(
    path="/path/to/project",
    services=["api", "worker"],
    since="30m"
)
```

#### `get_error_chain`

Trace error root causes through related log entries.

```python
get_error_chain(
    path="/var/log/app.log",
    error_pattern="database",
    time_window=60
)
```

#### `search_logs`

Semantic search through log entries.

```python
search_logs(
    path="/var/log/app.log",
    query="connection timeout",
    top_k=10
)
```

#### `get_journald_logs`

Analyze systemd journal logs.

```python
get_journald_logs(
    unit="nginx.service",
    since="1h",
    priority="err"
)
```

#### `list_containers`

List running Docker containers.

```python
list_containers()
```

#### `list_docker_services`

List Docker Compose services in a project.

```python
list_docker_services(path="/path/to/project")
```

## Secret Redaction

Logs are automatically redacted before analysis to prevent leaking sensitive data to external LLMs.

### Redaction Modes

| Mode | Description |
|------|-------------|
| `moderate` | Default. Emails, IPs, credit cards, SSNs, phones, API keys |
| `strict` | All moderate patterns + high-entropy strings in key=value |
| `minimal` | Only obvious secrets (bearer tokens, API keys) |
| `disabled` | No redaction (use only for internal logs) |

### Output Format

Redacted values use the format `[TYPE:length?:hash4]`:

```
# Input
user@acme.com logged in from 192.168.1.50
Error processing payment for user@acme.com card 4111111111111111

# Output (same entity → same hash for correlation)
[EMAIL:a7f2] logged in from [IPV4:3bc1]
Error processing payment for [EMAIL:a7f2] card [CC:16:d4e8]
```

### Detected Patterns

**PII:**
- Email addresses
- IPv4 and IPv6 addresses
- Credit card numbers (Luhn-validated)
- Social Security Numbers (xxx-xx-xxxx)
- Phone numbers

**Secrets:**
- AWS access keys and secret keys
- GitHub tokens (ghp_, ghs_)
- Stripe API keys (sk_live_, sk_test_)
- JWT tokens
- Bearer tokens
- Private key headers
- Connection strings (postgres://, mongodb://, redis://)

## Example Output

```markdown
# Log Analysis Summary

**Format detected:** docker
**Total lines:** 15,432
**Unique patterns:** 47
**Semantic clusters:** 10

---

## Log Patterns by Frequency

### Cluster 1: Database Operations

**Occurrences:** 5,234 | **Patterns:** 8

- `Query executed in <*>ms` (2,341x)
- `Connection pool size: <*>` (1,892x)
- `Transaction committed` (1,001x)

**Example:**
```
2025-01-01T10:00:00Z INFO Query executed in 45ms
```

### Cluster 2: HTTP Requests

**Occurrences:** 4,123 | **Patterns:** 5

- `[IPV4:3bc1] - GET /api/<*> <*>` (3,456x)
- `Response time: <*>ms` (667x)
```

## Development

```bash
# Clone and install
git clone https://github.com/petebytes/log-essence
cd log-essence
uv sync --all-groups

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## License

MIT
