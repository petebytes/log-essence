# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0b1] - 2025-12-27

Initial beta release of log-essence.

### Added

#### Core Analysis
- MCP server with `get_logs` tool using FastMCP framework
- Drain3 template extraction for log pattern recognition
- FastEmbed semantic clustering with k-means algorithm
- Auto-detection of log formats: JSON, syslog, Apache, nginx, Docker, Kubernetes
- Markdown output with configurable token budget enforcement
- Severity filtering (ERROR, WARNING, INFO, DEBUG)
- Time filtering with `--since` flag (durations like `1h`, `30m` or ISO dates)
- Glob pattern support for analyzing multiple files

#### Multi-Source Support
- `get_docker_logs` - analyze Docker Compose service logs with auto-discovery
- `get_container_logs` - analyze standalone Docker container logs
- `list_containers` - list running Docker containers
- `get_journald_logs` - analyze systemd journal logs
- `list_docker_services` - list Docker Compose services

#### Causality & Intelligence
- `get_error_chain` - trace error root causes with temporal proximity analysis
- `search_logs` - semantic search using AI embeddings
- Stack trace extraction for Python, Java, JavaScript, Go, and Rust
- Exception type extraction and "Caused by:" pattern detection

#### Security
- Correlation-preserving secret/PII redaction before LLM analysis
- Redacts: emails, IPs (v4/v6), credit cards (Luhn validated), SSNs, phones
- Redacts: AWS keys, GitHub tokens, Stripe keys, JWTs, Bearer tokens
- Redacts: private keys, connection strings, URL credentials
- Three redaction modes: `strict`, `moderate` (default), `minimal`
- Consistent hashing enables debugging without exposing values
- Output format: `[TYPE:length?:hash4]` (e.g., `[EMAIL:a7f2]`)

#### CLI
- Standalone CLI mode: `log-essence /path/to/logs`
- MCP server mode: `log-essence --serve`
- Config file support with YAML and named profiles
- JSON output format: `-o json`
- Watch mode for continuous monitoring: `-w`

#### Web UI
- Streamlit-based web interface: `log-essence ui`
- Paste-and-copy workflow for quick analysis
- Real-time processing stats (time, tokens, redaction count, savings %)
- Copy to clipboard and download as markdown
- Configurable settings sidebar
- Optional dependency: `pip install log-essence[ui]`

#### Demo Generation
- Automated demo creation with Playwright
- OpenAI-compatible TTS for narration
- YAML-based demo scripts with Pydantic validation
- Optional dependency: `pip install log-essence[demo]`

#### Infrastructure
- GitHub Actions CI workflow (Python 3.11, 3.12, 3.13)
- PyPI publish workflow with trusted publishing
- Comprehensive test suite (140 tests)
- MIT license

### Technical Details

- **Token compression**: Significant reduction in log volume
- **Supported Python**: 3.11+
- **Dependencies**: fastmcp, drain3, fastembed, numpy, tiktoken, pyyaml

[0.1.0b1]: https://github.com/petebytes/log-essence/releases/tag/v0.1.0b1
