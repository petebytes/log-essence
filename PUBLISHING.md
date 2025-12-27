# Publishing Guide for log-essence

This document covers how to publish log-essence to PyPI and set up the GitHub repository for maximum visibility.

## Pre-publish Checklist

Before publishing, ensure:

- [ ] All tests pass: `uv run pytest`
- [ ] Linting passes: `uv run ruff check src/ tests/`
- [ ] Formatting correct: `uv run ruff format --check src/ tests/`
- [ ] `uv.lock` committed (for reproducibility)

## GitHub Repository Setup

### 1. Create Repository

```bash
# Create on GitHub first, then:
git remote add origin git@github.com:petebytes/log-essence.git
git push -u origin main
```

### 2. Repository Settings

**Description** (one-line for GitHub):
```
Intelligent log analysis for LLMs - extract patterns, redact secrets, compress 50%+. CLI, Web UI, and MCP server.
```

**Topics** (GitHub tags - add these in repository settings):
```
log-analysis
llm
mcp
claude
ai
devtools
cli
python
log-parser
template-extraction
secret-detection
docker-logs
semantic-search
```

### 3. Enable GitHub Actions

The repository includes CI/CD workflows:
- `.github/workflows/ci.yml` - Runs on every push/PR
- `.github/workflows/publish.yml` - Publishes to PyPI on release

## PyPI Setup

### 1. Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Enable 2FA (required for new projects)

### 2. Configure Trusted Publishing (Recommended)

This is the modern, secure way to publish from GitHub Actions without API tokens:

1. Go to PyPI → Your Account → Publishing
2. Add new pending publisher:
   - PyPI Project Name: `log-essence`
   - Owner: `petebytes`
   - Repository: `log-essence`
   - Workflow name: `publish.yml`
   - Environment: `pypi`

### 3. Create GitHub Environment

1. Go to GitHub repo → Settings → Environments
2. Create environment named `pypi`
3. (Optional) Add protection rules like required reviewers

## Publishing Process

### Automated Release

Publishing is fully automated. Version is derived from git tags via `hatch-vcs`:

```bash
git tag v0.2.0
git push origin main --tags
```

**That's it!** GitHub Actions will automatically:
- Run tests (fails fast if tests don't pass)
- Build the package (version from tag)
- Publish to PyPI via trusted publishing
- Create a GitHub Release with auto-generated notes

### Version Format

- Release: `v0.2.0` → PyPI version `0.2.0`
- Beta: `v0.2.0b1` → PyPI version `0.2.0b1`
- Dev builds show commit info: `0.2.0.dev3+g1234567`

### Option B: Manual Publish

```bash
# Build
uv build

# Upload (requires PyPI API token)
uv publish --token YOUR_PYPI_TOKEN

# Or using twine
pip install twine
twine upload dist/*
```

## Recommended Metadata

### PyPI Classifiers (already configured)

The `pyproject.toml` includes appropriate classifiers:
- Development Status :: 4 - Beta
- Environment :: Console
- Intended Audience :: Developers
- Topic :: System :: Logging
- Typing :: Typed

### Keywords for PyPI Search

Current keywords in `pyproject.toml`:
```toml
keywords = ["mcp", "llm", "logs", "analysis", "claude", "ai"]
```

**Expanded recommended keywords**:
```toml
keywords = [
    "mcp",
    "llm",
    "logs",
    "log-analysis",
    "log-parser",
    "claude",
    "ai",
    "devtools",
    "observability",
    "template-extraction",
    "drain3",
    "semantic-clustering",
    "secret-detection",
    "pii-redaction",
    "docker-logs",
]
```

### Description

Current: "MCP Log Consolidator for LLM Analysis"

**Recommended expanded description**:
```
Intelligent log analysis tool that makes logs LLM-friendly. Uses Drain3 template extraction and semantic clustering to significantly compress logs while preserving signal. Features automatic secret/PII redaction, multi-source support (files, Docker, journald), and runs as CLI, web UI, or MCP server for Claude.
```

## Version Numbering

Version is automatically derived from git tags via `hatch-vcs`. Follow [Semantic Versioning](https://semver.org/):

- Tag format: `v{MAJOR}.{MINOR}.{PATCH}` (e.g., `v0.2.0`, `v1.0.0`)
- Beta: `v0.2.0b1`
- Use `1.0.0` when stable with API guarantees

## Post-publish Verification

After publishing:

1. **Check PyPI page**: https://pypi.org/project/log-essence/
2. **Test installation**:
   ```bash
   pip install log-essence
   log-essence --help
   ```
3. **Test uvx**:
   ```bash
   uvx log-essence --help
   ```

## Marketing / Visibility

### README Badges

Add these badges at the top of README.md after publishing:

```markdown
[![PyPI version](https://badge.fury.io/py/log-essence.svg)](https://pypi.org/project/log-essence/)
[![Python versions](https://img.shields.io/pypi/pyversions/log-essence.svg)](https://pypi.org/project/log-essence/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/petebytes/log-essence/actions/workflows/ci.yml/badge.svg)](https://github.com/petebytes/log-essence/actions/workflows/ci.yml)
```

### Where to Share

1. **Reddit**: r/Python, r/devops, r/MachineLearning, r/ClaudeAI
2. **Hacker News**: Show HN post
3. **X/Twitter**: Tag @AnthropicAI if relevant
4. **Dev.to / Hashnode**: Write a tutorial post
5. **MCP Community**: Anthropic's MCP Discord/forum

### Key Selling Points to Emphasize

1. **Significant token reduction** - context savings for LLMs
2. **Security-first** - automatic secret/PII redaction
3. **Multi-source** - files, Docker, journald all supported
4. **Claude integration** - native MCP server for Claude Desktop
5. **Zero config** - works out of the box with sensible defaults
