"""Pydantic models for UI configuration and state."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from log_essence import __version__

RedactionModeType = Literal["disabled", "minimal", "moderate", "strict"]
SeverityLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
OutputFormat = Literal["markdown", "json"]


class AnalysisStats(BaseModel):
    """Statistics from log analysis."""

    processing_time_ms: float = Field(description="Processing time in milliseconds")
    redaction_count: int = Field(default=0, description="Number of redactions performed")
    original_tokens: int = Field(description="Token count of original input")
    output_tokens: int = Field(description="Token count of output")

    @property
    def savings_percent(self) -> float:
        """Calculate token savings percentage."""
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.output_tokens / self.original_tokens) * 100


class AnalysisResult(BaseModel):
    """Result of log analysis with stats."""

    markdown: str = Field(description="Markdown-formatted analysis output")
    stats: AnalysisStats = Field(description="Analysis statistics")
    # Additional fields for JSON output (optional for backwards compatibility)
    log_format: str = Field(default="unknown", description="Detected log format")
    lines_processed: int = Field(default=0, description="Total lines processed")
    severity_distribution: dict[str, int] = Field(
        default_factory=dict, description="Count by severity level"
    )
    clusters_data: list[ClusterOutput] | None = Field(
        default=None, description="Cluster data for JSON output"
    )


class UIConfig(BaseModel):
    """Configuration for log analysis from the UI."""

    token_budget: int = Field(
        default=8000,
        ge=100,
        le=100000,
        description="Maximum tokens in output",
    )
    num_clusters: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of semantic clusters",
    )
    redaction_mode: RedactionModeType = Field(
        default="moderate",
        description="Redaction mode for secrets/PII",
    )
    severity_filter: list[SeverityLevel] | None = Field(
        default=None,
        description="Filter by severity levels",
    )

    @field_validator("severity_filter", mode="before")
    @classmethod
    def empty_list_to_none(cls, v: list[str] | None) -> list[str] | None:
        """Convert empty list to None for no filtering."""
        if v is not None and len(v) == 0:
            return None
        return v


# JSON Output Schema Models


class TemplateOutput(BaseModel):
    """A log template in JSON output."""

    template: str = Field(description="Log message pattern with <*> placeholders")
    count: int = Field(description="Number of occurrences")
    severity: str | None = Field(default=None, description="Log severity level")
    examples: list[str] = Field(default_factory=list, description="Example log lines")


class ClusterOutput(BaseModel):
    """A semantic cluster in JSON output."""

    id: int = Field(description="Cluster identifier")
    summary: str = Field(description="Summary of the cluster's content")
    total_count: int = Field(description="Total occurrences across all templates")
    templates: list[TemplateOutput] = Field(description="Templates in this cluster")


class MetadataOutput(BaseModel):
    """Metadata about the analysis."""

    source: str = Field(description="Path or source of logs analyzed")
    lines_processed: int = Field(description="Total lines processed")
    log_format: str = Field(description="Detected log format")
    timestamp: datetime = Field(description="When analysis was performed")
    version: str = Field(default=__version__, description="log-essence version")


class JSONOutput(BaseModel):
    """Complete JSON output schema for log analysis."""

    metadata: MetadataOutput = Field(description="Analysis metadata")
    stats: AnalysisStats = Field(description="Processing statistics")
    severity_distribution: dict[str, int] = Field(
        default_factory=dict, description="Count of logs by severity level"
    )
    clusters: list[ClusterOutput] = Field(description="Semantic clusters of log patterns")
