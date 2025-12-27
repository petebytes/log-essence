"""Pydantic models for demo script definition."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions that can be performed in a demo."""

    NAVIGATE = "navigate"
    TYPE = "type"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXECUTE = "execute"
    CLEAR = "clear"


class NavigateAction(BaseModel):
    """Navigate to a URL."""

    type: Literal["navigate"] = "navigate"
    url: str = Field(..., description="URL to navigate to")


class TypeAction(BaseModel):
    """Type text into an element."""

    type: Literal["type"] = "type"
    selector: str = Field(..., description="CSS selector for the target element")
    text: str = Field(..., description="Text to type")
    delay: int = Field(default=50, description="Delay between keystrokes in ms")


class WaitAction(BaseModel):
    """Wait for a specified duration."""

    type: Literal["wait"] = "wait"
    duration: int = Field(..., description="Duration to wait in milliseconds")


class ScreenshotAction(BaseModel):
    """Take a screenshot."""

    type: Literal["screenshot"] = "screenshot"
    name: str | None = Field(default=None, description="Optional name for the screenshot")


class ExecuteAction(BaseModel):
    """Execute the current command in the terminal."""

    type: Literal["execute"] = "execute"


class ClearAction(BaseModel):
    """Clear the terminal."""

    type: Literal["clear"] = "clear"


class ClickAction(BaseModel):
    """Click on an element."""

    type: Literal["click"] = "click"
    selector: str = Field(..., description="CSS selector or text to click")
    text: str | None = Field(default=None, description="Text content to match (uses getByText)")


class FillAction(BaseModel):
    """Fill a form input with text (uses fill, not type)."""

    type: Literal["fill"] = "fill"
    selector: str = Field(..., description="CSS selector for the input element")
    text: str = Field(..., description="Text to fill")


Action = (
    NavigateAction
    | TypeAction
    | WaitAction
    | ScreenshotAction
    | ExecuteAction
    | ClearAction
    | ClickAction
    | FillAction
)


class Scene(BaseModel):
    """A scene in the demo with narration and actions."""

    id: str = Field(..., description="Unique identifier for the scene")
    narration: str = Field(..., description="Text to be spoken during this scene")
    actions: list[Action] = Field(default_factory=list, description="Actions to perform")


class DemoScript(BaseModel):
    """Complete demo script definition."""

    title: str = Field(..., description="Title of the demo")
    description: str | None = Field(default=None, description="Optional description")
    scenes: list[Scene] = Field(..., description="Scenes in the demo")
    viewport: tuple[int, int] = Field(
        default=(1280, 720), description="Browser viewport size (width, height)"
    )
    typing_speed: int = Field(default=50, description="Default typing speed in ms between keys")

    @classmethod
    def from_yaml(cls, path: str) -> DemoScript:
        """Load a demo script from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
