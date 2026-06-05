from typing import List, Literal, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class StraikerGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="Straiker DefendAI API key (Bearer token). Env: STRAIKER_API_KEY.",
        json_schema_extra={"secret": True},
    )

    api_base: Optional[str] = Field(
        default="https://api.prod.straiker.ai/api/v1/detect",
        description="Straiker /detect endpoint base URL. Use the regional variant for non-US tenants.",
    )

    agentic: bool = Field(
        default=False,
        description=(
            "When true, posts the full messages[] (with tool_calls + tool results) to "
            "/detect?agentic for multi-turn / tool-using agents."
        ),
    )

    source: str = Field(
        default="litellm",
        description="Application name registered in the Straiker Defend Console.",
    )

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Block when score > threshold. post_call is observability-only and never blocks.",
    )

    timeout: float = Field(
        default=5.0,
        gt=0.0,
        description="Per-attempt HTTP timeout in seconds.",
    )

    max_retries: int = Field(
        default=2,
        ge=0,
        description="Retries on transient HTTP (408/429/5xx) and network errors.",
    )

    unreachable_fallback: Literal["fail_open", "fail_closed"] = Field(
        default="fail_closed",
        description="Behavior when Straiker is unreachable after retries.",
    )

    skip_models: Optional[List[str]] = Field(
        default=None,
        description="Glob patterns to bypass scoring (e.g., coding-agent models).",
    )

    verbose: bool = Field(
        default=True,
        description=(
            "Sends Straiker-Debug header; block responses include the full per-category "
            "detection envelope and triggered_categories."
        ),
    )

    dedup_iterations: bool = Field(
        default=True,
        description=(
            "Agentic dedup: one pre + one post Straiker entry per user prompt regardless "
            "of LLM round-trips. Set false to score every iteration."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Straiker"
