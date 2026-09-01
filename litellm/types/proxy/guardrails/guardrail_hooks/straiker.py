from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import GuardrailConfigModel

StraikerWebhookEventType = Literal["pre_call", "post_call"]
StraikerWebhookStreamPhase = Literal["none", "assembled"]
StraikerWebhookAction = Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"]
StraikerHookEventName = Literal[
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
    "beforeSubmitPrompt",
    "beforeShellExecution",
    "beforeMCPExecution",
    "beforeReadFile",
    "postToolUse",
]
StraikerCodingAgentEnabled = Literal["auto", "off", "force"]
StraikerCodingAgentMode = Literal["monitor", "block"]
StraikerCodingAgentLatency = Literal["zero", "hold", "strict"]
StraikerCodingAgentKind = Literal["claude_code", "cursor", "codex"]
StraikerAppAttribution = Literal["default", "model", "key_alias", "team_alias"]

STRAIKER_WEBHOOK_SCHEMA_VERSION: Final = "1"
STRAIKER_DETECT_PATH: Final = "/api/v1/detect"
STRAIKER_AGENT_TOOL_HEADERS: Final[Mapping[StraikerCodingAgentKind, str]] = MappingProxyType(
    {
        "claude_code": "claude-code",
        "cursor": "cursor",
        "codex": "codex",
    }
)
TRUNCATION_MARKER: Final = "…[truncated by Straiker gateway]"

PROMPT_EVENTS: Final = frozenset({"UserPromptSubmit", "beforeSubmitPrompt"})
TOOL_EVENTS: Final = frozenset({"PreToolUse", "beforeShellExecution", "beforeMCPExecution", "beforeReadFile"})
RESULT_EVENTS: Final = frozenset({"PostToolUse", "postToolUse"})


class StraikerWebhookStream(BaseModel):
    phase: StraikerWebhookStreamPhase = "none"
    index: int | None = None


class StraikerWebhookEvent(BaseModel):
    type: StraikerWebhookEventType
    id: str
    stream: StraikerWebhookStream = Field(default_factory=StraikerWebhookStream)


class StraikerWebhookContent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    texts: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    structured_messages: list[dict[str, object]] | None = None
    tools: list[dict[str, object]] | None = None
    tool_calls: list[dict[str, object]] | None = None
    finish_reason: str | None = None


class StraikerWebhookUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class StraikerWebhookContext(BaseModel):
    call_surface: str
    mode: list[str] | None = None
    model: str | None = None
    model_provider: str | None = None
    destination: str | None = None
    session_id: str | None = None
    litellm_call_id: str | None = None
    litellm_trace_id: str | None = None
    litellm_version: str | None = None


class StraikerWebhookIdentity(BaseModel):
    litellm_key: str | None = None
    litellm_team: str | None = None
    litellm_user_id: str | None = None
    litellm_user_email: str | None = None
    litellm_org_id: str | None = None
    end_user_id: str | None = None


class StraikerWebhookApplication(BaseModel):
    source: str
    name: str | None = None


class StraikerWebhookRequest(BaseModel):
    schema_version: str = STRAIKER_WEBHOOK_SCHEMA_VERSION
    event: StraikerWebhookEvent
    request: StraikerWebhookContent
    response: StraikerWebhookContent | None = None
    context: StraikerWebhookContext
    identity: StraikerWebhookIdentity
    application: StraikerWebhookApplication
    usage: StraikerWebhookUsage | None = None
    metadata: dict[str, object] | None = None


class StraikerWebhookResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: StraikerWebhookAction = "NONE"
    blocked_reason: str | None = None
    texts: list[str] | None = None
    schema_version: str | None = None
    turn_id: str | None = Field(default=None, alias="turnId")


class StraikerHookEvent(BaseModel):
    """One reconstructed coding-agent hook event, shaped exactly as the native Claude
    Code hook handler posts it so both paths score through the same backend pipeline."""

    model_config = ConfigDict(extra="forbid")

    hook_event_name: StraikerHookEventName
    session_id: str
    user_name: str | None = None
    cwd: str | None = None
    model: str | None = None
    prompt: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, object] | None = None
    tool_response: str | None = None
    tool_use_id: str | None = None
    is_error: bool | None = None
    mcp_server_name: str | None = None
    mcp_tool_name: str | None = None
    app_response: str | None = None
    stop_reason: str | None = None
    attachments: int | None = None

    def dedup_key(self) -> str:
        """The agentic loop resends the whole transcript on every call, so request-side
        events would otherwise re-fire for the rest of the session."""
        if self.hook_event_name in PROMPT_EVENTS:
            return f"prompt:{hashlib.sha256((self.prompt or '').encode()).hexdigest()}"
        if self.hook_event_name in TOOL_EVENTS:
            return f"pre:{self.tool_use_id}"
        if self.hook_event_name in RESULT_EVENTS:
            return f"post:{self.tool_use_id}"
        return f"stop:{hashlib.sha256((self.app_response or '').encode()).hexdigest()}"


class StraikerDetectResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    turn_id: str | None = None
    score: float | None = None
    score_category: str | None = None
    severity: str | None = None
    reason: str | None = None
    action: str | None = None


class StraikerCodingAgentConfig(BaseModel):
    """Coding-agent path config. Claude Code traffic is routed to /api/v1/detect with
    ``x-tool: claude-code`` instead of the webhook envelope, so it needs its own key:
    the backend files an application under Coding Agents or Custom Agents based on the
    traffic it receives, and one key serving both surfaces mixes them."""

    model_config = ConfigDict(extra="forbid")

    enabled: StraikerCodingAgentEnabled = Field(
        default="auto",
        description=(
            "'auto' routes requests detected as a coding agent down the hook-event path; "
            "'force' routes every request; 'off' disables it."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description="Straiker detect key for the coding-agent application. Falls back to the guardrail api_key.",
        json_schema_extra={"secret": True},
    )
    detect_path: str = Field(
        default=STRAIKER_DETECT_PATH,
        description="Path appended to api_base for hook events.",
    )
    mode: StraikerCodingAgentMode = Field(
        default="monitor",
        description="'monitor' scores and surfaces only; 'block' denies tool calls whose verdict is block.",
    )
    chatter_filter: bool = Field(
        default=True,
        description=(
            "Drop Claude Code's zero-tool utility calls (title generation, suggestion mode, "
            "conversation recap) before scoring. These carry scaffolding, not user intent."
        ),
    )
    sign_payloads: bool = Field(
        default=True,
        description="Send X-Straiker-Webhook-Signature/-Timestamp (HMAC-SHA256 over '{timestamp}.{payload}').",
    )
    timeout: float | None = Field(
        default=None,
        gt=0.0,
        description="Per-event HTTP timeout in seconds. Falls back to the guardrail timeout.",
    )
    user_name_header: str | None = Field(
        default="X-Straiker-User-Name",
        description=(
            "Request header carrying the caller's identity, used when the virtual key has no "
            "user. Set to null to ignore it. A session's event trace is keyed on this, so a "
            "deployment without per-developer virtual keys needs it to tell developers apart."
        ),
    )
    fail_open: bool = Field(
        default=True,
        description=(
            "Whether traffic proceeds when scoring fails. False blocks the request instead, "
            "which trades developer availability for guaranteed coverage."
        ),
    )
    default_user_name: str = Field(
        default="litellm-coding",
        description=(
            "Identity used when the request carries no virtual-key user. Must be stable for a "
            "session or the backend cannot pair a prompt with its tool calls."
        ),
    )
    dedup_ttl: int = Field(
        default=3600,
        gt=0,
        description="Seconds an emitted event is remembered, so a resent transcript is not rescored.",
    )
    dedup_cache_size: int = Field(
        default=20000,
        gt=0,
        description=(
            "In-memory dedup entries. LiteLLM's default cache holds 200, which a busy gateway "
            "evicts in minutes; once a key is evicted the resent transcript is scored again."
        ),
    )
    latency: StraikerCodingAgentLatency | None = Field(
        default=None,
        description=(
            "Latency/assurance profile. 'zero': events are posted in the background and the "
            "response streams untouched, so time-to-first-token is unaffected; nothing can be "
            "blocked. 'hold': the response streams while the guardrail scores in flight and can "
            "cut the stream, which stops further output but does NOT stop a tool call -- by the "
            "time the verdict lands the tool_use has already reached the client, which runs it. "
            "'strict': the response is withheld until scored, the only profile that can stop a "
            "tool call before the agent executes it. Defaults to 'zero' when mode is monitor and "
            "'strict' when mode is block."
        ),
    )
    streaming_sampling_rate: int = Field(
        default=5,
        gt=0,
        description="With latency 'hold', score every Nth streamed chunk.",
    )
    max_event_bytes: int = Field(
        default=262144,
        gt=0,
        description=(
            "Cap on one serialized hook event. Oversized tool output and prompts are truncated "
            "with a marker rather than dropped. Deliberately far below max_payload_bytes: a hook "
            "event is a single prompt or tool result, so an unbounded one is a whole file."
        ),
    )
    agents: list[StraikerCodingAgentKind] = Field(
        default=["claude_code", "cursor", "codex"],
        description=(
            "Which coding agents to reconstruct events for. Each is routed under its own x-tool "
            "value so the backend attributes the traffic to the agent that produced it."
        ),
    )
    x_tool_override: str | None = Field(
        default=None,
        description=(
            "Force the x-tool routing header instead of deriving it from the detected agent. "
            "Only needed for a backend that routes an agent under a non-default value."
        ),
    )

    @field_validator("detect_path")
    @classmethod
    def _reject_webhook_path(cls, value: str) -> str:
        if value.rstrip("/").endswith("/webhook"):
            raise ValueError("detect_path must be the /api/v1/detect coding-agent path, not the webhook path")
        return value

    def resolved_latency(self) -> StraikerCodingAgentLatency:
        if self.latency is not None:
            return self.latency
        return "strict" if self.mode == "block" else "zero"

    def posts_in_background(self) -> bool:
        """Only 'zero' can post without waiting: the other profiles need the verdict before
        deciding whether the client may proceed."""
        return self.resolved_latency() == "zero"


class StraikerGuardrailConfigModelOptionalParams(BaseModel):
    timeout: float | None = Field(
        default=5.0,
        gt=0.0,
        description="Per-attempt HTTP timeout in seconds.",
    )
    max_retries: int | None = Field(
        default=2,
        ge=0,
        description="Retries on transient HTTP (408/429/5xx) and network errors.",
    )
    initial_backoff: float | None = Field(
        default=0.1,
        ge=0.0,
        description="Initial retry backoff in seconds.",
    )
    max_backoff: float | None = Field(
        default=2.0,
        ge=0.0,
        description="Maximum retry backoff in seconds.",
    )
    unreachable_fallback: Literal["fail_open", "fail_closed"] | None = Field(
        default="fail_closed",
        description="Behavior when Straiker is unreachable after retries.",
    )
    fail_on_error: bool | None = Field(
        default=True,
        description=(
            "Behavior on any guardrail error, not just unreachability. True (default) blocks "
            "the request on error; False logs and allows the request to proceed."
        ),
    )
    max_payload_bytes: int | None = Field(
        default=524288,
        gt=0,
        description="Maximum serialized webhook payload size sent to Straiker.",
    )
    custom_headers: dict[str, str] | None = Field(
        default=None,
        description="Additional HTTP headers sent to Straiker, excluding Authorization and the webhook-format header.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description=(
            "Default metadata key/values added to the webhook metadata bag on every request. "
            "On key conflict with request-derived metadata, these configured values win."
        ),
    )
    verbose: bool | None = Field(
        default=False,
        description="Log webhook request/response payloads and record action/turn_id in response hidden params.",
    )
    coding_agent_enabled: StraikerCodingAgentEnabled | None = Field(
        default="off",
        description=(
            "Coding-agent support. 'auto' scores a detected coding agent as the hook events it "
            "stands for instead of one flattened envelope; 'force' treats all traffic that way; "
            "'off' disables it. Ordinary application traffic is unaffected either way."
        ),
    )
    coding_agent_api_key: str | None = Field(
        default=None,
        description=(
            "Straiker key for the coding-agent application. Required when coding agents are "
            "enabled: Straiker files an application by the traffic it receives, so reusing the "
            "gateway key would merge coding sessions into the gateway's application."
        ),
        json_schema_extra={"secret": True},
    )
    coding_agent_mode: StraikerCodingAgentMode | None = Field(
        default="monitor",
        description="'monitor' scores and surfaces only; 'block' denies on a block verdict.",
    )
    coding_agent_latency: StraikerCodingAgentLatency | None = Field(
        default=None,
        description=(
            "Assurance traded against time to first token. 'zero' posts in the background and "
            "never blocks; 'hold' streams while scoring and blocks prompts and tool results; "
            "'strict' withholds output until scored and is the only one that stops a tool call. "
            "Defaults to 'zero' for monitor and 'strict' for block."
        ),
    )
    coding_agent_fail_open: bool | None = Field(
        default=True,
        description="Whether coding-agent traffic proceeds when scoring is unavailable.",
    )
    coding_agent_chatter_filter: bool | None = Field(
        default=True,
        description="Drop the agent's own scaffolding calls, which are not user intent.",
    )
    coding_agent_user_name_header: str | None = Field(
        default="X-Straiker-User-Name",
        description="Request header to take developer identity from when a virtual key has no user.",
    )
    app_attribution: StraikerAppAttribution | None = Field(
        default="default",
        description=(
            "How each request is attributed to an application in the Defend Console. 'default' "
            "reports every request as default_app; 'model', 'key_alias' and 'team_alias' give each "
            "one its own application. metadata.agent_id always wins when present."
        ),
    )


class StraikerGuardrailConfigModel(GuardrailConfigModel[StraikerGuardrailConfigModelOptionalParams]):
    api_key: str = Field(
        min_length=1,
        description="Straiker DefendAI environment API key (Bearer token). Env: STRAIKER_API_KEY.",
        json_schema_extra={"secret": True},
    )

    api_base: str | None = Field(
        default="https://api.prod.straiker.ai",
        description="Straiker API base URL. Use the regional variant for non-US tenants.",
    )

    default_app: str | None = Field(
        default="LiteLLM Gateway",
        description=(
            "Default application registered in the Straiker Defend Console. "
            "Overridden per-request by metadata.agent_id when present."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Straiker"
