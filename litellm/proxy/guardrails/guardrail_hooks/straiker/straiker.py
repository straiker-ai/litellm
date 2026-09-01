from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, cast
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.exceptions import (
    BadRequestError,
    GuardrailRaisedException,
    ModifyResponseException,
    Timeout,
)
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
    log_guardrail_information,
)
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import SpecialProxyStrings
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    STRAIKER_WEBHOOK_SCHEMA_VERSION,
    StraikerAppAttribution,
    StraikerCodingAgentConfig,
    StraikerCodingAgentKind,
    StraikerCodingAgentLatency,
    StraikerDetectResponse,
    StraikerGuardrailConfigModel,
    StraikerHookEvent,
    StraikerWebhookApplication,
    StraikerWebhookContent,
    StraikerWebhookContext,
    StraikerWebhookEvent,
    StraikerWebhookIdentity,
    StraikerWebhookRequest,
    StraikerWebhookResponse,
    StraikerWebhookStream,
    StraikerWebhookUsage,
)
from litellm.types.utils import GenericGuardrailAPIInputs

from . import coding_agent
from .detect_client import AsyncPoster, DetectFailure, DetectOutcome, is_block, post_hook_event

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME: Final = "straiker"
DEFAULT_BLOCK_MESSAGE: Final = "Content violates policy"
DEFAULT_API_BASE: Final = "https://api.prod.straiker.ai"
DEFAULT_MAX_PAYLOAD_BYTES: Final = 524288
WEBHOOK_PATH: Final = "/api/v1/detect/webhook"
RETRY_STATUS: Final = frozenset({408, 429, 500, 502, 503, 504})
UNREACHABLE_STATUS: Final = frozenset({502, 503, 504})
_APPLICATION_METADATA_KEYS: Final = frozenset({"agent_id", "app_name"})
_OPAQUE_METADATA_SCALAR_TYPES: Final = (str, int, float, bool)
_JSON_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _WebhookFailure:
    message: str
    is_unreachable: bool


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _merged_metadata(request_data: dict) -> dict:
    return {
        **_as_dict(request_data.get("metadata")),
        **_as_dict(request_data.get("litellm_metadata")),
    }


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _build_webhook_metadata(request_data: dict, default_metadata: dict[str, str]) -> dict[str, object] | None:
    out: Final[dict[str, object]] = {}
    for key, value in _as_dict(request_data.get("metadata")).items():
        if key in _APPLICATION_METADATA_KEYS or key.startswith("user_api"):
            continue
        if key == "session_id":
            continue
        if isinstance(value, _OPAQUE_METADATA_SCALAR_TYPES):
            out[key] = value
    out.update(default_metadata)
    return out or None


def _extract_identity(request_data: dict) -> StraikerWebhookIdentity:
    meta: Final = _merged_metadata(request_data)
    return StraikerWebhookIdentity(
        litellm_key=_as_optional_str(meta.get("user_api_key_alias"))
        or _as_optional_str(meta.get("user_api_key_hash"))
        or _as_optional_str(meta.get("user_api_key_token")),
        litellm_team=_as_optional_str(meta.get("user_api_key_team_alias"))
        or _as_optional_str(meta.get("user_api_key_team_id")),
        litellm_user_id=_as_optional_str(meta.get("user_api_key_user_id")),
        litellm_user_email=_as_optional_str(meta.get("user_api_key_user_email")),
        litellm_org_id=_as_optional_str(meta.get("user_api_key_org_id")),
        end_user_id=_as_optional_str(meta.get("user_api_key_end_user_id")),
    )


def _resolve_provider(request_data: dict, model: str | None) -> str | None:
    litellm_params: Final = _as_dict(request_data.get("litellm_params"))
    custom_llm_provider: Final = request_data.get("custom_llm_provider") or litellm_params.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider
    if not model:
        return None
    try:
        _, provider, _, _ = get_llm_provider(
            model=model,
            api_base=request_data.get("api_base") or litellm_params.get("api_base"),
            api_key=request_data.get("api_key") or litellm_params.get("api_key"),
        )
    except BadRequestError:
        return None
    return provider or None


def _resolve_destination(request_data: dict) -> str | None:
    litellm_params: Final = _as_dict(request_data.get("litellm_params"))
    api_base: Final = request_data.get("api_base") or litellm_params.get("api_base")
    if not isinstance(api_base, str):
        return None
    try:
        return urlsplit(api_base).hostname
    except ValueError:
        return None


def _route_has_translation(request_data: dict) -> bool:
    from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
    from litellm.llms import load_guardrail_translation_mappings

    route: Final = _as_dict(request_data.get("litellm_metadata")).get("user_api_key_request_route")
    if not isinstance(route, str) or not route:
        return False
    mappings: Final = load_guardrail_translation_mappings()
    return any(call_type in mappings for call_type in get_call_types_for_route(route) or ())


def _request_structured_messages(request_data: dict) -> list[dict[str, Any]] | None:
    messages: Final = request_data.get("messages")
    if messages:
        return messages if isinstance(messages, list) else None
    if not _route_has_translation(request_data):
        return None
    return resolve_structured_messages(messages=None, request_kwargs=request_data)


def _hook_name(value: object) -> str:
    return value.value if isinstance(value, GuardrailEventHooks) else str(value)


def _configured_modes(event_hook: object) -> list[str] | None:
    if isinstance(event_hook, list):
        names = [_hook_name(v) for v in event_hook]
    elif isinstance(event_hook, (str, GuardrailEventHooks)):
        names = [_hook_name(event_hook)]
    elif isinstance(event_hook, Mode):
        default: Final = event_hook.default if isinstance(event_hook.default, list) else [event_hook.default]
        tags: Final = [v for value in event_hook.tags.values() for v in (value if isinstance(value, list) else [value])]
        names = [_hook_name(v) for v in (*default, *tags) if v is not None]
    else:
        return None
    return list(dict.fromkeys(names)) or None


def _resolve_call_surface(logging_obj: LiteLLMLoggingObj | None, request_data: dict) -> str:
    call_type: Final = (
        (getattr(logging_obj, "call_type", None) if logging_obj is not None else None)
        or request_data.get("call_type")
        or request_data.get("litellm_call_type")
    )
    return call_type if isinstance(call_type, str) and call_type else "unknown"


def _jsonable_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, BaseModel):
        return _JSON_DICT_ADAPTER.validate_python(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return _JSON_DICT_ADAPTER.validate_python(value)
    return None


def _opaque_dict_list(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    items: Final = tuple(plain for item in value if (plain := _jsonable_dict(item)) is not None)
    return list(items) if items else None


def _choice_terminal_reason(choice: object) -> str | None:
    if isinstance(choice, dict):
        return _as_optional_str(choice.get("finish_reason")) or _as_optional_str(choice.get("stop_reason"))
    return _as_optional_str(getattr(choice, "finish_reason", None)) or _as_optional_str(
        getattr(choice, "stop_reason", None)
    )


def _response_finish_reason(response: Any) -> str | None:
    if response is None:
        return None
    if isinstance(response, dict):
        top = _as_optional_str(response.get("finish_reason")) or _as_optional_str(response.get("stop_reason"))
        if top:
            return top
        choices = response.get("choices")
        if not isinstance(choices, list):
            return None
        for choice in choices:
            reason = _choice_terminal_reason(choice)
            if reason:
                return reason
        return None

    top = _as_optional_str(getattr(response, "finish_reason", None)) or _as_optional_str(
        getattr(response, "stop_reason", None)
    )
    if top:
        return top
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list):
        return None
    for choice in choices:
        reason = _choice_terminal_reason(choice)
        if reason:
            return reason
    return None


def _as_optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _usage_token_count(usage: object, openai_key: str, anthropic_key: str) -> int | None:
    get: Final = usage.get if isinstance(usage, dict) else lambda key: getattr(usage, key, None)
    openai_count: Final = _as_optional_int(get(openai_key))
    return openai_count if openai_count is not None else _as_optional_int(get(anthropic_key))


def _build_usage(response: object) -> StraikerWebhookUsage | None:
    usage: Final = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens: Final = _usage_token_count(usage, "prompt_tokens", "input_tokens")
    output_tokens: Final = _usage_token_count(usage, "completion_tokens", "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return StraikerWebhookUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _is_streamed_request(request_data: dict) -> bool:
    if request_data.get("stream") is True:
        return True
    body: Final = _as_dict(_as_dict(request_data.get("proxy_server_request")).get("body"))
    return body.get("stream") is True


def _typed_request(request_data: object) -> dict[str, object]:
    """Proxy request data is a JSON body plus litellm's own string keys; the base hook
    signature just cannot say so."""
    if not isinstance(request_data, dict):
        return {}
    return cast("dict[str, object]", request_data)  # cast-ok: proxy request data is str-keyed


def _typed_child(container: Mapping[str, object], key: str) -> dict[str, object]:
    return _typed_request(container.get(key))


def _typed_metadata(request_data: Mapping[str, object]) -> dict[str, object]:
    return {**_typed_child(request_data, "metadata"), **_typed_child(request_data, "litellm_metadata")}


def _real_identity(value: object) -> str | None:
    """LiteLLM's proxy-admin placeholder is not a person."""
    identity = _as_optional_str(value)
    return None if identity == SpecialProxyStrings.default_user_id.value else identity


def _request_header(request_data: Mapping[str, object], name: str | None) -> str | None:
    if not name:
        return None
    headers = _typed_child(_typed_child(request_data, "proxy_server_request"), "headers")
    wanted = name.lower()
    return next(
        (value for key, value in headers.items() if key.lower() == wanted and isinstance(value, str) and value),
        None,
    )


def _request_user_agent(request_data: Mapping[str, object]) -> str | None:
    headers = _typed_child(_typed_child(request_data, "proxy_server_request"), "headers")
    return next(
        (value for key, value in headers.items() if key.lower() == "user-agent" and isinstance(value, str)),
        None,
    )


def _coding_agent_config(value: object) -> StraikerCodingAgentConfig | None:
    if value is None or isinstance(value, StraikerCodingAgentConfig):
        return value
    return StraikerCodingAgentConfig.model_validate(value)


def _response_text(inputs: GenericGuardrailAPIInputs) -> str | None:
    texts = [text for text in (inputs.get("texts") or []) if text]
    return "\n".join(texts) if texts else None


def _verdict_summary(outcome: DetectOutcome) -> dict[str, object]:
    if isinstance(outcome, DetectFailure):
        return {"error": outcome.message}
    return {
        "turn_id": outcome.turn_id,
        "score": outcome.score,
        "score_category": outcome.score_category,
        "severity": outcome.severity,
        "action": outcome.action,
    }


def _event_preview(event: StraikerHookEvent) -> dict[str, object]:
    """What was actually sent, abbreviated. Operators debugging a coding-agent app need to
    see the tool command or the tool output that was scored, not just its verdict."""
    body = event.model_dump(exclude_none=True)
    preview = {
        key: (value[:200] + "…" if isinstance(value, str) and len(value) > 200 else value)
        for key, value in body.items()
        if key
        in ("prompt", "tool_input", "tool_response", "app_response", "is_error", "attachments", "user_name", "model")
    }
    return {**preview, "bytes": len(event.model_dump_json(exclude_none=True).encode("utf-8"))}


def _block_reason(outcome: DetectOutcome) -> str:
    if isinstance(outcome, StraikerDetectResponse) and outcome.reason:
        return outcome.reason
    return DEFAULT_BLOCK_MESSAGE


def _coding_config_warnings(
    config: StraikerCodingAgentConfig | None, latency: StraikerCodingAgentLatency
) -> Iterator[tuple[str, str]]:
    """Config combinations that are accepted but cannot do what their name promises."""
    if config is None:
        return
    if not config.fail_open and latency == "zero":
        yield (
            "straiker.coding_fail_closed_ignored",
            "coding_agent_fail_open=false has no effect at latency=zero, which posts in the "
            "background and never waits for a verdict. Use latency=hold or strict.",
        )
    if config.mode == "block" and latency != "strict":
        yield (
            "straiker.coding_enforcement_limited",
            "mode=block cannot stop a tool call at this latency profile: the tool_use reaches "
            "the client before the verdict does. Use latency=strict for tool-call enforcement.",
        )


class StraikerGuardrail(CustomGuardrail):
    guardrail_provider: str = "straiker"

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel]:
        return StraikerGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        source: str = "LiteLLM Gateway",
        timeout: float = 5.0,
        max_retries: int = 2,
        initial_backoff: float = 0.1,
        max_backoff: float = 2.0,
        unreachable_fallback: Literal["fail_open", "fail_closed"] = "fail_closed",
        fail_on_error: bool = True,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        custom_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        verbose: bool = False,
        coding_agent: StraikerCodingAgentConfig | dict[str, object] | None = None,
        app_attribution: StraikerAppAttribution = "default",
        async_handler: httpx.AsyncClient | None = None,
        dedup_cache: DualCache | None = None,
        **kwargs: object,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if unreachable_fallback not in ("fail_open", "fail_closed"):
            raise ValueError(f"unreachable_fallback must be 'fail_open' or 'fail_closed'; got {unreachable_fallback!r}")

        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.source = source
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.initial_backoff = max(0.0, float(initial_backoff))
        self.max_backoff = max(self.initial_backoff, float(max_backoff))
        self.unreachable_fallback = unreachable_fallback
        self.fail_on_error = fail_on_error
        self.max_payload_bytes = int(max_payload_bytes)
        self.custom_headers = dict(custom_headers) if custom_headers else {}
        self.default_metadata = dict(metadata) if metadata else {}
        self.verbose = bool(verbose)
        self.coding_agent = _coding_agent_config(coding_agent)
        self.app_attribution = app_attribution

        # Streaming posture. LiteLLM reads these off the instance when it wraps the
        # stream, so they set the profile for this guardrail entry: buffering withholds
        # every chunk until the assembled response is scored (safest, worst
        # time-to-first-token), end-of-stream-only scores once at the end, and neither
        # scores in flight at a sampled cadence so tokens keep flowing while a block can
        # still stop the stream. A proxy that wants different postures for coding and
        # non-coding traffic registers two guardrail entries and tag-routes them.
        latency = self.coding_agent.resolved_latency() if self.coding_agent else "strict"
        self.streaming_buffer_until_moderated = latency == "strict"
        self.streaming_end_of_stream_only = latency != "hold"
        if self.coding_agent is not None:
            self.streaming_sampling_rate = self.coding_agent.streaming_sampling_rate

        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        # LiteLLM's default in-memory cache holds 200 entries, which a busy gateway evicts
        # in minutes; an evicted dedup key means the resent transcript is scored again.
        dedup_size = self.coding_agent.dedup_cache_size if self.coding_agent else 20000
        self.dedup_cache = dedup_cache or DualCache(
            in_memory_cache=InMemoryCache(max_size_in_memory=dedup_size, default_ttl=None)
        )
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._dedup_backend_resolved = False

        for event, detail in _coding_config_warnings(self.coding_agent, latency):
            verbose_proxy_logger.warning(json.dumps({"event": event, "detail": detail, "latency": latency}))
        if self.coding_agent is not None:
            verbose_proxy_logger.info(
                json.dumps(
                    {
                        "event": "straiker.coding_agent_ready",
                        "agents": list(self.coding_agent.agents),
                        "mode": self.coding_agent.mode,
                        "latency": latency,
                        "posts_in_background": self.coding_agent.posts_in_background(),
                        "streaming_buffer_until_moderated": self.streaming_buffer_until_moderated,
                        "note": "streaming posture applies to all traffic on this guardrail entry",
                    }
                )
            )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)

        self.configured_modes = _configured_modes(self.event_hook)

    def _webhook_url(self) -> str:
        return f"{self.api_base}{WEBHOOK_PATH}"

    def _headers(self) -> dict[str, str]:
        reserved: Final = {"authorization", "content-type", "x-straiker-webhook-format"}
        extra: Final = {k: v for k, v in self.custom_headers.items() if k.lower() not in reserved}
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Straiker-Webhook-Format": "litellm",
            **extra,
        }

    def _attributed_source(self, request_data: Mapping[str, object], model: str | None) -> str | None:
        """Give each application its own Console card instead of collapsing every request
        the gateway proxies into one."""
        meta: Final = _typed_metadata(request_data)
        if self.app_attribution == "model":
            return _as_optional_str(model) or _as_optional_str(request_data.get("model"))
        if self.app_attribution == "key_alias":
            return _as_optional_str(meta.get("user_api_key_alias"))
        if self.app_attribution == "team_alias":
            return _as_optional_str(meta.get("user_api_key_team_alias")) or _as_optional_str(
                meta.get("user_api_key_team_id")
            )
        return None

    def _build_application(self, request_data: dict, model: str | None = None) -> StraikerWebhookApplication:
        meta: Final = _merged_metadata(request_data)
        agent_id: Final = _as_optional_str(meta.get("agent_id"))
        return StraikerWebhookApplication(
            source=agent_id or self._attributed_source(_typed_request(request_data), model) or self.source,
            name=_as_optional_str(meta.get("app_name")),
        )

    def _build_context(
        self,
        request_data: dict,
        model: str | None,
        logging_obj: LiteLLMLoggingObj | None,
    ) -> StraikerWebhookContext:
        return StraikerWebhookContext(
            call_surface=_resolve_call_surface(logging_obj, request_data),
            mode=self.configured_modes,
            model=model,
            model_provider=_resolve_provider(request_data, model),
            destination=_resolve_destination(request_data),
            session_id=get_session_id_from_request_data(request_data),
            litellm_call_id=getattr(logging_obj, "litellm_call_id", None) if logging_obj else None,
            litellm_trace_id=getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None,
            litellm_version=litellm_version,
        )

    def _build_envelope(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None,
    ) -> StraikerWebhookRequest:
        model: Final = inputs.get("model") or request_data.get("model")
        call_id: Final = getattr(logging_obj, "litellm_call_id", None) if logging_obj else None
        event_id: Final = f"{call_id or 'litellm'}:{input_type}"

        content: Final = StraikerWebhookContent(
            texts=list(inputs.get("texts") or []),
            images=list(inputs.get("images") or []),
            structured_messages=_opaque_dict_list(inputs.get("structured_messages")),
            tools=_opaque_dict_list(inputs.get("tools")),
            tool_calls=_opaque_dict_list(inputs.get("tool_calls")),
        )

        if input_type == "request":
            event = StraikerWebhookEvent(type="pre_call", id=event_id)
            return StraikerWebhookRequest(
                event=event,
                request=content,
                context=self._build_context(request_data, model, logging_obj),
                identity=_extract_identity(request_data),
                application=self._build_application(request_data, model),
                metadata=_build_webhook_metadata(request_data, self.default_metadata),
            )

        response_obj: Final = request_data.get("response")
        content.finish_reason = _response_finish_reason(response_obj)
        request_content: Final = StraikerWebhookContent(
            structured_messages=_opaque_dict_list(_request_structured_messages(request_data)),
        )
        phase: Final[Literal["none", "assembled"]] = "assembled" if _is_streamed_request(request_data) else "none"
        event = StraikerWebhookEvent(type="post_call", id=event_id, stream=StraikerWebhookStream(phase=phase))
        return StraikerWebhookRequest(
            event=event,
            request=request_content,
            response=content,
            context=self._build_context(request_data, model, logging_obj),
            identity=_extract_identity(request_data),
            application=self._build_application(request_data, model),
            usage=_build_usage(response_obj),
            metadata=_build_webhook_metadata(request_data, self.default_metadata),
        )

    async def _post_webhook(self, payload: dict) -> tuple[StraikerWebhookResponse | None, _WebhookFailure | None]:
        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as error:
            return None, _WebhookFailure(f"request serialization failed: {error}", is_unreachable=False)
        body_bytes: Final = len(body)
        if body_bytes > self.max_payload_bytes:
            return None, _WebhookFailure(
                f"payload {body_bytes}B exceeds max_payload_bytes {self.max_payload_bytes}",
                is_unreachable=False,
            )

        url: Final = self._webhook_url()
        headers: Final = self._headers()
        attempts: Final = self.max_retries + 1
        last_failure: _WebhookFailure | None = None

        if self.verbose:
            verbose_proxy_logger.info(
                json.dumps(
                    {
                        "event": "straiker.webhook_request",
                        "url": url,
                        "bytes": body_bytes,
                        "payload": payload,
                    },
                    default=str,
                )
            )

        for attempt in range(attempts):
            try:
                resp = await self.async_handler.post(url, content=body, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                        parsed = StraikerWebhookResponse.model_validate(body)
                    except (ValidationError, json.JSONDecodeError) as ve:
                        return None, _WebhookFailure(f"invalid response schema: {ve}", is_unreachable=False)
                    if self.verbose:
                        verbose_proxy_logger.info(
                            json.dumps(
                                {
                                    "event": "straiker.webhook_response",
                                    "status_code": resp.status_code,
                                    "body": body,
                                },
                                default=str,
                            )
                        )
                    return parsed, None
                last_failure = _WebhookFailure(
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    is_unreachable=resp.status_code in UNREACHABLE_STATUS,
                )
                if resp.status_code not in RETRY_STATUS:
                    return None, last_failure
            except (httpx.RequestError, asyncio.TimeoutError, Timeout) as e:
                last_failure = _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=True)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                return None, _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=False)

            if attempt < attempts - 1:
                backoff = min(self.initial_backoff * (2**attempt), self.max_backoff)
                await asyncio.sleep(random.uniform(0, backoff))

        return None, last_failure or _WebhookFailure("unknown error", is_unreachable=True)

    def _record(
        self,
        *,
        request_data: dict,
        logging_obj: LiteLLMLoggingObj | None,
        parsed: StraikerWebhookResponse,
    ) -> None:
        if not self.verbose:
            return
        response_obj: Final = request_data.get("response")
        hidden: Final = getattr(response_obj, "_hidden_params", None)
        if isinstance(hidden, dict):
            straiker_hidden: Final = hidden.setdefault("straiker", {})
            if isinstance(straiker_hidden, dict):
                straiker_hidden.update({"action": parsed.action, "turn_id": parsed.turn_id})

    def _fail(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        error: str,
        is_unreachable: bool,
    ) -> GenericGuardrailAPIInputs:
        fail_open: Final = (is_unreachable and self.unreachable_fallback == "fail_open") or not self.fail_on_error
        verbose_proxy_logger.error(
            json.dumps(
                {
                    "event": "straiker.error",
                    "input_type": input_type,
                    "error": error,
                    "fail_open": fail_open,
                },
                default=str,
            )
        )
        if fail_open:
            return inputs
        self._block(
            request_data=request_data,
            input_type=input_type,
            message=f"Straiker detection unavailable: {error}",
        )

    def _block(
        self,
        *,
        request_data: dict,
        input_type: Literal["request", "response"],
        message: str,
        blocked_content: bool = False,
    ) -> NoReturn:
        if input_type == "request":
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
                message=message,
                should_wrap_with_default_message=False,
                blocked_content=blocked_content,
            )
        raise ModifyResponseException(
            message=message,
            model=request_data.get("model", "unknown") or "unknown",
            request_data=request_data,
            guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
            original_response=request_data.get("response"),
        )

    @staticmethod
    def _intervened_inputs(
        inputs: GenericGuardrailAPIInputs,
        parsed: StraikerWebhookResponse,
    ) -> GenericGuardrailAPIInputs:
        return_inputs: Final[GenericGuardrailAPIInputs] = {}
        return_inputs.update(inputs)
        if parsed.texts is not None:
            return_inputs["texts"] = parsed.texts
        return return_inputs

    def _detect_coding_agent(
        self, config: StraikerCodingAgentConfig, request_data: dict[str, object]
    ) -> StraikerCodingAgentKind | None:
        if config.enabled == "off":
            return None
        agent = coding_agent.detect_agent(request_data, _request_user_agent(request_data), config.agents)
        if agent is not None:
            return agent
        return "claude_code" if config.enabled == "force" else None

    def _coding_user_name(self, config: StraikerCodingAgentConfig, request_data: Mapping[str, object]) -> str:
        """The backend keys a session's event trace on the user name, so an unstable value
        splits one conversation into traces that never pair a prompt with its tool calls.

        ``default_user_id`` is LiteLLM's placeholder for the proxy master key rather than a
        real person, so it must not outrank an explicitly supplied header; otherwise every
        developer behind a master key collapses into one identity with no way to override.
        """
        meta = _typed_metadata(request_data)
        return (
            _real_identity(meta.get("user_api_key_user_email"))
            or _real_identity(meta.get("user_api_key_user_id"))
            or _real_identity(meta.get("user_api_key_alias"))
            or _request_header(request_data, config.user_name_header)
            or _as_optional_str(meta.get("user_api_key_user_id"))
            or config.default_user_name
        )

    def _coding_events(
        self,
        *,
        config: StraikerCodingAgentConfig,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        session_id: str,
        agent: StraikerCodingAgentKind,
    ) -> tuple[StraikerHookEvent, ...]:
        user_name = self._coding_user_name(config, request_data)
        model = _as_optional_str(request_data.get("model"))
        if input_type == "request":
            parsed = coding_agent.parse_request(
                request_data,
                chatter_filter=config.chatter_filter,
                tool_less_is_utility=config.enabled != "force",
            )
            events = coding_agent.request_events(parsed, session_id, user_name, agent, model)
        elif coding_agent.is_utility_call(request_data, config.chatter_filter, config.enabled != "force"):
            events = ()
        else:
            events = coding_agent.response_events(
                coding_agent.parse_tool_calls(inputs.get("tool_calls")),
                _response_text(inputs),
                _response_finish_reason(request_data.get("response")),
                session_id,
                user_name,
                agent,
                model,
            )
        return tuple(coding_agent.truncate_event(event, config.max_event_bytes) for event in events)

    def _attach_shared_dedup(self) -> None:
        """Borrow the proxy's Redis once it exists, so workers dedup against each other.

        Without it every uvicorn worker keeps its own view and the same event is scored
        once per worker. Lazy because the guardrail is constructed before the proxy's
        cache is; in-memory-only is a valid single-worker configuration.
        """
        if self._dedup_backend_resolved:
            return
        self._dedup_backend_resolved = True
        try:
            from litellm.proxy.proxy_server import user_api_key_cache
        except ImportError:
            return
        redis_cache = getattr(user_api_key_cache, "redis_cache", None)
        if redis_cache is None:
            return
        self.dedup_cache.attach_redis_cache(redis_cache)
        verbose_proxy_logger.info(json.dumps({"event": "straiker.dedup_backend", "backend": "redis"}))

    async def _claim_event(
        self,
        config: StraikerCodingAgentConfig,
        session_id: str,
        event: StraikerHookEvent,
    ) -> StraikerHookEvent | None:
        """Claim an event exactly once.

        The local claim is synchronous so no await can interleave between the read and the
        write. Across workers the claim is a Redis SETNX, which is atomic and costs one
        round trip; a read-then-write would let two workers both see a miss and both post.
        """
        self._attach_shared_dedup()
        key = f"straiker:coding:{session_id}:{event.dedup_key()}"
        local = self.dedup_cache.in_memory_cache
        if local.get_cache(key) is not None:
            return None
        local.set_cache(key, True, ttl=config.dedup_ttl)

        redis_cache = self.dedup_cache.redis_cache
        if redis_cache is None:
            return event
        claimed = await redis_cache.async_set_cache(key, True, ttl=config.dedup_ttl, nx=True)
        return event if claimed else None

    async def _release_claim(self, session_id: str, event: StraikerHookEvent) -> None:
        key: Final = f"straiker:coding:{session_id}:{event.dedup_key()}"
        self.dedup_cache.in_memory_cache.delete_cache(key)
        redis_cache = self.dedup_cache.redis_cache
        if redis_cache is not None:
            await redis_cache.async_delete_cache(key)

    async def _release_failed_claims(
        self,
        session_id: str,
        events: tuple[StraikerHookEvent, ...],
        outcomes: tuple[DetectOutcome, ...],
    ) -> None:
        """A claim taken before the post means a failed post is never retried, because the
        resent transcript finds the key already held."""
        failed = tuple(e for e, o in zip(events, outcomes) if isinstance(o, DetectFailure))
        if failed:
            await asyncio.gather(*(self._release_claim(session_id, e) for e in failed))

    async def _post_coding_event(
        self,
        config: StraikerCodingAgentConfig,
        event: StraikerHookEvent,
        agent: StraikerCodingAgentKind,
    ) -> DetectOutcome:
        return await post_hook_event(
            client=cast("AsyncPoster", self.async_handler),  # cast-ok: both client types satisfy the protocol
            url=f"{self.api_base}{config.detect_path}",
            api_key=config.api_key or self.api_key,
            x_tool=config.x_tool_override or coding_agent.x_tool_for(agent),
            event=event,
            sign=config.sign_payloads,
            timeout=config.timeout or self.timeout,
            max_retries=self.max_retries,
            initial_backoff=self.initial_backoff,
            max_backoff=self.max_backoff,
        )

    def _log_coding_outcomes(
        self,
        events: tuple[StraikerHookEvent, ...],
        outcomes: tuple[DetectOutcome, ...],
    ) -> None:
        for event, outcome in zip(events, outcomes):
            record = {
                "event": "straiker.coding_event",
                "hook_event_name": event.hook_event_name,
                "session_id": event.session_id,
                "tool_name": event.tool_name,
                "verdict": _verdict_summary(outcome),
                **({"content": _event_preview(event)} if self.verbose else {}),
            }
            if isinstance(outcome, DetectFailure):
                verbose_proxy_logger.error(json.dumps(record, default=str))
            elif self.verbose:
                verbose_proxy_logger.info(json.dumps(record, default=str))

    def _dispatch_background(
        self,
        config: StraikerCodingAgentConfig,
        events: tuple[StraikerHookEvent, ...],
        agent: StraikerCodingAgentKind,
        session_id: str,
    ) -> None:
        """Post without waiting, so scoring never sits in front of the model call.

        Delivery is best effort by construction: nothing can be blocked on a verdict that
        has not arrived, which is the trade the 'zero' profile exists to make.
        """

        async def deliver() -> None:
            outcomes = tuple(await asyncio.gather(*(self._post_coding_event(config, e, agent) for e in events)))
            self._log_coding_outcomes(events, outcomes)
            await self._release_failed_claims(session_id, events, outcomes)

        task = asyncio.create_task(deliver())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _apply_coding_agent(
        self,
        *,
        config: StraikerCodingAgentConfig,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        agent: StraikerCodingAgentKind,
    ) -> GenericGuardrailAPIInputs:
        """Score a coding-agent call as the hook events it stands for.

        Fail-open by contract: a session-less call, a parser error or an unreachable
        detect endpoint passes the traffic through rather than breaking a developer's
        session, matching the reference Kong implementation.
        """
        session_id = get_session_id_from_request_data(request_data)
        if not session_id:
            return inputs
        try:
            events = self._coding_events(
                config=config,
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                session_id=session_id,
                agent=agent,
            )
        except (AttributeError, KeyError, TypeError, ValidationError, ValueError) as error:
            verbose_proxy_logger.warning(
                json.dumps({"event": "straiker.coding_parse_error", "error": str(error)}, default=str)
            )
            return inputs

        claimed = await asyncio.gather(*(self._claim_event(config, session_id, event) for event in events))
        fresh = tuple(event for event in claimed if event is not None)
        if not fresh:
            return inputs

        if config.posts_in_background():
            self._dispatch_background(config, fresh, agent, session_id)
            return inputs

        outcomes = tuple(await asyncio.gather(*(self._post_coding_event(config, event, agent) for event in fresh)))
        self._log_coding_outcomes(fresh, outcomes)
        await self._release_failed_claims(session_id, fresh, outcomes)

        if not config.fail_open:
            failure = next((outcome for outcome in outcomes if isinstance(outcome, DetectFailure)), None)
            if failure is not None:
                self._block(
                    request_data=request_data,
                    input_type=input_type,
                    message=f"Straiker detection unavailable: {failure.message}",
                )

        if config.mode != "block":
            return inputs
        blocked = next((outcome for outcome in outcomes if is_block(outcome)), None)
        if blocked is None:
            return inputs
        self._block(
            request_data=request_data,
            input_type=input_type,
            message=_block_reason(blocked),
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        config = self.coding_agent
        coding_request = _typed_request(request_data)
        agent = self._detect_coding_agent(config, coding_request) if config is not None else None
        if config is not None and agent is not None:
            return await self._apply_coding_agent(
                config=config,
                inputs=inputs,
                request_data=coding_request,
                input_type=input_type,
                agent=agent,
            )
        try:
            envelope: Final = self._build_envelope(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
            )
            payload: Final = envelope.model_dump(mode="json", exclude_none=True)
        except (ValidationError, TypeError, ValueError) as error:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=str(error),
                is_unreachable=False,
            )

        parsed, failure = await self._post_webhook(payload)
        if failure is not None:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=failure.message,
                is_unreachable=failure.is_unreachable,
            )

        if parsed is None:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error="empty response from Straiker",
                is_unreachable=False,
            )
        self._record(request_data=request_data, logging_obj=logging_obj, parsed=parsed)

        if parsed.schema_version is not None and parsed.schema_version != STRAIKER_WEBHOOK_SCHEMA_VERSION:
            verbose_proxy_logger.warning(
                json.dumps(
                    {
                        "event": "straiker.schema_drift",
                        "expected": STRAIKER_WEBHOOK_SCHEMA_VERSION,
                        "received": parsed.schema_version,
                    }
                )
            )

        if parsed.action == "BLOCKED":
            self._block(
                request_data=request_data,
                input_type=input_type,
                message=parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE,
                blocked_content=True,
            )
        if parsed.action == "GUARDRAIL_INTERVENED":
            is_streamed_response: Final = input_type == "response" and _is_streamed_request(request_data)
            if parsed.texts is None or is_streamed_response:
                self._block(
                    request_data=request_data,
                    input_type=input_type,
                    message=parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE,
                    blocked_content=True,
                )
            return self._intervened_inputs(inputs, parsed)
        return inputs
