"""
Straiker DefendAI guardrail for LiteLLM proxy.

Implements pre-call, during-call (moderation), and post-call hooks against
the Straiker /api/v1/detect endpoint, with optional agentic mode for
multi-turn / tool-using agent workloads.

Behavior:
  - pre_call:    blocks request with HTTPException(403) when score > threshold
  - moderation:  runs in parallel during the LLM call (mode: during_call)
  - post_call:   fire-and-forget /detect for observability + late-detection
  - agentic dedup: pre skipped on tool/assistant continuations; post skipped
    on iterations whose response still has tool_calls so one logical user
    prompt produces one pre + one post Console turn, not 2N
  - tool_calls reshape: OpenAI {function:{name, arguments:JSONstring}} ->
    Straiker agentic {id, name, input:object}

Enterprise-grade features:
  - Pydantic schema validation on Straiker response
  - Retry with exponential backoff + jitter on transient failures
  - unreachable_fallback: "allow" (fail_open) or "block" (fail_closed) on
    sustained Straiker unavailability
  - Per-route opt-out via enabled_models / skip_models
  - Payload size guard for long-context workloads
  - Custom headers passthrough for multi-tenant signing / forwarded-for
  - Structured JSON logging with score / turn_id / verdict / execution_ms
  - Straiker correlation surfaced via HTTPException detail (block path) and
    structured logs (allow path)

Coding-agent guidance (Cursor / Claude Code / Copilot):
  Don't enable this guardrail on coding-agent routes/keys. Long context (50k+
  tokens) kills inline guardrail latency. Use Straiker IDE Hooks instead.
  See README.md "Coding agents" section.

Usage in config.yaml:

    guardrails:
      - guardrail_name: straiker-pre
        litellm_params:
          guardrail: straiker_guardrail.StraikerGuardrail
          mode: pre_call
          default_on: true
          api_key: os.environ/STRAIKER_API_KEY
          agentic: true
          source: my-agent-app
          destination: api.openai.com
          threshold: 0.5
          unreachable_fallback: block   # or "allow"
          max_retries: 2
          max_payload_bytes: 524288
          skip_models: ["claude-3-5-sonnet-coding", "cursor-*"]

      - guardrail_name: straiker-post
        litellm_params:
          guardrail: straiker_guardrail.StraikerGuardrail
          mode: post_call
          default_on: true
          api_key: os.environ/STRAIKER_API_KEY
          agentic: true
          source: my-agent-app
          destination: api.openai.com
          threshold: 0.5
          unreachable_fallback: allow   # never block on post-call errors
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Optional

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from litellm.integrations.custom_guardrail import CustomGuardrail

log = logging.getLogger("straiker.guardrail")


# ---------- Pydantic response schema ---------------------------------------


class DetectResponse(BaseModel):
    """Subset of fields Straiker /detect returns we depend on.

    Extra fields are allowed and ignored, so future Straiker additions don't
    break parsing. Unknown types in known fields fail validation loudly.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    turn_id: Optional[str] = Field(default=None, alias="turnId")
    verdict: Optional[bool] = None
    explanation: Optional[str] = None
    debug: Optional[dict] = Field(
        default=None,
        description="Full Straiker-Debug envelope when Straiker-Debug: TRUE was "
                    "sent on the request. Contains per-category detection scores "
                    "(detect + block + disabled), score_block, score_detect.",
    )
    custom: Optional[dict] = Field(
        default=None,
        description="Per-category boolean/score flags returned by the non-agentic "
                    "/detect endpoint (payment, contains_address, contains_id, ...).",
    )


# `from __future__ import annotations` makes the field types strings; Pydantic
# needs explicit rebuild to resolve Optional inside the model namespace.
DetectResponse.model_rebuild()


# ---------- Helpers (unchanged behavior, preserved from prior version) -----


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "") or ""
    return ""


def _last_user_prompt(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return _extract_text_content(m.get("content"))
    return ""


def _transform_tool_calls(tcs: Any) -> Optional[list[dict]]:
    if not isinstance(tcs, list):
        return None
    out = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        entry: dict[str, Any] = {"id": tc.get("id")}
        fn = tc.get("function") or tc.get("func")
        if isinstance(fn, dict):
            entry["name"] = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    entry["input"] = json.loads(args)
                except Exception:
                    entry["input"] = {"_raw": args}
            elif isinstance(args, dict):
                entry["input"] = args
        else:
            entry["name"] = tc.get("name")
            entry["input"] = tc.get("input")
        out.append(entry)
    return out


def _build_agentic_messages(req_messages: list[dict],
                            app_response: Optional[str]) -> list[dict]:
    out = []
    for m in (req_messages or []):
        if not isinstance(m, dict):
            continue
        entry: dict[str, Any] = {"role": m.get("role")}
        text = _extract_text_content(m.get("content"))
        if text:
            entry["content"] = text
        if m.get("tool_calls"):
            entry["tool_calls"] = _transform_tool_calls(m["tool_calls"])
        if m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            entry["tool_name"] = m["name"]
        elif m.get("tool_name"):
            entry["tool_name"] = m["tool_name"]
        out.append(entry)
    if app_response and app_response != "N/A":
        out.append({"role": "assistant", "content": app_response})
    return out


def _has_meaningful_tool_calls(tool_calls: Any) -> bool:
    """A response's tool_calls is meaningful only if non-empty list of dicts
    with at least one entry carrying a function name."""
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            if isinstance(fn, dict) and fn.get("name"):
                return True
            if tc.get("name"):
                return True
    return False


def _wildcard_match(patterns: list[str], value: str) -> bool:
    """Glob-style wildcard match. Supports * and ? in patterns."""
    if not patterns or not value:
        return False
    for p in patterns:
        if not p:
            continue
        if p == value:
            return True
        regex = "^" + re.escape(p).replace(r"\*", ".*").replace(r"\?", ".") + "$"
        if re.match(regex, value):
            return True
    return False


def _structured_log(level: int, event: str, **fields: Any) -> None:
    """Single-line JSON log so downstream tools can parse score/turn_id."""
    payload = {"event": event, **fields}
    try:
        log.log(level, json.dumps(payload, default=str))
    except Exception:
        log.log(level, "%s %r", event, fields)


# ---------- Guardrail -------------------------------------------------------


class StraikerGuardrail(CustomGuardrail):
    """Straiker DefendAI guardrail integrated with LiteLLM proxy."""

    # Transient HTTP status codes that warrant a retry
    _RETRY_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
        detect_url: str = "https://api.prod.straiker.ai/api/v1/detect",
        agentic: bool = False,
        source: str = "litellm",
        destination: str = "api.openai.com",
        threshold: float = 0.5,
        timeout: float = 5.0,
        # Fault tolerance
        fail_open: Optional[bool] = None,
        unreachable_fallback: Optional[str] = None,
        max_retries: int = 2,
        initial_backoff: float = 0.1,
        max_backoff: float = 2.0,
        # Per-route opt-out
        enabled_models: Optional[list[str]] = None,
        skip_models: Optional[list[str]] = None,
        # Payload guard
        max_payload_bytes: int = 524_288,  # 512 KB
        # Custom headers (multi-tenant signing, forwarded-for, etc.)
        custom_headers: Optional[dict[str, str]] = None,
        # Verbose detection response (operator-tunable).
        # True  → sets Straiker-Debug: TRUE header so /detect returns the full
        #         envelope: per-category scores, score_block, score_detect,
        #         and the disabled-controls set. Block responses surface
        #         triggered_categories and the full debug payload in
        #         HTTPException.detail — actionable for SIEM / ticketing.
        # False → minimal response: just score + turn_id. Smaller payloads,
        #         marginally less Straiker work, no per-category visibility.
        # Default True because most operators want the detail for incident
        # triage; flip to False in cost-sensitive paths.
        verbose: bool = True,
        # Agentic dedup behavior
        # True  = score only iteration 1 user turn + final assistant turn
        #         (1 pre + 1 post Console entry per logical user prompt).
        # False = "inter-call" mode: score EVERY model call in the agent loop,
        #         including intermediate tool-call iterations. Use when you
        #         want fine-grained per-iteration visibility in Straiker.
        dedup_iterations: bool = True,
        **kwargs,
    ):
        self.api_key = api_key or os.environ.get("STRAIKER_API_KEY", "")
        self.detect_url = detect_url
        self.agentic = bool(agentic)
        self.source = source
        self.destination = destination
        self.threshold = float(threshold)
        self.timeout = float(timeout)

        # Resolve fail mode: explicit unreachable_fallback wins, then legacy
        # fail_open flag, default fail_closed.
        # Canonical values match LiteLLM's LitellmParams schema: "fail_open"
        # and "fail_closed". "allow"/"block" accepted as friendly aliases.
        if unreachable_fallback is not None:
            ufb = str(unreachable_fallback).lower()
            aliases = {"allow": "fail_open", "block": "fail_closed"}
            ufb = aliases.get(ufb, ufb)
            if ufb not in ("fail_open", "fail_closed"):
                raise ValueError(
                    "unreachable_fallback must be 'fail_open' or 'fail_closed' "
                    f"(or alias 'allow'/'block'); got {unreachable_fallback!r}"
                )
            self.unreachable_fallback = ufb
        elif fail_open is True:
            self.unreachable_fallback = "fail_open"
        else:
            self.unreachable_fallback = "fail_closed"

        self.max_retries = max(0, int(max_retries))
        self.initial_backoff = max(0.0, float(initial_backoff))
        self.max_backoff = max(self.initial_backoff, float(max_backoff))

        self.enabled_models = list(enabled_models) if enabled_models else []
        self.skip_models = list(skip_models) if skip_models else []
        self.max_payload_bytes = int(max_payload_bytes)
        self.custom_headers = dict(custom_headers) if custom_headers else {}
        self.verbose = bool(verbose)
        self.dedup_iterations = bool(dedup_iterations)

        super().__init__(**kwargs)

        if not self.api_key:
            log.warning(
                "[straiker] STRAIKER_API_KEY not configured; guardrail will fail "
                "every call. Set api_key in litellm_params or STRAIKER_API_KEY env."
            )

    # ---------- URL / payload ----------------------------------------------

    def _detect_url(self) -> str:
        if not self.agentic:
            return self.detect_url
        sep = "&" if "?" in self.detect_url else "?"
        return f"{self.detect_url}{sep}agentic"

    def _should_skip_for_model(self, model: Optional[str]) -> Optional[str]:
        """Return reason string if this model should bypass the guardrail, else None."""
        if not model:
            return None
        if self.enabled_models and not _wildcard_match(self.enabled_models, model):
            return "model not in enabled_models"
        if _wildcard_match(self.skip_models, model):
            return "model in skip_models"
        return None

    def _build_payload(self, *, messages: list[dict], app_response: str,
                       data: dict, hook: str) -> dict:
        meta = data.get("metadata") or {}
        net = meta.get("network") or {}
        model = data.get("model", "unknown") or "unknown"

        prompt = _last_user_prompt(messages)

        session_id = (
            meta.get("session_id")
            or meta.get("requester_metadata", {}).get("session_id")
            or "litellm-session"
        )
        user_name = meta.get("user_name") or data.get("user") or "litellm"
        user_role = meta.get("user_role") or "public"
        trace_id = meta.get("trace_id")
        agent_role = meta.get("agent_role")
        ip = net.get("IP") or "127.0.0.1"  # MUST be IPv4/IPv6, never "N/A"
        ua = net.get("User-Agent") or "litellm-proxy"

        network = {"IP": ip, "User-Agent": ua, "Content-Type": "application/json"}
        # Hook tag combines source platform + hook point so a single Console
        # filter ("litellm/pre_call") instantly identifies which LiteLLM hook
        # generated the entry, when one Straiker app receives traffic from
        # multiple gateways or multiple hook registrations.
        hook_tag = f"litellm/{hook}"
        metadata = {
            "session_id": session_id, "user_name": user_name, "user_role": user_role,
            "remote_ip": ip, "app_name": self.source, "source": "litellm",
            "litellm_hook": hook,              # "pre_call" | "moderation" | "post_call"
            "integration": "litellm-straiker", # version-ready tag for the integration
            "hook_tag": hook_tag,              # "litellm/pre_call" etc.
            "trace_id": trace_id, "agent_role": agent_role,
        }
        annotations = {
            "source": "litellm", "model": model, "hook": hook,
            "litellm_hook": hook,
            "integration": "litellm-straiker",
            "hook_tag": hook_tag,
            "trace_id": trace_id, "agent_role": agent_role,
        }

        if self.agentic:
            return {
                "source": self.source,
                "destination": self.destination,
                "messages": _build_agentic_messages(messages, app_response),
                "session_id": session_id, "user_name": user_name, "user_role": user_role,
                "metadata": metadata, "network": network, "annotations": annotations,
            }
        return {
            "prompt": prompt,
            "app_response": app_response or "N/A",
            "rag_content": "N/A",
            "session_id": session_id, "user_name": user_name, "user_role": user_role,
            "metadata": metadata, "network": network, "annotations": annotations,
        }

    # ---------- HTTP / retry -----------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.verbose:
            headers["Straiker-Debug"] = "TRUE"
        for k, v in self.custom_headers.items():
            # Never let custom_headers override Authorization.
            if k.lower() == "authorization":
                continue
            headers[k] = v
        return headers

    @staticmethod
    def _triggered_categories(debug_envelope: Optional[dict]) -> list[str]:
        """Extract names of controls that scored > 0 in the block category.

        Straiker-Debug response shape:
            {"debug": {"detections": {"block": {"<name>": 1, ...},
                                       "detect": {...},
                                       "disabled": {...}}}}
        """
        if not isinstance(debug_envelope, dict):
            return []
        detections = debug_envelope.get("detections") or {}
        block = detections.get("block") or {}
        if not isinstance(block, dict):
            return []
        return sorted(
            name for name, score in block.items()
            if isinstance(score, (int, float)) and score > 0
        )

    async def _call_straiker(
        self, payload: dict
    ) -> tuple[Optional[float], Optional[str], Optional[str], Optional["DetectResponse"]]:
        """Call Straiker /detect with retry + jitter.

        Returns (score, turn_id, error, parsed_response).
          - On success: (score, turn_id, None, parsed). parsed has .debug
            populated when verbose=True so callers can surface
            triggered_categories.
          - On terminal failure: (None, None, error_msg, None). Retries on
            transient HTTP codes and network errors up to ``max_retries``
            attempts.
        """
        body_bytes = 0
        try:
            body_bytes = len(json.dumps(payload, default=str).encode("utf-8"))
        except Exception:
            pass
        if body_bytes > self.max_payload_bytes:
            return None, None, (
                f"payload {body_bytes}B exceeds max_payload_bytes "
                f"{self.max_payload_bytes}; skipping guardrail call"
            ), None

        url = self._detect_url()
        headers = self._build_headers()
        last_err: Optional[str] = None
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            t0 = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=self.timeout, verify=True) as c:
                    resp = await c.post(url, json=payload, headers=headers)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if resp.status_code == 200:
                    try:
                        parsed = DetectResponse.model_validate(resp.json())
                    except (ValidationError, json.JSONDecodeError) as ve:
                        return None, None, f"invalid response schema: {ve}", None
                    return parsed.score, parsed.turn_id, None, parsed
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in self._RETRY_STATUS:
                    return None, None, last_err, None
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                last_err = f"{type(e).__name__}: {e}"
            except (json.JSONDecodeError, ValueError) as e:
                return None, None, f"{type(e).__name__}: {e}", None

            if attempt < attempts - 1:
                backoff = min(
                    self.initial_backoff * (2 ** attempt),
                    self.max_backoff,
                )
                # Full jitter
                sleep_s = random.uniform(0, backoff)
                _structured_log(
                    logging.DEBUG,
                    "straiker.retry",
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    error=last_err,
                    sleep_s=round(sleep_s, 3),
                    elapsed_ms=elapsed_ms,
                )
                await asyncio.sleep(sleep_s)

        return None, None, last_err or "unknown error", None

    # ---------- LiteLLM hooks ----------------------------------------------

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        msgs = data.get("messages") or []
        prompt = _last_user_prompt(msgs)
        if not prompt:
            return data

        model = data.get("model")
        skip_reason = self._should_skip_for_model(model)
        if skip_reason:
            _structured_log(
                logging.DEBUG,
                "straiker.skip",
                hook="pre_call",
                model=model,
                reason=skip_reason,
            )
            return data

        # Agentic dedup: only score iteration 1 (last role == user).
        # When dedup_iterations=False ("inter-call" mode), score every model
        # call including intermediate tool-call iterations — useful when you
        # want a Straiker entry per LLM round-trip rather than per logical
        # user prompt. Default True for parity with Kong/APIM behavior.
        if self.agentic and self.dedup_iterations:
            last = msgs[-1] if msgs else {}
            last_role = last.get("role") if isinstance(last, dict) else None
            if last_role in ("tool", "assistant"):
                return data
            if last_role != "user":
                return data

        payload = self._build_payload(messages=msgs, app_response="N/A",
                                      data=data, hook="pre_call")
        t0 = time.monotonic()
        score, turn_id, err, parsed = await self._call_straiker(payload)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if err is not None:
            _structured_log(
                logging.ERROR,
                "straiker.error",
                hook="pre_call",
                error=err,
                fallback=self.unreachable_fallback,
                elapsed_ms=elapsed_ms,
            )
            if self.unreachable_fallback == "fail_open":
                return data
            raise HTTPException(status_code=503, detail={
                "error": {
                    "message": f"Straiker unavailable: {err}",
                    "code": "503",
                    "x-straiker-verdict": "error",
                }
            })

        verdict = "block" if (score is not None and score > self.threshold) else "allow"
        debug_envelope = getattr(parsed, "debug", None) if parsed is not None else None
        triggered = self._triggered_categories(debug_envelope)
        _structured_log(
            logging.INFO,
            "straiker.score",
            hook="pre_call",
            score=score,
            turn_id=turn_id,
            verdict=verdict,
            execution_ms=elapsed_ms,
            model=model,
            triggered_categories=triggered or None,
        )

        if verdict == "block":
            detail = {
                "message": "Straiker: threat detected (pre-call)",
                "score": score,
                "turn_id": turn_id,
                "code": "403",
                "x-straiker-score": score,
                "x-straiker-turn-id": turn_id,
                "x-straiker-verdict": "block",
            }
            if self.verbose:
                detail["x-straiker-triggered-categories"] = triggered
                detail["straiker_debug"] = debug_envelope
            raise HTTPException(status_code=403, detail={"error": detail})
        return data

    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        """Run in parallel with the LLM call (mode: during_call).

        Same scoring semantics as pre_call but doesn't block the request from
        starting. Useful for streaming where pre_call would delay TTFC and
        post_call can't block mid-stream.
        """
        msgs = data.get("messages") or []
        if not _last_user_prompt(msgs):
            return data

        model = data.get("model")
        skip_reason = self._should_skip_for_model(model)
        if skip_reason:
            _structured_log(
                logging.DEBUG,
                "straiker.skip",
                hook="moderation",
                model=model,
                reason=skip_reason,
            )
            return data

        if self.agentic and self.dedup_iterations:
            last = msgs[-1] if msgs else {}
            last_role = last.get("role") if isinstance(last, dict) else None
            if last_role in ("tool", "assistant"):
                return data
            if last_role != "user":
                return data

        payload = self._build_payload(messages=msgs, app_response="N/A",
                                      data=data, hook="moderation")
        t0 = time.monotonic()
        score, turn_id, err, parsed = await self._call_straiker(payload)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if err is not None:
            _structured_log(
                logging.ERROR,
                "straiker.error",
                hook="moderation",
                error=err,
                fallback=self.unreachable_fallback,
                elapsed_ms=elapsed_ms,
            )
            if self.unreachable_fallback == "fail_open":
                return data
            raise HTTPException(status_code=503, detail={
                "error": {
                    "message": f"Straiker unavailable: {err}",
                    "code": "503",
                    "x-straiker-verdict": "error",
                }
            })

        verdict = "block" if (score is not None and score > self.threshold) else "allow"
        debug_envelope = getattr(parsed, "debug", None) if parsed is not None else None
        triggered = self._triggered_categories(debug_envelope)
        _structured_log(
            logging.INFO,
            "straiker.score",
            hook="moderation",
            score=score,
            turn_id=turn_id,
            verdict=verdict,
            execution_ms=elapsed_ms,
            model=model,
            triggered_categories=triggered or None,
        )

        if verdict == "block":
            detail = {
                "message": "Straiker: threat detected (during-call)",
                "score": score,
                "turn_id": turn_id,
                "code": "403",
                "x-straiker-score": score,
                "x-straiker-turn-id": turn_id,
                "x-straiker-verdict": "block",
            }
            if self.verbose:
                detail["x-straiker-triggered-categories"] = triggered
                detail["straiker_debug"] = debug_envelope
            raise HTTPException(status_code=403, detail={"error": detail})
        return data

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        msgs = list(data.get("messages") or [])
        if not _last_user_prompt(msgs):
            return response

        model = data.get("model")
        skip_reason = self._should_skip_for_model(model)
        if skip_reason:
            _structured_log(
                logging.DEBUG,
                "straiker.skip",
                hook="post_call",
                model=model,
                reason=skip_reason,
            )
            return response

        choice = getattr(response, "choices", [None])[0]
        msg = getattr(choice, "message", None) if choice is not None else None
        app_response = (getattr(msg, "content", None) or "") if msg is not None else ""
        tool_calls = getattr(msg, "tool_calls", None) if msg is not None else None

        # Agentic dedup: skip post on intermediate tool-calling iterations.
        # Strengthened: only skip if tool_calls list is non-empty AND has
        # meaningful function entries; empty arrays still get post-scored.
        # When dedup_iterations=False ("inter-call" mode) every iteration is
        # scored, including intermediate tool-call rounds.
        if (
            self.agentic
            and self.dedup_iterations
            and _has_meaningful_tool_calls(tool_calls)
        ):
            return response
        if not app_response and not _has_meaningful_tool_calls(tool_calls):
            # In inter-call mode an iteration with tool_calls but no content
            # still represents a real LLM round-trip worth scoring; only skip
            # when both content AND tool_calls are absent.
            return response

        # Append the assistant's message so the post-call payload carries the
        # full conversation (including any tool calls/results in `msgs` already).
        msgs.append({"role": "assistant", "content": app_response})

        payload = self._build_payload(messages=msgs, app_response=app_response,
                                      data=data, hook="post_call")
        t0 = time.monotonic()
        score, turn_id, err, parsed = await self._call_straiker(payload)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if err is not None:
            _structured_log(
                logging.ERROR,
                "straiker.error",
                hook="post_call",
                error=err,
                fallback=self.unreachable_fallback,
                elapsed_ms=elapsed_ms,
            )
        else:
            verdict = "block" if (score is not None and score > self.threshold) else "allow"
            debug_envelope = getattr(parsed, "debug", None) if parsed is not None else None
            triggered = self._triggered_categories(debug_envelope)
            _structured_log(
                logging.INFO,
                "straiker.score",
                hook="post_call",
                score=score,
                turn_id=turn_id,
                verdict=verdict,
                execution_ms=elapsed_ms,
                model=model,
                triggered_categories=triggered or None,
            )
            # Attach to response.litellm_metadata if available so downstream
            # callbacks (Langfuse, Datadog, etc) can surface the verdict.
            try:
                if hasattr(response, "_hidden_params") and isinstance(
                    response._hidden_params, dict
                ):
                    payload = {
                        "score": score,
                        "turn_id": turn_id,
                        "verdict": verdict,
                    }
                    if self.verbose:
                        payload["triggered_categories"] = triggered
                        payload["straiker_debug"] = debug_envelope
                    response._hidden_params.setdefault("straiker", {}).update(payload)
            except Exception:
                pass
        return response
