from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.exceptions import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.straiker.straiker import (
    StraikerGuardrail,
    _has_meaningful_tool_calls,
    _last_user_prompt,
    _resolve_unreachable_fallback,
    _wildcard_match,
)
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    StraikerGuardrailConfigModel,
)


def test_straiker_in_initializer_registry():
    assert "straiker" in guardrail_initializer_registry


def test_straiker_in_class_registry():
    assert "straiker" in guardrail_class_registry
    assert guardrail_class_registry["straiker"] is StraikerGuardrail


def test_ui_friendly_name():
    assert StraikerGuardrailConfigModel.ui_friendly_name() == "Straiker"
    assert StraikerGuardrail.get_config_model() is StraikerGuardrailConfigModel


@pytest.fixture
def pre_call_guardrail():
    return StraikerGuardrail(
        api_key="test-key",
        api_base="https://test.straiker.ai/api/v1/detect",
        threshold=0.5,
        max_retries=0,
        guardrail_name="straiker-pre",
        event_hook="pre_call",
    )


def _mock_response(score: float, debug: dict = None) -> httpx.Response:
    body = {"score": score, "turnId": "test-turn-id"}
    if debug is not None:
        body["debug"] = debug
    return httpx.Response(status_code=200, json=body)


@pytest.mark.asyncio
async def test_pre_call_blocks_when_score_above_threshold(pre_call_guardrail):
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "test prompt"}],
    }
    with patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(
            return_value=_mock_response(
                0.9, debug={"detections": {"block": {"prompt_injection": 1}}}
            )
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await pre_call_guardrail.async_pre_call_hook(
                user_api_key_dict=None, cache=None, data=data, call_type="completion"
            )
    assert exc.value.status_code == 403
    err = exc.value.detail["error"]
    assert err["x-straiker-verdict"] == "block"
    assert err["x-straiker-score"] == 0.9
    assert err["x-straiker-triggered-categories"] == ["prompt_injection"]


@pytest.mark.asyncio
async def test_pre_call_allows_when_score_below_threshold(pre_call_guardrail):
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "benign"}],
    }
    with patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(return_value=_mock_response(0.1)),
    ):
        result = await pre_call_guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    assert result is data


@pytest.mark.asyncio
async def test_pre_call_fail_open_returns_data_when_straiker_unreachable():
    g = StraikerGuardrail(
        api_key="test-key",
        unreachable_fallback="fail_open",
        max_retries=0,
        guardrail_name="straiker-pre",
        event_hook="pre_call",
    )
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}
    with patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectError("boom")),
    ):
        result = await g.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    assert result is data


@pytest.mark.asyncio
async def test_pre_call_fail_closed_raises_503_when_straiker_unreachable():
    g = StraikerGuardrail(
        api_key="test-key",
        unreachable_fallback="fail_closed",
        max_retries=0,
        guardrail_name="straiker-pre",
        event_hook="pre_call",
    )
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}
    with patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectError("boom")),
    ):
        with pytest.raises(HTTPException) as exc:
            await g.async_pre_call_hook(
                user_api_key_dict=None, cache=None, data=data, call_type="completion"
            )
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["x-straiker-verdict"] == "error"


@pytest.mark.asyncio
async def test_skip_models_bypasses_guardrail_without_calling_straiker(
    pre_call_guardrail,
):
    pre_call_guardrail.skip_models = ["cursor-*"]
    data = {"model": "cursor-claude", "messages": [{"role": "user", "content": "x"}]}
    mock_post = AsyncMock()
    with patch.object(httpx.AsyncClient, "post", new=mock_post):
        result = await pre_call_guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    assert result is data
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_agentic_dedup_skips_pre_when_last_role_is_tool():
    g = StraikerGuardrail(
        api_key="test-key",
        agentic=True,
        dedup_iterations=True,
        max_retries=0,
        guardrail_name="straiker-pre",
        event_hook="pre_call",
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "tool result"},
        ],
    }
    mock_post = AsyncMock()
    with patch.object(httpx.AsyncClient, "post", new=mock_post):
        result = await g.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    assert result is data
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_payload_size_guard_skips_oversized():
    g = StraikerGuardrail(
        api_key="test-key",
        max_payload_bytes=100,
        max_retries=0,
        guardrail_name="straiker-pre",
        event_hook="pre_call",
    )
    huge_content = "x" * 1000
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": huge_content}],
    }
    mock_post = AsyncMock()
    with patch.object(httpx.AsyncClient, "post", new=mock_post):
        result = await g.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    assert result is data
    mock_post.assert_not_called()


def test_resolve_unreachable_fallback_accepts_aliases():
    assert _resolve_unreachable_fallback("allow", None) == "fail_open"
    assert _resolve_unreachable_fallback("block", None) == "fail_closed"
    assert _resolve_unreachable_fallback("fail_open", None) == "fail_open"
    assert _resolve_unreachable_fallback("fail_closed", None) == "fail_closed"
    assert _resolve_unreachable_fallback(None, True) == "fail_open"
    assert _resolve_unreachable_fallback(None, False) == "fail_closed"
    assert _resolve_unreachable_fallback(None, None) == "fail_closed"


def test_resolve_unreachable_fallback_rejects_invalid():
    with pytest.raises(ValueError):
        _resolve_unreachable_fallback("invalid", None)


def test_wildcard_match_supports_globs():
    assert _wildcard_match(["cursor-*"], "cursor-claude") is True
    assert _wildcard_match(["cursor-*"], "gpt-4o-mini") is False
    assert _wildcard_match(["claude-?-coding"], "claude-3-coding") is True
    assert _wildcard_match(["exact"], "exact") is True
    assert _wildcard_match([], "anything") is False
    assert _wildcard_match(["something"], "") is False


def test_last_user_prompt_returns_most_recent_user_turn():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "second"},
    ]
    assert _last_user_prompt(msgs) == "second"


def test_last_user_prompt_handles_multimodal_content():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert _last_user_prompt(msgs) == "hello"


def test_last_user_prompt_returns_empty_when_no_user_turn():
    assert _last_user_prompt([{"role": "assistant", "content": "x"}]) == ""
    assert _last_user_prompt([]) == ""


def test_has_meaningful_tool_calls_requires_function_name():
    assert _has_meaningful_tool_calls([{"function": {"name": "get_weather"}}]) is True
    assert _has_meaningful_tool_calls([{"name": "get_weather"}]) is True
    assert _has_meaningful_tool_calls([{"function": {}}]) is False
    assert _has_meaningful_tool_calls([]) is False
    assert _has_meaningful_tool_calls(None) is False
