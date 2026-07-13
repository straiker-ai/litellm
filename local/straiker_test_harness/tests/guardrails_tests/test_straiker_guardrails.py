from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from starlette.exceptions import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.straiker.straiker import (
    StraikerGuardrail,
    _has_meaningful_tool_calls,
    _last_user_prompt,
    _resolve_session_id,
    _resolve_user_name,
    _wildcard_match,
)
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    StraikerGuardrailConfigModel,
)


def _mock_response(score: float, debug: dict = None) -> MagicMock:
    body = {"score": score, "turnId": "test-turn-id"}
    if debug is not None:
        body["debug"] = debug
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = body
    resp.text = ""
    return resp


def _make_guardrail(**overrides) -> StraikerGuardrail:
    defaults = {
        "api_key": "test-key",
        "api_base": "https://test.straiker.ai/api/v1/detect",
        "threshold": 0.5,
        "max_retries": 0,
        "guardrail_name": "straiker-pre",
        "event_hook": "pre_call",
        "async_handler": MagicMock(spec=httpx.AsyncClient),
    }
    defaults.update(overrides)
    g = StraikerGuardrail(**defaults)
    g.async_handler.post = AsyncMock()
    return g


def test_straiker_in_initializer_registry():
    assert "straiker" in guardrail_initializer_registry


def test_straiker_in_class_registry():
    assert "straiker" in guardrail_class_registry
    assert guardrail_class_registry["straiker"] is StraikerGuardrail


def test_ui_friendly_name():
    assert StraikerGuardrailConfigModel.ui_friendly_name() == "Straiker"
    assert StraikerGuardrail.get_config_model() is StraikerGuardrailConfigModel


def test_invalid_unreachable_fallback_rejected_at_init():
    with pytest.raises(ValueError):
        StraikerGuardrail(api_key="test", unreachable_fallback="invalid")


@pytest.mark.asyncio
async def test_pre_call_blocks_when_score_above_threshold():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(
        0.9, debug={"detections": {"block": {"prompt_injection": 1}}}
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "test prompt"}],
    }
    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    assert exc.value.status_code == 403
    err = exc.value.detail["error"]
    assert err["x-straiker-verdict"] == "block"
    assert err["x-straiker-score"] == 0.9
    assert err["x-straiker-triggered-categories"] == ["prompt_injection"]


@pytest.mark.asyncio
async def test_pre_call_allows_when_score_below_threshold():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "benign"}],
    }
    result = await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    assert result is data


@pytest.mark.asyncio
async def test_pre_call_fail_open_returns_data_when_straiker_unreachable():
    g = _make_guardrail(unreachable_fallback="fail_open")
    g.async_handler.post.side_effect = httpx.ConnectError("boom")
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}
    result = await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    assert result is data


@pytest.mark.asyncio
async def test_pre_call_fail_closed_raises_503_when_straiker_unreachable():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    g.async_handler.post.side_effect = httpx.ConnectError("boom")
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["x-straiker-verdict"] == "error"


@pytest.mark.asyncio
async def test_skip_models_bypasses_guardrail_without_calling_straiker():
    g = _make_guardrail(skip_models=["cursor-*"])
    data = {"model": "cursor-claude", "messages": [{"role": "user", "content": "x"}]}
    result = await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    assert result is data
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_agentic_dedup_skips_pre_when_last_role_is_tool():
    g = _make_guardrail(agentic=True, dedup_iterations=True)
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "tool result"},
        ],
    }
    result = await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    assert result is data
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_agentic_dedup_disabled_calls_straiker_on_tool_continuation():
    g = _make_guardrail(agentic=True, dedup_iterations=False)
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "tool result"},
        ],
    }
    await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    g.async_handler.post.assert_called_once()


@pytest.mark.asyncio
async def test_payload_size_guard_skips_oversized():
    g = _make_guardrail(max_payload_bytes=100)
    huge_content = "x" * 1000
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": huge_content}],
    }
    result = await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    assert result is data
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_sends_app_session_id_when_provided():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "x"}],
        "metadata": {"session_id": "my-app-session-42"},
    }
    await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    sent_payload = g.async_handler.post.call_args.kwargs["json"]
    assert sent_payload["session_id"] == "my-app-session-42"


@pytest.mark.asyncio
async def test_pre_call_falls_back_to_litellm_call_id_when_no_app_session():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "x"}],
        "litellm_call_id": "litellm-uuid-abc-123",
    }
    await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    sent_payload = g.async_handler.post.call_args.kwargs["json"]
    assert sent_payload["session_id"] == "litellm-uuid-abc-123"


@pytest.mark.asyncio
async def test_pre_call_falls_back_to_placeholder_when_no_session_id_anywhere():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}
    await g.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )
    sent_payload = g.async_handler.post.call_args.kwargs["json"]
    assert sent_payload["session_id"] == "litellm-session"


def test_resolve_session_id_precedence():
    assert _resolve_session_id({}, {"session_id": "from-meta"}) == "from-meta"
    assert (
        _resolve_session_id(
            {"litellm_call_id": "from-call"},
            {"requester_metadata": {"session_id": "from-requester"}},
        )
        == "from-requester"
    )
    assert (
        _resolve_session_id(
            {
                "litellm_metadata": {"session_id": "from-litellm-meta"},
                "litellm_call_id": "from-call",
            },
            {},
        )
        == "from-litellm-meta"
    )
    assert _resolve_session_id({"litellm_call_id": "from-call"}, {}) == "from-call"
    assert _resolve_session_id({}, {}) == "litellm-session"


def test_resolve_user_name_precedence():
    assert _resolve_user_name({}, {"user_name": "explicit"}) == "explicit"
    assert _resolve_user_name({"user": "from-data"}, {}) == "from-data"
    assert _resolve_user_name({}, {}) == "litellm"


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
