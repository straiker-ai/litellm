"""
A guardrail that scans a live stream once, at its end, can ask for tool calls to be held until that
verdict (``streaming_hold_tool_calls``). Text still streams live. From the first item that starts a
tool call the client would run, everything is held, then released on a pass or dropped on a block.
"""

import json
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import UnifiedLLMGuardrails
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    GenericGuardrailAPIInputs,
    ModelResponseStream,
    StreamingChoices,
)

BLOCK_MESSAGE: Final = "This tool call was blocked by policy."
TOOL_NAME: Final = "run_shell"
TOOL_ARGUMENTS: Final = '{"command": "rm -rf /"}'


def _chat_chunk(delta: Delta, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-live",
        created=1724900000,
        model="gpt-5.4-mini",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)],
    )


def _chat_tool_delta(name: str | None, arguments: str) -> Delta:
    call: Final = ChatCompletionDeltaToolCall(
        id="call_1" if name else None, type="function", index=0, function=Function(name=name, arguments=arguments)
    )
    return Delta(tool_calls=[call])


def _chat_items(with_tool_call: bool) -> tuple[object, ...]:
    text: Final = (
        _chat_chunk(Delta(role="assistant", content="Let me ")),
        _chat_chunk(Delta(content="clean that up.")),
    )
    if not with_tool_call:
        return (*text, _chat_chunk(Delta(), finish_reason="stop"))
    return (
        *text,
        _chat_chunk(_chat_tool_delta(TOOL_NAME, "")),
        _chat_chunk(_chat_tool_delta(None, TOOL_ARGUMENTS)),
        _chat_chunk(Delta(), finish_reason="tool_calls"),
    )


def _sse(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _messages_items(with_tool_call: bool) -> tuple[object, ...]:
    message: Final = {"id": "msg_live", "type": "message", "role": "assistant", "model": "claude", "content": []}
    text: Final = (
        _sse("message_start", {"type": "message_start", "message": message}),
        _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Let me clean that up."},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
    )
    stop_reason: Final = "tool_use" if with_tool_call else "end_turn"
    end: Final = (
        _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason}}),
        _sse("message_stop", {"type": "message_stop"}),
    )
    if not with_tool_call:
        return (*text, *end)
    tool_block: Final = {"type": "tool_use", "id": "toolu_1", "name": TOOL_NAME, "input": {}}
    return (
        *text,
        _sse("content_block_start", {"type": "content_block_start", "index": 1, "content_block": tool_block}),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": TOOL_ARGUMENTS},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 1}),
        *end,
    )


def _responses_items(with_tool_call: bool) -> tuple[object, ...]:
    envelope: Final = {"id": "resp_live", "model": "gpt-5.4-mini", "status": "in_progress", "output": []}
    message_item: Final = {"id": "msg_1", "type": "message", "role": "assistant", "content": []}
    text: Final = (
        {"type": "response.created", "response": envelope},
        {"type": "response.output_item.added", "output_index": 0, "item": message_item},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Let me clean that up.",
        },
    )
    done_message: Final = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Let me clean that up.", "annotations": []}],
    }
    call_item: Final = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": TOOL_NAME,
        "arguments": TOOL_ARGUMENTS,
        "status": "completed",
    }
    output: Final = [done_message, call_item] if with_tool_call else [done_message]
    completed: Final = {
        "type": "response.completed",
        "response": {"id": "resp_live", "model": "gpt-5.4-mini", "status": "completed", "output": output},
    }
    if not with_tool_call:
        return (*text, completed)
    return (
        *text,
        {"type": "response.output_item.added", "output_index": 1, "item": {**call_item, "arguments": ""}},
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 1,
            "delta": TOOL_ARGUMENTS,
        },
        {"type": "response.output_item.done", "output_index": 1, "item": call_item},
        completed,
    )


ROUTES: Final[dict[str, tuple[str, Callable[[bool], tuple[object, ...]], int]]] = {
    # route, the stream it relays, and how many leading items are text the client may see live
    "chat": ("/v1/chat/completions", _chat_items, 2),
    "messages": ("/v1/messages", _messages_items, 4),
    "responses": ("/v1/responses", _responses_items, 3),
}


class _RecordingGuardrail(CustomGuardrail):
    """Records what the client had already received when the end-of-stream scan ran."""

    def __init__(self, released: Sequence[object], blocks: bool) -> None:
        super().__init__(guardrail_name="hold-tool-calls-test", event_hook="post_call")
        self._released: Final = released
        self._blocks: Final = blocks
        self.released_at_scan: tuple[object, ...] | None = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: object | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.released_at_scan = tuple(self._released)
        if self._blocks:
            raise ModifyResponseException(
                message=BLOCK_MESSAGE, model="m", request_data=request_data, guardrail_name=self.guardrail_name
            )
        return inputs


async def _as_stream(items: Sequence[object]) -> AsyncGenerator[object, None]:
    for item in items:
        yield item


async def _relay(
    endpoint: str, *, with_tool_call: bool, hold_tool_calls: bool, blocks: bool
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    """The stream's items, what the client had at scan time, and everything the client received."""
    route, build, _ = ROUTES[endpoint]
    items: Final = build(with_tool_call)
    released: Final[list[object]] = []
    guardrail: Final = _RecordingGuardrail(released, blocks)
    guardrail.streaming_end_of_stream_only = True
    guardrail.streaming_buffer_until_moderated = False
    guardrail.streaming_hold_tool_calls = hold_tool_calls
    request_data: Final = {
        "messages": [{"role": "user", "content": "clean up"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    async for chunk in UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="test", request_route=route),
        response=_as_stream(items),
        request_data=request_data,
    ):
        released.append(chunk)
    assert guardrail.released_at_scan is not None, "the end-of-stream scan never ran"
    return items, guardrail.released_at_scan, tuple(released)


def _mentions_tool_call(chunks: Sequence[object]) -> bool:
    raw: Final = "".join(
        chunk.decode() if isinstance(chunk, bytes) else json.dumps(chunk, default=str) for chunk in chunks
    )
    return TOOL_NAME in raw or "rm -rf" in raw


@pytest.mark.parametrize("endpoint", ["chat", "messages", "responses"])
@pytest.mark.asyncio
async def test_a_tool_call_is_held_until_the_verdict_while_the_text_before_it_streams_live(endpoint: str) -> None:
    items, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=True, blocks=False)
    text_items: Final = ROUTES[endpoint][2]
    assert at_scan == items[:text_items]
    assert received == items


@pytest.mark.parametrize("endpoint", ["chat", "messages", "responses"])
@pytest.mark.asyncio
async def test_a_blocked_tool_call_never_reaches_the_client(endpoint: str) -> None:
    items, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=True, blocks=True)
    text_items: Final = ROUTES[endpoint][2]
    assert at_scan == items[:text_items]
    assert received[:text_items] == items[:text_items]
    assert not _mentions_tool_call(received)
    assert BLOCK_MESSAGE in "".join(
        chunk.decode() if isinstance(chunk, bytes) else json.dumps(chunk, default=str) for chunk in received
    )


@pytest.mark.parametrize("endpoint", ["chat", "messages", "responses"])
@pytest.mark.asyncio
async def test_without_the_flag_a_tool_call_streams_before_the_verdict(endpoint: str) -> None:
    items, at_scan, received = await _relay(endpoint, with_tool_call=True, hold_tool_calls=False, blocks=False)
    assert _mentions_tool_call(at_scan)
    assert received == items


@pytest.mark.parametrize("endpoint", ["chat", "messages", "responses"])
@pytest.mark.asyncio
async def test_a_text_only_stream_is_released_the_same_with_and_without_the_flag(endpoint: str) -> None:
    items, held_at_scan, held_received = await _relay(
        endpoint, with_tool_call=False, hold_tool_calls=True, blocks=False
    )
    _, live_at_scan, live_received = await _relay(endpoint, with_tool_call=False, hold_tool_calls=False, blocks=False)
    assert held_at_scan == live_at_scan
    assert held_received == live_received == items
