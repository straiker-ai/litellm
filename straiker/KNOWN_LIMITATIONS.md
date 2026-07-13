# Known limitations & gotchas

## LiteLLM `custom_code` (inline YAML) does not receive `messages` in post_call

LiteLLM lets you define a guardrail inline in YAML via `guardrail: custom_code`. In `post_call` mode the inline code only receives the response — not the original request `messages[]`. Without `messages`, you can't extract the user prompt or build an agentic payload.

**Fix:** use the `CustomGuardrail` class registered as `guardrail: straiker_guardrail.StraikerGuardrail`, which is what this repo ships. Pre and post both receive `data` with the full request.

## `network.IP` must be a valid IPv4/IPv6 address

The `/api/v1/detect` schema validates `network.IP` as `IPvAnyAddress`. Sending `"N/A"` returns a 422. The class defaults to `"127.0.0.1"` when the IP isn't provided in `metadata.network`.

## Streaming responses can't be blocked at post-call

LiteLLM's `async_post_call_success_hook` runs after streaming chunks have been delivered to the client — it's audit-only for streaming. Pre-call still fires normally. If you need to block streaming, either:
- Use `mode: during_call` (`async_moderation_hook`) which runs in parallel with the LLM call and can raise to abort, OR
- Implement `async_post_call_streaming_iterator_hook` to filter chunks (not in this repo's v0.1).

## `mode` is per-registration, not per-class

LiteLLM determines the hook to run based on the `mode` field on each `guardrails:` entry, not on which methods the class implements. To run both pre and post, register the same class twice (see `config.yaml`).

## Don't add the class to `litellm_settings.callbacks`

Older LiteLLM examples register guardrails via `litellm_settings.callbacks`. Don't do that for this class — it'll cause double-registration. Use the `guardrails:` block alone.

## `default_on` must live under `litellm_params`

Putting `default_on: true` at the top level of a guardrail entry silently does nothing. It must be inside `litellm_params`.

## Pre-call latency is on the request critical path

Each pre-call adds one HTTPS round trip to Straiker (typical p50 ~80–150 ms). Tune `timeout` based on your latency budget; `fail_open: true` lets you survive Straiker outages without breaking customer requests.

## Post-call must never block

Set `fail_open: true` on the post-call registration. A post-call failure has no security value (the response has already been generated) and propagating it to the user creates spurious 5xx responses.

## `source` must match an existing agentic Straiker app

For `agentic: true`, the `source` value is used to look up the app in the Straiker Console. If the app doesn't exist (and auto-creation isn't configured), `/detect?agentic` returns an error. Create the app once in the Console before pointing traffic at it.
