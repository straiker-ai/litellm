# feat(guardrails): add Straiker DefendAI guardrail provider

## Summary

Adds Straiker DefendAI as a native Partner Guardrail in the LiteLLM proxy. Straiker is a runtime AI security platform purpose-built for agentic workloads — multi-turn conversation scoring, tool-call inspection, and PII / prompt-injection / LLM-evasion blocking with published benchmarks (98.1% TPR, 0.70% FPR, <120ms p50).

## What's added

| File | Purpose |
|---|---|
| `litellm/proxy/guardrails/guardrail_hooks/straiker/__init__.py` | `initialize_guardrail()` + registry entries |
| `litellm/proxy/guardrails/guardrail_hooks/straiker/straiker.py` | `StraikerGuardrail(CustomGuardrail)` implementation |
| `litellm/types/proxy/guardrails/guardrail_hooks/straiker.py` | `StraikerGuardrailConfigModelOptionalParams` Pydantic schema |
| `litellm/types/guardrails.py` | one-line enum entry: `STRAIKER = "straiker"` |
| `tests/test_litellm/proxy/guardrails/guardrail_hooks/test_straiker.py` | 96 offline unit tests via `respx` |
| `tests/test_litellm/proxy/guardrails/guardrail_hooks/conftest.py` | `MockStraikerServer` fixture + helpers |
| `docs/my-website/docs/proxy/guardrails/straiker.md` | docs page |

## Capabilities

- Three hook modes implemented: `pre_call`, `during_call` (moderation, streaming-safe), `post_call` (observability)
- Agentic mode (`agentic: true`) posts full `messages[]` including `tool_calls` to `/api/v1/detect?agentic`
- Agent-loop dedup: 1 pre + 1 post Straiker Console entry per logical user prompt regardless of round-trip count
- Operator-tunable `dedup_iterations: false` for inter-call mode (score every LLM round-trip)
- Pydantic schema validation on Straiker responses
- Retry with exponential backoff + full jitter on transient failures
- `unreachable_fallback: fail_open | fail_closed` aligns with LiteLLM's `LitellmParams` literal type
- Per-route opt-out via `enabled_models` / `skip_models` (glob)
- Payload size guard (`max_payload_bytes`, default 512KB) for long-context workloads
- Custom headers passthrough (Authorization protected from override)
- `verbose` mode: sets `Straiker-Debug: TRUE` header, surfaces `triggered_categories` + full per-category detection envelope in block responses

## Block response shape

```json
{
  "error": {
    "message": "Straiker: threat detected (pre-call)",
    "score": 1.0,
    "turn_id": "pf-...",
    "code": "403",
    "x-straiker-score": 1.0,
    "x-straiker-turn-id": "pf-...",
    "x-straiker-verdict": "block",
    "x-straiker-triggered-categories": ["llm_evasion"],
    "straiker_debug": { ...full per-category envelope... }
  }
}
```

## Local testing

```bash
make format && make lint && make test-unit
```

96 unit tests pass. Offline-only — uses `respx` to mock the Straiker /detect endpoint; no real Straiker API key required for CI.

### Out-of-tree validation prior to this PR

- 1500+ live requests through `litellm-straiker-agentic` reference stack
- 5/5 block-mode end-to-end tests (email, credit card, 2× llm_evasion, benign control)
- Agentic tool-call dedup verified in the Straiker Defend Console with `web_search` / `web_fetch` / `rag_search` / `calculate` tool sequences

## Checklist

- [x] Scoped change (guardrail only, no unrelated edits)
- [x] At least 1 test added (96 added)
- [x] `make test-unit` passes locally
- [x] `make lint` passes locally
- [x] PR title follows `feat(guardrails): add <Provider> guardrail provider` convention
- [x] No secrets in source / test fixtures
- [x] CLA signed

## Reviewers

cc: @krrishdholakia @ishaan-jaffer
