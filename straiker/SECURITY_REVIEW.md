# Security Review Notes — Straiker LiteLLM Guardrail

**Status:** Internal review checklist completed by the integration author. Awaiting 2nd-reviewer sign-off before upstream submission.
**Scope:** `straiker_guardrail.py` and its test suite (`tests/test_unit.py`, `tests/conftest.py`).

---

## Threats considered

| Threat | Mitigation | Verified |
|---|---|---|
| API key leakage in logs | API key never printed; only sent in `Authorization` header. Structured logs JSON-encode fields with `default=str` so non-serializable values are coerced safely | ✅ `TestSecurity::test_api_key_not_in_log_messages` |
| API key in `repr(StraikerGuardrail)` | Class inherits `object.__repr__`; no `__str__` override; no `__repr__` formatting that includes secrets | ✅ `TestSecurity::test_api_key_not_in_repr` |
| Header injection via `custom_headers` | `_build_headers` skips any custom header named `authorization` (case-insensitive). Bearer token cannot be overridden | ✅ `TestHeaders::test_custom_headers_cannot_override_authorization` |
| Log-line injection via user content | All logged values are JSON-encoded via `_structured_log`, which escapes newlines/quotes | ✅ implicit via JSON encoding |
| TLS downgrade / cert validation bypass | `httpx.AsyncClient(verify=True)` — explicit, line 480 | ✅ visual code review |
| SSRF via configurable `detect_url` | Operator-controlled config; not user-controlled. Documented constraint in README. Future hardening: enforce HTTPS scheme + allowlist hostnames | ✅ documented |
| Pydantic schema bypass | `DetectResponse` uses strict bounded `Field(ge=0.0, le=1.0)` on score. Invalid responses return error path rather than score=0 | ✅ `TestDetectResponse::test_score_clamped_validation` |
| Unbounded payload size | `max_payload_bytes` default 512KB. Payload larger than limit bypasses guardrail with structured-log warning | ✅ `TestCallStraiker::test_payload_size_guard_skips_call` |
| Retry storm on persistent failure | `max_retries` default 2 (3 total attempts). Exponential backoff with full jitter, ceiling at `max_backoff` | ✅ `TestCallStraiker::test_retry_exhausted_returns_error` |
| Side-channel info disclosure in 403 detail | `HTTPException(403).detail.error` includes Straiker score + turn_id — these are not secrets but are operator-visible metadata. Required for correlation. No system internals leaked | ✅ visual code review |
| Side-channel via timing diff | Score-based block vs allow path takes nearly identical time (single Straiker call, parsed identically); no leak | ✅ visual code review |
| Race condition on shared client | `httpx.AsyncClient` is created per-call inside `async with`. No shared mutable state between concurrent hooks | ✅ visual code review |
| Forced agentic dedup bypass via empty content | Strengthened dedup (`_has_meaningful_tool_calls`) treats empty `tool_calls: []` arrays as non-meaningful, so they still get scored. Prevents an attacker from crafting a response shape that escapes scoring | ✅ `TestPostCallHook::test_empty_tool_calls_array_still_scored` |
| Fail-open on Straiker outage by default | Default `unreachable_fallback=fail_closed`. Operator must explicitly opt into fail-open. Matches LiteLLM conventions | ✅ `TestInit::test_init_unreachable_fallback_defaults_to_fail_closed` |
| Post-call exception causing 5xx to client | Post-call swallows all `_call_straiker` errors and never raises. Response already delivered by the time post-call runs | ✅ `TestPostCallHook::test_post_call_never_raises_on_error` |
| Untrusted skip_models pattern | Glob patterns use `re.escape()` before substituting `*` and `?` placeholders. Cannot inject arbitrary regex | ✅ `TestHelpers::test_wildcard_match_*` |

## Code-level checks (visual review)

- [x] No `eval`, `exec`, `compile`, or dynamic import of user-supplied data
- [x] No `subprocess` / shell calls
- [x] No filesystem writes outside the venv
- [x] No raw SQL or database access
- [x] `os.environ.get` only used for default values, never to construct paths or commands
- [x] All HTTP calls go to operator-configured `detect_url`
- [x] No backdoor flags / debug toggles left enabled
- [x] No hardcoded credentials in source

## Dependencies

| Package | Version pin | Reason |
|---|---|---|
| litellm[proxy] | ≥1.55.0,<2.0.0 | Base proxy + `CustomGuardrail` API |
| httpx | ≥0.27.0,<1.0.0 | Async HTTP client |
| pydantic | ≥2.6.0,<3.0.0 | Schema validation |
| fastapi | ≥0.110.0,<1.0.0 | `HTTPException` |
| pytest, pytest-asyncio, respx | latest pinned | Tests only — not shipped to runtime |

No dependencies pulled from untrusted sources. All on PyPI.

## What we did NOT do

- **Mutation testing.** Vigil Guard (the reference integration) ran mutmut and achieved 100% mutation coverage. We have not run mutation testing on this diff. Tracked as a follow-up before upstream PR.
- **Fuzz testing of payload shapes.** Pydantic validation is the primary defense. No hypothesis-based fuzz harness yet.
- **External pentest.** Out of scope per integration author + Phimm's decision; internal eng review only.
- **SAST scan (Semgrep / Bandit).** Recommended for the upstream-port commit but not run for the local version.

## Reviewer sign-off (pending)

| Reviewer | Role | Sign-off | Date |
|---|---|---|---|
| _TBD_ | Detection backend lead | ☐ | |
| _TBD_ | Platform / SDK lead | ☐ | |

## Recommended pre-merge actions for upstream PR

1. Run mutmut and capture mutation-coverage score in PR description
2. Run Semgrep with default ruleset
3. Confirm `verify=True` survives the port to upstream layout
4. Confirm `custom_headers` Authorization-protection survives the port
5. Re-run `TestSecurity` suite against the upstream-shaped class
