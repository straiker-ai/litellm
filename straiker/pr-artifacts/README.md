# litellm-straiker

Upstream-ready files for adding **Straiker DefendAI** as a native Partner Guardrail in [BerriAI/litellm](https://github.com/BerriAI/litellm).

The directory layout in this repo mirrors the relative paths these files will occupy inside the BerriAI/litellm tree. Submission is a copy-paste: clone BerriAI/litellm, drop these files into the matching paths, apply the one-line enum addition (see `ENUM_DIFF.md`), open a PR.

## What this gets us

Today the [Straiker LiteLLM integration](https://docs.straiker.ai/defend-ai/litellm-integration) works as a `CustomGuardrail` subclass loaded via YAML class-path registration. It's fully functional at runtime, but the LiteLLM admin UI surfaces almost nothing about it because LiteLLM's dashboard reads from a Pydantic schema that only natively-registered partners provide.

Landing these files upstream lights up the LiteLLM admin UI:

- **Provider dropdown** on "+ Add New Guardrail" includes Straiker
- **Settings tab** renders every `litellm_params` field as a labeled, type-aware form input
- **Test Playground** can target Straiker guardrails
- **Guardrails Monitor** populates `LiteLLM_DailyGuardrailMetrics` with per-Straiker counters

## File layout

```
litellm/
├── proxy/guardrails/guardrail_hooks/straiker/
│   ├── __init__.py          # initialize_guardrail() + registries
│   └── straiker.py          # StraikerGuardrail(CustomGuardrail)
└── types/
    └── proxy/guardrails/guardrail_hooks/
        └── straiker.py      # StraikerGuardrailConfigModelOptionalParams (Pydantic)

tests/
└── test_litellm/proxy/guardrails/guardrail_hooks/
    ├── conftest.py          # MockStraikerServer + pytest fixtures
    └── test_straiker.py     # 96 offline unit tests (mock via respx)

docs/
└── my-website/docs/proxy/guardrails/
    └── straiker.md          # docs page rendered at docs.litellm.ai

ENUM_DIFF.md                 # one-line addition to SupportedGuardrailIntegrations
PR_DESCRIPTION.md            # ready-to-paste PR description
```

## Submission steps

1. Fork `BerriAI/litellm`
2. Copy the four directories above into the matching paths in the fork
3. Apply the one-line enum addition from `ENUM_DIFF.md` to `litellm/types/guardrails.py`
4. Sign the [CLA](https://cla-assistant.io/BerriAI/litellm)
5. `make format && make lint && make test-unit` until green
6. Open PR using `PR_DESCRIPTION.md` as the body

## Local validation (before submission)

Validation against an out-of-tree mount (see `PhimmStraiker/litellm-straiker-agentic` for the working dev stack):

- 96 unit tests pass (`respx`-mocked, 100% offline)
- 1500+ live requests sent through the LiteLLM proxy stack
- Block-mode end-to-end: email, credit card, and llm_evasion categories all return HTTP 403 with the full `straiker_debug` envelope
- Agentic tool-call dedup verified in the Straiker Defend Console

## Authoritative source

- Production-grade integration repo (private dev/test surface): `PhimmStraiker/litellm-straiker-agentic`
- Straiker docs: https://docs.straiker.ai/defend-ai/litellm-integration
- Straiker website: https://straiker.ai

## License

This contribution is offered under the same license as the LiteLLM project ([MIT](https://github.com/BerriAI/litellm/blob/main/LICENSE)).
