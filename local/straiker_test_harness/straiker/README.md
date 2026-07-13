# Straiker integration tooling (fork-local)

This directory holds the demo stack, agent harness, and packaging artifacts for the Straiker DefendAI native partner guardrail that lives in this fork. It is **not** part of the upstream BerriAI/litellm PR. The upstream PR contains only the source under `litellm/`, `tests/`, `docs/`, and `ui/`; this directory is fork-only material for running local demos and coordinating the submission.

## Contents

```
straiker/
  Dockerfile                          Custom image that layers our patched litellm/
                                      and rebuilt admin UI onto ghcr.io/berriai/litellm.
  docker-compose.yml                  Local stack: LiteLLM proxy + Postgres.
  config.yaml                         Proxy config wiring the Straiker guardrail
                                      at pre_call, during_call, and post_call.
  requirements.txt                    Python deps for the agent harness.
  agent_harness.py                    Real multi-iteration tool-calling agent
                                      against the local proxy. Includes weather,
                                      research, injection, and followup scenarios.
  straiker_guardrail_classpath.py     Class-path fallback (StraikerGuardrail as
                                      CustomGuardrail subclass) for setups that
                                      cannot use the native partner registration.
                                      Same runtime behavior; no admin UI form.
  DEMO_SCRIPT.md                      Copy-pasteable curl demos for calls with
                                      the local proxy.
  KNOWN_LIMITATIONS.md                Current limitations of the integration.
  SECURITY_REVIEW.md                  Threat model and security review notes.
  VERSION.txt                         Integration version, upstream image digest,
                                      and verification date.
  pr-artifacts/
    README.md                         Historical upstream-port packaging notes.
    ENUM_DIFF.md                      One-line diff summary for the enum entry.
    PR_DESCRIPTION.md                 Draft PR body for the upstream submission.
```

## Where the source lives

The actual integration source is in the fork under the standard LiteLLM paths.

```
litellm/types/guardrails.py                                           enum entry
litellm/types/proxy/guardrails/guardrail_hooks/straiker.py            Pydantic config
litellm/proxy/guardrails/guardrail_hooks/straiker/__init__.py         registration
litellm/proxy/guardrails/guardrail_hooks/straiker/straiker.py         hook implementation
tests/guardrails_tests/test_straiker_guardrails.py                    unit tests
docs/my-website/docs/proxy/guardrails/straiker.md                     docs page
ui/litellm-dashboard/src/components/guardrails/                       UI wiring
ui/litellm-dashboard/public/assets/logos/straiker.svg                 logo asset
```

## Build the image

Run from this directory. Fork root is the build context.

```
docker build -f Dockerfile -t ghcr.io/phimmstraiker/litellm:straiker-latest ..
```

If the UI source changed, rebuild the UI first so the Dockerfile picks up fresh chunks.

```
cd ../ui/litellm-dashboard
npm install
npm run build
cd ../../straiker
docker build -f Dockerfile -t ghcr.io/phimmstraiker/litellm:straiker-latest ..
```

## Run the local stack

```
export OPENAI_API_KEY=sk-...
export STRAIKER_API_KEY=...
docker compose up -d
curl -s http://localhost:4000/health/liveliness      # "I'm alive!"
open http://localhost:4000/ui                        # admin UI (sk-1234)
```

## Drive a real agent loop

The harness implements OpenAI's tool-calling loop against the local proxy, so LiteLLM sees multiple model calls per user turn and the Straiker Console shows the deduped one-pre-plus-one-post-per-user-turn view.

```
python3 agent_harness.py --scenario weather
python3 agent_harness.py --scenario research
python3 agent_harness.py --scenario injection    # adversarial: tool result carries injection
python3 agent_harness.py --scenario followup     # multi-turn: several user messages
```

Filter both UIs by the session_id printed at the end of each run.

## Submitting the upstream PR

See `pr-artifacts/PR_DESCRIPTION.md` for a draft PR body. When ready to submit, the flow is a standard community PR from this fork's `main` into `BerriAI/litellm:main`. Comment `@greptileai` on the PR to trigger their review bot; they require Confidence Score >= 4 before a maintainer review. Ping `#pr-review` on the LiteLLM OSS Slack if the PR sits for more than a few days.
