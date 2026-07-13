ARG LITELLM_BASE=ghcr.io/berriai/litellm:main-stable

FROM $LITELLM_BASE

USER root

COPY litellm/types/guardrails.py /app/litellm/types/guardrails.py
COPY litellm/types/guardrails.py /app/.venv/lib/python3.13/site-packages/litellm/types/guardrails.py

COPY litellm/types/proxy/guardrails/guardrail_hooks/straiker.py /app/litellm/types/proxy/guardrails/guardrail_hooks/straiker.py
COPY litellm/types/proxy/guardrails/guardrail_hooks/straiker.py /app/.venv/lib/python3.13/site-packages/litellm/types/proxy/guardrails/guardrail_hooks/straiker.py

COPY litellm/proxy/guardrails/guardrail_hooks/straiker /app/litellm/proxy/guardrails/guardrail_hooks/straiker
COPY litellm/proxy/guardrails/guardrail_hooks/straiker /app/.venv/lib/python3.13/site-packages/litellm/proxy/guardrails/guardrail_hooks/straiker

COPY ui/litellm-dashboard/out /app/litellm/proxy/_experimental/out
COPY ui/litellm-dashboard/out /app/.venv/lib/python3.13/site-packages/litellm/proxy/_experimental/out

LABEL straiker.integration="1.0.0"
LABEL straiker.upstream="$LITELLM_BASE"
