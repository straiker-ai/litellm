from typing import TYPE_CHECKING

import litellm
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .straiker import StraikerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    _callback = StraikerGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        agentic=getattr(litellm_params, "agentic", False),
        source=getattr(litellm_params, "source", "litellm"),
        destination=getattr(litellm_params, "destination", "api.openai.com"),
        threshold=getattr(litellm_params, "threshold", 0.5),
        timeout=getattr(litellm_params, "timeout", 5.0),
        max_retries=getattr(litellm_params, "max_retries", 2),
        initial_backoff=getattr(litellm_params, "initial_backoff", 0.1),
        max_backoff=getattr(litellm_params, "max_backoff", 2.0),
        unreachable_fallback=getattr(
            litellm_params, "unreachable_fallback", "fail_closed"
        ),
        enabled_models=getattr(litellm_params, "enabled_models", None),
        skip_models=getattr(litellm_params, "skip_models", None),
        max_payload_bytes=getattr(litellm_params, "max_payload_bytes", 524288),
        custom_headers=getattr(litellm_params, "custom_headers", None),
        verbose=getattr(litellm_params, "verbose", True),
        dedup_iterations=getattr(litellm_params, "dedup_iterations", True),
        guardrail_name=guardrail.get("guardrail_name", "straiker"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.STRAIKER.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.STRAIKER.value: StraikerGuardrail,
}
