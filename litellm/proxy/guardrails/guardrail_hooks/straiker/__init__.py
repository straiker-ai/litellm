from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import litellm
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .straiker import StraikerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

_OPTIONAL_INIT_FIELDS: Final = (
    "timeout",
    "max_retries",
    "initial_backoff",
    "max_backoff",
    "unreachable_fallback",
    "fail_on_error",
    "max_payload_bytes",
    "custom_headers",
    "metadata",
    "verbose",
    "app_attribution",
)

_CODING_AGENT_FIELDS = (
    "enabled",
    "api_key",
    "mode",
    "latency",
    "fail_open",
    "chatter_filter",
    "user_name_header",
)


def _get_config_value(litellm_params: "LitellmParams", optional_params: object, attribute_name: str) -> object:
    if optional_params is not None:
        if isinstance(optional_params, dict):
            value = optional_params.get(attribute_name)
        else:
            value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def _reject_nested_coding_agent(litellm_params: "LitellmParams", optional_params: object) -> None:
    """Refuse the pre-flat nested ``coding_agent:`` block instead of reading it as absent.

    The flat keys are read by name, so a nested block matches nothing and leaves coding-agent
    scoring switched off with no error. Booting without a security control is worse than not
    booting, so this fails loudly and names the replacement.
    """
    if isinstance(_get_config_value(litellm_params, optional_params, "coding_agent"), Mapping):
        raise ValueError(
            "straiker: nested `coding_agent:` config is not supported. Use the flat "
            "`coding_agent_*` keys instead (coding_agent_enabled, coding_agent_api_key, "
            "coding_agent_mode, coding_agent_latency, ...). A nested block is read as absent, "
            "which would leave coding-agent scoring silently disabled."
        )


def _coding_agent_settings(litellm_params: "LitellmParams", optional_params: object) -> dict[str, object] | None:
    """Assemble the coding-agent settings from flat ``coding_agent_*`` config.

    They are flat rather than nested so each one renders as its own control in the admin UI,
    and so a reader can see at a glance whether coding agents are on and which key they use.
    """
    _reject_nested_coding_agent(litellm_params, optional_params)
    values = {
        field: value
        for field in _CODING_AGENT_FIELDS
        for value in [_get_config_value(litellm_params, optional_params, f"coding_agent_{field}")]
        if value is not None
    }
    if values.get("enabled", "off") == "off":
        return None
    return values


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    optional_params: Final = getattr(litellm_params, "optional_params", None)
    api_key: Final = litellm_params.api_key
    if not api_key:
        raise ValueError("api_key is required for straiker")

    api_base: Final = litellm_params.api_base or "https://api.prod.straiker.ai"
    default_app: Final = getattr(litellm_params, "default_app", None) or getattr(litellm_params, "source", None)
    source: Final = default_app if isinstance(default_app, str) and default_app else "LiteLLM Gateway"
    coding_agent: Final = _coding_agent_settings(litellm_params, optional_params)
    kwargs: Final[dict[str, object]] = {
        field: value
        for field in _OPTIONAL_INIT_FIELDS
        for value in [_get_config_value(litellm_params, optional_params, field)]
        if value is not None
    } | ({"coding_agent": coding_agent} if coding_agent is not None else {})
    _callback: Final = StraikerGuardrail(
        api_key=api_key,
        api_base=api_base if isinstance(api_base, str) else "https://api.prod.straiker.ai",
        source=source,
        guardrail_name=guardrail.get("guardrail_name", "straiker"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        **kwargs,
    )

    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.STRAIKER.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.STRAIKER.value: StraikerGuardrail,
}
