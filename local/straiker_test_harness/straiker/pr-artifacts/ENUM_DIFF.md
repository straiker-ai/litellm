# One-line enum addition

Add `STRAIKER = "straiker"` to the `SupportedGuardrailIntegrations` enum in `litellm/types/guardrails.py`.

This is the single line that makes Straiker discoverable by LiteLLM's registry and visible in the admin UI's Provider dropdown.

## Diff

```diff
--- a/litellm/types/guardrails.py
+++ b/litellm/types/guardrails.py
@@ class SupportedGuardrailIntegrations(Enum):
     APORIA = "aporia"
     BEDROCK = "bedrock"
     GUARDRAILS_AI = "guardrails_ai"
     LAKERA = "lakera"
     # ... existing entries ...
     ZSCALER_AI_GUARD = "zscaler_ai_guard"
+    STRAIKER = "straiker"
     JAVELIN = "javelin"
     ENKRYPTAI = "enkryptai"
     # ... rest of entries ...
```

Alphabetical placement is conventional but not enforced. Place near the bottom-of-list cluster of recently-added partners.

## Why a separate file

Diff documented separately so reviewers can see the full set of changes at a glance. The other four artifact directories (`litellm/proxy/...`, `litellm/types/...`, `tests/...`, `docs/...`) are full file additions; this single enum entry is the only in-place edit.
