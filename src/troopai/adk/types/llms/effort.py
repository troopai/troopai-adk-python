"""Typed effort levels for reasoning-capable models.

``EffortLevel`` names the depth/token-spend setting exposed by
providers as an output-level control (e.g. Anthropic's
``output_config.effort``). The framework never sets an effort by
default — the field is opt-in on provider configs and omitted from the
request when unset.
"""

from __future__ import annotations

from typing import Literal

type EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]
"""Reasoning/output effort: ``low`` < ``medium`` < ``high`` < ``xhigh`` < ``max``.

``xhigh`` and ``max`` are only accepted by models that support them;
the provider returns a request error otherwise.
"""

__all__ = ["EffortLevel"]
