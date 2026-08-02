"""LLM configuration as a dataclass.

Provides ``LLMConfig`` — a ``@dataclass`` that holds provider-agnostic
configuration parameters for language model calls.  Inspired by OpenAI's
``ModelSettings`` pattern but with more explicit field names.

Each field uses ``None`` to indicate "not set" — only non-``None`` values
are forwarded to the LLM provider.  The concrete ``LLM`` implementation
(e.g., ``LiteLLM``) is responsible for mapping these fields to
provider-specific parameter names.

Example::

    config = LLMConfig(temperature=0.7, max_output_tokens=2000)

    # Merge two configs (override takes precedence)
    merged = config.resolve(LLMConfig(temperature=0.2))
    # → LLMConfig(temperature=0.2, max_output_tokens=2000)
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from httpx import Timeout

from troopai.adk.types.common import Body, Headers, Metadata, Query
from troopai.adk.types.llms import LLMRetryPolicy
from troopai.adk.types.tools import ToolChoice, ToolExecutionMode


@dataclass
class LLMConfig:
    """Provider-agnostic configuration for LLM calls.

    A ``@dataclass`` where every field defaults to ``None`` (unset).
    Only non-``None`` fields are forwarded to the LLM provider.

    The concrete ``LLM`` implementation (e.g., ``LiteLLM``) reads
    these fields directly and maps them to provider-specific parameter
    names in the API call.

    Attributes:
        temperature: Sampling temperature (0–1).  Lower = more
            deterministic.
        top_k: Top-k filtering.  Limits to *k* highest-probability tokens.
        top_p: Nucleus sampling.  Considers smallest set of tokens with
            cumulative probability >= *top_p*.
        max_output_tokens: Maximum tokens in the response.
        frequency_penalty: Penalise tokens by how often they appear.
        presence_penalty: Penalise tokens by whether they appeared at all.
        response_logprobs: Whether to return log probabilities for chosen
            tokens.
        top_logprobs: Number of top candidate tokens to return log probs for.
        stop_sequences: Stop generation when any of these strings appears.
        seed: Random seed for reproducibility.
        metadata: Arbitrary metadata passed to the LLM API.
        extra_body: Extra fields merged into the API request body.
        extra_query: Extra query parameters for the API request.
        extra_headers: Extra HTTP headers for the API request.
        extra_args: Catch-all for provider-specific parameters.
            Spread into the LLM API call as ``**extra_args``.
        timeout: Request timeout in seconds or ``httpx.Timeout``.
        num_retries: Number of retries for transient API errors.
        retry_policy: Framework-level retry policy for transient LLM
            failures.  Runs outside SDK-level retries with backoff,
            jitter, and error-kind filters.  Non-streaming only.
        fallbacks: Alternative model names to try on failure.
        include_usage: Include token usage in streaming responses.
            Defaults to ``True`` when unset.
        tool_choice: Tool selection strategy (``"auto"``, ``"required"``,
            ``"none"``, or a tool name).
        tool_execution_mode: Sequential or parallel tool execution.
        reset_tool_choice: Reset ``tool_choice`` to ``"auto"`` after tools
            execute.  ``None`` is treated as ``True``.
    """

    temperature: float | None = None
    """Sampling temperature (0–1).  Higher = more random."""

    top_k: float | None = None
    """Top-k filtering.  Keeps only the *k* highest-probability tokens."""

    top_p: float | None = None
    """Nucleus sampling.  Considers tokens with cumulative probability >= top_p."""

    max_output_tokens: int | None = None
    """Maximum number of tokens to generate in the response."""

    frequency_penalty: float | None = None
    """Penalise repeated tokens based on their frequency so far."""

    presence_penalty: float | None = None
    """Penalise repeated tokens based on whether they appeared at all."""

    response_logprobs: bool | None = None
    """Whether to return log probabilities for the chosen tokens."""

    top_logprobs: int | None = None
    """Number of top candidate tokens to return log probs for."""

    stop_sequences: list[str] | None = None
    """Stop generation when any of these strings is encountered."""

    seed: int | None = None
    """Random seed for reproducibility."""

    metadata: Metadata | None = None
    """Arbitrary metadata passed to the LLM API."""

    extra_body: Body | None = None
    """Extra fields merged into the API request body."""

    extra_query: Query | None = None
    """Extra query parameters for the API request."""

    extra_headers: Headers | None = None
    """Extra HTTP headers for the API request."""

    extra_args: dict[str, Any] | None = None
    """Catch-all for provider-specific parameters.

    These are spread into the LLM API call as additional keyword
    arguments, allowing any provider-specific parameter to be passed
    without adding a dedicated field.
    """

    timeout: float | Timeout | None = None
    """Request timeout override, in seconds or as ``httpx.Timeout``."""

    num_retries: int | None = None
    """Number of retries for transient API errors (429, 500, timeouts).

    The LLM implementation retries the same API call up to this many
    times on transient failures (rate limits, server errors, network
    timeouts).  Each retry targets the same model with the same
    parameters.

    This is distinct from ``FunctionTool.max_retries``, which controls
    how many times the *LLM* can retry a failed *tool call*.

    - ``None`` (default): Use the LLM implementation's default.
    - ``0``: No retries — fail immediately on transient errors.
    - ``N > 0``: Retry up to N times before raising the error.
    """

    retry_policy: LLMRetryPolicy | None = None
    """Framework-level retry policy for transient LLM failures.

    When set, the ``LLM`` implementation wraps each non-streaming call
    in a retry loop that honours the policy's backoff, jitter, and
    error-kind filters. This runs *outside* any SDK-level retry
    (``num_retries``) and gives developers control over total retry
    budget, delay shape, and which error categories are retried.

    Distinct from :attr:`num_retries`: ``num_retries`` is an SDK-level
    hint forwarded to the provider library; :attr:`retry_policy` runs
    at the framework layer and can express jitter, exponential
    backoff, and category selection.

    Streaming calls are **not** retried — reconnecting mid-stream is
    unsafe, so streaming errors surface immediately.
    """

    fallbacks: list[str] | None = None
    """Alternative model names to try if the primary model fails.

    When the primary model returns an error after exhausting retries,
    the LLM implementation tries each fallback model in order until
    one succeeds.

    - ``None`` (default): No fallbacks — only the primary model is used.
    - ``["gpt-4o", "claude-sonnet-4-20250514"]``: Try these models in order.

    Example::

        config = LLMConfig(
            num_retries=2,
            fallbacks=["gpt-4o-mini", "claude-sonnet-4-20250514"],
        )
    """

    include_usage: bool | None = None
    """Include token usage in streaming responses.

    When ``True`` (the default when unset), the LLM implementation
    requests usage data in the final streaming chunk so that token
    counts are available.  Set to ``False`` to omit usage data from
    streaming responses.

    Not passed directly to the LLM API — used to build
    ``stream_options``.
    """

    tool_choice: ToolChoice | None = None
    """Tool selection strategy for LLM requests.

    - ``"auto"`` — LLM decides when to call tools (default when ``None``).
    - ``"required"`` — LLM must call a tool every turn.
    - ``"none"`` — LLM cannot call tools (text only).
    - Any other string — force a specific tool by name.

    Maps to the ``tool_choice`` API parameter.
    """

    tool_execution_mode: ToolExecutionMode | None = None
    """Whether the LLM may invoke multiple tools in a single turn.

    - ``None`` / ``ToolExecutionMode.SEQUENTIAL`` — one tool call per turn.
    - ``ToolExecutionMode.PARALLEL`` — multiple concurrent tool calls.

    Maps to the ``parallel_tool_calls`` API parameter.
    """

    reset_tool_choice: bool | None = None
    """Reset ``tool_choice`` to ``"auto"`` after tools execute.

    When ``tool_choice`` is ``"required"``, the LLM must call a tool
    every turn. Without reset, this creates an infinite loop when
    ``tool_use_behavior`` is ``"run_llm_again"``.

    - ``None`` (default): treated as ``True`` by the Runner.
    - ``True``: after tools execute, the next LLM call uses ``"auto"``.
    - ``False``: ``tool_choice`` stays as-is between turns.

    Set to ``False`` when pairing ``"required"`` with exit behaviors
    like ``"stop_on_first_tool"`` or ``StopAtTools``.
    """

    def resolve(self, override: LLMConfig | None = None) -> LLMConfig:
        """Produce a new LLMConfig by overlaying non-None values from *override*.

        Args:
            override: Optional config with override values.

        Returns:
            A new ``LLMConfig`` with merged values.

        Example::

            base = LLMConfig(temperature=0.7, max_output_tokens=1000)
            override = LLMConfig(temperature=0.2)
            merged = base.resolve(override)
            # → LLMConfig(temperature=0.2, max_output_tokens=1000)
        """
        if override is None:
            return self

        self_names = {f.name for f in fields(self)}
        override_names = {f.name for f in fields(override)}

        # Build on whichever config carries the richer field set so no field is
        # dropped or rejected by the constructor. A subclass config (e.g. one
        # adding provider-specific fields) has a superset of the base's fields;
        # iterating one side and blindly reading each field off the other would
        # either drop the subclass-only fields or raise ``AttributeError`` when
        # the base instance lacks them. Field sets nest for a base/subclass pair
        # (the realistic case), so the superset instance is the safe target.
        if override_names >= self_names:
            base, base_names = override, override_names
        else:
            base, base_names = self, self_names

        changes: dict[str, Any] = {}
        for name in base_names:
            override_value = getattr(override, name) if name in override_names else None
            if override_value is not None:
                changes[name] = override_value
            elif name in self_names:
                self_value = getattr(self, name)
                if self_value is not None:
                    changes[name] = self_value

        # Deep-merge extra_args (both configs may contribute; override wins per key)
        if self.extra_args is not None or override.extra_args is not None:
            merged_args: dict[str, Any] = {}
            if self.extra_args is not None:
                merged_args.update(self.extra_args)
            if override.extra_args is not None:
                merged_args.update(override.extra_args)
            changes["extra_args"] = merged_args if len(merged_args) > 0 else None

        return replace(base, **changes)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns:
            A dict with all non-None fields, using field names as keys.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}
