"""Schema models for declarative agent configuration.

These Pydantic models are the source of truth for the JSON config format and
for the generated JSON Schema. Validation is strict: unknown keys are
rejected (``extra="forbid"``) so a typo fails loudly instead of being
silently ignored. The one tolerated meta-key is ``$schema`` — an optional
pointer to the published schema for editor and CI tooling, never required
and ignored at construction time.

Where a framework type is already JSON-shaped it is reused directly (so the
schema can never drift from the runtime type): ``SystemPrompt`` for a
structured prompt, ``StopAtTools`` for that tool-use-behavior variant. A
config-only model is introduced only when the real type carries fields that
cannot live in JSON (e.g. ``VerboseConfig`` holds an output stream and a
style table — only its scalar knobs are surfaced here).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from troopai.adk.prompts.system_prompt import SystemPrompt
from troopai.adk.types.config.guardrail_config import GuardrailsConfig
from troopai.adk.types.config.llm_config import LLMConfigBlock, LLMProviderConfig
from troopai.adk.types.config.prompt_config import DynamicPromptRef
from troopai.adk.types.config.references import OutputSchemaRef
from troopai.adk.types.config.tool_config import HostedToolRef
from troopai.adk.types.tools.tool_use_behavior import StopAtTools
from troopai.adk.verbose.config import VerboseMode

# Keys naming behavior that can only be expressed as Python callables/objects.
# They will never have a declarative form; intercept them with a guiding
# message rather than letting strict validation emit a generic "extra field".
CODE_ONLY_KEYS: dict[str, str] = {
    "hooks": "lifecycle hooks",
    "middleware": "middleware",
    "skills": "skills",
}


class VerboseConfigRef(BaseModel):
    """JSON-friendly subset of ``VerboseConfig``.

    Only the scalar knobs are declarable. The output stream and per-event
    style table on the runtime ``VerboseConfig`` have no JSON form and keep
    their defaults.

    Attributes:
        enabled: Master switch for verbose output.
        mode: Render backend selector.
        use_color: Whether to emit ANSI/Rich color.
        use_rich: Prefer the Rich renderer when available.
        show_timestamps: Prefix each line with an ``HH:MM:SS`` stamp.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    """Master switch for verbose output."""

    mode: VerboseMode = "auto"
    """Render backend selector (reuses the runtime ``VerboseMode`` literal)."""

    use_color: bool = True
    """Whether to emit ANSI/Rich color."""

    use_rich: bool = True
    """Prefer the Rich renderer when available."""

    show_timestamps: bool = False
    """Prefix each line with an ``HH:MM:SS`` stamp."""


class AgentConfig(BaseModel):
    """Declarative configuration for a single ``Agent``.

    The static, data-heavy surface of an agent, including the reference-
    bearing fields ``llm``, ``tools``, and ``output_schema``. Code-only
    behavior is rejected with a guiding error.

    Attributes:
        schema_ref: Optional pointer to the published JSON Schema, carried as
            the ``$schema`` key. Tolerated under strict validation and
            ignored when the agent is built.
        name: Unique agent identifier.
        description: Short description of what the agent does.
        system_prompt: Instructions as a plain string, a structured
            ``SystemPrompt``, or a ``{dynamic: ref}`` callable reference.
        llm: Model selection — a bare model-name string or a typed provider
            block (``{provider, model, …, config}``).
        llm_config: Standalone provider-agnostic LLM configuration, paired
            with the string ``llm`` form (mutually exclusive with a provider
            block's ``config``).
        guardrails: Input/output guardrails — dotted refs to guardrail
            objects. ``None`` means no guardrails.
        tools: Tools — dotted ``ref`` strings resolving to ``FunctionTool``,
            or ``{type, args}`` provider-hosted tool references.
        output_schema: Structured-output schema — a class ``ref`` or
            ``{ref, enforcement}``.
        skill_activation: When skill instructions enter the system prompt.
        tool_use_behavior: What happens after tool execution — a string
            literal or a ``StopAtTools`` selection.
        verbose: Optional per-agent verbose-output override.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_ref: str | None = Field(default=None, alias="$schema")
    """Optional ``$schema`` pointer; ignored at construction time."""

    name: str = Field(min_length=1)
    """Unique agent identifier."""

    description: str | None = None
    """Short description of what the agent does."""

    system_prompt: str | SystemPrompt | DynamicPromptRef
    """Instructions — a plain string, a structured ``SystemPrompt``, or a
    ``{dynamic: ref}`` reference to a ``DynamicSystemPrompt`` callable."""

    llm: str | LLMProviderConfig | None = None
    """Model selection. A bare model-name string for the active backend, or a
    typed provider block (``{provider, model, …, config}``) selecting a
    provider-native LLM with its own configuration."""

    llm_config: LLMConfigBlock | None = None
    """Standalone provider-agnostic LLM configuration. Pairs with the string
    ``llm`` form; mutually exclusive with a provider block's ``config``."""

    guardrails: GuardrailsConfig | None = None
    """Input/output guardrails — dotted refs to guardrail objects."""

    tools: list[Annotated[str, Field(min_length=1)] | HostedToolRef] = Field(default_factory=list)
    """Tools — a non-empty dotted ``ref`` string resolving to a
    ``FunctionTool``, or a ``{type, args}`` provider-hosted tool reference."""

    output_schema: Annotated[str, Field(min_length=1)] | OutputSchemaRef | None = None
    """Structured-output schema — a bare class ``ref`` or ``{ref, enforcement}``."""

    skill_activation: Literal["eager", "lazy"] = "lazy"
    """When skill instructions enter the system prompt."""

    tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools = "run_llm_again"
    """What happens after tool execution."""

    verbose: VerboseConfigRef | None = None
    """Optional per-agent verbose-output override."""

    @model_validator(mode="before")
    @classmethod
    def reject_code_only_keys(cls, data: object) -> object:
        """Turn a code-only key into a guiding error before strict rejection.

        Args:
            data: The raw mapping (or any value) being validated.

        Returns:
            ``data`` unchanged when no code-only key is present.

        Raises:
            ValueError: If a code-only key is present, with a message that
                directs the user to construct the agent in Python.
        """
        if isinstance(data, dict):
            for key, label in CODE_ONLY_KEYS.items():
                if key in data:
                    raise ValueError(
                        f"'{key}' ({label}) cannot be configured declaratively — it requires "
                        "Python callables or objects. Build the Agent in Python to set it, or "
                        "reference a Python symbol where a config field supports it."
                    )
        return data

    @model_validator(mode="after")
    def reject_split_llm_config(self) -> AgentConfig:
        """Enforce one source of LLM configuration.

        A provider block carries its own ``config``; the standalone
        ``llm_config`` pairs with the string ``llm`` form. Declaring both is
        ambiguous and rejected.

        Returns:
            ``self`` when the configuration comes from a single source.

        Raises:
            ValueError: If ``llm`` is a provider block and ``llm_config`` is
                also present.
        """
        is_provider_block = self.llm is not None and not isinstance(self.llm, str)
        if is_provider_block and self.llm_config is not None:
            raise ValueError(
                "Set LLM configuration in one source: a provider block's 'config' OR a top-level "
                "'llm_config' (with a string 'llm'), not both."
            )
        return self
