"""Exceptions raised by hosted-tool converter dispatch."""

from __future__ import annotations


class UnsupportedHostedToolError(NotImplementedError):
    """Raised when a hosted tool is not supported by the active provider.

    Each provider's converter dispatches over the
    :class:`HostedTool` subclass set. When a converter encounters
    a variant it does not support, it raises this error rather than
    silently dropping the tool. The error message names the tool class,
    the active provider, and the providers that DO support the tool —
    so the developer can fix the agent constructor.
    """

    def __init__(
        self,
        tool: object,
        provider: str,
        *,
        supported_providers: tuple[str, ...] = (),
    ) -> None:
        """Construct the error with a helpful diagnostic message.

        Args:
            tool: The hosted-tool instance that was not supported.
                The class name is extracted for the error message.
            provider: Short identifier for the active provider (e.g.
                ``"anthropic"``, ``"openai-responses"``).
            supported_providers: Tuple of provider identifiers that DO
                support this tool. Included in the message to guide the
                developer toward a working configuration.
        """
        self.tool = tool
        self.provider = provider
        self.supported_providers = supported_providers
        tool_name = type(tool).__name__
        if len(supported_providers) > 0:
            supported = ", ".join(sorted(supported_providers))
            message = (
                f"{tool_name} is not supported by provider {provider!r}. "
                f"Supported providers: {supported}. "
                "Switch the agent's LLM or remove the tool."
            )
        else:
            message = (
                f"{tool_name} is not supported by provider {provider!r}. Switch the agent's LLM or remove the tool."
            )
        super().__init__(message)
