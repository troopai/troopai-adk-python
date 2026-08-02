from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

from .tokens import InputTokensDetails, OutputTokensDetails


@dataclass
class LLMSingleRequestUsage:
    """Token usage statistics for a single request.

    Attributes:
        input_tokens: Number of input tokens used in the request.
        output_tokens: Number of output tokens generated in the response.
        total_tokens: Total number of tokens used (input + output).
        input_tokens_details: Detailed breakdown of the input tokens.
        output_tokens_details: Detailed breakdown of the output tokens.
    """

    input_tokens: int
    """The number of input tokens used in the request."""

    output_tokens: int
    """The number of output tokens generated in the response."""

    total_tokens: int
    """The total number of tokens used (input + output)."""

    input_tokens_details: InputTokensDetails
    """A detailed breakdown of the input tokens."""

    output_tokens_details: OutputTokensDetails
    """A detailed breakdown of the output tokens."""


@dataclass(kw_only=True)
class LLMUsage:
    """Token usage statistics for an API request.

    Attributes:
        requests: Number of requests made to the LLM.
        tool_calls: Number of framework-dispatched tool calls.
        total_tokens: Total tokens used (input + output) across all requests.
        input_tokens: Number of input tokens used.
        input_tokens_details: Detailed breakdown of input tokens.
        output_tokens: Number of output tokens generated.
        output_tokens_details: Detailed breakdown of output tokens.
        usage: Per-request breakdown; each entry corresponds to one request.
    """

    requests: int = 0
    """The number of requests made to the LLM through API."""

    tool_calls: int = 0
    """The number of framework-dispatched tool calls."""

    total_tokens: int = 0
    """The total number of tokens used (input + output) across all requests."""

    input_tokens: int = 0
    """The number of input tokens used in the request."""

    input_tokens_details: InputTokensDetails | None = field(
        default_factory=lambda: InputTokensDetails(cached_tokens=0, cache_creation_input_tokens=0)
    )
    """A detailed breakdown of the input tokens."""

    output_tokens: int = 0
    """The number of output tokens generated in the response."""

    output_tokens_details: OutputTokensDetails | None = field(
        default_factory=lambda: OutputTokensDetails(reasoning_tokens=0)
    )
    """A detailed breakdown of the output tokens."""

    usage: list[LLMSingleRequestUsage] = field(default_factory=list)
    """A list of usage details for each individual request.

    Each entry in the list corresponds to a single request made to the LLM,
    providing detailed token usage statistics for that request.

    Example:
        For an API call that made 3 requests to the LLM this list would contain 3
        `SingleRequestUsage` objects, each detailing the token usage for that specific request.
        If the input tokens for the requests were [100, 150, 200] and the output tokens were [50, 75, 100],
        the `usage` list would look like:
        [
            SingleRequestUsage(input_tokens=100, output_tokens=50, total_tokens=150, ...),
            SingleRequestUsage(input_tokens=150, output_tokens=75, total_tokens=225, ...),
            SingleRequestUsage(input_tokens=200, output_tokens=100, total_tokens=300, ...),
        ]

        and the total number of input tokens would be 450, output tokens would be 225,
        and total tokens would be 675 across all requests. This is helpful for tracking costs
        and understanding how tokens are distributed across multiple requests within a single API call.
    """

    def __add__(self, other: LLMUsage) -> LLMUsage:

        new_usage = copy(self)
        new_usage.usage = list(self.usage)  # own copy — shallow copy shares the list

        new_usage.requests += other.requests
        new_usage.tool_calls += other.tool_calls
        new_usage.input_tokens += other.input_tokens
        new_usage.output_tokens += other.output_tokens
        new_usage.total_tokens += other.total_tokens

        # None guards: fields may be None when LLMUsage is constructed without defaults.
        other_in = other.input_tokens_details
        other_out = other.output_tokens_details
        self_in = new_usage.input_tokens_details

        other_cached_tokens = other_in.cached_tokens if other_in is not None else 0
        other_cache_creation = other_in.cache_creation_input_tokens if other_in is not None else 0
        other_reasoning_tokens = other_out.reasoning_tokens if other_out is not None else 0

        self_cached = self_in.cached_tokens if self_in is not None else 0
        self_cache_creation = self_in.cache_creation_input_tokens if self_in is not None else 0

        new_usage.input_tokens_details = InputTokensDetails(
            cached_tokens=self_cached + other_cached_tokens,
            cache_creation_input_tokens=self_cache_creation + other_cache_creation,
        )

        self_out = new_usage.output_tokens_details
        new_usage.output_tokens_details = OutputTokensDetails(
            reasoning_tokens=(self_out.reasoning_tokens if self_out is not None else 0) + other_reasoning_tokens,
        )

        # Automatically preserve usage.
        # If the other Usage represents a single request with tokens, record it.
        if other.requests == 1 and other.total_tokens > 0:
            input_tokens_details = (
                other.input_tokens_details
                if other.input_tokens_details is not None
                else InputTokensDetails(cached_tokens=0)
            )
            output_tokens_details = (
                other.output_tokens_details
                if other.output_tokens_details is not None
                else OutputTokensDetails(reasoning_tokens=0)
            )

            single_request_usage = LLMSingleRequestUsage(
                input_tokens=other.input_tokens,
                output_tokens=other.output_tokens,
                total_tokens=other.total_tokens,
                input_tokens_details=input_tokens_details,
                output_tokens_details=output_tokens_details,
            )
            new_usage.usage.append(single_request_usage)
        else:
            # If the other Usage has multiple requests, extend the usage list.
            new_usage.usage.extend(other.usage)

        return new_usage


@dataclass(repr=False, kw_only=True)
class LLMUsageLimits:
    """Limits on model usage.

    The request count is tracked by the Runner, and the request limit is
    checked before each request to the model. Token counts are provided in
    responses from the model, and the token limits are checked after each
    response.

    Each limit can be set to ``None`` to disable it.

    Attributes:
        request_limit: Maximum number of requests allowed to the model.
        tool_calls_limit: Maximum number of framework-dispatched tool calls
            allowed.
        input_tokens_limit: Maximum number of input tokens allowed.
        output_tokens_limit: Maximum number of output tokens allowed.
        total_tokens_limit: Maximum total tokens allowed across requests and
            responses combined.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""

    tool_calls_limit: int | None = None
    """The maximum number of framework-dispatched tool calls allowed."""

    input_tokens_limit: int | None = None
    """The maximum number of input/prompt tokens allowed."""

    output_tokens_limit: int | None = None
    """The maximum number of output/response tokens allowed."""

    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""
