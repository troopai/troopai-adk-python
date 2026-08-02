"""Provider-declared computer-use tool with local action execution.

``ComputerTool`` is a hybrid hosted/local primitive: the LLM provider
knows it as a tool type so the model can emit structured actions
(click, type, screenshot, …), but execution runs in the developer's
environment via a user-supplied :class:`Computer` callable.

``SafetyCheck`` lets the developer approve / reject individual
actions before they reach the executor — useful for sandboxes,
restricted environments, or audit logging.

References:
- OpenAI computer use: https://platform.openai.com/docs/guides/tools-computer-use
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Generic, Literal, Protocol, TypeVar, runtime_checkable

from troopai.adk.tools.hosted.hosted_tool import HostedTool
from troopai.adk.utils import MaybeAwaitable


@dataclass(frozen=True)
class SafetyCheck:
    """A pending action plus rationale, passed to ``on_safety_check``.

    The framework constructs one of these before invoking the user's
    :class:`Computer` callable. The ``on_safety_check`` callback
    returns ``True`` to approve the action, ``False`` to reject — a
    rejection skips execution and surfaces a rejection message to the
    LLM so it can choose differently.

    Attributes:
        action: The action kind (``"click"``, ``"type"``, …).
        reason: Provider-supplied or framework-supplied rationale for
            why this action is being requested.
        target: Optional target identifier (URL, window title, file
            path, …) the action would affect — provider-dependent.
    """

    action: str
    """The action kind matching ``ComputerAction.type``."""

    reason: str
    """Why this action is being requested (provider-supplied or
    derived from preceding messages)."""

    target: str | None = None
    """Optional human-readable target. ``None`` when the action is
    target-agnostic (e.g. ``"screenshot"``, ``"wait"``)."""


@runtime_checkable
class Computer(Protocol):
    """Local execution interface for computer-use actions.

    Each method runs in the developer's environment. The framework
    routes ``ComputerToolCall`` items from the provider's response
    through the matching method and feeds the return value back as a
    ``ComputerToolCallResult``.

    Implementations are typically thin shims over Playwright, an
    Anthropic ``BashBrowser``, an X11 client, or similar.

    Protocol enforcement note: ``@runtime_checkable`` only verifies
    that each method *name* exists on the object — it does not
    introspect signatures, async-ness, or arity. Calls that pass the
    isinstance check but fail at await time surface as ``TypeError``
    from the executor, not from ``ComputerTool.__post_init__``.
    """

    async def screenshot(self) -> str:
        """Capture the current screen as a base64-encoded PNG."""
        ...

    async def click(
        self,
        x: int,
        y: int,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
    ) -> None:
        """Click at screen coordinates with the given mouse button.

        Args:
            x: Horizontal screen coordinate in pixels.
            y: Vertical screen coordinate in pixels.
            button: Mouse button to click. Defaults to ``"left"``.
        """
        ...

    async def double_click(self, x: int, y: int) -> None:
        """Double-click at screen coordinates with the primary button.

        Args:
            x: Horizontal screen coordinate in pixels.
            y: Vertical screen coordinate in pixels.
        """
        ...

    async def type(self, text: str) -> None:
        """Type the given text at the current keyboard focus.

        Args:
            text: The text string to type.
        """
        ...

    async def keypress(self, keys: list[str]) -> None:
        """Press one or more keys (chord) and release.

        Args:
            keys: List of key names (``"Enter"``, ``"ctrl"``, ``"a"``).
                When more than one, they are pressed simultaneously
                (chord) and released together.
        """
        ...

    async def move(self, x: int, y: int) -> None:
        """Move the mouse cursor to screen coordinates without clicking.

        Args:
            x: Horizontal screen coordinate in pixels.
            y: Vertical screen coordinate in pixels.
        """
        ...

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at coordinates by the given horizontal / vertical deltas.

        Args:
            x: Horizontal screen coordinate to scroll at.
            y: Vertical screen coordinate to scroll at.
            scroll_x: Horizontal scroll delta in pixels (positive = right).
            scroll_y: Vertical scroll delta in pixels (positive = down).
        """
        ...

    async def drag(self, path: list[tuple[int, int]]) -> None:
        """Drag along a path of screen coordinates.

        Args:
            path: Ordered list of ``(x, y)`` waypoints. The drag starts
                at ``path[0]`` and ends at ``path[-1]``.
        """
        ...

    async def wait(self, seconds: float) -> None:
        """Pause for the given number of seconds.

        Args:
            seconds: Duration to pause in seconds.
        """
        ...


ComputerT = TypeVar("ComputerT", bound=Computer)
"""TypeVar preserving the concrete ``Computer`` subtype on
``ComputerTool[YourComputer]``. Internal — not part of the public
``troopai.adk.tools`` import surface."""


type SafetyCheckHandler = Callable[..., MaybeAwaitable[bool]]
"""Type for the ``on_safety_check`` callback.

The runtime dispatch supplies ``(ToolContext, SafetyCheck)`` and
expects ``bool`` or ``Awaitable[bool]`` back; the variadic form
accommodates one-arg callbacks too. ``ToolContext`` is intentionally
not in the typing chain to avoid a circular import — its concrete
type is documented on :attr:`ComputerTool.on_safety_check`."""


type ApprovalRequiredCallback = Callable[[SafetyCheck], MaybeAwaitable[bool]]
"""Type for the callable form of :attr:`ComputerTool.requires_approval`.

Returns ``True`` when the action needs HITL approval, ``False`` for
auto-approve."""


@dataclass(kw_only=True)
class ComputerTool(HostedTool, Generic[ComputerT]):
    """Hybrid hosted/local computer-use tool.

    Declares the computer-use capability to the LLM provider so the
    model can emit structured actions (``computer_call`` items), and
    carries the local :class:`Computer` callable plus optional
    safety + approval gates.

    Provider matrix:
    - **OpenAI Responses**: native via the ``computer_use_preview``
      tool type. Honours every attribute below.
    - All other providers raise
      :class:`UnsupportedHostedToolError` when the converter encounters
      a ``ComputerTool``. Anthropic computer-use is not supported here:
      its action vocabulary (e.g. ``triple_click``, ``hold_key``,
      ``cursor_position``) requires :class:`Computer` Protocol extensions
      this ADK does not implement.

    Treat instances as immutable after construction: mutating
    ``.computer`` post-construction bypasses the Protocol check and
    the dispatch wiring. Use :func:`dataclasses.replace` to swap
    executors. (The dataclass cannot be marked ``frozen=True``
    because the parent :class:`HostedTool` is non-frozen — Python
    forbids frozen subclasses of non-frozen parents.)

    Attributes:
        computer: User-supplied execution backend conforming to the
            :class:`Computer` protocol.
        display_width: Reported display width in pixels.
            **OpenAI Responses only.**
        display_height: Reported display height in pixels.
            **OpenAI Responses only.**
        environment: Provider environment hint (``"browser"``,
            ``"mac"``, ``"windows"``, ``"linux"``, ``"ubuntu"``).
            **OpenAI Responses only.**
        on_safety_check: Optional callback ``(ToolContext, SafetyCheck)
            → bool`` invoked before each action. Return ``False`` to
            reject. Both sync and async are supported.
            **Applies to all supported providers.**
        requires_approval: Whether each action requires user approval
            via the framework's HITL deferral surface. Cost-and-safety-
            conservative default: ``True``. A bool, or a callable
            ``(SafetyCheck) → bool`` evaluated per-action.
            **Applies to all supported providers.**
    """

    SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]] = ("openai-responses",)

    computer: ComputerT
    """The user-supplied :class:`Computer` executor."""

    display_width: int = 1024
    """Reported display width in pixels. **OpenAI Responses only.**"""

    display_height: int = 768
    """Reported display height in pixels. **OpenAI Responses only.**"""

    environment: Literal["browser", "mac", "windows", "linux", "ubuntu"] = "browser"
    """Provider environment hint. **OpenAI Responses only.**"""

    on_safety_check: SafetyCheckHandler | None = field(default=None, repr=False)
    """Optional pre-execution gate. Receives ``(ToolContext, SafetyCheck)``
    and returns whether to proceed. ``None`` allows every action.
    Both sync and async forms supported.
    **Applies to all supported providers.**"""

    requires_approval: bool | ApprovalRequiredCallback = True
    """Whether each action requires HITL approval.

    Cost-and-safety-conservative default: ``True``. Computer-use
    actions can mutate the developer's environment (open apps, click
    links, type into forms); requiring opt-out is the right default.
    A callable receives the :class:`SafetyCheck` and returns
    ``True`` for "needs approval", ``False`` for "auto-approve".
    **Applies to all supported providers.**
    """

    def __post_init__(self) -> None:
        if self.display_width <= 0:
            raise ValueError(f"display_width must be positive, got {self.display_width}")
        if self.display_height <= 0:
            raise ValueError(f"display_height must be positive, got {self.display_height}")
        if self.computer is None:
            raise TypeError("ComputerTool.computer is required and must not be None.")
        if not isinstance(self.computer, Computer):
            raise TypeError(
                "ComputerTool.computer must implement the Computer protocol — at "
                "least one method name is missing on the supplied object "
                f"({type(self.computer).__name__}). Note: this check verifies "
                "method *names* only; signature mismatches (wrong arity, sync "
                "instead of async) surface as TypeError when the executor is "
                "invoked."
            )
