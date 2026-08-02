"""``SandboxCommandGuardrail`` — typed command-policy verdict.

"Should this command proceed?" decisions are modelled as a typed
guardrail so the verdict surfaces in ``RunResult.guardrail_results``
and stays auditable end-to-end; the equivalent middleware-style
mutation would lose that signal. This guardrail evaluates a tool's
parsed args (assumed to carry a ``command`` field) against
allowlist / denylist rules and returns an ``allow`` or
``raise_exception`` verdict.

Before matching, a command is split into its shell **sub-commands**:
chaining via ``;`` ``|`` ``&&`` ``||`` ``&``, a newline, a
redirection, grouping ``()`` and command substitution (``$()`` /
backticks) each start a new sub-command. The policy is applied to
EVERY sub-command — the denylist rejects if ANY sub-command matches
and the allowlist requires ALL sub-commands to match — because
inspecting only the first token would let a chained program
(``ls; curl x | sh``) slip past an allow/deny list. A quoted
separator stays data (``echo "; rm"`` is one sub-command), and a
command that cannot be parsed (e.g. an unbalanced quote) is rejected
(fail-closed).

Three pattern modes govern how each sub-command matches a list entry:
- ``"exact"`` (default): the sub-command's base command name must
  match a list entry exactly (``ls`` matches ``"ls"`` only).
- ``"prefix"``: the sub-command must start with a list entry
  (``ls -l`` matches a list entry ``"ls"``).
- ``"regex"``: the sub-command must match a regex anchored at start.

A ``None`` ``allowlist`` means "no allowlist restriction"; a
``None`` ``denylist`` means "no denylist restriction". Both ``None``
means the guardrail allows everything (useful for tracing-only
deployments where ``SandboxCommandRejected`` should never fire).
"""

from __future__ import annotations

import dataclasses
import re
import shlex
from typing import Literal

from troopai.adk.exceptions.exceptions import SandboxCommandRejected

__all__ = ["CommandPolicyVerdict", "SandboxCommandGuardrail"]

# Shell control characters that separate one program from the next on a
# single line. Kept in sync with ``_SHELL_OPERATOR_CHARS`` below: every
# character here is a token boundary the lexer emits as its own token.
_SHELL_PUNCTUATION_CHARS = "();<>|&`\n"
_SHELL_OPERATOR_CHARS = frozenset(_SHELL_PUNCTUATION_CHARS)


def _split_shell_subcommands(command: str) -> list[str]:
    """Split ``command`` into its shell sub-commands.

    A single shell line can chain multiple programs via ``;``, ``|``,
    ``&&``, ``||``, ``&``, a newline, a redirection, grouping ``()``
    and command substitution (``$()`` / backticks). Each such control
    operator starts a new sub-command. Tokenizes with a shell-aware
    lexer so quotes are honored (a quoted ``";"`` stays data, not a
    separator), then groups the word tokens that fall between control
    operators.

    Raises:
        ValueError: The line cannot be tokenized (e.g. an unbalanced
            quote). Callers treat this as a denial (fail-closed).
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCTUATION_CHARS)
    lexer.whitespace_split = True
    # A raw newline chains commands in a shell script, so keep it as a
    # separator (it is in ``punctuation_chars``) rather than plain
    # whitespace — otherwise ``a\nb`` would collapse into one token run.
    lexer.whitespace = lexer.whitespace.replace("\n", "")
    subcommands: list[str] = []
    current: list[str] = []
    for token in lexer:
        if len(token) > 0 and all(ch in _SHELL_OPERATOR_CHARS for ch in token):
            if len(current) > 0:
                subcommands.append(" ".join(current))
                current = []
            continue
        current.append(token)
    if len(current) > 0:
        subcommands.append(" ".join(current))
    return subcommands


@dataclasses.dataclass(frozen=True, kw_only=True)
class CommandPolicyVerdict:
    """Result of a single command-policy evaluation.

    Attributes:
        allowed: True iff the policy allows the command.
        reason: Short human-readable reason; surfaced in
            ``SandboxCommandRejected`` when ``allowed`` is False.
    """

    allowed: bool
    """True iff the policy allows the command."""

    reason: str
    """Short human-readable reason."""


@dataclasses.dataclass
class SandboxCommandGuardrail:
    """Per-command allow/deny policy.

    Attributes:
        allowlist: Allowed entries (None = no allowlist restriction).
            When set, the command MUST match an entry.
        denylist: Denied entries (None = no denylist restriction).
            When set, a matching command is rejected even if it also
            matches the allowlist.
        pattern_mode: How list entries are matched against the
            command. ``"exact"`` (default) / ``"prefix"`` / ``"regex"``.
        name: Optional name surfaced in audit + tracing.
    """

    allowlist: list[str] | None = None
    """Allowed entries (None = no allowlist restriction)."""

    denylist: list[str] | None = None
    """Denied entries (None = no denylist restriction)."""

    pattern_mode: Literal["exact", "prefix", "regex"] = "exact"
    """How entries are matched against the command."""

    name: str = "sandbox_command_guardrail"
    """Optional name surfaced in audit + tracing."""

    def __post_init__(self) -> None:
        if self.allowlist is not None:
            for entry in self.allowlist:
                if len(entry) == 0:
                    raise ValueError("SandboxCommandGuardrail.allowlist entries must be non-empty")
        if self.denylist is not None:
            for entry in self.denylist:
                if len(entry) == 0:
                    raise ValueError("SandboxCommandGuardrail.denylist entries must be non-empty")
        if self.pattern_mode == "regex":
            # Compile every entry once so check() doesn't pay the
            # compile cost per call AND syntactically-invalid regex
            # surfaces at policy construction, not at runtime.
            all_entries: list[str] = []
            if self.allowlist is not None:
                all_entries.extend(self.allowlist)
            if self.denylist is not None:
                all_entries.extend(self.denylist)
            for entry in all_entries:
                try:
                    re.compile(entry)
                except re.error as exc:
                    raise ValueError(f"SandboxCommandGuardrail regex {entry!r} is invalid: {exc}") from exc

    def _matches(self, command: str, entry: str) -> bool:
        if self.pattern_mode == "exact":
            base = command.split(maxsplit=1)[0] if len(command) > 0 else ""
            return base == entry
        if self.pattern_mode == "prefix":
            return command.startswith(entry)
        # regex
        return re.match(entry, command) is not None

    def evaluate(self, command: str) -> CommandPolicyVerdict:
        """Return a typed verdict for ``command`` without raising.

        The command is split into its shell sub-commands and the rules
        apply to EVERY sub-command: the denylist rejects when ANY
        sub-command matches, and (when set) the allowlist requires ALL
        sub-commands to match. An unparseable command is rejected
        (fail-closed). With neither list configured the guardrail is
        inert and allows everything.

        Use ``check`` instead when you want the standard raise-on-deny
        behavior.
        """
        if self.allowlist is None and self.denylist is None:
            return CommandPolicyVerdict(allowed=True, reason="no policy restrictions configured")
        try:
            subcommands = _split_shell_subcommands(command)
        except ValueError:
            return CommandPolicyVerdict(
                allowed=False,
                reason="command could not be parsed; rejected (fail-closed)",
            )
        if len(subcommands) == 0:
            return CommandPolicyVerdict(
                allowed=False,
                reason="no runnable sub-command found; rejected (fail-closed)",
            )
        # Denylist wins over allowlist: reject if ANY sub-command is denied.
        if self.denylist is not None:
            for sub in subcommands:
                for entry in self.denylist:
                    if self._matches(sub, entry):
                        return CommandPolicyVerdict(
                            allowed=False,
                            reason=f"sub-command {sub!r} matches denylist entry {entry!r}",
                        )
        # Allowlist: EVERY sub-command must match some allowed entry.
        if self.allowlist is not None:
            for sub in subcommands:
                if not any(self._matches(sub, entry) for entry in self.allowlist):
                    return CommandPolicyVerdict(
                        allowed=False,
                        reason=f"sub-command {sub!r} does not match any allowlist entry",
                    )
        return CommandPolicyVerdict(allowed=True, reason="all sub-commands satisfy the policy")

    def check(self, command: str) -> None:
        """Raise ``SandboxCommandRejected`` if ``command`` is denied."""
        verdict = self.evaluate(command)
        if not verdict.allowed:
            raise SandboxCommandRejected(command=command, reason=verdict.reason)
