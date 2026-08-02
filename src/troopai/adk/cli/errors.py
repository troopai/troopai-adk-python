"""Map framework load/config errors to clean CLI failures.

Config and resolution errors are user-input problems: the framework
already produces guiding messages for them, so commands surface the
message verbatim as a usage error (exit code 2, no traceback).
Everything else — including runtime errors raised by user tool code
during execution — propagates untouched (exit 1, full traceback),
because mislabeling a tool bug as a usage error hides the stack the
developer needs. Missing config files are guarded at the load seam,
not here.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import click

from troopai.adk.exceptions import ConfigParseError, ConfigResolutionError


def framework_errors[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """Decorate a command so framework load errors become usage errors.

    Args:
        f: The click command callback to wrap.

    Returns:
        The wrapped callback; config parse/resolution errors re-raise as
        :class:`click.UsageError` with the original guiding message.
    """

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return f(*args, **kwargs)
        except (ConfigParseError, ConfigResolutionError) as exc:
            raise click.UsageError(str(exc)) from exc

    return wrapper
