"""Reference resolution for declarative agent configuration.

:func:`resolve_dotted_spec` turns a dotted-path string in a config document
(a tool function, an output-schema class, an edge-condition predicate) into
the live Python object it names.

This module is the single sanctioned dynamic-import boundary in the config
layer: it imports by a runtime-supplied string. That is acceptable here
precisely because it runs only at config-load time on operator-supplied
input — never on a hot path.

Security: resolving a reference imports the named module, which runs that
module's top-level code, and the resolved object is later called. Loading a
config therefore executes Python from the modules it references — load only
config files you trust, exactly as you would for a Python entry point.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from troopai.adk.exceptions import ConfigResolutionError

if TYPE_CHECKING:
    from troopai.adk.agents.agent_guardrails import AgentInputGuardrail, AgentOutputGuardrail
    from troopai.adk.prompts.system_prompt import DynamicSystemPrompt
    from troopai.adk.schemas import SchemaEnforcement
    from troopai.adk.schemas.agent_output_schema import AgentOutputSchema
    from troopai.adk.tools import FunctionTool

logger = logging.getLogger(__name__)


@contextmanager
def importable_dir(directory: Path) -> Iterator[None]:
    """Temporarily make a config file's directory importable.

    Dotted references in a config document name importable modules. Putting the
    config file's own directory on the import path for the duration of the load
    lets a sibling module resolve by its bare name regardless of how the config
    is loaded (a script entrypoint, a test runner, a library import). The same
    mechanism makes a ``config_path`` sub-agent file's own directory importable
    while that sub-agent is built.

    The directory is **appended** to ``sys.path`` (lowest precedence), not
    prepended: a sibling module resolves by its bare name only when that name
    does not already belong to an installed or standard-library module. A config
    directory that happens to contain a file named like a real module (``types.py``,
    ``tools.py``) therefore cannot shadow the genuine one during the load.

    The directory is resolved to an absolute path so the entry is stable across
    later working-directory changes. If it is already on ``sys.path`` (or is not
    an existing directory) the path is left untouched. Only the entry this
    manager itself adds is removed on exit; an entry it did not add is left in
    place. Config loading is single-threaded; this manager is not safe under
    concurrent loads (two callers can both pass the "already present" guard and
    each append, leaving a duplicate after one removes its entry).

    Args:
        directory: The directory to make importable for the duration of the
            ``with`` body.

    Yields:
        ``None``; the prepared ``sys.path`` is in effect for the body.
    """
    resolved = directory.resolve()
    entry = str(resolved)
    if not resolved.is_dir() or entry in sys.path:
        logger.debug("importable_dir: leaving sys.path unchanged for %r", entry)
        yield
        return

    sys.path.append(entry)
    logger.debug("importable_dir: appended %r to sys.path", entry)
    try:
        yield
    finally:
        # Remove only the entry we added. A concurrent loader could in principle
        # have removed it already; tolerate that in this cleanup path rather than
        # raising over it.
        try:
            sys.path.remove(entry)
            logger.debug("importable_dir: removed %r from sys.path", entry)
        except ValueError:
            logger.debug("importable_dir: %r already absent from sys.path", entry)


def resolve_dotted_spec(spec: str) -> Any:
    """Resolve a dotted-path reference to the Python object it names.

    Accepts two forms:

    - ``"package.module:attr"`` — the explicit, unambiguous colon form.
    - ``"package.module.attr"`` — the dotted form; the final dotted segment
      is taken as the attribute and the rest as the module path.

    Args:
        spec: A dotted-path reference to an importable Python symbol.

    Returns:
        The resolved object (function, class, or any module attribute).

    Raises:
        ConfigResolutionError: If ``spec`` has no module/attribute separator
            or a relative module path, names a module that cannot be imported
            (or that raises while importing), or names an attribute the module
            does not define. The raised error quotes ``spec`` so the offending
            config entry is easy to locate.
    """
    # Prefer the explicit colon form; fall back to the final-dot split. The
    # colon form is unambiguous; the dotted fallback always takes the final
    # segment as the attribute, which cannot express nested attribute access.
    if ":" in spec:
        module_path, attr = spec.split(":", 1)
    elif "." in spec:
        module_path, attr = spec.rsplit(".", 1)
    else:
        raise ConfigResolutionError(
            f"Cannot resolve reference {spec!r}: expected the form 'package.module:attr' or 'package.module.attr'."
        )

    if len(module_path) == 0 or len(attr) == 0:
        raise ConfigResolutionError(
            f"Cannot resolve reference {spec!r}: both a module path and an attribute name are required."
        )

    if module_path.startswith("."):
        raise ConfigResolutionError(
            f"Cannot resolve reference {spec!r}: relative module paths are not supported; use an absolute path."
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigResolutionError(
            f"Cannot resolve reference {spec!r}: module {module_path!r} could not be imported ({exc})."
        ) from exc
    except Exception as exc:
        # The module exists but raised while importing (its top-level code
        # failed). Surface it as a config error with the offending spec rather
        # than leaking an opaque error from deep inside an unrelated module.
        raise ConfigResolutionError(
            f"Cannot resolve reference {spec!r}: importing module {module_path!r} raised {type(exc).__name__}: {exc}."
        ) from exc

    try:
        resolved = getattr(module, attr)
    except AttributeError as exc:
        raise ConfigResolutionError(
            f"Cannot resolve reference {spec!r}: module {module_path!r} has no attribute {attr!r}."
        ) from exc

    logger.debug("Resolved config reference %r -> %r", spec, resolved)
    return resolved


def resolve_function_tool(ref: str) -> FunctionTool:
    """Resolve a dotted reference to a ``FunctionTool``.

    Args:
        ref: A dotted-path reference to a ``FunctionTool`` (e.g. a function
            decorated with ``@function_tool``).

    Returns:
        The resolved ``FunctionTool``.

    Raises:
        ConfigResolutionError: If ``ref`` is unresolvable or resolves to
            something that is not a ``FunctionTool``.
    """
    from troopai.adk.tools import FunctionTool

    obj = resolve_dotted_spec(ref)
    if not isinstance(obj, FunctionTool):
        raise ConfigResolutionError(
            f"Tool reference {ref!r} resolved to {type(obj).__name__}, expected a FunctionTool. "
            "Decorate the function with @function_tool."
        )
    return obj


def resolve_output_schema(ref: str, enforcement: SchemaEnforcement) -> AgentOutputSchema:
    """Resolve a dotted reference to an output-schema class and wrap it.

    Args:
        ref: A dotted-path reference to the output-schema class.
        enforcement: How the generated JSON schema is processed.

    Returns:
        An ``AgentOutputSchema`` wrapping the resolved class.

    Raises:
        ConfigResolutionError: If ``ref`` is unresolvable or resolves to
            something that is not a class.
    """
    from troopai.adk.schemas.agent_output_schema import AgentOutputSchema

    obj = resolve_dotted_spec(ref)
    if not isinstance(obj, type):
        raise ConfigResolutionError(
            f"output_schema reference {ref!r} resolved to {type(obj).__name__}, expected a class."
        )
    return AgentOutputSchema(obj, schema_enforcement=enforcement)


def resolve_input_guardrail(ref: str) -> AgentInputGuardrail[Any]:
    """Resolve a dotted reference to an ``AgentInputGuardrail``.

    Args:
        ref: A dotted-path reference to an ``AgentInputGuardrail``.

    Returns:
        The resolved guardrail.

    Raises:
        ConfigResolutionError: If ``ref`` is unresolvable or resolves to a
            non-``AgentInputGuardrail``.
    """
    from troopai.adk.agents.agent_guardrails import AgentInputGuardrail

    obj = resolve_dotted_spec(ref)
    if not isinstance(obj, AgentInputGuardrail):
        raise ConfigResolutionError(
            f"Guardrail reference {ref!r} resolved to {type(obj).__name__}, expected an AgentInputGuardrail."
        )
    return obj


def resolve_output_guardrail(ref: str) -> AgentOutputGuardrail[Any]:
    """Resolve a dotted reference to an ``AgentOutputGuardrail``.

    Args:
        ref: A dotted-path reference to an ``AgentOutputGuardrail``.

    Returns:
        The resolved guardrail.

    Raises:
        ConfigResolutionError: If ``ref`` is unresolvable or resolves to a
            non-``AgentOutputGuardrail``.
    """
    from troopai.adk.agents.agent_guardrails import AgentOutputGuardrail

    obj = resolve_dotted_spec(ref)
    if not isinstance(obj, AgentOutputGuardrail):
        raise ConfigResolutionError(
            f"Guardrail reference {ref!r} resolved to {type(obj).__name__}, expected an AgentOutputGuardrail."
        )
    return obj


def resolve_dynamic_prompt(ref: str) -> DynamicSystemPrompt:
    """Resolve a dotted reference to a ``DynamicSystemPrompt`` callable.

    Args:
        ref: A dotted-path reference to the callable.

    Returns:
        The resolved callable.

    Raises:
        ConfigResolutionError: If ``ref`` is unresolvable or not callable.
    """
    obj = resolve_dotted_spec(ref)
    if not callable(obj):
        raise ConfigResolutionError(
            f"Dynamic system-prompt reference {ref!r} resolved to a non-callable {type(obj).__name__}."
        )
    # DynamicSystemPrompt is a Callable type alias, not a class, so it cannot be
    # isinstance-checked; callability is verified above and the referenced callable's
    # signature is the config author's contract (the runner invokes it at run time).
    return cast("DynamicSystemPrompt", obj)
