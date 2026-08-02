from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from griffe import Docstring, DocstringSectionKind, DocstringStyle, parse_auto
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from troopai.adk.exceptions import UserError
from troopai.adk.run.context import RunContext
from troopai.adk.tools.tool_context import ExecutionAwareToolContext, HistoryAwareToolContext, ToolContext

logger = logging.getLogger(__name__)

# Type alias for schema which can be a Pydantic model class or a JSON schema dict
FunctionToolSchema = type[BaseModel] | dict[str, Any]


@dataclass
class FunctionSchema:
    """Captures the schema for a Python function, in preparation for sending it to an LLM as a tool.

    Attributes:
        name: The name of the function.
        description: The description of the function.
        signature: The introspected signature of the function.
        schema: The Pydantic model (or JSON schema dict) for the function's
            parameters.  Used for input validation and to generate the JSON
            schema for the LLM.
        takes_context: Whether the function takes a ``RunContext`` or
            ``ToolContext`` argument as its first parameter.
        execution_aware: Whether the function's context parameter is
            specifically ``ExecutionAwareToolContext``.  When ``True``, the
            Runner constructs an extended context with read-only execution
            state snapshots (usage, turns, messages, tokens).
        history_aware: Whether the function's context parameter is
            specifically ``HistoryAwareToolContext``.  When ``True``, the
            Runner constructs a context with a read-only snapshot of
            conversation history as Layer 3 ``RunItem``\\s, plus all
            execution state.  Implies ``execution_aware=True``.
    """

    name: str
    """The name of the function."""

    description: str | None
    """The description of the function."""

    signature: inspect.Signature
    """The signature of the function."""

    schema: FunctionToolSchema
    """The Pydantic model for the function's parameters.
    Used for input validation and to generate the JSON schema for the LLM."""

    takes_context: bool = False
    """Whether the function takes a RunContext or ToolContext argument (must be the first argument)."""

    execution_aware: bool = False
    """Whether the function's context param is specifically ``ExecutionAwareToolContext``.

    When ``True``, the Runner constructs an extended context with
    read-only execution state snapshots (usage, turns, messages, tokens).
    """

    history_aware: bool = False
    """Whether the function's context param is specifically ``HistoryAwareToolContext``.

    When ``True``, the Runner constructs a context with a read-only
    snapshot of conversation history as Layer 3 RunItems, plus all
    execution state.  Implies ``execution_aware=True``.
    """

    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """Convert validated data from the Pydantic model into ``(args, kwargs)``.

        Produces positional and keyword argument lists suitable for calling
        the original function.

        Args:
            data: The validated Pydantic model instance.

        Returns:
            A two-tuple ``(positional_args, keyword_args)`` ready to be
            unpacked as ``func(*positional_args, **keyword_args)``.
        """
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        seen_var_positional = False

        # Use enumerate() so we can skip the first parameter if it's context.
        for idx, (name, param) in enumerate(self.signature.parameters.items()):
            # If the function takes context and this is the first parameter, skip it.
            if self.takes_context and idx == 0:
                continue

            value = getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                # e.g. *args: extend positional args and mark that *args is now seen
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                # e.g. **kwargs handling
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                # Before *args, add to positional args. After *args, add to keyword args.
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                # For KEYWORD_ONLY parameters, always use keyword args.
                keyword_args[name] = value

        return positional_args, keyword_args


@dataclass
class FunctionDocumentation:
    """Contains metadata about a Python function, extracted from its docstring.

    Attributes:
        name: The name of the function, via ``__name__``.
        description: The description of the function, derived from the
            docstring.
        parameters_description: A mapping of parameter names to their
            descriptions, derived from the docstring.  ``None`` if the
            docstring contains no parameter section.
    """

    name: str
    """The name of the function, via `__name__`."""

    description: str | None
    """The description of the function, derived from the docstring."""

    parameters_description: Mapping[str, str] | None
    """The parameters description of the function, derived from the docstring."""


@contextlib.contextmanager
def _suppress_griffe_logging():
    """Suppresses warnings about missing annotations for params."""
    griffe_logger = logging.getLogger("griffe")
    previous_level = griffe_logger.getEffectiveLevel()
    griffe_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        griffe_logger.setLevel(previous_level)


def _strip_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return the underlying annotation and any metadata from ``typing.Annotated``.

    Args:
        annotation: A type annotation, potentially wrapped in one or more
            ``Annotated[T, ...]`` layers.

    Returns:
        A two-tuple ``(inner_annotation, metadata)`` where
        *inner_annotation* is the unwrapped base type and *metadata*
        collects all extra ``Annotated`` arguments in order.
    """
    metadata: tuple[Any, ...] = ()
    ann = annotation

    # Unwrap ONLY typing.Annotated layers. CPython special-cases
    # get_origin(Annotated[T, ...]) to return Annotated, so this is the exact
    # discriminator. The previous `len(args) > 1` test matched ANY multi-arg
    # generic and silently stripped it to its first argument — corrupting
    # dict[k, v] -> k, tuple[a, b] -> a, X | Y -> X, and Literal[a, b] -> a.
    while get_origin(ann) is Annotated:
        args = get_args(ann)
        ann = args[0]
        metadata = (*metadata, *args[1:])

    return ann, metadata


def _extract_description_from_metadata(metadata: tuple[Any, ...]) -> str | None:
    """Extract a human-readable description from ``Annotated`` metadata if present.

    Args:
        metadata: The extra arguments collected from ``Annotated[T, ...]``
            wrappers.

    Returns:
        The first ``str`` item found in *metadata*, or ``None`` if no
        string description is present.
    """
    for item in metadata:
        if isinstance(item, str):
            return item
    return None


def _extract_field_infos_from_metadata(metadata: tuple[Any, ...]) -> tuple[FieldInfo, ...]:
    """Extract Pydantic ``Field`` metadata from ``Annotated`` metadata."""
    return tuple(item for item in metadata if isinstance(item, FieldInfo))


def _build_model_field(
    *,
    default: Any,
    description: str | None,
    metadata: tuple[Any, ...],
) -> Any:
    """Build a Pydantic field definition while preserving ``Annotated`` metadata."""
    field_infos = _extract_field_infos_from_metadata(metadata)
    if len(field_infos) > 0:
        overrides: dict[str, Any] = {
            "default": ... if default is inspect.Parameter.empty else default,
        }
        if description is not None:
            overrides["description"] = description
        return FieldInfo.merge_field_infos(*field_infos, **overrides)

    if default is inspect.Parameter.empty:
        if description is not None:
            return Field(description=description)
        return ...

    return Field(default=default, description=description)


def _is_context_type(annotation: Any) -> bool:
    """Check whether *annotation* is a context type (``RunContext`` or ``ToolContext``).

    Args:
        annotation: A type annotation or ``inspect.Parameter.empty``.

    Returns:
        ``True`` if the annotation is ``RunContext`` or any ``ToolContext``
        subclass (including ``ExecutionAwareToolContext`` and
        ``HistoryAwareToolContext``).
    """
    if annotation is inspect.Parameter.empty:
        return False

    annotation, _ = _strip_annotated(annotation)
    origin = get_origin(annotation) or annotation

    # Handle RunContext
    if isinstance(origin, type) and issubclass(origin, RunContext):
        return True

    # Handle ToolContext (includes ExecutionAwareToolContext via subclass)
    return bool(isinstance(origin, type) and issubclass(origin, ToolContext))


def _is_execution_aware_context(annotation: Any) -> bool:
    """Check whether *annotation* is specifically ``ExecutionAwareToolContext``.

    Args:
        annotation: A type annotation or ``inspect.Parameter.empty``.

    Returns:
        ``True`` if the annotation is ``ExecutionAwareToolContext`` or a
        subclass of it.
    """
    if annotation is inspect.Parameter.empty:
        return False

    annotation, _ = _strip_annotated(annotation)
    origin = get_origin(annotation) or annotation

    return bool(isinstance(origin, type) and issubclass(origin, ExecutionAwareToolContext))


def _is_history_aware_context(annotation: Any) -> bool:
    """Check whether *annotation* is specifically ``HistoryAwareToolContext``.

    Args:
        annotation: A type annotation or ``inspect.Parameter.empty``.

    Returns:
        ``True`` if the annotation is ``HistoryAwareToolContext`` or a
        subclass of it.
    """
    if annotation is inspect.Parameter.empty:
        return False

    annotation, _ = _strip_annotated(annotation)
    origin = get_origin(annotation) or annotation

    return bool(isinstance(origin, type) and issubclass(origin, HistoryAwareToolContext))


def generate_function_documentation(
    function: Callable[..., Any], style: DocstringStyle | None = None
) -> FunctionDocumentation:
    """Generate documentation for a Python function by parsing its docstring.

    Args:
        function: The Python function to document.
        style: The docstring style to use for parsing.  If ``None``,
            auto-detects the style.

    Returns:
        The extracted function documentation.
    """
    docstring_text = inspect.getdoc(function)

    if docstring_text is None:
        return FunctionDocumentation(name=function.__name__, description=None, parameters_description=None)

    docstring = Docstring(docstring_text, lineno=1)

    with _suppress_griffe_logging():
        if style is not None:
            # Use the explicitly provided style
            sections = docstring.parse(style)
        else:
            # Auto-detect the style
            style_order: list[DocstringStyle] = ["google", "numpy", "sphinx"]
            sections = parse_auto(
                docstring,
                style_order=style_order,
                default="google",  # Fallback if detection fails
            )

    description: str | None = next(
        (section.value for section in sections if section.kind == DocstringSectionKind.text),
        None,
    )

    parameters_description: Mapping[str, str] = {
        param.name: param.description.strip() if param.description is not None else ""
        for section in sections
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }

    return FunctionDocumentation(
        name=function.__name__,
        description=description.strip() if description is not None else None,
        parameters_description=parameters_description if len(parameters_description) > 0 else None,
    )


def function_schema(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    docstring_style: DocstringStyle | None = None,
    parse_docstring: bool = False,
) -> FunctionSchema:
    """Extract the schema of a Python function, including its name, description, and input parameters.

    Args:
        function: The Python function to extract the schema from.
        name: Custom name for the function.  If ``None``, uses
            ``function.__name__``.
        description: Custom description for the function.  If ``None``
            and *parse_docstring* is ``True``, attempts to extract the
            description from the function's docstring.
        docstring_style: The docstring style to use for parsing.  If
            ``None``, auto-detects the style.
        parse_docstring: Whether to parse the function's docstring for
            its description and parameter descriptions.

    Returns:
        The extracted function schema.

    Raises:
        UserError: If the function has a ``*args`` or ``**kwargs``
            parameter — variadic parameters are not supported as tool
            inputs.
        ValueError: If a ``RunContext`` or ``ToolContext`` parameter
            appears at a non-first position in the function signature.
    """
    # Extract docstring info if requested
    documentation: FunctionDocumentation | None = None
    parameters_description: dict[str, str] = {}

    if parse_docstring:
        documentation = generate_function_documentation(function, style=docstring_style)
        parameters_description = dict(documentation.parameters_description or {})

    # Get function signature and type hints
    sig = inspect.signature(function)
    params = list(sig.parameters.items())

    # Get type hints with evaluated forward references
    try:
        type_hints = get_type_hints(function, include_extras=True)
    except NameError as e:
        logger.warning("get_type_hints failed for %s: %s", function.__name__, e)
        type_hints = {}

    # Detect if function takes context and filter it out
    takes_context = False
    execution_aware = False
    history_aware = False
    filtered_params: list[tuple[str, inspect.Parameter]] = []

    if len(params) > 0:
        first_name, first_param = params[0]
        ann = type_hints.get(first_name, first_param.annotation)

        if _is_context_type(ann):
            takes_context = True
            history_aware = _is_history_aware_context(ann)
            execution_aware = history_aware or _is_execution_aware_context(ann)
        else:
            filtered_params.append((first_name, first_param))

        # For parameters other than the first, raise error if any use context types
        for param_name, param in params[1:]:
            ann = type_hints.get(param_name, param.annotation)
            if _is_context_type(ann):
                raise ValueError(
                    f"RunContext/ToolContext param found at non-first position in function {function.__name__}"
                )
            filtered_params.append((param_name, param))
    else:
        filtered_params = params

    # Build Pydantic field definitions
    fields: dict[str, Any] = {}

    for field_name, param in filtered_params:
        ann = type_hints.get(field_name, param.annotation)
        default = param.default

        # Strip Annotated wrapper if present
        ann, metadata = _strip_annotated(ann)

        # If there's no type hint, assume Any
        if ann is inspect.Parameter.empty:
            ann = Any

        # Get field description from docstring or Annotated metadata
        field_description = parameters_description.get(field_name)
        if field_description is None:
            field_description = _extract_description_from_metadata(metadata)

        # Handle different parameter kinds.
        #
        # ``*args`` and ``**kwargs`` are REJECTED at construction time:
        # a variadic tool signature has no clean JSON-Schema equivalent
        # the LLM can reliably target (every provider has different
        # quirks around array-of-anything and dict-of-anything, and
        # silently rewriting ``*args: int`` as ``args: list[int]`` was
        # the kind of hidden behaviour the plan calls out as forbidden).
        # Authors who actually need a list / dict parameter spell it
        # explicitly: ``def t(items: list[int])`` / ``def t(meta: dict[str, str])``.
        if param.kind == param.VAR_POSITIONAL:
            raise UserError(
                f"Function tool {function.__name__!r} has a *args parameter "
                f"({field_name!r}). Variadic positional parameters are not "
                "supported — declare a list parameter explicitly instead "
                f"(e.g. `{field_name}: list[int]`)."
            )

        if param.kind == param.VAR_KEYWORD:
            raise UserError(
                f"Function tool {function.__name__!r} has a **kwargs parameter "
                f"({field_name!r}). Variadic keyword parameters are not "
                "supported — declare a dict parameter explicitly instead "
                f"(e.g. `{field_name}: dict[str, str]`)."
            )

        fields[field_name] = (
            ann,
            _build_model_field(
                default=default,
                description=field_description,
                metadata=metadata,
            ),
        )

    # Create dynamic Pydantic model
    func_name = name or function.__name__
    dynamic_model = create_model(
        f"{func_name}_args",
        __base__=BaseModel,
        **fields,
    )

    return FunctionSchema(
        name=func_name,
        description=description or (documentation.description if documentation is not None else None),
        schema=dynamic_model,  # Store Pydantic model for validation
        signature=sig,
        takes_context=takes_context,
        execution_aware=execution_aware,
        history_aware=history_aware,
    )
