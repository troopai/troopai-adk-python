from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeVar, Union

T = TypeVar("T")

# Type alias for functions that may be sync or async
MaybeAwaitable = Union[T, Awaitable[T]]
