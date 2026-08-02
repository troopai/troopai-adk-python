"""Cost-aware sandbox backend selection.

A pluggable backend chooser: a ``SandboxSelector`` picks one
``SandboxCandidate`` (a client paired with its options) given the run's
``SandboxRequirements``, raising ``SandboxSelectionError`` when no
candidate qualifies. ``CheapestFirstSelector`` filters candidates whose
backend capabilities satisfy the requirements, then returns the one with
the lowest per-minute rate.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
from typing import TYPE_CHECKING, override

from troopai.adk.exceptions.exceptions import SandboxSelectionError

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.base import BaseSandboxClient, BaseSandboxClientOptions
    from troopai.adk.types.sandbox.cost import SandboxRequirements

logger = logging.getLogger(__name__)

__all__ = ["CheapestFirstSelector", "SandboxCandidate", "SandboxSelector"]


@dataclasses.dataclass(frozen=True)
class SandboxCandidate:
    """A backend client paired with the options used to create a session.

    Attributes:
        client: The backend client (carries ``cost`` + ``capabilities``).
        options: Backend-specific creation options, or ``None`` when the
            client supports default options.
    """

    client: BaseSandboxClient
    """The backend client to create a session from; the selector reads its
    ``cost`` rate card and ``capabilities``."""

    options: BaseSandboxClientOptions | None = None
    """Backend-specific creation options, or ``None`` when the client
    supports default options."""


class SandboxSelector(abc.ABC):
    """Picks one ``SandboxCandidate`` for a run from a candidate list."""

    @abc.abstractmethod
    def select(
        self,
        candidates: list[SandboxCandidate],
        requirements: SandboxRequirements,
    ) -> SandboxCandidate:
        """Return the chosen candidate or raise ``SandboxSelectionError``."""


class CheapestFirstSelector(SandboxSelector):
    """Choose the lowest-rate backend that satisfies the requirements.

    Unpriced backends (``cost is None``) sort after every priced one, and
    ties are broken by candidate order (the first stays).
    """

    @override
    def select(
        self,
        candidates: list[SandboxCandidate],
        requirements: SandboxRequirements,
    ) -> SandboxCandidate:
        if len(candidates) == 0:
            raise SandboxSelectionError("CheapestFirstSelector: no candidates provided")
        eligible = [c for c in candidates if c.client.capabilities.satisfies(requirements)]
        if len(eligible) == 0:
            raise SandboxSelectionError(f"No sandbox candidate satisfies requirements {requirements!r}")
        chosen = min(
            eligible,
            key=lambda c: c.client.cost.rate_key() if c.client.cost is not None else float("inf"),
        )
        logger.info("sandbox.selector chose backend=%s", chosen.client.backend_id)
        return chosen
