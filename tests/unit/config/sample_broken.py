"""A module that raises on import.

Referenced (never imported directly) by resolver tests to verify that a
module which fails during import surfaces as a ConfigResolutionError rather
than leaking the raw error.
"""

from __future__ import annotations

raise RuntimeError("boom on import")
