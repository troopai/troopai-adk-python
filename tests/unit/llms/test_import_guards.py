"""Subprocess regression tests for the optional-provider import guards in
``troopai.adk.llms.__init__``.

The Gemini guard must swallow BOTH ``ModuleNotFoundError(name="google")`` and
``ModuleNotFoundError(name="google.genai")``. The latter is the real failure
shape when another ``google.*`` distribution (google-auth, google-cloud-*)
already created the ``google`` namespace package but the ``google-genai``
wheel is absent — swallowing only ``"google"`` let that common case crash the
entire ``llms`` import.

Each case runs in a fresh subprocess so a meta-path blocker installed before
import sees an interpreter with no cached ``google`` / ``troopai`` modules.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_blocked_import(block_name: str) -> subprocess.CompletedProcess[str]:
    """Import ``troopai.adk.llms`` with ``block_name`` masked as uninstalled."""
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        BLOCK = {block_name!r}

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == BLOCK or name.startswith(BLOCK + "."):
                    raise ModuleNotFoundError("No module named " + repr(name), name=BLOCK)
                return None

        sys.meta_path.insert(0, _Blocker())
        for cached in list(sys.modules):
            if cached == BLOCK or cached.startswith(BLOCK + "."):
                del sys.modules[cached]

        from troopai.adk.llms import LLM, GeminiLLM, GeminiConfig, OpenAIResponsesLLM
        assert LLM is not None, "LLM ABC must import"
        assert GeminiLLM is None, "GeminiLLM must degrade to None"
        assert GeminiConfig is None, "GeminiConfig must degrade to None"
        assert OpenAIResponsesLLM is not None, "OpenAI must remain importable"
        print("IMPORT_OK")
        """
    )
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)


def test_llms_import_survives_missing_google_genai() -> None:
    # THE regression: name == "google.genai" (google namespace present, genai
    # wheel absent) must not crash the whole llms import.
    result = _run_blocked_import("google.genai")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "IMPORT_OK" in result.stdout


def test_llms_import_survives_missing_google_namespace() -> None:
    # The original covered case: name == "google" stays swallowed.
    result = _run_blocked_import("google")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "IMPORT_OK" in result.stdout
