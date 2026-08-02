from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_import_troopai_adk_only_installs_null_handler() -> None:
    code = """
import json
import logging
import sys
import troopai.adk

logger = logging.getLogger("troopai.adk")
sys.stdout.write(json.dumps([type(handler).__name__ for handler in logger.handlers]))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(completed.stdout) == ["NullHandler"]
