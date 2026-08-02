"""Batch runner for the examples suite.

Discovers every runnable example (a file with an ``if __name__ ==
"__main__"`` guard under ``examples/``), classifies each by the API keys
and external infrastructure it needs, skips the ones whose prerequisites
are absent (with a logged reason), and runs the rest as isolated
subprocesses with a per-example timeout. Results — PASSED / FAILED /
SKIPPED / TIMEOUT — go to per-example log files plus a console summary,
and the failures are written to a rerun list.

Auto mode (``--auto-mode``, or ``TROOPAI_EXAMPLES_INTERACTIVE_MODE=auto``
in the environment) is injected into every subprocess so examples wired
with the ``auto_mode`` helpers run without a human at the terminal.

Run from the repo root, e.g.::

    python examples/run_examples.py --list                 # classify only
    python examples/run_examples.py --filter flows         # one topic
    python examples/run_examples.py --auto-mode --jobs 4    # full auto run

Logs are written under ``logs/run_examples/<timestamp>/`` (gitignored).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, Flag, auto
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

logger = logging.getLogger("run_examples")

AUTO_MODE_ENV = "TROOPAI_EXAMPLES_INTERACTIVE_MODE"
DEFAULT_TIMEOUT = 120.0
DEFAULT_JOBS = min(8, os.cpu_count() or 4)
DEFAULT_LOGS_DIR = REPO_ROOT / "logs" / "run_examples"
MAIN_GUARD = re.compile(r"if\s+__name__\s*==\s*[\"']__main__[\"']")
# Matches a bare ``input(`` builtin call — not the English word "input ("
# in prose (which has a space) nor an ``obj.input(`` attribute call.
RAW_INPUT = re.compile(r"(?<![\w.])input\(")
AUTO_MODE_MARKERS = ("is_auto_mode", "input_with_fallback", "confirm_with_fallback")
EXCLUDED_NAMES = {"run_examples.py", "auto_mode.py", "__init__.py"}


class Prereq(Flag):
    """A bitmask of the prerequisites an example needs to run for real."""

    NONE = 0
    ANTHROPIC = auto()
    OPENAI = auto()
    GEMINI = auto()
    E2B = auto()
    A2A_SERVER = auto()
    TEMPORAL = auto()
    RESTATE = auto()
    DOCKER = auto()
    K8S = auto()
    MCP_STDIO = auto()
    MCP_SERVER = auto()
    INTERACTIVE = auto()
    SERVER_MODE = auto()
    NETWORK = auto()
    PYMUPDF = auto()


class Status(Enum):
    """Terminal status of an example run."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"


# Resource prerequisites whose presence is checked against the local
# environment (keys set, ports open, tools installed). The policy flags
# (INTERACTIVE, SERVER_MODE, NETWORK) are never "satisfied" — they gate on CLI
# opt-in.
RESOURCE_FLAGS: tuple[Prereq, ...] = (
    Prereq.ANTHROPIC,
    Prereq.OPENAI,
    Prereq.GEMINI,
    Prereq.E2B,
    Prereq.A2A_SERVER,
    Prereq.TEMPORAL,
    Prereq.RESTATE,
    Prereq.DOCKER,
    Prereq.K8S,
    Prereq.MCP_STDIO,
    Prereq.MCP_SERVER,
    Prereq.PYMUPDF,
)

# Heuristic misclassifications corrected by hand: maps an example key to
# ``(flags_to_add, flags_to_remove)``. Kept small on purpose — the heuristics
# carry the bulk; this only patches the genuinely ambiguous files.
OVERRIDE_TABLE: dict[str, tuple[Prereq, Prereq]] = {
    # The A2A server IS the server, not a client of one.
    "examples/a2a/server_basic.py": (Prereq.SERVER_MODE, Prereq.A2A_SERVER),
    # These entrypoints load Claude-backed agents from sibling JSON files.
    "examples/config/run_config_agent.py": (Prereq.ANTHROPIC, Prereq.NONE),
    "examples/config/run_graph.py": (Prereq.ANTHROPIC, Prereq.NONE),
    "examples/config/run_swarm.py": (Prereq.ANTHROPIC, Prereq.NONE),
    "examples/config/run_topology.py": (Prereq.ANTHROPIC, Prereq.NONE),
}


@dataclass
class ExampleSpec:
    """A discovered example and the prerequisites it was classified with.

    Attributes:
        path: Absolute path to the example file.
        key: Repo-root-relative POSIX path, used for display and filtering.
        prereqs: The prerequisites detected for this example.
    """

    path: Path
    key: str
    prereqs: Prereq


@dataclass
class ExampleResult:
    """Outcome of attempting to run a single example.

    Attributes:
        key: Repo-root-relative POSIX path of the example.
        status: Terminal status of the attempt.
        prereqs: The prerequisites the example was classified with.
        duration_s: Wall-clock seconds the subprocess ran (0 when skipped).
        returncode: Subprocess exit code, or ``None`` when not run or killed
            by timeout.
        skip_reason: Human-readable reason, set only when ``SKIPPED``.
        log_file: Per-example log path, or ``None`` when skipped.
        error: Runner-internal error text (e.g. spawn failure), if any.
    """

    key: str
    status: Status
    prereqs: Prereq
    duration_s: float = 0.0
    returncode: int | None = None
    skip_reason: str = ""
    log_file: Path | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _contains_any(source: str, needles: tuple[str, ...]) -> bool:
    """Return whether any needle appears in the source text."""
    return any(needle in source for needle in needles)


def _llm_prereqs(source: str) -> Prereq:
    """Detect which LLM provider key(s) an example needs."""
    prereqs = Prereq.NONE
    if _contains_any(source, ("AnthropicLLM", "anthropic/", "claude-", "ANTHROPIC_API_KEY")):
        prereqs |= Prereq.ANTHROPIC
    if _contains_any(
        source,
        ("OpenAIChatCompletionsLLM", "OpenAIResponsesLLM", "openai/", "OPENAI_API_KEY", '"gpt-', "text-embedding-3"),
    ):
        prereqs |= Prereq.OPENAI
    if _contains_any(source, ("GeminiLLM", "gemini/", "gemini-", "GEMINI_API_KEY", "GOOGLE_API_KEY")):
        prereqs |= Prereq.GEMINI
    if "E2B_API_KEY" in source:
        prereqs |= Prereq.E2B
    # A bare ``Agent(`` with no explicit provider uses the default Anthropic
    # model, so it needs the Anthropic key.
    if "Agent(" in source and prereqs == Prereq.NONE:
        prereqs |= Prereq.ANTHROPIC
    return prereqs


def _infra_prereqs(source: str) -> Prereq:
    """Detect external infrastructure an example needs (servers, daemons)."""
    prereqs = Prereq.NONE
    if _contains_any(source, ("uvicorn.run(", "hypercorn")):
        prereqs |= Prereq.SERVER_MODE
    if _contains_any(source, ("temporalio", "localhost:7233")):
        prereqs |= Prereq.TEMPORAL
    if _contains_any(source, ("import restate", "from restate")):
        prereqs |= Prereq.RESTATE
    if _contains_any(source, ("DockerSandboxClient", "sandbox.clients.docker")):
        prereqs |= Prereq.DOCKER
    if _contains_any(source, ("K8sPodSandboxClient", "K8sSandboxClient")):
        prereqs |= Prereq.K8S
    if "MCPServerStdio" in source:
        prereqs |= Prereq.MCP_STDIO
    if _contains_any(source, ("MCPServerStreamableHttp", "MCPServerSse", "MCPServerWebsocket")):
        prereqs |= Prereq.MCP_SERVER
    if "localhost:8080" in source:
        prereqs |= Prereq.A2A_SERVER
    return prereqs


def _interaction_prereqs(source: str) -> Prereq:
    """Detect whether an example blocks on interactive stdin."""
    # Examples wired for auto mode handle non-interactive runs themselves.
    if _contains_any(source, AUTO_MODE_MARKERS):
        return Prereq.NONE
    if "run_demo_loop" in source:
        return Prereq.INTERACTIVE
    if RAW_INPUT.search(source) is not None and "except EOFError" not in source:
        return Prereq.INTERACTIVE
    return Prereq.NONE


def _marker_prereqs(source: str) -> Prereq:
    """Detect explicit ``# requires:`` markers an example declares.

    An example that downloads over the network and parses a PDF marks itself
    ``# requires: network`` so it is skipped by default (network runs are flaky)
    and only runs under ``--allow-network`` with the PDF deps installed.
    """
    if "# requires: network" in source:
        return Prereq.NETWORK | Prereq.PYMUPDF
    return Prereq.NONE


def classify_example(key: str, source: str) -> Prereq:
    """Classify an example's prerequisites from its source text.

    Combines heuristics (LLM keys, infrastructure, interactivity) then
    applies any hand-tuned override for ``key``.

    Args:
        key: Repo-root-relative POSIX path of the example.
        source: Full source text of the example file.

    Returns:
        The combined prerequisite bitmask.
    """
    prereqs = _llm_prereqs(source) | _infra_prereqs(source) | _interaction_prereqs(source) | _marker_prereqs(source)
    add, remove = OVERRIDE_TABLE.get(key, (Prereq.NONE, Prereq.NONE))
    return (prereqs | add) & ~remove


def _prereq_names(prereqs: Prereq) -> str:
    """Render a prereq bitmask as a compact ``A|B|C`` string."""
    names = [flag.name for flag in Prereq if flag != Prereq.NONE and flag in prereqs]
    if len(names) == 0:
        return "-"
    return "|".join(name for name in names if name is not None)


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------


def _env_key_present(*names: str) -> bool:
    """Return whether any of the named environment variables is non-empty."""
    return any(len(os.environ.get(name, "")) > 0 for name in names)


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return whether a TCP connection to ``host:port`` succeeds quickly."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _module_available(name: str) -> bool:
    """Return whether an importable module/package is installed."""
    return importlib.util.find_spec(name) is not None


def _command_ok(command: list[str]) -> bool:
    """Return whether ``command`` runs and exits 0 within a short timeout."""
    if shutil.which(command[0]) is None:
        return False
    try:
        completed = subprocess.run(command, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


class PrereqChecker:
    """Evaluates (once, then caches) whether each prerequisite is satisfied."""

    def __init__(self) -> None:
        self._cache: dict[Prereq, bool] = {}

    def satisfied(self, prereq: Prereq) -> bool:
        """Return whether a single resource prerequisite is met locally.

        Args:
            prereq: A single resource flag (not a combination).

        Returns:
            Whether the prerequisite is satisfied in this environment.
        """
        if prereq not in self._cache:
            self._cache[prereq] = self._evaluate(prereq)
        return self._cache[prereq]

    def _evaluate(self, prereq: Prereq) -> bool:
        if prereq is Prereq.ANTHROPIC:
            return _env_key_present("ANTHROPIC_API_KEY")
        if prereq is Prereq.OPENAI:
            return _env_key_present("OPENAI_API_KEY")
        if prereq is Prereq.GEMINI:
            return _env_key_present("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if prereq is Prereq.E2B:
            return _env_key_present("E2B_API_KEY")
        if prereq is Prereq.A2A_SERVER:
            return _tcp_open("localhost", 8080)
        if prereq is Prereq.TEMPORAL:
            return _module_available("temporalio") and _tcp_open("localhost", 7233)
        if prereq is Prereq.RESTATE:
            return _module_available("restate") and _tcp_open("localhost", 9080)
        if prereq is Prereq.DOCKER:
            return _command_ok(["docker", "info"])
        if prereq is Prereq.K8S:
            return _command_ok(["kubectl", "cluster-info"])
        if prereq is Prereq.MCP_STDIO:
            return shutil.which("npx") is not None
        if prereq is Prereq.MCP_SERVER:
            return _tcp_open("localhost", 4000)
        if prereq is Prereq.PYMUPDF:
            return _module_available("pymupdf") and _module_available("lingua")
        raise ValueError(f"unhandled resource prerequisite: {prereq!r}")


# Policy flags gate on a CLI opt-in rather than an environment check.
POLICY_SKIPS: tuple[tuple[Prereq, str, str], ...] = (
    (Prereq.SERVER_MODE, "server-mode (blocks the process)", "include_server"),
    (Prereq.INTERACTIVE, "interactive stdin", "include_interactive"),
    (Prereq.NETWORK, "network access", "allow_network"),
)


def _missing_resources(prereqs: Prereq, checker: PrereqChecker) -> list[Prereq]:
    """Return the resource prerequisites that are not satisfied locally."""
    return [flag for flag in RESOURCE_FLAGS if flag in prereqs and not checker.satisfied(flag)]


def should_skip(prereqs: Prereq, checker: PrereqChecker, args: argparse.Namespace) -> str | None:
    """Return a skip reason for an example, or ``None`` if it should run.

    Args:
        prereqs: The example's prerequisite bitmask.
        checker: Resource-presence checker.
        args: Parsed CLI arguments (consulted for force/include flags).

    Returns:
        A human-readable skip reason, or ``None`` to run the example.
    """
    for flag, reason, attr in POLICY_SKIPS:
        if flag in prereqs and not getattr(args, attr):
            return f"{reason}; pass --{attr.replace('_', '-')}"
    if args.force:
        return None
    missing = _missing_resources(prereqs, checker)
    if len(missing) > 0:
        return "missing prerequisite(s): " + ", ".join(_prereq_names(flag) for flag in missing)
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _is_candidate(path: Path) -> bool:
    """Return whether a path could be a runnable example entry point."""
    if path.name in EXCLUDED_NAMES:
        return False
    return "__pycache__" not in path.parts


def discover(filters: Sequence[str]) -> list[ExampleSpec]:
    """Discover and classify runnable examples, optionally filtered.

    Args:
        filters: Substrings; an example is kept if its key contains any of
            them. An empty sequence keeps everything.

    Returns:
        Classified example specs, sorted by key.
    """
    specs: list[ExampleSpec] = []
    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        if not _is_candidate(path):
            continue
        source = path.read_text(encoding="utf-8")
        if MAIN_GUARD.search(source) is None:
            continue
        key = path.relative_to(REPO_ROOT).as_posix()
        if len(filters) > 0 and not any(needle in key for needle in filters):
            continue
        specs.append(ExampleSpec(path=path, key=key, prereqs=classify_example(key, source)))
    return specs


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _log_path(run_dir: Path, spec: ExampleSpec) -> Path:
    """Return the per-example log file path under ``run_dir``."""
    safe = spec.key.replace("/", "__").removesuffix(".py")
    return run_dir / f"{safe}.log"


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    """Kill a timed-out subprocess and its whole process group.

    Falls back to ``proc.kill()`` if the process-group signal fails, and logs
    (rather than raises) if even that fails, so one unkillable process cannot
    abort the whole batch run.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError) as exc:
        logger.debug("killpg failed (%s); falling back to proc.kill()", exc)
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError) as kill_exc:
            logger.warning("could not kill timed-out subprocess pid %s: %s", proc.pid, kill_exc)
    proc.wait()


def _result(
    spec: ExampleSpec,
    status: Status,
    start: float,
    *,
    log_file: Path | None = None,
    returncode: int | None = None,
    error: str = "",
) -> ExampleResult:
    """Build an ``ExampleResult``, stamping elapsed wall-clock since ``start``."""
    return ExampleResult(
        key=spec.key,
        status=status,
        prereqs=spec.prereqs,
        duration_s=time.monotonic() - start,
        returncode=returncode,
        log_file=log_file,
        error=error,
    )


def run_example(spec: ExampleSpec, log_file: Path, timeout: float, auto_mode: bool) -> ExampleResult:
    """Run one example as a subprocess, capturing output to ``log_file``.

    The subprocess runs from the repo root with stdin closed and, in auto
    mode, ``TROOPAI_EXAMPLES_INTERACTIVE_MODE=auto`` in its environment. It is
    placed in its own session so a timeout can kill the whole process tree.

    Args:
        spec: The example to run.
        log_file: Destination for combined stdout/stderr.
        timeout: Seconds before the subprocess is killed.
        auto_mode: Whether to inject the auto-mode environment variable.

    Returns:
        The result, with status PASSED / FAILED / TIMEOUT.
    """
    env = dict(os.environ)
    if auto_mode:
        env[AUTO_MODE_ENV] = "auto"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_file.open("wb") as handle:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(spec.path)],
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return _result(spec, Status.FAILED, start, log_file=log_file, error=str(exc))
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(proc)
            return _result(spec, Status.TIMEOUT, start, log_file=log_file)
    status = Status.PASSED if returncode == 0 else Status.FAILED
    return _result(spec, status, start, log_file=log_file, returncode=returncode)


def _log_progress(done: int, total: int, result: ExampleResult) -> None:
    """Log a single completed example's status line."""
    detail = ""
    if result.returncode is not None and result.returncode != 0:
        detail = f" (rc={result.returncode})"
    elif len(result.error) > 0:
        detail = f" (error: {result.error})"
    logger.info(
        "[%3d/%3d] %-8s %6.1fs  %s%s",
        done,
        total,
        result.status.value,
        result.duration_s,
        result.key,
        detail,
    )


def run_all(
    specs: Sequence[ExampleSpec],
    checker: PrereqChecker,
    args: argparse.Namespace,
    run_dir: Path,
) -> list[ExampleResult]:
    """Skip-or-run every spec and collect results.

    Args:
        specs: Classified examples to consider.
        checker: Resource-presence checker.
        args: Parsed CLI arguments.
        run_dir: Directory for this run's per-example logs.

    Returns:
        One result per spec (SKIPPED entries included).
    """
    results: list[ExampleResult] = []
    runnable: list[ExampleSpec] = []
    for spec in specs:
        reason = should_skip(spec.prereqs, checker, args)
        if reason is None:
            runnable.append(spec)
        else:
            results.append(ExampleResult(key=spec.key, status=Status.SKIPPED, prereqs=spec.prereqs, skip_reason=reason))
            logger.info("[skip]    %-8s %s  (%s)", Status.SKIPPED.value, spec.key, reason)
    total = len(runnable)
    logger.info("Running %d example(s); jobs=%d timeout=%.0fs", total, args.jobs, args.timeout)
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_example, spec, _log_path(run_dir, spec), args.timeout, args.auto_mode): spec
            for spec in runnable
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # The runner itself failing on one example must not abort the
                # whole batch — record it as FAILED and keep going.
                logger.error("runner error on %s: %s", spec.key, exc, exc_info=True)
                result = _result(spec, Status.FAILED, time.monotonic(), error=f"runner error: {exc}")
            results.append(result)
            done += 1
            _log_progress(done, total, result)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _rerun_path(args: argparse.Namespace) -> Path:
    """Return the rerun-list path (explicit, or the logs-dir default)."""
    if args.rerun_file is not None:
        return args.rerun_file
    return args.logs_dir / "latest_failures.txt"


def _rel(path: Path) -> str:
    """Render ``path`` relative to the repo root when possible, else as-is.

    Handles both absolute paths (the default log locations) and relative
    paths (e.g. a user-supplied ``--rerun-file``/``--logs-dir``), which
    ``Path.relative_to`` alone would reject.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_rerun(failures: Sequence[ExampleResult], args: argparse.Namespace) -> None:
    """Write the failing example keys to the rerun list, one per line."""
    path = _rerun_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(result.key for result in failures)
    path.write_text("\n".join(keys) + "\n", encoding="utf-8")
    logger.info("Wrote %d failing example(s) to %s", len(failures), _rel(path))


def write_summary(results: Sequence[ExampleResult], args: argparse.Namespace) -> int:
    """Log the run summary and write the rerun list; return the exit code.

    Args:
        results: All results from the run.
        args: Parsed CLI arguments.

    Returns:
        ``0`` when nothing failed or timed out, else ``1``.
    """
    counts = dict.fromkeys(Status, 0)
    for result in results:
        counts[result.status] += 1
    logger.info("=" * 64)
    logger.info(
        "Summary: %d passed, %d failed, %d timeout, %d skipped (of %d)",
        counts[Status.PASSED],
        counts[Status.FAILED],
        counts[Status.TIMEOUT],
        counts[Status.SKIPPED],
        len(results),
    )
    failures = [r for r in results if r.status in (Status.FAILED, Status.TIMEOUT)]
    for result in sorted(failures, key=lambda r: r.key):
        location = _rel(result.log_file) if result.log_file is not None else "-"
        logger.info("  %-8s %s  (log: %s)", result.status.value, result.key, location)
    if args.write_rerun and len(failures) > 0:
        _write_rerun(failures, args)
    return 1 if len(failures) > 0 else 0


def cmd_list(specs: Sequence[ExampleSpec], checker: PrereqChecker, args: argparse.Namespace) -> int:
    """Print the classification + would-run/skip verdict for each example.

    Args:
        specs: Classified examples.
        checker: Resource-presence checker.
        args: Parsed CLI arguments.

    Returns:
        Always ``0`` — this is a cost-free dry run.
    """
    logger.info("%-58s  %-9s  %s", "EXAMPLE", "VERDICT", "PREREQS")
    would_run = 0
    for spec in specs:
        reason = should_skip(spec.prereqs, checker, args)
        verdict = "RUN" if reason is None else "SKIP"
        if reason is None:
            would_run += 1
        suffix = "" if reason is None else f"  ({reason})"
        logger.info("%-58s  %-9s  %s%s", spec.key, verdict, _prereq_names(spec.prereqs), suffix)
    logger.info("=" * 64)
    logger.info("%d example(s): %d would run, %d would skip", len(specs), would_run, len(specs) - would_run)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    """Install a single stdout handler so all output flows through logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def _load_env() -> None:
    """Load the repo-root ``.env`` so prereq checks and subprocesses see keys.

    Each example also calls ``load_dotenv()`` itself; loading here means the
    runner's prerequisite checks observe the same keys, and they propagate to
    every subprocess through the environment overlay.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; relying on the ambient environment")
        return
    load_dotenv(REPO_ROOT / ".env")


def _resolve_filters(args: argparse.Namespace) -> list[str]:
    """Resolve filters from ``--filter`` or, failing that, a rerun file."""
    filters = list(args.filters)
    if len(filters) == 0 and args.rerun_file is not None and args.rerun_file.exists():
        lines = args.rerun_file.read_text(encoding="utf-8").splitlines()
        filters = [line.strip() for line in lines if len(line.strip()) > 0]
        logger.info("Reran %d example(s) from %s", len(filters), args.rerun_file)
    return filters


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Register the which-examples and how-to-run flags."""
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        dest="filters",
        metavar="SUBSTR",
        help="Run only examples whose path contains SUBSTR (repeatable, OR).",
    )
    parser.add_argument(
        "--jobs", type=int, default=DEFAULT_JOBS, metavar="N", help="Parallel workers (default: %(default)s)."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help="Per-example timeout before kill (default: %(default)s).",
    )
    parser.add_argument(
        "--auto-mode",
        action="store_true",
        dest="auto_mode",
        help="Inject TROOPAI_EXAMPLES_INTERACTIVE_MODE=auto into each subprocess.",
    )
    parser.add_argument(
        "--include-server",
        action="store_true",
        dest="include_server",
        help="Also run server-mode examples (they block; pair with --filter).",
    )
    parser.add_argument(
        "--include-interactive",
        action="store_true",
        dest="include_interactive",
        help="Also run interactive-stdin examples.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        dest="allow_network",
        help="Also run examples that download over the network (need the PDF deps installed).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Run examples even when their key/infra prerequisites are unmet."
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register the logging / reporting / dry-run flags."""
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        metavar="DIR",
        help="Root directory for per-run logs (default: logs/run_examples).",
    )
    parser.add_argument(
        "--rerun-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Failure list path. If it exists and --filter is absent, rerun its entries.",
    )
    parser.add_argument(
        "--no-write-rerun", action="store_false", dest="write_rerun", help="Do not write the failure rerun list."
    )
    parser.add_argument(
        "--list", action="store_true", help="Classify and show would-run/skip verdicts, then exit (no runs)."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python examples/run_examples.py",
        description="Discover, classify, and run the examples suite.",
    )
    _add_selection_args(parser)
    _add_output_args(parser)
    parser.set_defaults(write_rerun=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, discover examples, and run or list them.

    Args:
        argv: Optional explicit argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code: ``0`` on success, ``1`` when an example failed.
    """
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    _load_env()
    if not args.auto_mode and os.environ.get(AUTO_MODE_ENV, "").lower() == "auto":
        args.auto_mode = True
        logger.info("auto mode enabled via %s", AUTO_MODE_ENV)
    specs = discover(_resolve_filters(args))
    if len(specs) == 0:
        logger.warning("No examples matched the given filters.")
        return 0
    checker = PrereqChecker()
    if args.list:
        return cmd_list(specs, checker, args)
    run_dir = args.logs_dir / datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Logs: %s", _rel(run_dir))
    results = run_all(specs, checker, args, run_dir)
    return write_summary(results, args)


if __name__ == "__main__":
    sys.exit(main())
