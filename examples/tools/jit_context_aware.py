"""JIT Context Aware Tool — multi-turn conversation with context pressure.

Demonstrates how JITContextAwareTool tools become naturally useful in a
multi-turn human-in-the-loop conversation where context accumulates.
The system prompt says nothing about JIT tools — the LLM discovers them
from their descriptions and decides when to use them.

The ``query_knowledge_base`` tool simulates a RAG lookup returning large
text blocks. Over multiple turns, the human keeps asking new questions
and the tight context budget (4K tokens) forces the agent to actively
manage its context — or lose earlier findings.

Usage:
    python examples/tools/jit_context_aware.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.context import CompactionConfig, ContextManagementConfig
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools import JITContextAwareTool
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.types.input import LLMInputContentItem, LLMInputEasyMessage
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated knowledge base — large text blocks
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: dict[str, str] = {
    "quantum_computing_overview": (
        "Quantum computing leverages quantum mechanical phenomena — superposition, "
        "entanglement, and interference — to process information in fundamentally "
        "different ways than classical computers. A quantum bit (qubit) can exist in "
        "a superposition of states, enabling parallel exploration of solution spaces. "
        "Key milestones: Google's Sycamore achieved quantum supremacy in 2019 (53 qubits). "
        "IBM's Condor reached 1,121 qubits in 2023. Error correction remains the primary "
        "bottleneck — current NISQ (Noisy Intermediate-Scale Quantum) devices have error "
        "rates of ~0.1-1% per gate. Logical qubits require ~1,000 physical qubits for "
        "error correction. Applications: cryptography (Shor's algorithm breaks RSA), "
        "optimization (QAOA), drug discovery (molecular simulation), and machine learning "
        "(quantum kernels). The quantum advantage threshold for practical problems is "
        "estimated at ~10,000 logical qubits, projected for 2028-2032."
    ),
    "quantum_error_correction": (
        "Quantum error correction (QEC) is the central challenge in scaling quantum "
        "computers. Unlike classical bits that are either 0 or 1, qubits decohere — "
        "losing their quantum state through interaction with the environment. Surface "
        "codes are the leading QEC approach: they arrange physical qubits in a 2D grid "
        "where data qubits are interspersed with 'syndrome' qubits that detect errors "
        "without collapsing the quantum state. The distance d of a surface code determines "
        "its error-correcting power: it can correct up to (d-1)/2 errors. Current state "
        "of the art: Google's Willow chip (2024) demonstrated that increasing code distance "
        "exponentially reduces error rates — a key threshold. IBM's roadmap targets "
        "100,000 physical qubits by 2033. Alternative approaches: bosonic codes (cat qubits), "
        "topological qubits (Microsoft's approach using Majorana fermions), and "
        "dynamically corrected gates. The overhead ratio (physical:logical qubits) is "
        "improving from ~1000:1 toward ~100:1 with better hardware and codes."
    ),
    "fusion_energy_status": (
        "Nuclear fusion — the process powering stars — promises virtually limitless "
        "clean energy. The fuel (deuterium and tritium) is abundant: deuterium from "
        "seawater, tritium bred from lithium. Key approaches: (1) Magnetic confinement "
        "(tokamaks): ITER in France is the largest, targeting first plasma in 2025 and "
        "D-T operations by 2035. It aims for Q=10 (10x energy output vs input). "
        "(2) Inertial confinement: NIF at Lawrence Livermore achieved ignition in Dec 2022, "
        "producing 3.15 MJ from 2.05 MJ of laser energy (Q=1.5). (3) Private startups: "
        "Commonwealth Fusion Systems (SPARC tokamak, high-temperature superconducting "
        "magnets), TAE Technologies (field-reversed configuration), Helion Energy "
        "(pulsed field-reversed), First Light Fusion (projectile fusion). Challenges: "
        "plasma confinement stability, neutron damage to reactor walls, tritium breeding "
        "ratio must exceed 1.0 for self-sufficiency, and economic competitiveness with "
        "renewables+storage. Timeline: first commercial fusion power projected 2035-2045."
    ),
    "fusion_materials_challenge": (
        "The materials challenge is arguably fusion's hardest engineering problem. "
        "D-T fusion produces 14.1 MeV neutrons — far more energetic than fission "
        "neutrons (2 MeV). These fast neutrons cause: (1) Atomic displacement cascades "
        "that create voids and swelling in structural materials. (2) Transmutation — "
        "neutron capture changes element composition, producing helium and hydrogen "
        "that cause embrittlement. (3) Activation — materials become radioactive, "
        "though with much shorter half-lives than fission waste. Leading candidate "
        "materials: Reduced Activation Ferritic-Martensitic (RAFM) steels like "
        "EUROFER97, silicon carbide composites (SiC/SiC) for high-temperature "
        "components, and tungsten alloys for plasma-facing components. The IFMIF-DONES "
        "facility will test materials under fusion-relevant neutron flux. No material "
        "has been qualified for the full 14 MeV neutron spectrum at the fluence levels "
        "needed for a commercial reactor (~150 dpa over 5 years)."
    ),
    "space_elevator_concept": (
        "A space elevator is a proposed structure connecting Earth's surface to "
        "geostationary orbit (35,786 km altitude) via a tether under tension. The "
        "concept: a counterweight beyond GEO keeps the tether taut through centrifugal "
        "force, while climbers ascend the tether carrying payloads. Cost reduction: "
        "current launch costs are ~$2,000/kg to LEO (SpaceX Falcon 9); a space elevator "
        "could potentially reduce this to ~$100/kg. The fundamental challenge is the "
        "tether material — it must have a specific strength (strength/density ratio) "
        "exceeding 50 MPa·m³/kg. Steel: ~0.05, Kevlar: ~2.5, Carbon nanotubes (theoretical): "
        "~47-60. Carbon nanotubes are the only known material approaching the requirement, "
        "but the longest lab-grown CNT fiber is ~50 cm with bulk strength far below "
        "theoretical. Alternative: lunar space elevator using existing materials (Zylon), "
        "since lunar gravity is 1/6 Earth's."
    ),
    "space_elevator_engineering": (
        "Engineering a space elevator involves challenges beyond the tether material. "
        "Tether dynamics: the 35,786 km cable would experience oscillations from Coriolis "
        "forces, lunar/solar gravitational perturbations, and wind loads in the lower "
        "atmosphere. Active damping systems and taper profiles are required. Climber "
        "design: electromagnetic or friction-based drive systems, power delivery via "
        "laser beaming from ground stations or solar panels. Speed: at 200 km/h, the "
        "trip to GEO takes ~7.5 days. Debris avoidance: the tether must dodge ~34,000 "
        "tracked objects in LEO/MEO, requiring lateral displacement capability or "
        "debris remediation. Anchor point: equatorial location required (candidates: "
        "Ecuador, Brazil, Indonesia, or ocean platform). Construction sequence: deploy "
        "initial seed tether from GEO, then bootstrap by climbing additional material. "
        "Estimated cost: $10-20 billion for the first elevator, with subsequent ones "
        "cheaper. ISEC (International Space Elevator Consortium) projects feasibility "
        "by 2050 if CNT manufacturing breakthroughs occur."
    ),
    "renewable_energy_comparison": (
        "For context on fusion's economic competition: solar PV costs dropped 99% since "
        "1976 (Swanson's law), reaching ~$20-30/MWh unsubsidized in 2025. Onshore wind: "
        "~$25-50/MWh. Battery storage (lithium-ion): ~$120/kWh in 2025, projected $50/kWh "
        "by 2030. The 'baseload' argument for fusion weakens as storage costs fall — a "
        "solar+storage system at $50/MWh competes with fusion's projected $60-80/MWh. "
        "However, fusion has advantages for: (1) high-latitude regions with poor solar "
        "resources, (2) industrial heat (>1000°C for steel, cement), (3) energy density "
        "for space applications, (4) minimal land use (~1 km² vs ~50 km² for equivalent "
        "solar farm). The real question is not 'will fusion work?' but 'will fusion be "
        "economically competitive by the time it's ready?' Grid-scale energy storage, "
        "advanced geothermal, and next-gen nuclear fission are all competitors."
    ),
    "advanced_computing_landscape": (
        "The computing landscape in 2026 spans multiple paradigms: (1) Classical: Moore's "
        "Law continues via chiplet architectures (AMD, Intel), 3D stacking (TSMC SoIC), "
        "and new transistor designs (GAAFETs at 2nm). (2) Quantum: NISQ era with 1000+ "
        "qubit processors, early fault-tolerant demonstrations. (3) Neuromorphic: Intel "
        "Loihi 2, IBM NorthPole — brain-inspired chips for AI inference at ~100x energy "
        "efficiency. (4) Photonic: Lightmatter, PsiQuantum — optical computing for "
        "specific ML workloads. (5) Analog: Mythic, Syntiant — in-memory computing "
        "for edge AI. The convergence trend: heterogeneous computing where quantum "
        "processors handle optimization subroutines, neuromorphic chips handle inference, "
        "and classical CPUs/GPUs orchestrate. Key bottleneck shifting from compute to "
        "memory bandwidth (the 'memory wall') — addressed by HBM4, CXL 3.0, and "
        "processing-in-memory architectures."
    ),
}


@function_tool(
    name="query_knowledge_base",
    description=(
        "Query an internal knowledge base on advanced technology topics. "
        "Returns detailed technical information. Available topics: "
        "quantum_computing_overview, quantum_error_correction, "
        "fusion_energy_status, fusion_materials_challenge, "
        "space_elevator_concept, space_elevator_engineering, "
        "renewable_energy_comparison, advanced_computing_landscape."
    ),
)
def query_knowledge_base(ctx: ToolContext, topic: str) -> str:
    """Query the knowledge base for a specific topic."""
    result = _KNOWLEDGE_BASE.get(topic)
    if result is None:
        available = ", ".join(_KNOWLEDGE_BASE.keys())
        return f"Topic '{topic}' not found. Available: {available}"
    return result


# ---------------------------------------------------------------------------
# Agent — system prompt says NOTHING about JIT tools
# ---------------------------------------------------------------------------

analyst = Agent(
    name="Technology Analyst",
    system_prompt=(
        "You are a senior technology analyst. Answer questions by querying the knowledge base. Be thorough and precise."
    ),
    tools=[
        query_knowledge_base,
        JITContextAwareTool(),
    ],
)

# Tight context budget with full JIT context management:
# - truncation: hard enforcement (drop oldest if over budget)
# - pressure_feedback: inject developer message so LLM sees pressure
# - forced_tool: FORCE manage_context call when pressure exceeds 80%
# No passive compaction — the agent must manage its own context.
# Budget: 8K tokens leaves room for the system prompt + tool schemas
# (~1.5K) and a few topic lookups before pressure kicks in around turn 3.
from troopai.adk.context import ContextEditingConfig

config = RunConfig(
    verbose=VerboseConfig(),
    context_management=ContextManagementConfig(
        max_context_tokens=8_000,
        truncation=True,
        pressure_feedback=True,
        forced_tool="manage_context",
        compaction=CompactionConfig(enabled=False),
        editing=ContextEditingConfig(clear_tool_results=False, clear_thinking_blocks=False),
    ),
)

# Pre-scripted human turns that progressively build context pressure
HUMAN_TURNS = [
    "Look up quantum_computing_overview and quantum_error_correction for me.",
    "Now look up fusion_energy_status and fusion_materials_challenge.",
    "Look up space_elevator_concept and space_elevator_engineering.",
    "Also check renewable_energy_comparison and advanced_computing_landscape.",
    (
        "Based on everything you've researched across all 8 topics, which "
        "technology is closest to practical impact and why? Reference specific "
        "data points from your earlier research."
    ),
]


# ---------------------------------------------------------------------------
# Multi-turn conversation loop
# ---------------------------------------------------------------------------


def _extract_tool_calls(new_items: list) -> list[dict]:
    """Extract tool call info from new_items (RunItems) for logging."""
    from troopai.adk.types.items.items import ToolCallItem, ToolCallOutputItem

    calls = []
    for item in new_items:
        if isinstance(item, ToolCallItem):
            calls.append(
                {
                    "tool": item.raw.name,
                    "args": item.raw.arguments[:80],
                }
            )
        elif isinstance(item, ToolCallOutputItem):
            output = str(item.raw.output)
            preview = output[:100] + "..." if len(output) > 100 else output
            calls.append(
                {
                    "result": preview,
                }
            )
    return calls


async def main() -> None:
    logger.info("=== JIT Context Aware Tool — Multi-Turn Demo ===\n")

    # Enable context management logging
    ctx_logger = logging.getLogger("troopai.adk.context")
    ctx_logger.setLevel(logging.INFO)
    jit_logger = logging.getLogger("troopai.adk.tools.builtin.jit_context_aware_tool")
    jit_logger.setLevel(logging.INFO)

    # Accumulate conversation history across turns
    history: list[LLMInputContentItem] = []
    jit = next(t for t in analyst.tools if isinstance(t, JITContextAwareTool))
    assert jit.store is not None  # Populated by JITContextAwareTool.__post_init__
    store = jit.store

    for turn_num, human_msg in enumerate(HUMAN_TURNS, 1):
        logger.info(f"\n{'━' * 60}")
        logger.info(f"  TURN {turn_num}")
        logger.info(f"{'━' * 60}")
        logger.info(f"  Human: {human_msg[:80]}{'...' if len(human_msg) > 80 else ''}")
        logger.info(f"  History messages: {len(history)}")

        # Build input
        user_msg: LLMInputEasyMessage = {"role": "user", "content": human_msg}
        input_messages: list[LLMInputContentItem] = [*history, user_msg]

        # Snapshot notes/directives before run
        notes_before = store.count()

        result = await Runner.arun(
            analyst,
            input_messages,
            max_turns=30,
            run_config=config,
        )

        # Accumulate history for next turn — convert RunItems back to Layer 1
        # for continued conversation input. Guard the empty case: slicing with
        # [-0:] is [0:], which would re-append the ENTIRE input list and
        # duplicate the whole history, so extend only when there are new items.
        history.append(user_msg)
        new_count = len(result.new_items)
        if new_count > 0:
            history.extend(result.to_input_list()[-new_count:])

        # ── Tool calls made this turn ──
        tool_calls = _extract_tool_calls(result.new_items)
        if len(tool_calls) > 0:
            logger.info("\n  Tool activity this turn:")
            for call in tool_calls:
                if "tool" in call:
                    logger.info(f"    → Called: {call['tool']}({call['args']})")
                elif "result" in call:
                    logger.info(f"    ← Result: {call['result']}")

        # ── Notes delta ──
        notes_after = store.count()
        if notes_after > notes_before:
            logger.info(f"\n  Notes: {notes_before} → {notes_after} (+{notes_after - notes_before} new)")
            for note in store.recall():
                if note.turn >= 0:  # Show all
                    logger.info(f"    [{note.importance}] {note.key}")

        # ── Response preview ──
        output = result.final_output or "(no output)"
        preview = output[:300] + "..." if len(output) > 300 else output
        logger.info(f"\n  Assistant: {preview}")

        # ── Stats ──
        assert result.context is not None
        usage = result.context.usage
        llm_calls = usage.requests or 1
        logger.info(f"\n  LLM calls this turn: {llm_calls}")
        logger.info(f"  History size: {len(history)} messages")

    # ══════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'━' * 60}")
    logger.info("  FINAL SUMMARY")
    logger.info(f"{'━' * 60}")

    notes = store.recall()
    if len(notes) > 0:
        logger.info(f"\n  Notes saved by agent ({len(notes)}):")
        for note in notes:
            logger.info(f"    [{note.importance}] {note.key}:")
            # Show full content, indented
            for line in note.content.split("\n")[:10]:
                logger.info(f"      {line}")
            if note.content.count("\n") > 10:
                logger.info(f"      ... ({note.content.count(chr(10)) - 10} more lines)")
    else:
        logger.info("\n  (Agent chose not to save any notes)")

    logger.info(f"\n  Pending directives: {jit.directives.count}")
    logger.info(f"  History messages: {len(history)}")
    logger.info("")


if __name__ == "__main__":
    asyncio.run(main())
