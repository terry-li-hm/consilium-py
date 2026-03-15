"""Pre-mortem mode: assume failure happened and reason backward to causes."""

import asyncio
import time

from .models import (
    JUDGE_MODEL,
    SessionResult,
    query_model,
    run_parallel,
    sanitize_speaker_content,
)
from .prompts import (
    PREMORTEM_HOST_FRAMING,
    PREMORTEM_HOST_MITIGATION,
    PREMORTEM_HOST_SYNTHESIS,
    PREMORTEM_PANELIST_SYSTEM,
)
from .forecast import _run_parallel_different_messages


def run_premortem(
    question: str,
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    verbose: bool = True,
    timeout: float = 300.0,
) -> SessionResult:
    """Run pre-mortem mode: frame failure → generate scenarios → synthesise → mitigate."""
    start_time = time.time()
    cost_accumulator: list[float] = []
    transcript_parts: list[str] = []
    conversation_history: list[tuple[str, str]] = []

    judge_name = JUDGE_MODEL.split("/")[-1]

    if verbose:
        print("=" * 60)
        print("PRE-MORTEM")
        print("=" * 60)
        print()

    # Phase 1: Setup — host frames the failure scenario
    if verbose:
        print("## Setup")
        print("### Host (Claude)")

    setup_messages = [
        {"role": "system", "content": PREMORTEM_HOST_FRAMING},
        {"role": "user", "content": question},
    ]
    host_setup = query_model(
        api_key, JUDGE_MODEL, setup_messages,
        max_tokens=400, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## Setup\n\n### Host ({judge_name})\n{host_setup}")
    conversation_history.append(("Host (Claude)", host_setup))

    # Phase 2: Failure Scenarios (parallel, per-panelist)
    if verbose:
        print("=" * 60)
        print(f"## Failure Scenarios\n(querying {len(panelists)} panelists in parallel...)")

    sanitized_setup = sanitize_speaker_content(host_setup)
    scenario_messages_list = [
        [
            {"role": "system", "content": PREMORTEM_PANELIST_SYSTEM(name)},
            {"role": "user", "content": (
                f"Plan/decision for pre-mortem:\n\n{question}"
                f"\n\nHost framing:\n\n{sanitized_setup}"
            )},
        ]
        for name, _, _ in panelists
    ]

    scenario_results = asyncio.run(_run_parallel_different_messages(
        panelists, scenario_messages_list, api_key, google_api_key,
        max_tokens=700, cost_accumulator=cost_accumulator, verbose=verbose, timeout=timeout,
    ))

    transcript_parts.append("## Failure Scenarios")
    for name, model_name, response in scenario_results:
        if verbose and not response.startswith("["):
            print(f"### {name}\n{response}\n")
        transcript_parts.append(f"### {name}\n{response}")
        conversation_history.append((name, response))

    # Phase 3: Host Synthesis
    if verbose:
        print("=" * 60)
        print("## Host Synthesis")
        print("### Host (Claude)")

    narratives_text = "\n\n".join(
        f"**{speaker}**: {sanitize_speaker_content(text)}"
        for speaker, text in conversation_history[1:]  # skip the host framing
    )

    synthesis_messages = [
        {"role": "system", "content": PREMORTEM_HOST_SYNTHESIS},
        {"role": "user", "content": f"Plan/decision:\n{question}\n\nFailure narratives:\n\n{narratives_text}"},
    ]
    host_synthesis = query_model(
        api_key, JUDGE_MODEL, synthesis_messages,
        max_tokens=600, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## Host Synthesis\n\n### Host ({judge_name})\n{host_synthesis}")
    conversation_history.append(("Host (Claude)", host_synthesis))

    # Phase 4: Mitigation Map
    if verbose:
        print("=" * 60)
        print("## Mitigation Map")
        print("### Host (Claude)")

    full_history_text = "\n\n".join(
        f"**{speaker}**: {sanitize_speaker_content(text)}"
        for speaker, text in conversation_history
    )

    mitigation_messages = [
        {"role": "system", "content": PREMORTEM_HOST_MITIGATION},
        {"role": "user", "content": f"Plan/decision:\n{question}\n\nPre-mortem discussion so far:\n\n{full_history_text}"},
    ]
    host_mitigation = query_model(
        api_key, JUDGE_MODEL, mitigation_messages,
        max_tokens=600, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## Mitigation Map\n\n### Host ({judge_name})\n{host_mitigation}")

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    return SessionResult(
        transcript="\n\n".join(transcript_parts),
        cost=total_cost,
        duration=duration,
    )
