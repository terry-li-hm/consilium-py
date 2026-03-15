"""Debate mode: multi-round cross-critique between panelists."""

import asyncio
import time

from .models import (
    JUDGE_MODEL,
    SessionResult,
    is_error_response,
    query_model,
    sanitize_speaker_content,
)
from .prompts import (
    DEBATE_CROSS_CRITIQUE_SYSTEM,
    DEBATE_SYNTHESIS_SYSTEM,
)
from .forecast import _run_parallel_different_messages


def run_debate(
    question: str,
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    rounds: int = 2,
    verbose: bool = True,
    timeout: float = 300.0,
) -> SessionResult:
    """Run debate mode: independent reviews → cross-critique rounds → judge synthesis.

    Round 1: All panelists review independently (parallel).
    Round 2+: Each panelist cross-critiques all others' previous outputs (parallel).
    Final: Judge synthesises across all rounds.
    """
    start_time = time.time()
    cost_accumulator: list[float] = []
    transcript_parts: list[str] = []

    if verbose:
        print("=" * 60)
        print(f"DEBATE ({rounds} rounds, {len(panelists)} panelists)")
        print("=" * 60)
        print()

    # Track per-panelist outputs across rounds: {name: [round1_output, round2_output, ...]}
    round_outputs: dict[str, list[str]] = {name: [] for name, _, _ in panelists}
    panelist_model_names: dict[str, str] = {
        name: model.split("/")[-1] for name, model, _ in panelists
    }

    for round_num in range(1, rounds + 1):
        round_label = f"ROUND {round_num}"
        if verbose:
            print("=" * 60)
            print(f"{round_label}: {'Independent Review' if round_num == 1 else 'Cross-Critique'}")
            print("=" * 60)
            print(f"(querying {len(panelists)} panelists in parallel...)")

        transcript_parts.append(f"## {round_label}")

        if round_num == 1:
            # Round 1: simple independent review
            messages_list = [
                [{"role": "user", "content": question}]
                for _ in panelists
            ]
        else:
            # Round 2+: cross-critique — each panelist reads all others' previous outputs
            prev_round = round_num - 1
            messages_list = []
            for name, model, _ in panelists:
                model_name = panelist_model_names[name]
                # Build the other-reviewers section
                other_sections = []
                for other_name, _, _ in panelists:
                    if other_name == name:
                        continue
                    other_model = panelist_model_names[other_name]
                    other_prev_output = round_outputs[other_name][-1] if round_outputs[other_name] else ""
                    if other_prev_output and not is_error_response(other_prev_output):
                        other_sections.append(
                            f"--- {other_name} ({other_model}) ---\n"
                            f"{sanitize_speaker_content(other_prev_output)}\n"
                            "---"
                        )

                others_text = "\n\n".join(other_sections) if other_sections else "(No other reviews available.)"

                user_content = (
                    f"Original input:\n\n{question}\n\n"
                    f"Here are the other reviewers' Round {prev_round} assessments:\n\n"
                    f"{others_text}"
                )

                messages_list.append([
                    {"role": "system", "content": DEBATE_CROSS_CRITIQUE_SYSTEM(name, model_name, prev_round)},
                    {"role": "user", "content": user_content},
                ])

        results = asyncio.run(_run_parallel_different_messages(
            panelists, messages_list, api_key, google_api_key,
            max_tokens=800, cost_accumulator=cost_accumulator,
            verbose=False,  # We print manually below for cleaner output
            timeout=timeout,
        ))

        for name, model_name, response in results:
            round_outputs[name].append(response)
            if verbose:
                print(f"\n### {name} ({model_name})")
                if is_error_response(response):
                    print(response)
                else:
                    print(response)
            transcript_parts.append(f"### {name} ({model_name})\n{response}")

        if verbose:
            print()

    # Final: Judge synthesis across all rounds
    judge_name = JUDGE_MODEL.split("/")[-1]
    if verbose:
        print("=" * 60)
        print(f"## Final Synthesis ({judge_name})")

    # Build full debate transcript for the judge
    synthesis_input_parts = [f"Question:\n{question}\n"]
    for round_num in range(1, rounds + 1):
        round_label = f"Round {round_num} ({'Independent Review' if round_num == 1 else 'Cross-Critique'})"
        synthesis_input_parts.append(f"### {round_label}")
        for name, model, _ in panelists:
            model_name = panelist_model_names[name]
            if len(round_outputs[name]) >= round_num:
                output = round_outputs[name][round_num - 1]
                if not is_error_response(output):
                    synthesis_input_parts.append(
                        f"**{name} ({model_name}):**\n{sanitize_speaker_content(output)}"
                    )

    synthesis_messages = [
        {"role": "system", "content": DEBATE_SYNTHESIS_SYSTEM},
        {"role": "user", "content": "\n\n".join(synthesis_input_parts)},
    ]

    judge_synthesis = query_model(
        api_key, JUDGE_MODEL, synthesis_messages,
        max_tokens=1200, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## Final Synthesis ({judge_name})\n{judge_synthesis}")

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    return SessionResult(
        transcript="\n\n".join(transcript_parts),
        cost=total_cost,
        duration=duration,
    )
