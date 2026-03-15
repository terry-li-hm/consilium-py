"""Forecast mode: independent probabilities, divergence analysis, and reconciliation."""

import asyncio
import time

from .models import (
    JUDGE_MODEL,
    SessionResult,
    is_error_response,
    query_model,
    run_parallel,
    sanitize_speaker_content,
)
from .prompts import (
    FORECAST_BLIND_SYSTEM,
    FORECAST_HOST_DIVERGENCE,
    FORECAST_HOST_SYNTHESIS,
    FORECAST_RECONCILE_SYSTEM,
)


def run_forecast(
    question: str,
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    verbose: bool = True,
    timeout: float = 300.0,
) -> SessionResult:
    """Run forecast mode: blind estimates → divergence analysis → reconciliation → final distribution."""
    start_time = time.time()
    cost_accumulator: list[float] = []
    transcript_parts: list[str] = []

    if verbose:
        print("=" * 60)
        print("FORECAST")
        print("=" * 60)
        print()

    # Phase 1: Blind Estimates (parallel, per-panelist prompts)
    if verbose:
        print(f"## Blind Estimates\n(querying {len(panelists)} panelists in parallel...)")

    blind_messages_list = [
        [
            {"role": "system", "content": FORECAST_BLIND_SYSTEM(name)},
            {"role": "user", "content": f"Forecast question:\n\n{question}"},
        ]
        for name, _, _ in panelists
    ]

    blind_results = asyncio.run(_run_parallel_different_messages(
        panelists, blind_messages_list, api_key, google_api_key,
        max_tokens=500, cost_accumulator=cost_accumulator, verbose=verbose, timeout=timeout,
    ))

    transcript_parts.append("## Blind Estimates")
    blind_estimates: list[tuple[str, str]] = []
    for name, model_name, response in blind_results:
        if verbose and not response.startswith("["):
            print(f"### {name}\n{response}\n")
        transcript_parts.append(f"### {name}\n{response}")
        blind_estimates.append((name, response))

    all_blind_text = "\n\n".join(
        f"**{name}**:\n{sanitize_speaker_content(response).strip()}"
        for name, response in blind_estimates
        if not is_error_response(response)
    )

    # Phase 2: Divergence Analysis (judge/host)
    if verbose:
        print("=" * 60)
        print("## Divergence Analysis")
        print("### Host (Claude)")

    judge_name = JUDGE_MODEL.split("/")[-1]
    divergence_messages = [
        {"role": "system", "content": FORECAST_HOST_DIVERGENCE(all_blind_text)},
        {"role": "user", "content": f"Question:\n{question}\n\nBlind estimates:\n\n{all_blind_text}"},
    ]
    host_divergence = query_model(
        api_key, JUDGE_MODEL, divergence_messages,
        max_tokens=600, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## Divergence Analysis\n\n### Host ({judge_name})\n{host_divergence}")
    sanitized_divergence = sanitize_speaker_content(host_divergence)

    # Phase 3: Reconciliation (parallel, per-panelist)
    if verbose:
        print("=" * 60)
        print(f"## Reconciliation\n(querying {len(panelists)} panelists in parallel...)")

    reconcile_messages_list = [
        [
            {"role": "system", "content": FORECAST_RECONCILE_SYSTEM(name, all_blind_text, sanitized_divergence)},
            {"role": "user", "content": f"Forecast question:\n\n{question}"},
        ]
        for name, _, _ in panelists
    ]

    reconcile_results = asyncio.run(_run_parallel_different_messages(
        panelists, reconcile_messages_list, api_key, google_api_key,
        max_tokens=600, cost_accumulator=cost_accumulator, verbose=verbose, timeout=timeout,
    ))

    transcript_parts.append("## Reconciliation")
    final_estimates: list[tuple[str, str]] = []
    for name, model_name, response in reconcile_results:
        if verbose and not response.startswith("["):
            print(f"### {name}\n{response}\n")
        transcript_parts.append(f"### {name}\n{response}")
        final_estimates.append((name, response))

    final_estimates_text = "\n\n".join(
        f"**{name}**:\n{sanitize_speaker_content(response).strip()}"
        for name, response in final_estimates
        if not is_error_response(response)
    )

    # Phase 4: Final Distribution (judge/host)
    if verbose:
        print("=" * 60)
        print("## Final Distribution")
        print("### Host (Claude)")

    synthesis_messages = [
        {"role": "system", "content": FORECAST_HOST_SYNTHESIS},
        {"role": "user", "content": f"Question:\n{question}\n\nFinal reconciled estimates:\n\n{final_estimates_text}"},
    ]
    host_distribution = query_model(
        api_key, JUDGE_MODEL, synthesis_messages,
        max_tokens=600, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## Final Distribution\n\n### Host ({judge_name})\n{host_distribution}")

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    return SessionResult(
        transcript="\n\n".join(transcript_parts),
        cost=total_cost,
        duration=duration,
    )


async def _run_parallel_different_messages(
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    messages_list: list[list[dict]],
    api_key: str,
    google_api_key: str | None,
    max_tokens: int,
    cost_accumulator: list[float],
    verbose: bool,
    timeout: float,
) -> list[tuple[str, str, str]]:
    """Run parallel queries where each panelist gets a different message list."""
    import httpx
    from .models import query_model_async

    indexed_results: list[tuple[int, tuple[str, str, str]]] = []

    async def _query(idx: int, name: str, model: str, fallback, messages: list[dict], client: httpx.AsyncClient):
        result = await query_model_async(
            client, model, messages, name, fallback,
            google_api_key, max_tokens=max_tokens,
            cost_accumulator=cost_accumulator,
        )
        indexed_results.append((idx, result))

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    ) as client:
        tasks = [
            _query(i, name, model, fallback, messages_list[i], client)
            for i, (name, model, fallback) in enumerate(panelists)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    indexed_results.sort(key=lambda x: x[0])
    return [r for _, r in indexed_results]
