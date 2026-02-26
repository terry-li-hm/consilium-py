---
title: "feat: Consilium quality-of-life improvements"
type: feat
status: completed
date: 2026-02-18
deepened: 2026-02-18
---

# feat: Four quality-of-life improvements

## Enhancement Summary

**Deepened on:** 2026-02-18
**Sections enhanced:** All
**Research agents used:** architecture-strategist, code-simplicity-reviewer, kieran-python-reviewer, best-practices-researcher (Oxford debate)

### Key Changes from Deepening
1. **Cut pipeline mode** — YAGNI, zero usage history, doubles dispatch complexity
2. **Cut closure extraction** — 3-line closures don't warrant shared helpers
3. **Oxford redesigned** — 2 models (not 4), random sides, "This House Believes..." format, prior/posterior judge
4. **Simplified SessionResult** — 3 fields only (transcript, cost, duration), no failed_models/confidences fields
5. **Fixed union type** — explicit keyword args instead of isinstance sniffing
6. **Loosened confidence regex** — handles missing bold, spaces, "out of 10"

## Overview

Four improvements to consilium: extract shared parallel helper, confidence tracking, session analytics, and Oxford debate format.

## Implementation Order

1. Extract `run_parallel()` — dedup, everything else is easier
2. `SessionResult` + log cost/duration to history — enables analytics
3. Confidence tracking — parse + display in council/solo
4. `--stats` — session analytics
5. `--oxford` — Oxford debate format

Each step is independently shippable.

---

## 1. Extract `run_parallel()` into models.py

**Problem:** Parallel query helper copy-pasted across 4 modules (shared-messages variant).

**Scope:** Extract the shared-messages variant ONLY. Leave redteam's `_run_attacks()` and solo's `_run_blind()` alone — they have genuinely different message patterns (per-panelist), and the cost of maintaining 15 duplicate lines across 2 files is near zero.

**Modules to update:**

| Module | Function | Action |
|--------|----------|--------|
| `discuss.py:26` | `_run_discuss_parallel()` | Replace with import |
| `socratic.py:26` | `_run_socratic_parallel()` | Replace with import |
| `council.py:35` | `run_blind_phase_parallel()` | Replace with import |
| `quick.py:14` | `_run_quick_async()` | Replace with import |
| `redteam.py:105` | `_run_attacks()` | **Keep as-is** (per-panelist messages) |
| `solo.py:94` | `_run_blind()` | **Keep as-is** (per-panelist messages) |

```python
# consilium/models.py — new function

async def run_parallel(
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    messages: list[dict],
    api_key: str,
    google_api_key: str | None = None,
    max_tokens: int = 500,
    cost_accumulator: list[float] | None = None,
) -> list[tuple[str, str, str]]:
    """Parallel query panelists with shared messages. Returns [(name, model_name, response)]."""
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=180.0,
    ) as client:
        tasks = [
            query_model_async(
                client, model, messages, name, fallback,
                google_api_key, max_tokens=max_tokens,
                cost_accumulator=cost_accumulator,
            )
            for name, model, fallback in panelists
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for i, result in enumerate(results):
        name, model, fallback = panelists[i]
        model_name = model.split("/")[-1]
        if isinstance(result, Exception):
            out.append((name, model_name, f"[Error: {result}]"))
        else:
            out.append(result)
    return out
```

**quick.py migration:** Adapt `QUICK_MODELS` from 2-tuples to 3-tuples so it works with `run_parallel()`:
```python
# models.py — change from:
QUICK_MODELS = [("Claude", JUDGE_MODEL)] + [(n, m) for n, m, _ in COUNCIL]
# to:
QUICK_MODELS = [("Claude", JUDGE_MODEL, None)] + [(n, m, fb) for n, m, fb in COUNCIL]
```

**Note:** This gives quick mode fallback support (Gemini → AI Studio). Intentional behavioral improvement, not just a refactor.

**Closures (`_build_history_text`, etc.):** NOT extracted. They're 3-line closures over local scope — extraction adds parameter passing with zero net simplification.

---

## 2. `SessionResult` + Cost/Duration Logging

**Problem:** Mode functions return only transcript (str). Cost and duration are computed internally, printed, then discarded. This blocks analytics.

### SessionResult dataclass

```python
# consilium/models.py — new dataclass

from dataclasses import dataclass

@dataclass
class SessionResult:
    transcript: str
    cost: float
    duration: float
```

Three fields only. No `failed_models` (stays local to council.py, printed inline). No `confidences` (stays local to the debate loop, printed in transcript footer).

**Migration:** Each `run_*()` function changes from:
```python
return "\n\n".join(transcript_parts)
```
to:
```python
return SessionResult(
    transcript="\n\n".join(transcript_parts),
    cost=round(sum(cost_accumulator), 4),
    duration=time.time() - start_time,
)
```

**council.py special case:** Currently returns `(transcript, failed_models)`. Change to return `SessionResult`. Move `failed_models` handling: council prints failure summary inline (already does this at cli.py:648), so the return value doesn't need to carry it. The CLI block changes from:
```python
transcript, failed_models = run_council(...)
```
to:
```python
result = run_council(...)
# failed_models now printed inside run_council() directly
transcript = result.transcript
```

### Log cost/duration to history.jsonl

Add `cost` and `duration` to `_log_history()`:
```python
def _log_history(question, mode, session_path, gist_url, extra, cost=None, duration=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question[:200],
        "mode": mode,
        "cost": cost,        # NEW
        "duration": duration, # NEW
        "session": str(session_path) if session_path else None,
        "gist": gist_url,
    }
```

Each CLI dispatch block passes `cost=result.cost, duration=result.duration`.

---

## 3. Confidence Tracking

**Where it applies:** Council debate rounds + solo debate rounds only. Not discuss/socratic/redteam.

### Parse confidence

```python
# consilium/models.py — new function

import re

_CONFIDENCE_RE = re.compile(
    r'\*{0,2}Confidence:?\s*\*{0,2}\s*(\d{1,2})\s*(?:/\s*10|out\s+of\s+10)',
    re.IGNORECASE,
)

def parse_confidence(response: str) -> int | None:
    """Extract Confidence: N/10 from a debate response."""
    match = _CONFIDENCE_RE.search(response)
    if match:
        value = int(match.group(1))
        return value if 0 <= value <= 10 else None
    return None
```

Handles: `**Confidence: 7/10**`, `Confidence: 7/10`, `Confidence: 7 / 10`, `confidence: 7 out of 10`, `**Confidence**: 7/10`.

### Track in debate loops

In `council.py` and `solo.py` debate loops:
```python
confidences: dict[str, list[int]] = {}

# After each debate response:
conf = parse_confidence(response)
if conf is not None:
    confidences.setdefault(name, []).append(conf)
```

### Display confidence drift

After debate phase, before judge/synthesis:
```python
if confidences and verbose:
    drift_parts = []
    for name, scores in confidences.items():
        if len(scores) >= 2:
            drift_parts.append(f"{name} {scores[0]}→{scores[-1]}")
        elif scores:
            drift_parts.append(f"{name} {scores[0]}")
    if drift_parts:
        print(f"  Confidence: {', '.join(drift_parts)}")
        print()
```

Also append to transcript:
```
Confidence drift: GPT 7→8, Gemini 9→6, Grok 5→8
```

**No `detect_false_consensus()`** — speculative, the warning doesn't change behavior, and 8/10 is reasonable for clear-cut questions. Cut until we see the pattern in real sessions.

---

## 4. Session Analytics (`consilium --stats`)

**Data source:** `~/.consilium/history.jsonl`

```python
# cli.py — new argument
parser.add_argument("--stats", action="store_true", help="Show session analytics")
```

**Handler** (before question requirement check):
```python
if args.stats:
    history_file = get_sessions_dir().parent / "history.jsonl"
    if not history_file.exists():
        print("No history found.")
        sys.exit(0)

    entries = []
    for line in history_file.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))

    if not entries:
        print("No sessions recorded.")
        sys.exit(0)

    # Group by mode
    from collections import defaultdict
    by_mode = defaultdict(list)
    for e in entries:
        by_mode[e["mode"]].append(e)

    # Date range
    first = entries[0].get("timestamp", "?")[:10]
    last = entries[-1].get("timestamp", "?")[:10]

    print(f"Consilium Stats ({len(entries)} sessions, {first} — {last})\n")
    print(f"{'Mode':<14s} {'Sessions':>8s} {'Avg Cost':>10s} {'Total Cost':>12s} {'Avg Time':>10s}")

    total_cost = 0.0
    for mode in sorted(by_mode.keys()):
        mode_entries = by_mode[mode]
        count = len(mode_entries)
        costs = [e.get("cost") for e in mode_entries if e.get("cost") is not None]
        durations = [e.get("duration") for e in mode_entries if e.get("duration") is not None]

        avg_cost = f"${sum(costs)/len(costs):.2f}" if costs else "—"
        sum_cost = sum(costs) if costs else 0
        total_cost += sum_cost
        total_cost_str = f"${sum_cost:.2f}" if costs else "—"
        avg_dur = f"{sum(durations)/len(durations):.0f}s" if durations else "—"

        print(f"{mode:<14s} {count:>8d} {avg_cost:>10s} {total_cost_str:>12s} {avg_dur:>10s}")

    print(f"\n{'Total':<14s} {len(entries):>8d} {'':>10s} ${total_cost:>11.2f}")

    # Last 7 days
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    recent = [e for e in entries if e.get("timestamp", "") >= cutoff]
    recent_cost = sum(e.get("cost", 0) or 0 for e in recent)
    print(f"\nLast 7 days: {len(recent)} sessions, ${recent_cost:.2f}")

    sys.exit(0)
```

**Note:** Older entries without `cost` field gracefully show "—". Stats become more useful over time.

---

## 5. Oxford Debate (`consilium --oxford`)

### Research-Grounded Design

Based on Oxford Union formal rules, ICLR 2025 MAD analysis, Zhang et al. 2026 dynamic role assignment, and UCL debate research.

### Key Design Decisions

| Decision | Choice | Research basis |
|----------|--------|---------------|
| How many debaters? | **2** (not 4) | Oxford is bipolar; more models muddies it |
| Side assignment? | **Random** | ICLR 2025: fixed assignment = worst approach. Zhang 2026: static underperforms by up to 74.8% |
| Motion format? | **"This House Believes..."** | Oxford Union formal convention |
| Judge sees sides? | **Yes** | Needed to evaluate argument relevance to assigned position |
| How judge scores? | **Prior/posterior probability shift** | Mirrors Oxford pre-vote/post-vote mechanism |
| Rebuttal rounds? | **1** (v1) | Safety debate paper: 3 max, but 1 is sufficient for v1 |
| Cross-examination? | **No** | That's Lincoln-Douglas, not Oxford. Rebuttals serve the same function |

### Flow

```
Phase 1: MOTION        Host transforms question to "This House Believes..." (or --motion override)
Phase 2: PRIOR         Judge states prior probability (0-100) before hearing arguments
Phase 3: CONSTRUCTIVE  Proposition argues FOR (~400 words, parallel isn't possible — sequential)
                       Opposition argues AGAINST (~400 words)
Phase 4: REBUTTAL      Proposition rebuts Opposition's specific points (~300 words)
                       Opposition rebuts Proposition's specific points (~300 words)
Phase 5: CLOSING       Proposition summary, NO new arguments (~200 words)
                       Opposition summary, NO new arguments (~200 words)
Phase 6: VERDICT       Judge: argument analysis + posterior probability + verdict
```

**API calls:** 1 motion transform + 1 prior + 2 constructive + 2 rebuttal + 2 closing + 1 verdict = **9 calls, ~$0.25**

### Models

```python
# models.py
OXFORD_MODELS = COUNCIL[:2]  # GPT, Gemini — debaters
# Judge: JUDGE_MODEL (Claude) — already separate
```

Side assignment at runtime:
```python
import random

def _assign_sides(debaters):
    """Random side assignment for Oxford debate."""
    sides = list(debaters)
    random.shuffle(sides)
    return sides[0], sides[1]  # (proposition, opposition)
```

### Prompts

```python
# prompts.py — new Oxford prompts

OXFORD_MOTION_TRANSFORM = """Convert this question into a binary Oxford debate motion.

Rules:
1. Must start with "This House Believes" or "This House Would"
2. Must be arguable from both sides
3. Must be specific enough to debate
4. Preserve the user's intent

Question: {question}

Output ONLY the motion, starting with "This House Believes" or "This House Would"."""

OXFORD_CONSTRUCTIVE_SYSTEM = """You are {name}, arguing {side} the motion:

"{motion}"

Build your strongest case. Present 2-3 clear arguments with evidence or reasoning. Be specific — vague claims are weak claims. ~400 words.

You are assigned this side regardless of your personal view. Argue it convincingly."""

OXFORD_REBUTTAL_SYSTEM = """You are {name}, arguing {side} the motion:

"{motion}"

Your opponent argued:
{opponent_argument}

Respond to their STRONGEST point directly. Concede what you must — selective concession is persuasive, blanket denial is not. Then counter with your most compelling evidence. ~300 words.

Do NOT introduce entirely new arguments. Build on your constructive case."""

OXFORD_CLOSING_SYSTEM = """You are {name}, giving your closing statement {side} the motion:

"{motion}"

This is your final word. Summarize your case in its strongest form. What is the single most compelling reason the judge should side with you?

FORBIDDEN: No new arguments. No new evidence. Summation only. ~200 words."""

OXFORD_JUDGE_PRIOR = """You are about to judge an Oxford-style debate on this motion:

"{motion}"

Before hearing any arguments, what probability (0-100) would you assign to this motion being true/correct? State your prior and briefly explain why (2-3 sentences).

Format: **Prior: N/100** followed by your reasoning."""

OXFORD_JUDGE_VERDICT = """You judged an Oxford-style debate on:

"{motion}"

{proposition_name} argued FOR. {opposition_name} argued AGAINST.

INSTRUCTIONS:
1. Evaluate ARGUMENT QUALITY, not which position you personally agree with.
2. Do NOT reward verbosity. Concise arguments that land beat lengthy ones that meander.
3. Note logical fallacies, strawmanning, or evasion by either side.

Provide:

## Argument Analysis

### Proposition ({proposition_name})
- Strongest argument:
- Weakest argument:
- Rebuttal effectiveness:

### Opposition ({opposition_name})
- Strongest argument:
- Weakest argument:
- Rebuttal effectiveness:

## Verdict

**Prior:** [your pre-debate probability]/100
**Posterior:** [your post-debate probability]/100
**Winner:** [Proposition/Opposition] — the side that shifted your probability more
**Margin:** [Decisive / Narrow / Too close to call]
**Reasoning:** [2-3 sentences on what decided it]"""
```

### New file: `consilium/oxford.py`

~180 lines following the established mode module pattern. Uses `run_parallel` for nothing (all phases are sequential in Oxford since each speech responds to the previous). Uses `query_model()` directly for each speech.

### CLI

```python
parser.add_argument("--oxford", action="store_true",
    help="Oxford debate: binary for/against with rebuttals and verdict")
parser.add_argument("--motion", help="Override auto-generated debate motion for --oxford")
```

Validation:
- `--oxford` added to mode_flags mutual exclusivity
- `--oxford` incompatible with `--challenger`, `--followup`, `--format`
- `--motion` requires `--oxford`

Epilog example:
```
Oxford debate (binary for/against with verdict):
  consilium "Should we use microservices?" --oxford
  consilium "Remote work" --oxford --motion "THB fully remote work produces better outcomes"
```

---

## Files to Modify

| File | Changes | ~Lines |
|------|---------|--------|
| `consilium/models.py` | `run_parallel()`, `SessionResult`, `parse_confidence()`, `OXFORD_MODELS`, `_assign_sides()` | +60 |
| `consilium/prompts.py` | 6 Oxford prompt constants | +60 |
| `consilium/oxford.py` | New module — `run_oxford()` | +180 |
| `consilium/cli.py` | `--stats`, `--oxford`, `--motion` flags, cost/duration logging, stats handler | +90 |
| `consilium/discuss.py` | Replace `_run_discuss_parallel` with `run_parallel` import | -25 |
| `consilium/socratic.py` | Same | -25 |
| `consilium/council.py` | Replace parallel helper, return SessionResult, add confidence tracking | -20, +25 |
| `consilium/quick.py` | Use `run_parallel`, adapt to 3-tuples | -30, +5 |
| `consilium/solo.py` | Add confidence tracking | +15 |
| `consilium/__init__.py` | New exports, version bump | +5 |

**Net:** ~+180 new code, ~-100 removed duplication, +180 for oxford.py. Total ~+260 net lines.

## Acceptance Criteria

- [x] `run_parallel()` in models.py used by discuss, socratic, council, quick (4 modules)
- [x] All mode functions return `SessionResult` with transcript, cost, duration
- [x] `_log_history()` logs cost and duration for every session
- [x] `parse_confidence()` extracts N/10 with forgiving regex
- [x] Confidence drift displayed in council/solo transcript footer
- [x] `consilium --stats` prints session analytics table
- [x] `consilium --oxford "We should X"` runs 2-model debate with random sides
- [x] `--motion` flag overrides auto-generated motion
- [x] Oxford judge uses prior/posterior probability shift mechanism
- [x] All existing modes pass regression: `consilium "2+2" --quick --quiet`
- [x] Version bumped to 0.10.0

## What NOT to Build

- No pipeline mode — YAGNI, zero demonstrated need
- No closure extraction — 3-line closures don't warrant shared helpers
- No `detect_false_consensus()` — speculative, warning doesn't change behavior
- No `--swap` flag for Oxford — future enhancement if position bias is observed
- No confidence tracking in discuss/socratic/redteam — only council and solo have debate rounds
- No `--auto` routing for oxford — always explicit
- No structured output (JSON/YAML) for oxford — prose only
- No README updates
- No tests (except consider one for parse_confidence edge cases)
