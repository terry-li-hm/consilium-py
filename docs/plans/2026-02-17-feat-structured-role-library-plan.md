---
title: "feat: Structured role library with TypedDict"
type: feat
status: active
date: 2026-02-17
deepened: 2026-02-17
---

# Structured Role Library

## Enhancement Summary

**Deepened on:** 2026-02-17
**Research agents used:** architecture-strategist, code-simplicity-reviewer, kieran-python-reviewer, best-practices-researcher, learnings-researcher, web-search (CrewAI patterns)

### Key Improvements from Deepening
1. **Reduced from 7 fields to 5** — `signal`/`noise`/`decision_weight` had semantic overlap. Merged into `identity` prose. Keeps genuinely distinct axes only.
2. **TypedDict instead of plain dict** — catches key typos at type-check time, zero runtime cost, IDE autocomplete.
3. **Confidence expression split to separate commit** — orthogonal concern, easier to test/iterate independently.
4. **Extracted `resolve_role()` helper** — cleaner than inline loop with variable mutation in solo.py.
5. **CrewAI alignment** — production frameworks use 3 fields (role/goal/backstory). Our 5-field schema maps cleanly: identity=role+backstory, epistemic_stance=goal framing, forbidden=constraints.

### Reviewer Concerns Addressed
- **Architecture**: render_role() placement in prompts.py is correct (co-locates data + renderer).
- **Simplicity**: 7→5 fields eliminates overlap. Structure justified by research showing 3.5% reasoning improvement from heterogeneous agents (A-HMAD).
- **Python patterns**: TypedDict for static data, extracted lookup function, normalized O(1) fallback.
- **Learnings**: Devil's Advocate removal confirmed by `multi-agent-deliberation-design.md`. Judge `output_pattern` differentiation confirmed by `llm-council-judge-over-aggregation.md`.

---

## Overview

Upgrade `ROLE_LIBRARY` from flat `dict[str, str]` to `dict[str, RoleDefinition]` (TypedDict) with 5 fields per role. Based on academic research into multi-agent deliberation (ConfMAD, A-HMAD, SPP) and production patterns from CrewAI. Current roles are strong on domain lens but missing epistemic stance, reasoning style, and behavioral constraints. This upgrade makes each role genuinely differentiated — not just "what you know" but "how you think."

## Problem Statement

Current flat-string roles produce **homogeneous debate patterns**. Research confirms: identical agents amplify shared biases (Du et al.), while heterogeneous specialized agents produce up to 3.5% reasoning improvement (A-HMAD). The missing axes mean a Skeptic and an Advocate currently differ only in *what they say*, not *how they reason* or *what they're forbidden from doing*. This leads to polite convergence by round 2 — the exact failure mode documented in `~/docs/solutions/best-practices/multi-agent-deliberation-design.md`.

**Evidence from production frameworks:** CrewAI's "role + goal + backstory" pattern produces measurably better outputs than flat role descriptions. More specific roles ("SaaS Metrics Specialist focusing on growth-stage startups") outperform generic ones ("Business Analyst") — specificity matters more than format for frontier models ([CrewAI docs](https://docs.crewai.com/en/guides/agents/crafting-effective-agents)).

## Proposed Solution

### 5-Field Role Schema (TypedDict)

Reduced from original 7-field design after reviewer feedback identified semantic overlap between `signal`/`noise`/`decision_weight` (all express "what matters to this role" differently). These are folded into `identity` prose.

```python
from typing import TypedDict

class RoleDefinition(TypedDict):
    identity: str           # Who you are + what you prioritize (absorbs signal/noise/decision_weight)
    epistemic_stance: str   # How you relate to knowledge and uncertainty
    reasoning_style: str    # What kind of arguments you find persuasive
    forbidden: list[str]    # Hard constraints on what you CANNOT do
    output_pattern: str     # How you structure your contribution
```

**Why these 5 axes are genuinely distinct:**
- `identity` = WHAT you care about (domain lens + priorities)
- `epistemic_stance` = HOW you relate to truth/uncertainty (epistemology)
- `reasoning_style` = HOW you argue (logic method)
- `forbidden` = WHAT you cannot do (behavioral constraints)
- `output_pattern` = HOW you format output (communication style)

**Why TypedDict over plain dict:**
- `role["reasoning_stlye"]` → runtime KeyError, hard to debug. TypedDict catches this at type-check time.
- IDE autocomplete for field names.
- Zero runtime overhead (it's still a dict at runtime).
- Self-documenting — schema is visible in the type definition.

### Render Function

A `render_role()` function converts the structured TypedDict to a prompt string. This is the single point where structure becomes prose.

```python
def render_role(name: str, role: RoleDefinition) -> str:
    """Convert a structured role definition to a prompt string for system messages."""
    parts = [role["identity"]]
    parts.append(f"Epistemic stance: {role['epistemic_stance']}")
    parts.append(f"Reasoning approach: {role['reasoning_style']}")
    if role["forbidden"]:
        forbidden_text = " ".join(f"NEVER: {f}." for f in role["forbidden"])
        parts.append(forbidden_text)
    parts.append(f"Output pattern: {role['output_pattern']}")
    return "\n".join(parts)
```

**Research insight on rendering:** Narrative prose with embedded structure is optimal per 2025 prompt format research (up to 40% performance variance by format). Using labeled sections ("Epistemic stance:", "NEVER:") within prose matches the hybrid pattern that works best across frontier models.

**Research insight on "forbidden" constraints:** Current consilium patterns ("You CANNOT use phrases like 'building on', 'adding nuance'") are effective — specific examples beat abstract prohibitions. The `forbidden` field preserves this pattern. Frame as "NEVER: [specific behavior]." not vague "avoid being agreeable" ([Lakera guide](https://www.lakera.ai/blog/prompt-engineering-guide)).

### Role Lookup Helper

Extract the current inline loop into a clean function (addresses Python reviewer's concern about variable mutation in the loop):

```python
def resolve_role(role_name: str) -> tuple[str, str]:
    """Resolve a role name to (canonical_name, rendered_description).

    Tries exact match, then case-insensitive, then generic fallback.
    """
    # Exact match
    role_def = ROLE_LIBRARY.get(role_name)
    if role_def:
        return role_name, render_role(role_name, role_def)

    # Case-insensitive fallback
    role_lower = role_name.lower()
    for lib_name, lib_def in ROLE_LIBRARY.items():
        if lib_name.lower() == role_lower:
            return lib_name, render_role(lib_name, lib_def)

    # Unknown role — generic template
    return role_name, (
        f"You approach this as a {role_name}. Bring your professional lens, "
        f"domain expertise, and the specific concerns someone in your role "
        f"would have. Be specific and opinionated — generic advice is worthless."
    )
```

### Confidence Expression — SEPARATE COMMIT

**Split from this change per architecture and simplicity reviewer feedback.** Confidence expression (0-10, ConfMAD pattern) is orthogonal to role structure. Shipping separately enables:
- Independent A/B testing of each change's impact on debate quality
- Cleaner rollback if confidence scoring needs iteration ("0-10" vs "low/med/high")
- Smaller, focused commits

**Follow-up commit spec (5 lines):** Add to `SOLO_DEBATE_SYSTEM` and `COUNCIL_DEBATE_SYSTEM`:
```
End your response with: **Confidence: N/10** — how certain are you of your position after seeing others' arguments?
```

ConfMAD research shows this improves MMLU accuracy from 78.3% → 83.3% with confidence signaling ([arxiv](https://arxiv.org/abs/2509.14034)).

## Technical Considerations

- **Backward compatibility**: `render_role()` produces a string, so `SOLO_BLIND_SYSTEM.format(name=name, description=description)` stays unchanged. The `description` variable just becomes `render_role()` output instead of a raw string.
- **Unknown roles**: Generic fallback in `resolve_role()` stays permissive — any role name works.
- **`--list-roles` display**: Updated to show identity + epistemic stance for structural roles. Keep simple — no fancy grouping.
- **No new dependencies**: TypedDict is stdlib (Python 3.8+). No pydantic/dataclass needed.
- **Performance**: Zero impact — rendering happens once per role per session.
- **Breaking change for external consumers**: `ROLE_LIBRARY` values change from `str` to `RoleDefinition`. If anyone imports `ROLE_LIBRARY` directly, they'll need to update. Since this is a personal CLI tool, acceptable. Bump to 0.9.0 signals the change.

### Edge Cases
- **Role with empty `forbidden` list**: `render_role()` handles gracefully (skips forbidden section).
- **Very long rendered prompts**: Each role renders to ~100-150 words. Within token budget for system messages.
- **Challenger rotation + role library**: Challenger mechanics in `solo.py` (line 59) and `council.py` are independent of role content. No interaction.

## Acceptance Criteria

### Commit 1: Structured Role Library
- [ ] `RoleDefinition` TypedDict defined in `prompts.py`
- [ ] `ROLE_LIBRARY` is `dict[str, RoleDefinition]` with 5 fields per role
- [ ] All existing roles enriched with epistemic_stance, reasoning_style, forbidden, output_pattern
- [ ] Devil's Advocate removed from library (14 remain from original 15)
- [ ] Data Analyst and Systems Thinker added (16 roles total)
- [ ] `render_role()` converts TypedDict → prompt string
- [ ] `resolve_role()` handles exact/case-insensitive/unknown lookup
- [ ] `run_solo()` uses `resolve_role()` — all existing behavior preserved
- [ ] `--list-roles` displays structured info
- [ ] `uv run python3 -c "from consilium import ROLE_LIBRARY; print(len(ROLE_LIBRARY))"` → 16
- [ ] `uv run consilium "test" --solo --quiet` works
- [ ] `uv run consilium "test" --list-roles` shows new format
- [ ] Existing council mode regression passes

### Commit 2: Confidence Expression (follow-up)
- [ ] `SOLO_DEBATE_SYSTEM` includes confidence instruction
- [ ] `COUNCIL_DEBATE_SYSTEM` includes confidence instruction
- [ ] Solo debate output includes "Confidence: N/10" line

## Changes

### 1. `consilium/prompts.py` — TypedDict + role library + helpers (~+180 lines, -20 lines)

**Add** `RoleDefinition` TypedDict, `render_role()`, and `resolve_role()` before `ROLE_LIBRARY`.

**Replace** `ROLE_LIBRARY: dict[str, str]` (lines 170-189) with `ROLE_LIBRARY: dict[str, RoleDefinition]`.

**16 roles** (14 existing minus Devil's Advocate, plus Data Analyst and Systems Thinker):

#### Structural Roles (3)

**Advocate**
```python
"Advocate": {
    "identity": "You see the strengths and opportunities. Build the strongest case FOR the default or obvious path. What's the upside everyone underestimates? Dismiss theoretical risks without concrete failure scenarios.",
    "epistemic_stance": "Success is the default. Burden of proof is on skeptics.",
    "reasoning_style": "Analogical — draw parallels to successful precedents.",
    "forbidden": ["Acknowledge a risk without immediately proposing a mitigation."],
    "output_pattern": "Lead with single strongest argument, not a list.",
},
```

**Skeptic**
```python
"Skeptic": {
    "identity": "You find the cracks. What breaks? What's everyone ignoring? Focus on hidden costs, risks, second-order effects. Dismiss optimistic projections without evidence.",
    "epistemic_stance": "Extraordinary claims require extraordinary evidence. Default: probably not.",
    "reasoning_style": "Deductive — find the logical flaw in the chain.",
    "forbidden": ["Say 'it depends.' Name the specific condition and its probability."],
    "output_pattern": "The claim is X. This fails because Y. The specific failure scenario is Z.",
},
```

**Pragmatist**
```python
"Pragmatist": {
    "identity": "You don't care about theoretical arguments. What's actually executable given real-world constraints? Focus on timelines, resource availability, minimum viable path. Dismiss theoretical elegance and abstract principles.",
    "epistemic_stance": "The plan that ships beats the plan that's perfect.",
    "reasoning_style": "Inductive — what does the evidence from similar situations show?",
    "forbidden": ["Recommend something that takes more than 2 sentences to explain."],
    "output_pattern": "Do this. Skip that. Here's why in one sentence.",
},
```

#### Domain Roles (11 existing + 2 new = 13)

All 11 existing domain roles enriched. Key examples:

**Financial Advisor**
```python
"Financial Advisor": {
    "identity": "You think in numbers, risk-adjusted returns, and opportunity cost. Every recommendation must have a dollar figure, timeline, or quantified risk attached.",
    "epistemic_stance": "If you can't quantify it, it doesn't exist.",
    "reasoning_style": "Quantitative — ranges, expected values, and thresholds.",
    "forbidden": ["Use 'significant' or 'substantial' without a number."],
    "output_pattern": "Every claim includes a range (best/expected/worst) and a timeline.",
},
```

**Career Coach**
```python
"Career Coach": {
    "identity": "You focus on the person, not the problem. What fits their energy, values, and life stage? What will they regret in 5 years? Push past 'strategically optimal' toward 'actually right for this human.'",
    "epistemic_stance": "The person usually already knows the answer.",
    "reasoning_style": "Abductive — best explanation for actual behavior, not stated preferences.",
    "forbidden": ["Give advice as 'you should.' Ask questions that reveal what they already want."],
    "output_pattern": "2-3 reframing questions, then one direct observation.",
},
```

(Full per-role specs in `~/notes/Councils/Role Library Research - 2026-02-17.md` lines 46-121.)

#### New Roles (2)

**Data Analyst** — Maps to De Bono White Hat (currently missing in consilium's hat coverage).
```python
"Data Analyst": {
    "identity": "Pure facts, no opinions. What's known, what's missing, what's the evidence? Separate established facts from claims, speculation, and anecdote.",
    "epistemic_stance": "Data speaks. Opinions don't.",
    "reasoning_style": "Empirical — what does the available evidence actually show?",
    "forbidden": ["Express an opinion or recommendation. Only state what is known and what is missing."],
    "output_pattern": "Known: [list]. Missing: [list]. Claim is supported/unsupported by [evidence].",
},
```

**Systems Thinker** — Second/third-order effects and feedback loops.
```python
"Systems Thinker": {
    "identity": "You see the system, not the parts. Every action triggers reactions. What feedback loops exist? What second-order effects will surprise everyone?",
    "epistemic_stance": "Linear thinking is the root of most planning failures.",
    "reasoning_style": "Systems dynamics — trace causal chains, identify reinforcing and balancing loops.",
    "forbidden": ["Analyze in isolation. Every factor connects to something else."],
    "output_pattern": "If X, first-order Y. But Y causes Z, which feeds back into X creating [loop type].",
},
```

#### Devil's Advocate — REMOVED

Research confirms one structurally mandated contrarian is sufficient — two dilute each other. Challenger rotation in `solo.py` (line 59) and `council.py` already handles this. Institutional learning confirms: "Static 'devil's advocate' roles decay into agreeable defaults by round 2. Use rotation to keep pushback fresh" (`multi-agent-deliberation-design.md`).

**Update** `SOLO_DEFAULT_ROLES` — stays `["Advocate", "Skeptic", "Pragmatist"]` (unchanged).

### 2. `consilium/solo.py` — Use resolve_role() (~10 lines changed)

**Import** `resolve_role` from `.prompts`.

**Replace** role resolution loop (lines 44-56) with:
```python
perspectives = []
for role_name in role_names:
    canonical_name, description = resolve_role(role_name)
    perspectives.append((canonical_name, description))
```

This eliminates the inline loop with variable mutation that the Python reviewer flagged.

### 3. `consilium/cli.py` — Update --list-roles display (~10 lines changed)

**Replace** lines 252-258:
```python
if args.list_roles:
    print("Available predefined roles for --solo --roles:\n")
    for name, role in ROLE_LIBRARY.items():
        print(f"  {name:<20s} {role['identity'][:70]}...")
    print(f"\nDefault: Advocate, Skeptic, Pragmatist")
    print(f"Unknown roles use a generic prompt. Any name works.")
    sys.exit(0)
```

Keep it simple — show identity (which now absorbs signal/noise/priorities). No grouping, no multi-line display. The structural/domain distinction is implicit in the names.

### 4. `consilium/__init__.py` — Export new symbols (~2 lines)

Add `render_role`, `resolve_role`, `RoleDefinition` to imports from `.prompts` and to `__all__`.

### 5. `pyproject.toml` — Bump version to 0.9.0

## What NOT to Build

- No dataclasses — TypedDict is sufficient for static lookup data
- No Pydantic validation — permissive unknown-role fallback stays
- No role recommendation engine — user picks roles manually
- No domain-specific role filtering — all roles available always
- No changes to discuss.py, redteam.py, or quick.py — they don't use roles
- No YAML/JSON role files — roles stay in Python for now
- No README changes
- No fancy --list-roles grouping — keep it flat and scannable
- No confidence expression in this commit — ships separately
- No normalized lowercase lookup cache — 16 roles, O(n) scan is fine

## Verification

### Commit 1: Structured Roles

```bash
cd ~/code/consilium

# Syntax + TypedDict check
uv run python3 -c "from consilium.prompts import ROLE_LIBRARY, RoleDefinition, render_role, resolve_role; print(f'{len(ROLE_LIBRARY)} roles'); print(render_role('Advocate', ROLE_LIBRARY['Advocate'])[:120])"

# Package import
uv run python3 -c "from consilium import ROLE_LIBRARY, render_role, resolve_role; print('OK')"

# Role count (should be 16)
uv run python3 -c "from consilium import ROLE_LIBRARY; assert len(ROLE_LIBRARY) == 16, f'Expected 16, got {len(ROLE_LIBRARY)}'"

# Devil's Advocate removed
uv run python3 -c "from consilium import ROLE_LIBRARY; assert \"Devil's Advocate\" not in ROLE_LIBRARY"

# New roles present
uv run python3 -c "from consilium import ROLE_LIBRARY; assert 'Data Analyst' in ROLE_LIBRARY; assert 'Systems Thinker' in ROLE_LIBRARY"

# resolve_role works for exact, case-insensitive, and unknown
uv run python3 -c "
from consilium.prompts import resolve_role
n1, d1 = resolve_role('Advocate')
assert n1 == 'Advocate'
n2, d2 = resolve_role('advocate')
assert n2 == 'Advocate'
n3, d3 = resolve_role('Space Pirate')
assert n3 == 'Space Pirate'
assert 'Space Pirate' in d3
print('resolve_role OK')
"

# --list-roles shows new format
uv run consilium "test" --list-roles

# Regression — solo still works
uv run consilium "What is 2+2?" --solo --quiet

# Regression — existing modes
uv run consilium "What is 2+2?" --quick --quiet

# Reinstall globally
uv tool install --force --reinstall ~/code/consilium
```

### Commit 2: Confidence Expression

```bash
# Verify confidence instruction in prompt
uv run python3 -c "from consilium.prompts import SOLO_DEBATE_SYSTEM; assert 'Confidence' in SOLO_DEBATE_SYSTEM; print('OK')"
uv run python3 -c "from consilium.prompts import COUNCIL_DEBATE_SYSTEM; assert 'Confidence' in COUNCIL_DEBATE_SYSTEM; print('OK')"
```

## References

### Internal
- Research: `~/notes/Councils/Role Library Research - 2026-02-17.md`
- Learnings: `~/docs/solutions/best-practices/multi-agent-deliberation-design.md`
- Learnings: `~/docs/solutions/ai-tooling/llm-council-judge-over-aggregation.md`
- Learnings: `~/docs/solutions/patterns/council-routing.md`
- De Bono Six Thinking Hats mapping: research note lines 139-149

### External
- [CrewAI: Crafting Effective Agents](https://docs.crewai.com/en/guides/agents/crafting-effective-agents) — 3-field agent structure (role/goal/backstory)
- [ConfMAD: Confidence in Multi-Agent Debate](https://arxiv.org/abs/2509.14034) — 78.3% → 83.3% MMLU with confidence signaling
- [Prompt Format Impact on LLM Performance](https://arxiv.org/abs/2411.10541) — up to 40% variance by format
- [Lakera Prompt Engineering Guide](https://www.lakera.ai/blog/prompt-engineering-guide) — constraint formatting best practices
