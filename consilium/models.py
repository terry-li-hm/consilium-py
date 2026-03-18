"""Model configurations, query functions, and shared helpers."""

import asyncio
import httpx
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SessionResult:
    """Structured return from all mode functions."""
    transcript: str
    cost: float
    duration: float
    failures: list[str] | None = None  # Model failure descriptions, if any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models"
BIGMODEL_URL = "https://api.z.ai/api/paas/v4/chat/completions"
XAI_URL = "https://api.x.ai/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Default council composition (static fallback — use resolved_council() at runtime)
# Format: (name, openrouter_model, fallback) - fallback is (provider, model) or None
# Fallback routing based on HK latency benchmark (2026-03-04):
#   GPT: None — gpt-5.2-pro is a standard chat/thinking model on OpenRouter
#   Claude: Anthropic direct (claude-sonnet-4-6 fallback)
#   Grok: xAI direct (5.8s) vs OR (13.0s) — direct much faster
#   DeepSeek: None — use OpenRouter only
#   GLM: z.ai direct (2.6s) vs OR (9.8s) — direct much faster
COUNCIL = [
    ("GPT", "openai/gpt-5.2-pro", None),
    ("Claude", "anthropic/claude-opus-4-6", ("anthropic", "claude-sonnet-4-6")),
    ("Grok-4.20\u03B2", "x-ai/grok-4", ("xai", "grok-4.20-experimental-beta-0304-reasoning")),
    ("DeepSeek", "deepseek/deepseek-v3.2", None),
    ("GLM", "z-ai/glm-5", ("zhipu", "glm-5")),
]

# Gemini is judge (not in council) — avoids judge self-bias
JUDGE_MODEL = "google/gemini-3.1-pro-preview"
# Critique model for CollabEval phase 2
CRITIQUE_MODEL = "anthropic/claude-sonnet-4-6"
# Classification model for auto-routing — Opus for accuracy since this gates mode selection
CLASSIFIER_MODEL = "anthropic/claude-opus-4-6"
# Compression model for context between rounds
COMPRESSION_MODEL = "meta-llama/llama-3.3-70b-instruct"
# Default xAI model slug (used by resolved_council and --grok flag)
XAI_DEFAULT_MODEL = "grok-4.20-experimental-beta-0304-reasoning"

# Env var names for model overrides
CONSILIUM_MODEL_M1_ENV = "CONSILIUM_MODEL_M1"
CONSILIUM_MODEL_M2_ENV = "CONSILIUM_MODEL_M2"
CONSILIUM_MODEL_M3_ENV = "CONSILIUM_MODEL_M3"
CONSILIUM_MODEL_M4_ENV = "CONSILIUM_MODEL_M4"
CONSILIUM_MODEL_M5_ENV = "CONSILIUM_MODEL_M5"
CONSILIUM_MODEL_JUDGE_ENV = "CONSILIUM_MODEL_JUDGE"
CONSILIUM_MODEL_CRITIQUE_ENV = "CONSILIUM_MODEL_CRITIQUE"
CONSILIUM_XAI_MODEL_ENV = "CONSILIUM_XAI_MODEL"
GLM_MAX_TOKENS_ENV = "GLM_MAX_TOKENS"

# Quick mode: judge + council (no judge conflict in quick mode)
QUICK_MODELS = [("Gemini", JUDGE_MODEL, None)] + [(n, m, fb) for n, m, fb in COUNCIL]

# Discussion mode: first 3 council panelists
DISCUSS_MODELS = COUNCIL[:3]  # GPT, Claude, Grok
DISCUSS_HOST = "anthropic/claude-opus-4-6"  # Claude hosts

# Red team mode: same 3-model panel
REDTEAM_MODELS = COUNCIL[:3]  # GPT, Claude, Grok

# Oxford debate: first 2 debaters
OXFORD_MODELS = COUNCIL[:2]  # GPT, Claude

# Thinking models - use non-streaming, higher tokens, longer timeout
THINKING_MODEL_SUFFIXES = {
    "claude-opus-4-6", "claude-opus-4.5",
    "gpt-5.2-pro", "gpt-5.2",
    "gemini-3.1-pro-preview",
    "grok-4",
    "deepseek-r1", "deepseek-v3.2",
    "glm-5",
}

# Keywords that suggest social/conversational context (auto-detect)
SOCIAL_KEYWORDS = [
    "interview", "ask him", "ask her", "ask them", "question to ask",
    "networking", "outreach", "message", "email", "linkedin",
    "coffee chat", "informational", "reach out", "follow up",
    "what should i say", "how should i respond", "conversation",
]

# Extraction model for structured summaries
EXTRACTION_MODEL = "anthropic/claude-haiku-4-5"


def is_thinking_model(model: str) -> bool:
    """Check if model is a thinking model that doesn't stream well."""
    model_name = model.split("/")[-1].lower()
    return (
        model_name in THINKING_MODEL_SUFFIXES
        or model_name.startswith("grok-4.2")  # covers all grok-4.20+ beta variants
    )


def model_max_output_tokens(model: str) -> int:
    """Return the known maximum output tokens for each model family."""
    m = model.lower()
    if "gemini-2.5" in m or "gemini-3" in m:
        return 65536
    if "gemini" in m:
        return 8192
    if "claude" in m or "anthropic" in m:
        return 32000
    if "gpt" in m or "openai" in m or "deepseek" in m:
        return 16384
    if "grok" in m or "xai" in m:
        return 32768
    if "kimi" in m or "moonshot" in m:
        return 16384
    if "glm" in m or "zhipu" in m:
        return 16000
    return 8192  # default fallback


def per_model_max_tokens(model: str, default: int) -> int:
    """Resolve per-model token budget overrides."""
    if "glm" in model.lower():
        val = _env_override(GLM_MAX_TOKENS_ENV)
        if val is not None:
            try:
                n = int(val)
                if n > 0:
                    return n
            except ValueError:
                pass
        return 16000
    return default


def _env_override(var: str) -> str | None:
    """Read an env var, returning None for empty/unset."""
    val = os.environ.get(var, "").strip()
    return val if val else None


def _normalize_model_override(value: str) -> str:
    """Resolve short aliases (sonnet, opus, gemini) to full model IDs."""
    trimmed = value.strip()
    match trimmed.lower():
        case "sonnet":
            return "anthropic/claude-sonnet-4-6"
        case "opus":
            return "anthropic/claude-opus-4-6"
        case "gemini":
            return "google/gemini-3.1-pro-preview"
        case _:
            return trimmed


def _display_name_from_model(model_id: str) -> str:
    """Generate a display name from a model ID (e.g. 'openai/gpt-5.2-pro' -> 'GPT-5.2-Pro')."""
    model_name = model_id.rsplit("/", 1)[-1]
    model_name = model_name.removesuffix("-preview")

    special = {"gpt": "GPT", "glm": "GLM", "deepseek": "DeepSeek"}
    parts = [p for p in model_name.split("-") if p]
    result = []
    for part in parts:
        low = part.lower()
        if low in special:
            result.append(special[low])
        else:
            result.append(part[0].upper() + part[1:] if part else "")
    return "-".join(result)


def _xai_model_label(model: str) -> str:
    """Short display label for xAI model slugs (condenses verbose beta names)."""
    if "4.20" in model:
        suffix = "-NR" if "non-reasoning" in model else ""
        return f"Grok-4.20\u03B2{suffix}"
    return _display_name_from_model(model)


def resolved_council() -> list[tuple[str, str, tuple[str, str] | None]]:
    """Resolve council models at runtime, applying env var overrides.

    This is the runtime source of truth for council composition.
    The COUNCIL constant is the default fallback.
    """
    # M1: GPT
    m1 = _env_override(CONSILIUM_MODEL_M1_ENV) or "openai/gpt-5.2-pro"
    m1_name = _display_name_from_model(m1)

    # M2: Claude
    m2 = _env_override(CONSILIUM_MODEL_M2_ENV) or "anthropic/claude-opus-4-6"
    m2_name = _display_name_from_model(m2)

    # M3: Grok
    m3 = _env_override(CONSILIUM_MODEL_M3_ENV) or "x-ai/grok-4"
    xai_model = _env_override(CONSILIUM_XAI_MODEL_ENV) or XAI_DEFAULT_MODEL
    m3_name = _xai_model_label(xai_model)

    # M4: DeepSeek
    m4 = _env_override(CONSILIUM_MODEL_M4_ENV) or "deepseek/deepseek-v3.2"
    m4_name = _display_name_from_model(m4)

    # M5: GLM
    m5_fallback = _env_override(CONSILIUM_MODEL_M5_ENV) or "glm-5"
    m5_name = _display_name_from_model("z-ai/glm-5")

    return [
        (m1_name, m1, None),
        (m2_name, m2, ("anthropic", "claude-sonnet-4-6")),
        (m3_name, m3, ("xai", xai_model)),
        (m4_name, m4, None),
        (m5_name, "z-ai/glm-5", ("zhipu", m5_fallback)),
    ]


def resolved_judge_model(cli_override: str | None = None) -> str:
    """Resolve judge model at runtime, applying CLI and env var overrides."""
    if cli_override:
        return _normalize_model_override(cli_override)
    env = _env_override(CONSILIUM_MODEL_JUDGE_ENV)
    if env:
        return _normalize_model_override(env)
    return JUDGE_MODEL


def resolved_critique_model(cli_override: str | None = None) -> str:
    """Resolve critique model at runtime, applying CLI and env var overrides."""
    if cli_override:
        return _normalize_model_override(cli_override)
    env = _env_override(CONSILIUM_MODEL_CRITIQUE_ENV)
    if env:
        return _normalize_model_override(env)
    return CRITIQUE_MODEL


def quick_models() -> list[tuple[str, str, tuple[str, str] | None]]:
    """Quick mode: judge + council models (no judge conflict in quick mode)."""
    judge = resolved_judge_model()
    judge_label = _display_name_from_model(judge)
    models: list[tuple[str, str, tuple[str, str] | None]] = [(judge_label, judge, None)]
    models.extend((n, m, fb) for n, m, fb in resolved_council() if m != judge)
    return models


def discuss_models() -> list[tuple[str, str, tuple[str, str] | None]]:
    """Discussion mode: first 3 council models."""
    return resolved_council()[:3]


def redteam_models() -> list[tuple[str, str, tuple[str, str] | None]]:
    """Red team mode: first 3 council models."""
    return resolved_council()[:3]


def oxford_models() -> list[tuple[str, str, tuple[str, str] | None]]:
    """Oxford debate: first 2 council models."""
    return resolved_council()[:2]


def is_error_response(content: str) -> bool:
    """Check if a response is an error string rather than real content."""
    return (
        not content
        or (
            content.startswith("[")
            and (
                content.startswith("[Error:")
                or content.startswith("[No response")
                or content.startswith("[Model still thinking")
            )
        )
    )


def fallback_also_failed_message(name: str, primary: str, fallback: str) -> str:
    """Build diagnostic when both primary and fallback attempts fail."""
    return f"[Fallback also failed for {name}: primary={primary}, fallback={fallback}]"


def sanitize_speaker_content(content: str) -> str:
    """Sanitize speaker content to prevent prompt injection."""
    sanitized = content.replace("SYSTEM:", "[SYSTEM]:")
    sanitized = sanitized.replace("INSTRUCTION:", "[INSTRUCTION]:")
    sanitized = sanitized.replace("IGNORE PREVIOUS", "[IGNORE PREVIOUS]")
    sanitized = sanitized.replace("OVERRIDE:", "[OVERRIDE]:")
    return sanitized


def detect_social_context(question: str) -> bool:
    """Auto-detect if the question is about social/conversational context."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in SOCIAL_KEYWORDS)


def classify_mode(
    question: str,
    api_key: str,
    cost_accumulator: list[float] | None = None,
) -> str:
    """Classify question into best deliberation mode. Returns mode name."""
    messages = [
        {"role": "system", "content": """Pick the best deliberation mode for this question. Respond with ONLY the mode name.

quick: Factual questions, straightforward comparisons, single-dimension — just need parallel opinions
council: Complex trade-offs, multi-stakeholder, strategic decisions with many interacting variables
oxford: Binary decisions with clear for/against framing — "should I X or Y?"
redteam: Stress-testing a plan, decision, or strategy — "what could go wrong with X?"
socratic: Exposing hidden assumptions, probing beliefs — "what am I not seeing about X?"
discuss: Open-ended exploration, no clear decision needed — "let's think about X"
solo: Niche — only when the user explicitly wants one model in multiple roles

Default to council when unsure."""},
        {"role": "user", "content": question},
    ]
    response = query_model(
        api_key, CLASSIFIER_MODEL, messages,
        max_tokens=10, timeout=15.0,
        cost_accumulator=cost_accumulator,
    )
    result = response.strip().lower().rstrip(".")
    valid_modes = ("quick", "council", "oxford", "redteam", "socratic", "discuss", "solo")
    if result in valid_modes:
        return result
    return "council"


def query_model(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
    stream: bool = False,
    retries: int = 2,
    cost_accumulator: list[float] | None = None,
) -> str:
    """Query a model via OpenRouter with retry logic for flaky models."""
    if is_thinking_model(model):
        max_tokens = max(max_tokens, 2500)
        timeout = max(timeout, 300.0)

    if stream:
        result = query_model_streaming(api_key, model, messages, max_tokens, timeout, cost_accumulator=cost_accumulator)
        if not result.startswith("["):
            return result
        print("(Streaming failed, retrying without streaming...)", flush=True)

    for attempt in range(retries + 1):
        if attempt > 0:
            # Exponential backoff with jitter: 2s, 4s base + random 0-1s
            backoff = (2 ** attempt) + random.random()
            time.sleep(backoff)

        try:
            response = httpx.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
        except (httpx.RequestError, httpx.RemoteProtocolError) as e:
            if attempt < retries:
                continue
            return f"[Error: Connection failed for {model}: {e}]"

        if response.status_code != 200:
            if attempt < retries:
                continue
            return f"[Error: HTTP {response.status_code} from {model}]"

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            if attempt < retries:
                continue
            return f"[Error: Invalid JSON response from {model}]"

        if "error" in data:
            if attempt < retries:
                continue
            return f"[Error: {data['error'].get('message', data['error'])}]"

        if "choices" not in data or not data["choices"]:
            if attempt < retries:
                continue
            return f"[Error: No response from {model}]"

        content = data["choices"][0]["message"]["content"]

        if not content or not content.strip():
            reasoning = data["choices"][0]["message"].get("reasoning", "")
            if reasoning and reasoning.strip():
                if attempt < retries:
                    continue
                return f"[Model still thinking - needs more tokens. Partial reasoning: {reasoning[:150]}...]"
            if attempt < retries:
                continue
            return f"[No response from {model} after {retries + 1} attempts]"

        if "<think>" in content:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        if cost_accumulator is not None:
            usage = data.get("usage", {})
            cost = usage.get("cost")
            if cost is not None:
                cost_accumulator.append(float(cost))

        # If streaming was requested but failed, print the non-streamed response
        if stream:
            print(content)

        return content

    return f"[Error: Failed to get response from {model}]"


def query_google_ai_studio(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 8192,
    timeout: float = 120.0,
    retries: int = 2,
) -> str:
    """Query Google AI Studio directly (fallback for Gemini models)."""
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})

    body = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        }
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = f"{GOOGLE_AI_STUDIO_URL}/{model}:generateContent?key={api_key}"

    for attempt in range(retries + 1):
        if attempt > 0:
            backoff = (2 ** attempt) + random.random()
            time.sleep(backoff)

        try:
            response = httpx.post(url, json=body, timeout=timeout)

            if response.status_code != 200:
                if attempt < retries:
                    continue
                return f"[Error: HTTP {response.status_code} from AI Studio {model}]"

            data = response.json()

            if "error" in data:
                if attempt < retries:
                    continue
                return f"[Error: {data['error'].get('message', data['error'])}]"

            candidates = data.get("candidates", [])
            if not candidates:
                if attempt < retries:
                    continue
                return f"[Error: No candidates from AI Studio {model}]"

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                if attempt < retries:
                    continue
                return f"[Error: No content from AI Studio {model}]"

            content = parts[0].get("text", "")
            if not content.strip():
                if attempt < retries:
                    continue
                return f"[No response from AI Studio {model} after {retries + 1} attempts]"

            return content

        except httpx.TimeoutException:
            if attempt < retries:
                continue
            return f"[Error: Timeout from AI Studio {model}]"
        except httpx.RequestError as e:
            if attempt < retries:
                continue
            return f"[Error: Request failed for AI Studio {model}]"

    return f"[Error: Failed to get response from AI Studio {model}]"


def query_xai_direct(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
    retries: int = 2,
) -> str:
    """Query xAI API directly (api.x.ai). OpenAI-compatible endpoint."""
    for attempt in range(retries + 1):
        if attempt > 0:
            backoff = (2 ** attempt) + random.random()
            time.sleep(backoff)

        try:
            response = httpx.post(
                XAI_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=timeout,
            )
        except (httpx.RequestError, httpx.RemoteProtocolError) as e:
            if attempt < retries:
                continue
            return f"[Error: xAI connection failed: {e}]"

        if response.status_code != 200:
            if attempt < retries:
                continue
            return f"[Error: HTTP {response.status_code} from xAI {model}]"

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            if attempt < retries:
                continue
            return f"[Error: Invalid JSON from xAI {model}]"

        if "error" in data:
            if attempt < retries:
                continue
            return f"[Error: {data['error'].get('message', data['error'])}]"

        if not data.get("choices"):
            if attempt < retries:
                continue
            return f"[Error: No choices from xAI {model}]"

        content = data["choices"][0]["message"].get("content", "")
        if not content or not content.strip():
            reasoning = data["choices"][0]["message"].get("reasoning", "")
            if reasoning and reasoning.strip():
                return f"[Model still thinking — increase max_tokens. Partial: {reasoning[:150]}...]"
            if attempt < retries:
                continue
            return f"[No response from xAI {model}]"

        if "<think>" in content:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        return content

    return f"[Error: Failed to get response from xAI {model}]"


def query_zhipu_direct(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
    retries: int = 2,
) -> str:
    """Query Zhipu/GLM API directly (api.z.ai). OpenAI-compatible endpoint."""
    for attempt in range(retries + 1):
        if attempt > 0:
            backoff = (2 ** attempt) + random.random()
            time.sleep(backoff)

        try:
            response = httpx.post(
                BIGMODEL_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=timeout,
            )
        except (httpx.RequestError, httpx.RemoteProtocolError) as e:
            if attempt < retries:
                continue
            return f"[Error: Zhipu connection failed: {e}]"

        if response.status_code != 200:
            if attempt < retries:
                continue
            return f"[Error: HTTP {response.status_code} from Zhipu {model}]"

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            if attempt < retries:
                continue
            return f"[Error: Invalid JSON from Zhipu {model}]"

        if "error" in data:
            if attempt < retries:
                continue
            return f"[Error: {data['error'].get('message', data['error'])}]"

        if not data.get("choices"):
            if attempt < retries:
                continue
            return f"[Error: No choices from Zhipu {model}]"

        content = data["choices"][0]["message"].get("content", "")
        if not content or not content.strip():
            if attempt < retries:
                continue
            return f"[No response from Zhipu {model}]"

        return content

    return f"[Error: Failed to get response from Zhipu {model}]"


def query_anthropic_direct(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
    retries: int = 2,
) -> str:
    """Query Anthropic API directly. Extracts system message per Anthropic format."""
    system_content = None
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            anthropic_messages.append(msg)

    body: dict = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
    }
    if system_content:
        body["system"] = system_content

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    for attempt in range(retries + 1):
        if attempt > 0:
            backoff = (2 ** attempt) + random.random()
            time.sleep(backoff)

        try:
            response = httpx.post(ANTHROPIC_URL, headers=headers, json=body, timeout=timeout)
        except (httpx.RequestError, httpx.RemoteProtocolError) as e:
            if attempt < retries:
                continue
            return f"[Error: Anthropic connection failed: {e}]"

        if response.status_code != 200:
            if attempt < retries:
                continue
            return f"[Error: HTTP {response.status_code} from Anthropic {model}]"

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            if attempt < retries:
                continue
            return f"[Error: Invalid JSON from Anthropic {model}]"

        if "error" in data:
            if attempt < retries:
                continue
            return f"[Error: {data['error'].get('message', data['error'])}]"

        content_blocks = data.get("content", [])
        if not content_blocks:
            if attempt < retries:
                continue
            return f"[No response from Anthropic {model}]"

        text = " ".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        ).strip()
        if not text:
            if attempt < retries:
                continue
            return f"[No text content from Anthropic {model}]"

        return text

    return f"[Error: Failed to get response from Anthropic {model}]"


def query_claude_print(
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
) -> str:
    """Query Claude via `claude --print` CLI (uses Max subscription via OAuth).

    Unsets CLAUDECODE env var to prevent hook recursion.
    Falls back with error string if claude CLI is unavailable or fails.
    """
    import subprocess

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] != "system"]

    prompt_parts: list[str] = []
    if system_parts:
        prompt_parts.append(system_parts[0])
        prompt_parts.append("")
    prompt_parts.extend(user_parts)
    prompt = "\n".join(prompt_parts)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE", None)

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", model, "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError:
        return "[Error: claude CLI not found]"
    except subprocess.TimeoutExpired:
        return f"[Error: claude --print timed out after {timeout}s]"

    if result.returncode != 0:
        return f"[Error: claude --print exited {result.returncode}: {result.stderr[:200]}]"

    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            content = data.get("result") or data.get("content") or data.get("message", "")
            if content:
                return str(content)
    except (json.JSONDecodeError, ValueError):
        pass

    text = result.stdout.strip()
    if text:
        return text

    return "[Error: Empty response from claude --print]"


def query_model_streaming(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
    cost_accumulator: list[float] | None = None,
) -> str:
    """Query a model with streaming output - prints tokens as they arrive."""
    import json as json_module

    full_content = []
    in_think_block = False
    error_msg = None

    try:
        with httpx.stream(
            "POST",
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                error_msg = f"[Error: HTTP {response.status_code} from {model}]"
            else:
                for line in response.iter_lines():
                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json_module.loads(data_str)
                            if "error" in data:
                                error_msg = f"[Error: {data['error'].get('message', data['error'])}]"
                                break

                            # Final chunk with usage/cost has empty choices
                            if cost_accumulator is not None and "usage" in data:
                                cost = data["usage"].get("cost")
                                if cost is not None:
                                    cost_accumulator.append(float(cost))

                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if "<think>" in content:
                                        in_think_block = True
                                    if in_think_block:
                                        if "</think>" in content:
                                            in_think_block = False
                                            content = content.split("</think>", 1)[-1]
                                        else:
                                            continue

                                    if content:
                                        print(content, end="", flush=True)
                                        full_content.append(content)
                        except json_module.JSONDecodeError:
                            pass

    except httpx.TimeoutException:
        error_msg = f"[Error: Timeout from {model}]"
    except (httpx.RequestError, httpx.RemoteProtocolError) as e:
        error_msg = f"[Error: Connection failed for {model}: {e}]"

    print()

    if error_msg:
        print(error_msg)
        return error_msg

    if not full_content:
        empty_msg = f"[No response from {model}]"
        print(empty_msg)
        return empty_msg

    return "".join(full_content)


async def query_model_async(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    name: str,
    fallback: tuple[str, str] | None = None,
    google_api_key: str | None = None,
    max_tokens: int = 500,
    retries: int = 2,
    cost_accumulator: list[float] | None = None,
) -> tuple[str, str, str]:
    """Async query for parallel phases. Returns (name, model_name, response).

    Routing order (matches Rust query_model_with_fallback):
      1. Direct provider first (if fallback specified and API key available)
      2. OpenRouter as fallback (if direct fails or no key)
    This matters for latency: xAI direct 5.8s vs OR 13s, Zhipu 2.6s vs OR 9.8s.
    """
    if is_thinking_model(model):
        max_tokens = max(max_tokens, 1500)

    model_name = model.split("/")[-1]

    # Step 1: Try direct provider FIRST (latency optimization)
    if fallback:
        fallback_provider, fallback_model = fallback
        direct_response = None

        if fallback_provider == "anthropic":
            # Claude: try claude --print (Max subscription) → Anthropic API key
            direct_response = await asyncio.to_thread(
                query_claude_print, fallback_model, messages, max_tokens
            )
            if is_error_response(direct_response):
                anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
                if anthropic_key:
                    direct_response = await asyncio.to_thread(
                        query_anthropic_direct, anthropic_key, fallback_model, messages, max_tokens
                    )
        elif fallback_provider == "xai":
            xai_key = os.environ.get("XAI_API_KEY")
            if xai_key:
                direct_response = await asyncio.to_thread(
                    query_xai_direct, xai_key, fallback_model, messages, max_tokens
                )
        elif fallback_provider == "zhipu":
            zhipu_key = os.environ.get("ZHIPU_API_KEY")
            if zhipu_key:
                direct_response = await asyncio.to_thread(
                    query_zhipu_direct, zhipu_key, fallback_model, messages, max_tokens
                )
        elif fallback_provider == "google" and google_api_key:
            direct_response = query_google_ai_studio(
                google_api_key, fallback_model, messages, max_tokens=max_tokens
            )

        if direct_response and not is_error_response(direct_response):
            return (name, fallback_model, direct_response)
        # Direct failed or no key — fall through to OpenRouter

    # Step 2: OpenRouter fallback
    for attempt in range(retries + 1):
        if attempt > 0:
            backoff = (2 ** attempt) + random.random()
            await asyncio.sleep(backoff)

        try:
            response = await client.post(
                OPENROUTER_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            )

            if response.status_code != 200:
                if attempt < retries:
                    continue
                break

            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError):
                if attempt < retries:
                    continue
                break

            if "error" in data:
                if attempt < retries:
                    continue
                break

            if "choices" not in data or not data["choices"]:
                if attempt < retries:
                    continue
                break

            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                if attempt < retries:
                    continue
                break

            if not content or not content.strip():
                reasoning = data["choices"][0]["message"].get("reasoning", "")
                if reasoning and reasoning.strip():
                    if attempt < retries:
                        continue
                    return (name, model_name, f"[Model still thinking - increase max_tokens. Partial: {reasoning[:200]}...]")
                if attempt < retries:
                    continue
                break

            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            if cost_accumulator is not None:
                usage = data.get("usage", {})
                cost = usage.get("cost")
                if cost is not None:
                    cost_accumulator.append(float(cost))

            return (name, model_name, content)

        except (httpx.RequestError, httpx.RemoteProtocolError):
            if attempt < retries:
                continue
            break

    return (name, model_name, f"[No response from {model_name} after {retries + 1} attempts]")


async def run_parallel(
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    messages: list[dict],
    api_key: str,
    google_api_key: str | None = None,
    max_tokens: int = 500,
    cost_accumulator: list[float] | None = None,
    verbose: bool = False,
) -> list[tuple[str, str, str]]:
    """Parallel query panelists with shared messages. Returns [(name, model_name, response)].

    When verbose=True, prints each model's response as soon as it completes.
    """
    indexed_results: list[tuple[int, tuple[str, str, str] | Exception]] = []

    async def _query_and_print(idx, name, model, fallback, client):
        try:
            result = await query_model_async(
                client, model, messages, name, fallback,
                google_api_key, max_tokens=max_tokens,
                cost_accumulator=cost_accumulator,
            )
        except Exception as e:
            indexed_results.append((idx, e))
            return
        if verbose:
            _, model_name, response = result
            if not response.startswith("["):
                print(f"\n### {name}")
                print(response)
                print(flush=True)
        indexed_results.append((idx, result))

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=300.0,
    ) as client:
        tasks = [
            _query_and_print(i, name, model, fallback, client)
            for i, (name, model, fallback) in enumerate(panelists)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    indexed_results.sort(key=lambda x: x[0])

    out = []
    for idx, result in indexed_results:
        name, model, fallback = panelists[idx]
        model_name = model.split("/")[-1]
        if isinstance(result, Exception):
            out.append((name, model_name, f"[Error: {result}]"))
        else:
            out.append(result)
    return out


_CONFIDENCE_RE = re.compile(
    r'\*{0,2}Confidence\*{0,2}:?\s*(\d{1,2})\s*(?:/\s*10|out\s+of\s+10)',
    re.IGNORECASE,
)


def parse_confidence(response: str) -> int | None:
    """Extract Confidence: N/10 from a debate response."""
    match = _CONFIDENCE_RE.search(response)
    if match:
        value = int(match.group(1))
        return value if 0 <= value <= 10 else None
    return None




# Brand name aliases for each council model — used to scrub judge transcript
_BRAND_ALIASES: dict[str, list[str]] = {
    "GPT":      ["GPT", "OpenAI", "ChatGPT", "gpt-5", "gpt-4", "gpt 5", "gpt 4"],
    "Claude":   ["Claude", "Anthropic", "claude-opus", "claude-sonnet", "claude 4", "claude opus"],
    "Grok-4.20\u03B2": ["Grok", "xAI", "x.ai", "grok-4", "grok 4"],
    "DeepSeek": ["DeepSeek", "deepseek-r1", "deepseek r1", "deepseek-v3", "deepseek v3"],
    "GLM":      ["GLM", "Zhipu", "ChatGLM", "glm-5", "glm 5", "zhipuai"],
}


def anonymise_for_judge(
    deliberation_text: str,
    display_names: dict[str, str],
    council_config: list[tuple],
) -> str:
    """Scrub model brand names from deliberation text before judge sees it.

    Layer 2 of judge anonymisation (Layer 1 = display_names substitution already done).
    Catches in-prose self-references like 'as an OpenAI model' or 'GPT suggested'.
    Case-insensitive, word-boundary regex to avoid mangling URLs or technical strings.
    """
    text = deliberation_text
    for name, _, _ in council_config:
        speaker_alias = display_names.get(name, name)
        aliases = _BRAND_ALIASES.get(name, [name])
        for alias in aliases:
            # Word boundary match, case-insensitive; skip terms inside URLs (after /)
            pattern = r'(?<![/.\w])' + re.escape(alias) + r'\b'
            text = re.sub(pattern, speaker_alias, text, flags=re.IGNORECASE)
    return text


def detect_consensus(
    conversation: list[tuple[str, str]],
    council_config: list[tuple[str, str, tuple[str, str] | None]],
    current_challenger_idx: int | None = None,
) -> tuple[bool, str]:
    """Detect if council has converged. Returns (converged, reason).

    Excludes the current challenger from consensus count since they're
    structurally incentivized to disagree.
    """
    council_size = len(council_config)

    if len(conversation) < council_size:
        return False, "insufficient responses"

    recent = conversation[-council_size:]

    # Exclude challenger from consensus count
    if current_challenger_idx is not None:
        challenger_name = council_config[current_challenger_idx][0]
        recent = [(name, text) for name, text in recent if name != challenger_name]

    effective_size = len(recent)
    if effective_size == 0:
        return False, "no non-challenger responses"

    threshold = effective_size - 1  # Need all-but-one non-challengers to agree

    consensus_count = sum(1 for _, text in recent if "CONSENSUS:" in text.upper())
    agreement_phrases = ["i agree with", "i concur", "we all agree", "consensus emerging"]
    agreement_count = sum(
        1 for _, text in recent
        if any(phrase in text.lower() for phrase in agreement_phrases)
    )

    if consensus_count >= threshold:
        potential_consensus, reason = True, "explicit consensus signals"
    elif agreement_count >= threshold:
        potential_consensus, reason = True, "agreement language detected"
    else:
        return False, "no consensus"

    # If consensus reached, ensure the challenger isn't actively dissenting
    if potential_consensus and current_challenger_idx is not None:
        challenger_name = council_config[current_challenger_idx][0].lower()
        full_recent = conversation[-council_size:]
        dissent_phrases = [
            "i disagree", "i challenge", "this is wrong",
            "critical flaw", "fundamental problem", "overlooking", "must object",
        ]
        for name, text in full_recent:
            if name.lower() == challenger_name:
                lower = text.lower()
                if any(phrase in lower for phrase in dissent_phrases):
                    return False, "challenger actively dissenting"

    return True, reason


EXTRACTION_PROMPT = """Extract a structured JSON summary from this judge synthesis.

The text uses ACH (Analysis of Competing Hypotheses) with Competing Hypotheses (H1, H2, etc.), Points of Disagreement, and a Synthesis section.

Return ONLY valid JSON (no markdown fences) matching this schema:

{
  "decision": "The core recommendation in 1-2 actionable sentences",
  "confidence": "high|medium|low",
  "winning_hypothesis": "H label + one-line summary of the endorsed hypothesis",
  "reasoning_summary": "2-3 sentence summary of the Synthesis section",
  "dissents": [{"model": "speaker/model name", "concern": "what they disagreed on"}]
}

Rules:
- decision: summarize the Synthesis into what should actually be done, not a section heading
- winning_hypothesis: which H was endorsed or closest to the final position
- dissents: unresolved disagreements from Points of Disagreement with the speaker who raised them
- If no content for a field, use "" or []"""


def _parse_recommendation_items(judge_response: str) -> dict:
    """Extract Do Now, Consider Later, Skip items from Recommendation section using regex."""
    result: dict = {}

    rec_match = re.search(r'## Recommendation[^\n]*\n(.*?)(?=\n## |\Z)', judge_response, re.DOTALL)
    if not rec_match:
        return result
    rec_text = rec_match.group(1)

    # Do Now: bold numbered items like **1. Title here.**
    do_now_match = re.search(r'### Do Now[^\n]*\n(.*?)(?=\n### |\Z)', rec_text, re.DOTALL)
    if do_now_match:
        items = re.findall(r'\*\*\d+\.\s*(.+?)\*\*', do_now_match.group(1))
        result["do_now"] = [item.strip().rstrip('.') for item in items if len(item.strip()) > 5]

    # Consider Later: bullet items with bold lead
    consider_match = re.search(r'### Consider Later[^\n]*\n(.*?)(?=\n### |\Z)', rec_text, re.DOTALL)
    if consider_match:
        items = re.findall(r'^[-*]\s+\*\*(.+?)\*\*', consider_match.group(1), re.MULTILINE)
        if not items:
            items = [line.lstrip('-* ').strip() for line in consider_match.group(1).split('\n')
                     if line.strip().startswith(('-', '*')) and len(line.strip()) > 5]
        result["consider_later"] = items

    # Skip: bullet items with bold lead
    skip_match = re.search(r'### Skip[^\n]*\n(.*?)(?=\n### |\n---|\Z)', rec_text, re.DOTALL)
    if skip_match:
        items = re.findall(r'^[-*]\s+\*\*(.+?)\*\*', skip_match.group(1), re.MULTILINE)
        if not items:
            items = [line.lstrip('-* ').strip() for line in skip_match.group(1).split('\n')
                     if line.strip().startswith(('-', '*')) and len(line.strip()) > 5]
        result["skip"] = items

    return result


def _extract_for_llm(judge_response: str) -> str:
    """Extract interpretive sections for LLM summarization (not the structured Recommendation items)."""
    parts = []
    for header in ("Competing Hypotheses", "Points of Disagreement", "Synthesis"):
        pattern = rf"## {header}[^\n]*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, judge_response, re.DOTALL)
        if match:
            parts.append(f"## {header}\n{match.group(1).strip()}")
    if parts:
        return "\n\n".join(parts)
    return judge_response


def extract_structured_summary(
    judge_response: str,
    question: str,
    models_used: list[str],
    rounds: int,
    duration: float,
    cost: float,
    api_key: str | None = None,
    cost_accumulator: list[float] | None = None,
) -> dict:
    # Step 1: Code-level extraction of structured items (deterministic)
    code_items = _parse_recommendation_items(judge_response)

    # Step 2: LLM extraction for interpretive fields only
    extracted = {}
    if api_key:
        try:
            focused_input = _extract_for_llm(judge_response)
            messages = [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": focused_input},
            ]
            raw = query_model(api_key, EXTRACTION_MODEL, messages, max_tokens=600, timeout=30.0, cost_accumulator=cost_accumulator)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            extracted = json.loads(raw)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Fallback for LLM fields
    if not extracted:
        synth_match = re.search(r'## Synthesis[^\n]*\n(.*?)(?=\n## |\Z)', judge_response, re.DOTALL)
        synth_text = synth_match.group(1).strip() if synth_match else ""

        # Decision: first substantial line from Synthesis
        decision = ""
        for line in (synth_text or judge_response).split('\n'):
            line = line.strip()
            if line and len(line) > 30 and not line.startswith('#'):
                decision = line
                break

        extracted = {
            "decision": decision[:500] if decision else "See transcript for details",
            "confidence": "medium",
            "winning_hypothesis": "",
            "reasoning_summary": synth_text[:500] if synth_text else "See transcript for details",
            "dissents": [],
        }

    # Step 3: Merge code-extracted items (override LLM for structured fields)
    if code_items.get("do_now"):
        extracted["do_now"] = code_items["do_now"]
    if code_items.get("consider_later"):
        extracted["consider_later"] = code_items["consider_later"]
    if code_items.get("skip"):
        extracted["skip"] = code_items["skip"]

    # Derive action_items from do_now if not provided by LLM
    if not extracted.get("action_items") and code_items.get("do_now"):
        extracted["action_items"] = [{"action": item, "priority": "high"} for item in code_items["do_now"]]

    # Always add meta and question
    extracted["schema_version"] = "1.0"
    extracted["question"] = question
    extracted["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "models_used": models_used,
        "rounds": rounds,
        "duration_seconds": duration,
        "estimated_cost_usd": cost,
    }
    return extracted
