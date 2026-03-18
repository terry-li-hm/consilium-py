"""CLI entry point for consilium."""

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODE_TITLES = {
    "quick": "Quick Council",
    "discuss": "Roundtable Discussion",
    "redteam": "Red Team",
    "socratic": "Socratic Examination",
    "oxford": "Oxford Debate",
    "solo": "Solo Council",
    "council": "Council Deliberation",
    "deep": "Deep Council",
    "forecast": "Forecast",
    "premortem": "Pre-Mortem",
    "debate": "Multi-Round Debate",
}


def get_sessions_dir() -> Path:
    """Get the sessions directory, creating if needed."""
    sessions_dir = Path.home() / ".consilium" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text[:max_len].strip('-')


class LiveWriter:
    """Tee stdout to a file for live watching from another terminal."""

    def __init__(self, original, live_file):
        self._original = original
        self._file = live_file

    def write(self, data):
        self._original.write(data)
        if not self._file.closed:
            self._file.write(data)
            self._file.flush()

    def flush(self):
        self._original.flush()
        if not self._file.closed:
            self._file.flush()

    def isatty(self):
        return self._original.isatty()

    @property
    def encoding(self):
        return self._original.encoding

    def fileno(self):
        return self._original.fileno()


def _save_session(
    question: str,
    transcript: str,
    mode: str,
    header_extra: str = "",
    no_save: bool = False,
    output: str | None = None,
    quiet: bool = False,
) -> Path | None:
    """Save transcript to user-specified location and auto-save to sessions directory."""
    if output:
        Path(output).write_text(transcript)
        if not quiet:
            print(f"Transcript saved to: {output}")

    if no_save:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(question)
    filename = f"{timestamp}-{mode}-{slug}.md" if mode != "council" else f"{timestamp}-{slug}.md"
    session_path = get_sessions_dir() / filename

    session_content = f"""# {MODE_TITLES.get(mode, 'Session')}

**Question:** {question}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Mode:** {mode}
{header_extra}
---

{transcript}
"""
    session_path.write_text(session_content)
    if not quiet:
        print(f"Session saved to: {session_path}")

    return session_path


def _share_gist(question: str, transcript: str, mode: str, quiet: bool = False) -> str | None:
    """Upload transcript to secret GitHub Gist and return URL."""
    try:
        import tempfile
        title = MODE_TITLES.get(mode, "Session")
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', prefix=f'council-{mode}-', delete=False
        ) as f:
            f.write(f"# {title}\n\n")
            f.write(f"**Question:** {question}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")
            f.write(transcript)
            temp_path = f.name

        result = subprocess.run(
            ["gh", "gist", "create", temp_path, "--desc", f"{title}: {question[:50]}"],
            capture_output=True, text=True
        )
        os.unlink(temp_path)

        if result.returncode == 0:
            gist_url = result.stdout.strip()
            print(f"\nShared: {gist_url}")
            return gist_url
        else:
            print(f"Gist creation failed: {result.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print("Error: 'gh' CLI not found. Install with: brew install gh", file=sys.stderr)

    return None


def _log_history(
    question: str,
    mode: str,
    session_path: Path | None = None,
    gist_url: str | None = None,
    extra: dict | None = None,
    cost: float | None = None,
    duration: float | None = None,
) -> None:
    """Append entry to history.jsonl."""
    history_file = get_sessions_dir().parent / "history.jsonl"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question[:200],
        "mode": mode,
        "cost": round(cost, 4) if cost is not None else None,
        "duration": round(duration, 1) if duration is not None else None,
        "session": str(session_path) if session_path else None,
        "gist": gist_url,
    }
    if extra:
        log_entry.update(extra)
    with open(history_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def _finish_session(
    args,
    question: str,
    result,
    mode: str,
    header_extra: str = "",
    history_extra: dict | None = None,
) -> None:
    """Post-run ceremony: save session, share gist, log history, exit."""
    session_path = _save_session(question, result.transcript, mode,
        header_extra=header_extra,
        no_save=args.no_save, output=args.output, quiet=args.quiet)
    gist_url = _share_gist(question, result.transcript, mode) if args.share else None
    if result.failures:
        if history_extra is None:
            history_extra = {}
        history_extra["failures"] = result.failures
    _log_history(question, mode, session_path, gist_url, history_extra,
        cost=result.cost, duration=result.duration)
    # In piped mode, print only the session path (machine-readable)
    if getattr(args, '_piped_mode', False) and session_path:
        print(session_path)
    sys.exit(0)


def main():
    from .models import (
        COUNCIL,
        QUICK_MODELS,
        DISCUSS_MODELS,
        REDTEAM_MODELS,
        OXFORD_MODELS,
        classify_mode,
        detect_social_context,
    )
    from .prompts import ROLE_LIBRARY, DOMAIN_CONTEXTS

    parser = argparse.ArgumentParser(
        description="Multi-model deliberation CLI. Auto-routes by question type, or pick a mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default: auto-routes by question type (quick, council, oxford, redteam, socratic, discuss, solo).

Modes:
  (no flag)   Auto-route — Opus classifies, then runs quick or council
  --quick     Parallel queries, no debate (~$0.10)
  --council   Full multi-round debate + judge (~$0.50)
  --discuss   Hosted roundtable exploration (~$0.30)
  --socratic  Socratic variant of discuss: probing questions (~$0.30)
  --oxford    Binary for/against with rebuttals + verdict (~$0.40)
  --redteam   Adversarial stress-test of a plan (~$0.20)
  --solo      Claude debates itself in multiple roles (~$0.40)
  --forecast  Superforecasting: blind probs + reconciliation (~$0.20)
  --premortem Pre-mortem: assume failure, reason backward (~$0.20)
  --debate    Multi-round cross-critique + synthesis (~$0.30)

Examples:
  consilium "Should we use microservices or monolith?"
  consilium "What are the tradeoffs of SSR vs CSR?" --quick
  consilium "Career decision" --council --persona "builder who hates process"
  consilium "Is AI consciousness possible?" --discuss --rounds 0
  consilium "Hire seniors or train juniors?" --socratic
  consilium "Should we use microservices?" --oxford
  consilium "My plan: migrate to microservices..." --redteam
  consilium "Pricing strategy" --solo --roles "investor,founder,customer"
  consilium "Will we hit Q3 target?" --forecast
  consilium "We're launching next month..." --premortem
  consilium "CrewAI vs raw API for multi-agent?" --debate --rounds 2

Session management:
  consilium --stats                        # Cost breakdown by mode
  consilium --watch                        # Live tail from another tmux tab
  consilium --tui                          # TUI with phase/cost/time tracking
  consilium --view                         # View latest session
  consilium --search "career"              # Search all sessions
        """,
    )
    parser.add_argument("question", nargs="?", help="The question for the council to deliberate")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    parser.add_argument("--output", "-o", help="Save transcript to file")
    parser.add_argument("--context", "-c", help="Context hint for the judge")
    parser.add_argument("--format", "-f", choices=["json", "yaml", "prose"], default="prose", dest="output_format", help="Output format")
    parser.add_argument("--share", action="store_true", help="Upload transcript to secret GitHub Gist")
    parser.add_argument("--persona", "-p", help="Context about the person asking")
    parser.add_argument("--domain", help="Regulatory domain context (banking, healthcare, eu, fintech, bio)")
    parser.add_argument("--challenger", help="Which model should argue contrarian")
    parser.add_argument("--followup", action="store_true", help="Enable followup mode after judge synthesis")
    parser.add_argument("--decompose", action="store_true", help="Decompose complex question into sub-questions before deliberation")
    parser.add_argument("--xpol", action="store_true", help="Cross-pollination: second parallel pass where models investigate gaps in each other's blind claims")
    parser.add_argument("--no-save", action="store_true", help="Don't auto-save transcript")
    parser.add_argument("--quick", action="store_true", help="Quick mode: parallel queries, no debate/judge")
    parser.add_argument("--council", action="store_true", help="Full council: skip auto-routing, always run debate + judge")
    parser.add_argument("--discuss", action="store_true", help="Discussion mode: hosted roundtable exploration")
    parser.add_argument("--redteam", action="store_true", help="Red team mode: adversarial stress-test")
    parser.add_argument("--solo", action="store_true", help="Solo council: Claude debates itself")
    parser.add_argument("--socratic", action="store_true", help="Socratic mode: probing questions to expose assumptions")
    parser.add_argument("--oxford", action="store_true", help="Oxford debate: binary for/against with rebuttals and verdict")
    parser.add_argument("--deep", action="store_true", help="Deep council: full debate + decompose + 2 rounds (~$0.80)")
    parser.add_argument("--forecast", action="store_true", help="Forecast mode: superforecasting blind estimates + reconciliation (~$0.20)")
    parser.add_argument("--premortem", action="store_true", help="Pre-mortem mode: assume failure, reason backward to causes (~$0.20)")
    parser.add_argument("--debate", action="store_true", help="Debate mode: multi-round cross-critique + judge synthesis (~$0.30)")
    parser.add_argument("--motion", help="Override auto-generated debate motion for --oxford")
    parser.add_argument("--roles", help="Custom perspectives for --solo (comma-separated, >= 2)")
    parser.add_argument("--rounds", type=int, default=None, metavar="N", help="Rounds for --discuss or --socratic (0 = unlimited)")
    parser.add_argument("--list-roles", action="store_true", help="List available predefined roles")
    parser.add_argument("--stats", action="store_true", help="Show session analytics")
    parser.add_argument("--sessions", action="store_true", help="List recent sessions")
    parser.add_argument("--watch", action="store_true", help="Watch live council output (rich formatted)")
    parser.add_argument("--tui", action="store_true", help="Watch live council output (TUI with phase/cost tracking)")
    parser.add_argument("--view", nargs="?", const="latest", default=None, metavar="TERM", help="View a session in pager")
    parser.add_argument("--search", metavar="TERM", help="Search session content")
    parser.add_argument("--timeout", type=float, default=300.0, metavar="SECS",
                        help="Per-model timeout in seconds (default: 300)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judge synthesis (for external judge integration)")
    parser.add_argument(
        "--order",
        metavar="MODELS",
        help="Override model speaking order (comma-separated names, e.g. gpt,gemini,grok,deepseek,glm)",
    )
    parser.add_argument(
        "--no-anon-judge",
        action="store_true",
        help="Disable judge anonymisation (for debugging; default: anonymise brand names)",
    )
    args = parser.parse_args()

    # Handle --sessions
    if args.sessions:
        sessions_dir = get_sessions_dir()
        sessions = sorted(sessions_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            print("No sessions found.")
        else:
            print(f"Sessions in {sessions_dir}:\n")
            for s in sessions[:20]:
                mtime = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  {mtime}  {s.name}")
            if len(sessions) > 20:
                print(f"\n  ... and {len(sessions) - 20} more")
        sys.exit(0)

    # Handle --list-roles
    if args.list_roles:
        print("Available predefined roles for --solo --roles:\n")
        for name, desc in ROLE_LIBRARY.items():
            short = desc[:80] + "..." if len(desc) > 80 else desc
            print(f"  {name:<20s} {short}")
        print(f"\nDefault: Advocate, Skeptic, Pragmatist")
        print(f"Unknown roles use a generic prompt. Any name works.")
        sys.exit(0)

    # Handle --watch
    if args.watch:
        from .watch import watch_live

        live_link = get_sessions_dir().parent / "live.md"
        # Check for active session: live-*.md files indicate a running council
        live_dir = get_sessions_dir().parent
        active_files = list(live_dir.glob("live-*.md"))
        if not active_files:
            # Clean up stale symlink
            if live_link.is_symlink():
                live_link.unlink(missing_ok=True)
            print("(no active session — waiting for next council...)", file=sys.stderr, flush=True)
        watch_live(live_link)
        sys.exit(0)

    # Handle --tui
    if args.tui:
        from .tui import run_tui

        live_link = get_sessions_dir().parent / "live.md"
        live_dir = get_sessions_dir().parent
        active_files = list(live_dir.glob("live-*.md"))
        if not active_files:
            if live_link.is_symlink():
                live_link.unlink(missing_ok=True)
        run_tui(live_link)
        sys.exit(0)

    # Handle --view
    if args.view is not None:
        sessions_dir = get_sessions_dir()
        sessions = sorted(sessions_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            print("No sessions found.")
            sys.exit(1)

        if args.view == "latest":
            target = sessions[0]
        else:
            term = args.view.lower()
            matches = [s for s in sessions if term in s.name.lower()]
            if not matches:
                matches = [s for s in sessions if term in s.read_text()[:2000].lower()]
            if not matches:
                print(f"No sessions matching '{args.view}'.")
                sys.exit(1)
            if len(matches) > 1:
                print(f"({len(matches)} matches, showing most recent)\n")
            target = matches[0]

        pager = os.environ.get("PAGER", "less")
        os.execvp(pager, [pager, str(target)])

    # Handle --search
    if args.search:
        sessions_dir = get_sessions_dir()
        sessions = sorted(sessions_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        term = args.search.lower()
        matches = []
        for s in sessions:
            if term in s.read_text()[:3000].lower():
                matches.append(s)

        if not matches:
            print(f"No sessions matching '{args.search}'.")
        else:
            print(f"Sessions matching '{args.search}':\n")
            for s in matches[:20]:
                mtime = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                content = s.read_text()[:500]
                q_match = re.search(r'\*\*Question:\*\* (.+)', content)
                question = q_match.group(1)[:60] if q_match else s.stem
                print(f"  {mtime}  {question}")
            if len(matches) > 20:
                print(f"\n  ... and {len(matches) - 20} more")
        sys.exit(0)

    # Handle --stats
    if args.stats:
        from collections import defaultdict
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

        by_mode = defaultdict(list)
        for e in entries:
            by_mode[e.get("mode", "unknown")].append(e)

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

            avg_cost = f"${sum(costs)/len(costs):.2f}" if costs else "\u2014"
            sum_cost = sum(costs) if costs else 0
            total_cost += sum_cost
            total_cost_str = f"${sum_cost:.2f}" if costs else "\u2014"
            avg_dur = f"{sum(durations)/len(durations):.0f}s" if durations else "\u2014"

            print(f"{mode:<14s} {count:>8d} {avg_cost:>10s} {total_cost_str:>12s} {avg_dur:>10s}")

        print(f"\n{'Total':<14s} {len(entries):>8d} {'':>10s} ${total_cost:>11.2f}")

        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        recent = [e for e in entries if e.get("timestamp", "") >= cutoff]
        recent_cost = sum(e.get("cost", 0) or 0 for e in recent)
        print(f"\nLast 7 days: {len(recent)} sessions, ${recent_cost:.2f}")

        sys.exit(0)

    # Require question for normal operation
    if not args.question:
        parser.error("the following arguments are required: question")

    # Validate explicit mode flags (mutually exclusive check only — full validation after auto-routing)
    mode_flags = [f for f in ["--quick", "--council", "--discuss", "--redteam", "--solo", "--socratic", "--oxford", "--deep", "--forecast", "--premortem", "--debate"] if getattr(args, f.lstrip("-"))]
    if len(mode_flags) > 1:
        parser.error(f"{' and '.join(mode_flags)} are mutually exclusive")

    # Auto-detect social context
    social_mode = detect_social_context(args.question)
    if social_mode and not args.quiet:
        print("(Auto-detected social context — enabling social calibration)")
        print()

    # Validate and resolve domain
    domain_context = None
    if args.domain:
        if args.domain.lower() not in DOMAIN_CONTEXTS:
            print(f"Error: Unknown domain '{args.domain}'. Valid domains: {', '.join(DOMAIN_CONTEXTS.keys())}", file=sys.stderr)
            sys.exit(1)
        domain_context = args.domain.lower()

    # Resolve challenger model
    challenger_idx = None
    if args.challenger:
        challenger_lower = args.challenger.lower()
        model_name_map = {n.lower(): i for i, (n, _, _) in enumerate(COUNCIL)}
        if challenger_lower not in model_name_map:
            print(f"Error: Unknown model '{args.challenger}'. Valid models: {', '.join(n for n, _, _ in COUNCIL)}", file=sys.stderr)
            sys.exit(1)
        challenger_idx = model_name_map[challenger_lower]
    elif args.domain:
        challenger_idx = 0

    if not args.quiet and challenger_idx is not None:
        challenger_name = COUNCIL[challenger_idx][0]
        print(f"(Contrarian challenger: {challenger_name})")
        print()

    # Get API keys
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    google_api_key = os.environ.get("GOOGLE_API_KEY")

    # Silent mode: when stdout is piped (not a tty), suppress TUI output
    # and only print the session file path at the end
    piped_mode = not sys.stdout.isatty()
    if piped_mode and not args.quiet:
        args.quiet = True
        args._piped_mode = True  # Flag to print session path at end
    else:
        args._piped_mode = False

    # Set up live file for watching from another terminal
    live_dir = get_sessions_dir().parent
    if not args.quiet:
        live_pid_path = live_dir / f"live-{os.getpid()}.md"
        live_link = live_dir / "live.md"

        # Clean up stale live files — only delete if the PID is dead
        for old_live in live_dir.glob("live-*.md"):
            if old_live != live_pid_path:
                try:
                    old_pid = int(old_live.stem.split("-", 1)[1])
                    os.kill(old_pid, 0)  # Check if process is alive (no signal sent)
                except (ValueError, ProcessLookupError):
                    # Dead PID or unparseable — safe to delete
                    old_live.unlink(missing_ok=True)
                except PermissionError:
                    pass  # PID exists but owned by another user — leave it

        _live_file = open(live_pid_path, "w")
        sys.stdout = LiveWriter(sys.stdout, _live_file)

        try:
            live_link.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            live_link.symlink_to(live_pid_path.name)
        except OSError:
            pass

        def _cleanup_live():
            _live_file.close()
            # Don't delete — watchers (--tui, --watch) need time to read.
            # Stale files cleaned up on next session start.

        atexit.register(_cleanup_live)

    # Default: auto-route by question type (unless an explicit mode was chosen)
    auto_mode = None
    explicit_mode = any(getattr(args, f) for f in ("quick", "council", "discuss", "redteam", "solo", "socratic", "oxford", "deep", "forecast", "premortem", "debate"))
    if not explicit_mode:
        if not args.quiet:
            print("Classifying question...", flush=True)
        auto_mode = classify_mode(args.question, api_key)
        if not args.quiet:
            print(f"Mode: {auto_mode}")
            print()

        setattr(args, auto_mode, True)

    # Validate flag compatibility (after auto-routing so it catches both explicit and auto-routed modes)
    if args.quick:
        debate_flags = []
        if args.challenger:
            debate_flags.append("--challenger")
        if args.followup:
            debate_flags.append("--followup")
        if debate_flags:
            parser.error(f"--quick is incompatible with: {', '.join(debate_flags)}")

    for mode in ("discuss", "redteam", "solo", "socratic", "oxford"):
        if getattr(args, mode):
            incompatible = []
            if args.challenger:
                incompatible.append("--challenger")
            if args.followup:
                incompatible.append("--followup")
            if args.output_format != "prose":
                incompatible.append("--format")
            if incompatible:
                parser.error(f"--{mode} is incompatible with: {', '.join(incompatible)}")

    if hasattr(args, 'roles') and args.roles and not args.solo:
        parser.error("--roles requires --solo")

    if args.rounds is not None and not (args.discuss or args.socratic or args.debate):
        parser.error("--rounds requires --discuss, --socratic, or --debate")

    if args.motion and not args.oxford:
        parser.error("--motion requires --oxford")

    try:
        # Quick mode
        if args.quick:
            from .quick import run_quick

            if not args.quiet:
                print(f"Running quick council ({len(QUICK_MODELS)} models)...")
                print()

            result = run_quick(
                question=args.question,
                models=QUICK_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                format=args.output_format,
                timeout=args.timeout,
            )

            _finish_session(args, args.question, result, "quick",
                header_extra=f"**Models:** {', '.join(m.split('/')[-1] for _, m, _ in QUICK_MODELS)}",
                history_extra={"models": [m.split("/")[-1] for _, m, _ in QUICK_MODELS]})

        # Discussion mode
        if args.discuss:
            from .discuss import run_discuss

            discuss_rounds = args.rounds if args.rounds is not None else 2
            if not args.quiet:
                rounds_label = "unlimited — Ctrl+C to wrap up" if discuss_rounds == 0 else f"{discuss_rounds} rounds"
                print(f"Running roundtable discussion ({len(DISCUSS_MODELS)} panelists, Claude hosting, {rounds_label})...")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_discuss(
                question=args.question,
                panelists=DISCUSS_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
                rounds=discuss_rounds,
            )

            _finish_session(args, args.question, result, "discuss",
                header_extra=f"**Panelists:** {', '.join(name for name, _, _ in DISCUSS_MODELS)}",
                history_extra={"models": [name for name, _, _ in DISCUSS_MODELS]})

        # Red team mode
        if args.redteam:
            from .redteam import run_redteam

            if not args.quiet:
                print(f"Running red team ({len(REDTEAM_MODELS)} attackers, Claude hosting)...")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_redteam(
                question=args.question,
                panelists=REDTEAM_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
            )

            _finish_session(args, args.question, result, "redteam",
                header_extra=f"**Attackers:** {', '.join(name for name, _, _ in REDTEAM_MODELS)}",
                history_extra={"models": [name for name, _, _ in REDTEAM_MODELS]})

        # Socratic mode (probing questions, via discuss engine)
        if args.socratic:
            from .discuss import run_discuss

            socratic_rounds = args.rounds if args.rounds is not None else 2
            if not args.quiet:
                rounds_label = "unlimited — Ctrl+C to wrap up" if socratic_rounds == 0 else f"{socratic_rounds} rounds"
                print(f"Running Socratic examination ({len(DISCUSS_MODELS)} panelists, Claude examining, {rounds_label})...")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_discuss(
                question=args.question,
                panelists=DISCUSS_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
                rounds=socratic_rounds,
                style="socratic",
            )

            _finish_session(args, args.question, result, "socratic",
                header_extra=f"**Panelists:** {', '.join(name for name, _, _ in DISCUSS_MODELS)}",
                history_extra={"models": [name for name, _, _ in DISCUSS_MODELS]})

        # Oxford debate mode
        if args.oxford:
            from .oxford import run_oxford

            if not args.quiet:
                print(f"Running Oxford debate ({', '.join(name for name, _, _ in OXFORD_MODELS)} debating, Claude judging)...")
                if args.motion:
                    print(f"(Motion override: {args.motion})")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_oxford(
                question=args.question,
                debaters=OXFORD_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
                motion_override=args.motion,
            )

            _finish_session(args, args.question, result, "oxford",
                header_extra=f"**Debaters:** {', '.join(name for name, _, _ in OXFORD_MODELS)}",
                history_extra={"models": [name for name, _, _ in OXFORD_MODELS]})

        # Solo council mode
        if args.solo:
            from .solo import run_solo

            custom_roles = [r.strip().title() for r in args.roles.split(",")] if args.roles else None

            if not args.quiet:
                if custom_roles:
                    print(f"Running solo council ({' / '.join(custom_roles)})...")
                else:
                    print("Running solo council (Advocate / Skeptic / Pragmatist)...")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_solo(
                question=args.question,
                api_key=api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
                roles=custom_roles,
            )

            roles_label = " / ".join(custom_roles) if custom_roles else "Advocate / Skeptic / Pragmatist"
            _finish_session(args, args.question, result, "solo",
                header_extra=f"**Model:** Claude ({roles_label})")

        # Forecast mode
        if args.forecast:
            from .forecast import run_forecast

            forecast_panelists = [(n, m, fb) for n, m, fb in COUNCIL[:3]]  # GPT, Gemini, Grok
            if not args.quiet:
                print(f"Running forecast ({len(forecast_panelists)} panelists)...")
                print()

            result = run_forecast(
                question=args.question,
                panelists=forecast_panelists,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                timeout=args.timeout,
            )

            _finish_session(args, args.question, result, "forecast",
                header_extra=f"**Panelists:** {', '.join(n for n, _, _ in forecast_panelists)}",
                history_extra={"models": [n for n, _, _ in forecast_panelists]})

        # Pre-mortem mode
        if args.premortem:
            from .premortem import run_premortem

            premortem_panelists = [(n, m, fb) for n, m, fb in COUNCIL[:3]]  # GPT, Gemini, Grok
            if not args.quiet:
                print(f"Running pre-mortem ({len(premortem_panelists)} panelists, Claude hosting)...")
                print()

            result = run_premortem(
                question=args.question,
                panelists=premortem_panelists,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                timeout=args.timeout,
            )

            _finish_session(args, args.question, result, "premortem",
                header_extra=f"**Panelists:** {', '.join(n for n, _, _ in premortem_panelists)}",
                history_extra={"models": [n for n, _, _ in premortem_panelists]})

        # Debate mode
        if args.debate:
            from .debate import run_debate

            debate_panelists = [(n, m, fb) for n, m, fb in COUNCIL[:3]]  # GPT, Gemini, Grok
            debate_rounds = args.rounds if args.rounds is not None else 2
            if not args.quiet:
                print(f"Running debate ({len(debate_panelists)} panelists, {debate_rounds} rounds)...")
                print()

            result = run_debate(
                question=args.question,
                panelists=debate_panelists,
                api_key=api_key,
                google_api_key=google_api_key,
                rounds=debate_rounds,
                verbose=not args.quiet,
                timeout=args.timeout,
            )

            _finish_session(args, args.question, result, "debate",
                header_extra=f"**Panelists:** {', '.join(n for n, _, _ in debate_panelists)}\n**Rounds:** {debate_rounds}",
                history_extra={"models": [n for n, _, _ in debate_panelists], "rounds": debate_rounds})

        # Full council mode (explicit --council or auto-routed moderate/complex)
        # --deep: council + forced decompose + 2 rounds
        _council_rounds = 1
        _is_deep = args.deep
        if args.deep:
            args.decompose = True
            _council_rounds = 2

        from .council import decompose_question, run_council, run_followup_discussion

        # Apply --order override if specified
        council_config = list(COUNCIL)
        if hasattr(args, 'order') and args.order:
            order_names = [n.strip().lower() for n in args.order.split(",")]
            name_to_entry = {name.lower(): (name, model, fb) for name, model, fb in council_config}
            reordered = []
            for n in order_names:
                if n in name_to_entry:
                    reordered.append(name_to_entry[n])
                else:
                    print(f"Warning: --order: model '{n}' not found in council. Ignoring.")
            if reordered:
                council_config = reordered

        if not args.quiet:
            mode_parts = ["anonymous", "blind"]
            if social_mode:
                mode_parts.append("social")
            print(f"Running LLM Council ({', '.join(mode_parts)})...")
            if google_api_key:
                print("(Fallback enabled: Gemini→AI Studio)")
            if args.persona:
                print(f"(Persona context: {args.persona})")
            active_challenger_idx = challenger_idx if challenger_idx is not None else 0
            active_challenger_name = council_config[active_challenger_idx][0]
            print(f"(Challenger: {active_challenger_name})")
            print()

        # Skip collabeval for moderate auto-routed questions; use for complex or explicit --council
        use_collabeval = True
        cli_cost_accumulator: list[float] = []  # Track costs outside run_council
        sub_questions = None
        if args.decompose:
            if not args.quiet:
                print("Decomposing question into sub-questions...")
            sub_questions = decompose_question(
                question=args.question,
                api_key=api_key,
                verbose=not args.quiet,
                cost_accumulator=cli_cost_accumulator,
            )
            if not args.quiet and sub_questions:
                print("Sub-questions:")
                for i, sub_q in enumerate(sub_questions, 1):
                    print(f"{i}. {sub_q}")
                print()

        result = run_council(
            question=args.question,
            council_config=council_config,
            api_key=api_key,
            google_api_key=google_api_key,
            rounds=_council_rounds,
            verbose=not args.quiet,
            anonymous=True,
            blind=True,
            context=args.context,
            social_mode=social_mode,
            persona=args.persona,
            domain=domain_context,
            challenger_idx=challenger_idx,
            format=args.output_format,
            collabeval=use_collabeval,
            judge=not args.no_judge,
            sub_questions=sub_questions,
            cross_pollinate=args.xpol,
            anon_judge=not args.no_anon_judge,
        )

        transcript = result.transcript

        # Followup mode
        if args.followup and not args.quiet:
            print("\n" + "=" * 60)
            print("Enter topic to explore further (or 'done'): ", end="", flush=True)
            topic = input().strip()

            if topic and topic.lower() != "done":
                domain_ctxt = DOMAIN_CONTEXTS.get(domain_context, "") if domain_context else ""
                followup_transcript = run_followup_discussion(
                    question=args.question,
                    topic=topic,
                    council_config=council_config,
                    api_key=api_key,
                    domain_context=domain_ctxt,
                    social_mode=social_mode,
                    persona=args.persona,
                    verbose=not args.quiet,
                    cost_accumulator=cli_cost_accumulator,
                )
                transcript += "\n\n" + followup_transcript

        _mode_prefix = "deep, " if _is_deep else ""
        mode_label = f"{_mode_prefix}anonymous, blind{', xpol' if args.xpol else ''}{', social' if social_mode else ''}{f', auto-routed' if auto_mode else ''}{', collabeval' if use_collabeval else ''}{', no-judge' if args.no_judge else ''}"
        header_lines = f"**Mode:** {mode_label}"
        if args.context:
            header_lines += f"\n**Context:** {args.context}"
        if args.persona:
            header_lines += f"\n**Persona:** {args.persona}"

        history_extra = {
            "context": args.context,
            "models": [name for name, _, _ in council_config],
            "collabeval": use_collabeval,
            "judge": "external" if args.no_judge else "internal",
        }
        if auto_mode:
            history_extra["auto_mode"] = auto_mode

        # Council uses transcript (may include followup), not result.transcript directly
        from .models import SessionResult
        total_cost = result.cost + round(sum(cli_cost_accumulator), 4)
        final_result = SessionResult(transcript, total_cost, result.duration)
        _finish_session(args, args.question, final_result, "deep" if _is_deep else "council",
            header_extra=header_lines, history_extra=history_extra)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
