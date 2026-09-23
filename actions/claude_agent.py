"""
actions/claude_agent.py — hand a hard task to Claude, and only with the user's say-so.

WHAT IT IS FOR
    Gemini Live is the assistant's voice: quick, and good at picking one tool.
    It is not built to work through a job — read a folder, decide, act, check,
    try again. Claude Code is. This action runs it headless (`claude -p`) on the
    user's own Claude plan, in the background, and hands the outcome back to
    the conversation when it is done.

    It is a last resort by design: every call spends the user's plan and takes
    minutes rather than seconds, so the declaration steers the model to its
    other tools first.

ALWAYS ASKED, AND THE MODEL CANNOT ANSWER FOR THE USER
    Two calls, never one:
        action='ask'  registers the task and tells the model to ask the user,
                      in its own words, whether to call Claude;
        action='run'  starts it — refused unless the user spoke or typed AFTER
                      the assistant finished asking (core/activity.py keeps
                      that ordering).
    The task is fixed at 'ask', so what the user agreed to is what runs. What
    the gate cannot do is tell "yes" from "no" in every language — that part is
    the model's judgement. The gate guarantees the question was asked and
    answered, not what the answer was.

INSIDE THE TASK
    Claude runs in Claude Code's `auto` permission mode with nobody to answer
    prompts: a classifier lets ordinary work through and blocks the risky kind
    — deleting or overwriting existing files the task did not name, sending
    data out, installing packages nobody asked for. A blocked step is reported,
    not worked around. Claude's changes are NOT on the undo stack; its report
    says what it changed.

SETUP (once)
    Claude Code must be signed in to the user's plan:  claude auth login
    The binary is found on PATH, else the copy bundled with the Claude desktop
    app. Optional keys in config/api_keys.json:
        claude_path             explicit path to the claude executable
        claude_workdir          where Claude works (default: the home folder)
        claude_model            e.g. "sonnet" or "opus" (default: the plan's)
        claude_permission_mode  default "auto"
        claude_timeout_min      default 20
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from core import activity

_BASE = Path(__file__).resolve().parent.parent
_CONFIG = _BASE / "config" / "api_keys.json"
_WINDOWS = platform.system() == "Windows"

ASK_TTL = 180.0         # seconds an unanswered question stays valid
_STEP_LOG_GAP = 2.5     # min seconds between progress lines in the activity log
_REPORT_CHARS = 1500    # how much of Claude's report goes back to the model

# The user chose their plan, so a stray key or endpoint in the environment must
# never move this onto API billing or someone else's proxy. And if the assistant
# was itself launched from inside a Claude session (a terminal in the Claude
# app, say), that session's plumbing must not leak into the child: CLAUDECODE
# alone makes the CLI refuse to start. The user's own Claude Code settings —
# the Git Bash path, a `claude setup-token` token — still pass through.
_STRIP_ENV = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
              "CLAUDECODE", "CLAUDE_PID", "CLAUDE_AGENT_SDK_VERSION"}
_KEEP_ENV = {"CLAUDE_CODE_GIT_BASH_PATH", "CLAUDE_CODE_OAUTH_TOKEN"}

# Appended to Claude Code's own system prompt for every handed-off task.
_SYSTEM = (
    "You were started by the user's desktop voice assistant to do one task the "
    "user approved out loud. Nobody can answer questions or approve prompts while "
    "you work: make sensible choices and finish the job. If a step is blocked or "
    "impossible, say so rather than working around it. Do not delete or overwrite "
    "the user's existing files unless the task names them. Your final message is "
    "shown to the user and summarised aloud, so end with plain language and no "
    "markdown: what you did, where the results are, and anything they need to do. "
    "Two to five sentences."
)


@dataclass
class _Job:
    task: str
    details: str
    started: float
    proc: subprocess.Popen | None = None
    steps: int = 0
    last_step: str = ""
    done: bool = False
    ok: bool = False
    report: str = ""
    killed: str = ""


_lock = threading.Lock()
_pending: dict | None = None     # {"task", "asked_at"} while the user decides
_job: _Job | None = None         # the running job, or the last finished one
_signed_in = False               # cached once confirmed; re-checked until then


# ── Locating and checking Claude Code ────────────────────────────────────────

def _cfg() -> dict:
    try:
        return json.loads(_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _version_key(name: str) -> tuple:
    try:
        return tuple(int(p) for p in name.split("."))
    except ValueError:
        return (0,)


def _find_claude(cfg: dict) -> str | None:
    explicit = str(cfg.get("claude_path") or "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("claude")
    if found:
        return found
    exe = "claude.exe" if _WINDOWS else "claude"
    native = Path.home() / ".local" / "bin" / exe
    if native.is_file():
        return str(native)
    # The Claude desktop app ships its own copy in a versioned folder. Take the
    # newest: an update can leave the previous one behind for a while. The
    # Store / MSIX build keeps %APPDATA%\Claude virtualised — only processes
    # inside the package see it there — so its real home is under Packages.
    if _WINDOWS:
        roots = [Path(os.environ.get("APPDATA", "")) / "Claude" / "claude-code"]
        roots += (Path(os.environ.get("LOCALAPPDATA", "")) / "Packages").glob(
            "Claude_*/LocalCache/Roaming/Claude/claude-code")
    else:
        roots = [Path.home() / "Library" / "Application Support" / "Claude" / "claude-code"]
    copies = [p for root in roots if root.is_dir()
              for p in root.glob(f"*/{exe}") if p.is_file()]
    if not copies:
        return None
    return str(max(copies, key=lambda p: _version_key(p.parent.name)))


def _child_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if k in _KEEP_ENV
            or (k not in _STRIP_ENV and not k.startswith("CLAUDE_CODE_"))}


def _check_signed_in(exe: str) -> bool | None:
    """True / False from `claude auth status`; None when it cannot be read —
    an unreadable check must not block a task the CLI might run fine."""
    global _signed_in
    if _signed_in:
        return True
    try:
        r = subprocess.run([exe, "auth", "status"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=20,
                           env=_child_env())
        _signed_in = bool(json.loads(r.stdout or "{}").get("loggedIn"))
        return _signed_in
    except Exception as e:
        print(f"[Claude] auth status unreadable: {e}")
        return None


# ── The four actions ─────────────────────────────────────────────────────────

def claude_agent(parameters: dict, player=None, speak=None) -> str:
    action = str(parameters.get("action") or "ask").strip().lower()
    if action == "ask":
        return _ask(parameters, player)
    if action == "run":
        return _run(parameters, player, speak)
    if action == "status":
        return _status()
    if action == "cancel":
        return _cancel()
    return "Unknown action. Use ask, run, status or cancel."


def _busy() -> str | None:
    with _lock:
        j = _job
    if j and not j.done:
        mins = (time.monotonic() - j.started) / 60
        return (f"Claude is still working on an earlier task ({j.task[:80]}), "
                f"started {mins:.0f} min ago. Tell the user; they can wait or ask "
                f"you to cancel it.")
    return None


def _ask(parameters: dict, player) -> str:
    global _pending
    task = str(parameters.get("task") or "").strip()
    if not task:
        return ("No task given. Call again with 'task' describing exactly what the "
                "user wants done.")
    busy = _busy()
    if busy:
        return busy

    exe = _find_claude(_cfg())
    if not exe:
        _log(player, "Claude Code not found. Install it, or set claude_path in "
                     "config/api_keys.json.")
        return ("Claude is not installed on this computer, so this cannot be handed "
                "off. Tell the user in one sentence; the details are in the log.")
    if _check_signed_in(exe) is False:
        _log(player, f'Claude is not signed in. Run this once in a terminal:  '
                     f'"{exe}" auth login')
        return ("Claude is installed but not signed in to the user's plan, so nothing "
                "was started. Tell the user in one sentence that Claude needs signing "
                "in once; the command is in the log.")

    with _lock:
        _pending = {"task": task, "asked_at": time.monotonic()}
    return (
        "[ASK_USER] Nothing has started. Ask the user now, in ONE short sentence in "
        "their language and your own words, whether they want you to call Claude "
        "for this. The gist: this one needs Claude, shall I call it? Name the task "
        "in a few words and vary the wording each time. Then stop and wait for "
        "their answer. If they agree, call claude_agent with action='run', putting "
        "anything they added in 'details'. If they decline, drop it."
    )


def _run(parameters: dict, player, speak) -> str:
    global _pending, _job
    with _lock:
        p = _pending
    if p is None or time.monotonic() - p["asked_at"] > ASK_TTL:
        with _lock:
            _pending = None
        return ("Nothing is waiting for the user's approval, so nothing was started. "
                "Call claude_agent with action='ask' first, then wait for the user "
                "to answer.")
    if not activity.user_answered_since(p["asked_at"]):
        return ("The user has not answered yet, so nothing was started. Ask them "
                "whether to call Claude, and use action='run' only after they reply.")

    exe = _find_claude(_cfg())
    if not exe:
        return "Claude is no longer available on this computer. Nothing was started."

    with _lock:
        if _job and not _job.done:
            return _busy() or "Claude is busy."
        _pending = None
        job = _Job(task=p["task"],
                   details=str(parameters.get("details") or "").strip(),
                   started=time.monotonic())
        _job = job

    _log(player, f"Claude started: {job.task[:90]}")
    threading.Thread(target=_work, args=(job, exe, player, speak),
                     name="claude-agent", daemon=True).start()
    return ("[CLAUDE_STARTED] Claude is working on it in the background. Tell the "
            "user in a few words that it is underway. Do not wait for it: the "
            "outcome arrives later as a [CLAUDE_RESULT] message.")


def _status() -> str:
    with _lock:
        j = _job
    if j is None:
        return "Claude has not been given a task this session."
    if not j.done:
        mins = (time.monotonic() - j.started) / 60
        latest = f" Latest step: {j.last_step.rstrip('.')}." if j.last_step else ""
        return (f"Claude is still working on: {j.task[:120]}. {mins:.0f} min in, "
                f"{j.steps} steps so far.{latest}")
    state = "finished" if j.ok else "did not finish"
    return f"Claude's last task ({j.task[:120]}) {state}. Its report: {j.report[:600]}"


def _cancel() -> str:
    global _pending
    with _lock:
        j = _job
        had_question = _pending is not None
        _pending = None
    if j and not j.done and j.proc is not None:
        _kill(j, "cancelled by the user")
        return "Claude has been stopped. Anything it already changed stays changed."
    if had_question:
        return "Dropped. Claude was never started."
    return "Claude is not working on anything."


# ── Running the job ──────────────────────────────────────────────────────────

def _work(job: _Job, exe: str, player, speak) -> None:
    try:
        ok, report = _run_claude(job, exe, player)
    except Exception as e:
        ok, report = False, f"Claude could not be run: {e}"
    job.ok, job.report, job.done = ok, report, True
    _log(player, "Claude finished — report on screen." if ok
         else "Claude did not finish — see the panel for why.")
    if player:
        try:
            player.show_content(f"CLAUDE — {job.task[:38]}", report)
        except Exception:
            pass
    if speak:
        # Speaking over the user, or over the assistant mid-sentence, would cut
        # one of them off. Wait for a gap; if none comes, say it anyway.
        activity.wait_quiet()
        state = "Finished" if ok else "Did not finish"
        speak(f"[CLAUDE_RESULT] {state}. Task: {job.task[:200]}\n\n"
              f"Claude's report:\n{report[:_REPORT_CHARS]}")


def _run_claude(job: _Job, exe: str, player) -> tuple[bool, str]:
    cfg = _cfg()
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose",
           "--permission-mode", str(cfg.get("claude_permission_mode") or "auto"),
           "--permission-prompts", "none",
           "--append-system-prompt", _SYSTEM]
    model = str(cfg.get("claude_model") or "").strip()
    if model:
        cmd += ["--model", model]

    workdir = Path(str(cfg.get("claude_workdir") or Path.home())).expanduser()
    if not workdir.is_dir():
        workdir = Path.home()

    prompt = job.task
    if job.details:
        prompt += f"\n\nWhen approving, the user added: {job.details}"

    proc = subprocess.Popen(
        cmd, cwd=str(workdir), env=_child_env(),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        # POSIX: own process group, so a cancel takes the shells it spawned too.
        **({} if _WINDOWS else {"start_new_session": True}),
    )
    job.proc = proc

    err_tail: deque[str] = deque(maxlen=20)
    threading.Thread(target=lambda: err_tail.extend(proc.stderr),
                     daemon=True).start()
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except Exception:
        pass

    timeout_s = float(cfg.get("claude_timeout_min") or 20) * 60
    timer = threading.Timer(timeout_s, _kill, args=(job, "timed out"))
    timer.daemon = True
    timer.start()

    result, last_text, last_log = None, "", 0.0
    try:
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = ev.get("type")
            if kind == "assistant":
                for block in (ev.get("message") or {}).get("content") or []:
                    if block.get("type") == "tool_use":
                        job.steps += 1
                        job.last_step = _describe_step(block)
                        print(f"[Claude] → {job.last_step}")
                        # Every step goes to the console; the activity log gets
                        # a sample, so a 60-step job cannot bury the conversation.
                        now = time.monotonic()
                        if now - last_log >= _STEP_LOG_GAP:
                            last_log = now
                            _log(player, f"Claude → {job.last_step}", console=False)
                    elif block.get("type") == "text" and str(block.get("text", "")).strip():
                        last_text = block["text"].strip()
            elif kind == "result":
                result = ev
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()     # output is closed and the report is in; just reap it
    finally:
        timer.cancel()

    if job.killed:
        return False, f"Claude was stopped ({job.killed}) after {job.steps} steps. " \
                      f"Anything it already changed stays changed."
    if result is None:
        tail = " ".join(s.strip() for s in err_tail)[-400:]
        return False, (f"Claude exited without a result (code {proc.returncode})."
                       + (f" {tail}" if tail else ""))

    report = str(result.get("result") or last_text or "").strip() \
        or "Claude finished but wrote no report."
    denied = result.get("permission_denials") or []
    if denied:
        report += (f"\n\n({len(denied)} step{'s' if len(denied) != 1 else ''} "
                   f"blocked by Claude's safety check.)")
    ok = not result.get("is_error") and result.get("subtype", "success") == "success"
    return ok, report


_STEP_KEYS = ("command", "file_path", "path", "pattern", "query", "url",
              "description")


def _describe_step(block: dict) -> str:
    name = block.get("name") or "tool"
    inp = block.get("input") or {}
    for key in _STEP_KEYS:
        val = inp.get(key)
        if isinstance(val, str) and val.strip():
            detail = " ".join(val.split())
            if len(detail) > 70:
                detail = detail[:67] + "..."
            return f"{name}: {detail}"
    return name


def _kill(job: _Job, why: str) -> None:
    job.killed = why
    proc = job.proc
    if proc is None or proc.poll() is not None:
        return
    try:
        if _WINDOWS:
            # /T takes the whole tree: the shells and scripts Claude started.
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _log(player, message: str, console: bool = True) -> None:
    if console:
        print(f"[Claude] {message}")
    if player:
        try:
            player.write_log(f"SYS: {message}")
        except Exception:
            pass


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "claude_agent",
    "description": (
        "Hands a HARD, multi-step task to Claude, a much stronger AI agent that "
        "works through it by itself on this computer: sorting or analysing many "
        "files, substantial coding work, research that ends in a written result. "
        "Last resort: never use it for anything another tool can do directly "
        "(apps, settings, a single search, messages, reminders, the screen). "
        "It spends the user's Claude plan and takes minutes, so it is ALWAYS two "
        "calls: action='ask' first, which tells you how to ask the user, and "
        "action='run' only after they agree. The undo tool cannot reverse "
        "Claude's changes."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": (
                    "ask — propose the task and ask the user first | "
                    "run — start it, only after the user agreed | "
                    "status — how the running task is going | "
                    "cancel — stop it, or drop an unanswered question"
                ),
            },
            "task": {
                "type": "STRING",
                "description": (
                    "For ask: the complete task in English with every specific the "
                    "user gave (folders, file names, what finished looks like). "
                    "Claude sees only this, not the conversation."
                ),
            },
            "details": {
                "type": "STRING",
                "description": (
                    "For run: anything the user added when agreeing, e.g. 'only the "
                    "PDFs'. Leave empty otherwise."
                ),
            },
        },
        "required": ["action"],
    },
    "handler": claude_agent,
}
