"""
core/activity.py — who spoke last, readable from any thread.

WHY THIS EXISTS
    Some decisions depend on the ORDER of the conversation, not its content:
    "did the user answer after the assistant asked?" and "is anyone talking
    right now, or can a background result be spoken?". Only main.py sees the
    audio and transcript streams, and action handlers run in executor threads
    with no reference to it. So main.py reports here and anything can read.

    main.py reports:
        set_speaking(bool)   the assistant's voice started / stopped
        note_user_input()    a real user turn arrived — the microphone hearing
                             sustained speech, its transcript, a typed command,
                             or a phone dashboard command

    Text the application itself injects (briefings, proactive check-ins, tool
    results, plugin speech) is NOT user input and must never be reported as it.

COST
    Two short deques behind a lock. Reporting is an append; nothing here runs
    on its own.
"""

from __future__ import annotations

import threading
import time
from collections import deque

_lock = threading.Lock()
_speaking = False
# Monotonic timestamps, newest last. Bounded: only recent ordering matters.
_speech_ends: deque[float] = deque(maxlen=64)
_user_inputs: deque[float] = deque(maxlen=64)


def set_speaking(value: bool) -> None:
    """Record the assistant's voice starting or stopping. Only a True→False
    transition counts as the end of an utterance; main.py calls this once per
    audio batch, so repeated Trues are expected and ignored."""
    global _speaking
    with _lock:
        if _speaking and not value:
            _speech_ends.append(time.monotonic())
        _speaking = bool(value)


def note_user_input() -> None:
    """Record that the user said or typed something just now."""
    with _lock:
        _user_inputs.append(time.monotonic())


def speaking() -> bool:
    with _lock:
        return _speaking


def user_answered_since(t: float) -> bool:
    """True if the assistant finished speaking after `t` and the user then said
    or typed something.

    Anchoring on the end of speech rather than on `t` itself matters: a
    transcript of the request that prompted the question can arrive a moment
    after the question was triggered, and must not count as the answer to it.
    """
    with _lock:
        spoken = next((e for e in _speech_ends if e > t), None)
        if spoken is None:
            return False
        return any(u > spoken for u in _user_inputs)


def wait_quiet(timeout: float = 90.0, settle: float = 2.0) -> bool:
    """Block until the assistant is not speaking and the user has been silent
    for `settle` seconds. Returns False if `timeout` passed first — callers
    should go ahead anyway rather than drop what they wanted to say."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _lock:
            now = time.monotonic()
            last_user = _user_inputs[-1] if _user_inputs else 0.0
            last_end = _speech_ends[-1] if _speech_ends else 0.0
            quiet = (not _speaking
                     and now - last_user >= settle
                     and now - last_end >= settle)
        if quiet:
            return True
        time.sleep(0.25)
    return False
