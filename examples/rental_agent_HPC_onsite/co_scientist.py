#!/usr/bin/env python3
"""
co_scientist.py — Terminal portal for managing parallel co-scientist sessions.

Each "chat" is one controller.py process running inside a dedicated tmux
session.  The portal lets you:

  [1..N]   Attach to an existing investigation (blocks until you detach with
           Ctrl-B D).
  [N]      Start a new investigation — prompts for a name and research goal,
           then launches a tmux session running controller.py.
  [D1..DN] Delete investigation N — kills its tmux session, removes the
           checkpoint file, and removes it from chats.json.
  [R]      Refresh the status display.
  [Q]      Quit the portal (running investigations continue in the background).
"""

import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure this directory is on sys.path so chat_registry is found
# ---------------------------------------------------------------------------
_AGENT_DIR = Path(__file__).parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

import chat_registry  # noqa: E402 (after sys.path setup)

_AGENT_SCRIPT = _AGENT_DIR / "rental_agent.py"

# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "active":    "⚙ ",
    "waiting":   "⏳",
    "completed": "✅",
    "error":     "❌",
    "unknown":   "❓",
}

_STATUS_LABELS = {
    "active":    "ACTIVE   ",
    "waiting":   "WAITING  ",
    "completed": "DONE     ",
    "error":     "ERROR    ",
    "unknown":   "UNKNOWN  ",
}


def _tmux_session_alive(session_name: str) -> bool:
    """Return True if the tmux session exists."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("ERROR: tmux not found.  Please install tmux.")
        sys.exit(1)


def _relative_time(iso_ts: str) -> str:
    """Return a human-readable relative time string, e.g. '3m ago'."""
    if not iso_ts:
        return ""
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return "{}s ago".format(secs)
        if secs < 3600:
            return "{}m ago".format(secs // 60)
        if secs < 86400:
            return "{}h ago".format(secs // 3600)
        return "{}d ago".format(secs // 86400)
    except (ValueError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# Chat info enrichment
# ---------------------------------------------------------------------------

def _enrich_chats(chats: list) -> list:
    enriched = []
    for c in chats:
        ckpt  = chat_registry.read_checkpoint(c.get("checkpoint_file", ""))
        alive = _tmux_session_alive(c.get("tmux_session", ""))
        enriched.append({**c, "alive": alive, "checkpoint": ckpt})
    return enriched


# ---------------------------------------------------------------------------
# Auto-relaunch for sessions that died while waiting for HPC results
# ---------------------------------------------------------------------------

def _relaunch_session(chat: dict) -> None:
    """Restart the tmux session for a chat that was 'waiting' but whose
    tmux process died (e.g. the machine rebooted)."""
    session  = chat["tmux_session"]
    script   = chat.get("agent_script", str(_AGENT_SCRIPT))
    ckpt     = chat.get("checkpoint_file", "")
    chat_id  = chat["chat_id"]
    user_req = chat["checkpoint"].get("user_request", "")

    cmd = (
        f"python {shlex.quote(script)} "
        f"--chat-id {shlex.quote(chat_id)} "
        f"--checkpoint {shlex.quote(ckpt)} "
        f"--resume "
        f"{shlex.quote(user_req)}"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, cmd],
        capture_output=True,
    )
    print(f"[co_scientist] Relaunched session '{session}'.")


def _maybe_relaunch_dead_sessions(chats: list) -> None:
    """Auto-relaunch any dead sessions that were in 'waiting' or 'active' state."""
    for chat in chats:
        if chat["alive"]:
            continue
        status = chat["checkpoint"].get("status", "unknown")
        if status in ("waiting", "active"):
            print(
                f"[co_scientist] Session '{chat['tmux_session']}' died "
                f"(was {status}). Relaunching..."
            )
            _relaunch_session(chat)
            chat["alive"] = True


# ---------------------------------------------------------------------------
# Menu display
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 72


def _print_header() -> None:
    print()
    print("=" * 72)
    print("  AdaptiveComputing Co-Scientist  (HPC_onsite — LocalHero)")
    print("=" * 72)


def _print_chats(chats: list) -> None:
    if not chats:
        print("\n  (No investigations yet.  Press [N] to start one.)\n")
        return

    print()
    print("  {:>3}  {:<12} {:<40} {:<12}  {}".format(
        "#", "Status", "Name", "Updated", "Info"
    ))
    print("  " + _SEPARATOR)

    for i, chat in enumerate(chats, 1):
        ckpt   = chat["checkpoint"]
        status = ckpt.get("status", "unknown")
        if not chat["alive"] and status in ("waiting", "active"):
            status = "unknown"

        icon      = _STATUS_ICONS.get(status, "❓")
        label     = _STATUS_LABELS.get(status, "UNKNOWN  ")
        name      = chat.get("name", chat["chat_id"][:8])[:38]
        updated   = _relative_time(ckpt.get("updated_at", chat.get("created_at", "")))

        info_parts = []
        n_pending = ckpt.get("n_pending", 0)
        if status == "waiting" and n_pending:
            info_parts.append("{} pending".format(n_pending))
        best_y = ckpt.get("best_y")
        if best_y is not None:
            info_parts.append("best=${:.2f}".format(-best_y))
        info = "  ".join(info_parts)

        print("  {:>3}  {} {}  {:<40} {:<12}  {}".format(
            i, icon, label, name, updated, info
        ))

    print()


def _print_menu(n_chats: int) -> None:
    print("  [1..{}]     Attach to investigation".format(n_chats) if n_chats else
          "  (no investigations to attach)")
    print("  [N]         New investigation")
    if n_chats:
        print("  [D1..D{}]   Delete investigation (kill session + remove files)".format(
            n_chats
        ))
    print("  [R]         Refresh")
    print("  [Q]         Quit")
    print()
    print("  Tip: inside a session, press Ctrl-B then D to detach without stopping it.")
    print()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _attach(chat: dict) -> None:
    session = chat["tmux_session"]
    if not chat["alive"]:
        print("\n  Session is not running.  Starting a new run...")
        _relaunch_session(chat)
    print()
    print("  Attaching to '{}'".format(session))
    print("  Press Ctrl-B then D to detach and return to this menu.")
    print()
    subprocess.run(["tmux", "attach-session", "-t", session])


def _start_new(chats_raw: list) -> list:
    print()
    name = input("  Investigation name (short label): ").strip()
    if not name:
        print("  Cancelled.")
        return chats_raw

    chat_id   = str(uuid.uuid4())
    ckpt_path = str(chat_registry.checkpoint_path(chat_id))
    session   = "co-sci-{}".format(chat_id[:8])
    script    = str(_AGENT_SCRIPT)

    cmd = (
        f"python {shlex.quote(script)} "
        f"--chat-id {shlex.quote(chat_id)} "
        f"--checkpoint {shlex.quote(ckpt_path)}"
    )

    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, cmd],
        capture_output=True,
    )
    if result.returncode != 0:
        print("\n  ERROR: Could not start tmux session: {}".format(
            result.stderr.decode().strip()
        ))
        return chats_raw

    chat_registry.add_chat(
        chat_id=chat_id,
        name=name,
        agent_script=script,
        tmux_session=session,
        checkpoint_file=ckpt_path,
    )

    print()
    print("  Press Ctrl-B then D to detach and return to this menu.")
    print()
    subprocess.run(["tmux", "attach-session", "-t", session])
    return chat_registry.load_chats()


def _delete(chat: dict, chats_raw: list) -> list:
    session = chat["tmux_session"]
    ckpt    = chat.get("checkpoint_file", "")
    name    = chat.get("name", chat["chat_id"][:8])

    confirm = input(
        "\n  Delete '{}' (kill session + remove files)? [y/N] ".format(name)
    ).strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return chats_raw

    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    chat_registry.delete_checkpoint(ckpt)
    chat_registry.remove_chat(chat["chat_id"])
    print(f"  Deleted '{name}'.")
    return chat_registry.load_chats()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    chats_raw = chat_registry.load_chats()

    while True:
        chats = _enrich_chats(chats_raw)
        _maybe_relaunch_dead_sessions(chats)

        _print_header()
        _print_chats(chats)
        _print_menu(len(chats))

        try:
            raw = input("  Select: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not raw:
            continue

        upper = raw.upper()

        if upper == "Q":
            print("\n  Goodbye.  Running investigations continue in the background.")
            break

        if upper == "R":
            chats_raw = chat_registry.load_chats()
            continue

        if upper == "N":
            chats_raw = _start_new(chats_raw)
            continue

        if upper.startswith("D") and upper[1:].isdigit():
            idx = int(upper[1:]) - 1
            if 0 <= idx < len(chats):
                chats_raw = _delete(chats[idx], chats_raw)
            else:
                print("  Invalid number.")
            continue

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(chats):
                _attach(chats[idx])
                chats_raw = chat_registry.load_chats()
            else:
                print("  Invalid number.")
            continue

        print("  Unknown command: {}".format(raw))


if __name__ == "__main__":
    main()
