#!/usr/bin/env python3
"""
co_scientist.py — Terminal portal for managing parallel co-scientist sessions.

Each "chat" is one rental_agent.py process running inside a dedicated tmux
session.  The portal lets you:

  [1..N]   Attach to an existing investigation (blocks until you detach with
           Ctrl-B D).
  [N]      Start a new investigation — prompts for a name and research goal,
           then launches a tmux session running rental_agent.py.
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
    """
    Return a copy of *chats* with two extra fields added:
      alive     : bool  — tmux session is running
      checkpoint: dict  — loaded checkpoint (or {})
    """
    enriched = []
    for c in chats:
        ckpt = chat_registry.read_checkpoint(c.get("checkpoint_file", ""))
        alive = _tmux_session_alive(c.get("tmux_session", ""))
        enriched.append({**c, "alive": alive, "checkpoint": ckpt})
    return enriched


# ---------------------------------------------------------------------------
# Auto-relaunch for sessions that died while waiting for HPC results
# ---------------------------------------------------------------------------

def _relaunch_session(chat: dict) -> None:
    """Restart the tmux session for a chat that was 'waiting' but whose
    tmux process died (e.g. the machine rebooted)."""
    session = chat["tmux_session"]
    script  = chat.get("agent_script", str(_AGENT_SCRIPT))
    ckpt    = chat.get("checkpoint_file", "")
    chat_id = chat["chat_id"]
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
            chat["alive"] = True  # optimistic update for display


# ---------------------------------------------------------------------------
# Menu display
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 72


def _print_header() -> None:
    print()
    print("=" * 72)
    print("  AdaptiveComputing Co-Scientist")
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
            status = "unknown"  # session dead unexpectedly

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
    print("  [X]         Total reset — delete all chats and kill all daemons")
    print("               (agents, MCP server, manager). Does not delete the experiments registry.")
    print("  [R]         Refresh")
    print("  [Q]         Quit")
    print()
    print("  Tip: inside a session, press Ctrl-B then D to detach without stopping it.")
    print()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _attach(chat: dict) -> None:
    """Attach to a running tmux session (blocks until detach)."""
    session = chat["tmux_session"]
    if not chat["alive"]:
        print("\n  Session is not running.  Starting a new run...")
        _relaunch_session(chat)
    print()
    print("  Attaching to '{}'".format(session))
    print("  Press Ctrl-B then D to detach and return to this menu.")
    print()
    subprocess.run(["tmux", "attach-session", "-t", session])


def _start_new(chats: list) -> list:
    """Prompt for a short name, launch a new tmux session, then attach.

    The research goal is entered interactively inside the agent itself —
    rental_agent.py shows its own example prompt list and input when no
    goal is passed on the command line.
    """
    print()
    name = input("  Investigation name (short label): ").strip()
    if not name:
        print("  Cancelled.")
        return chats

    chat_id   = str(uuid.uuid4())
    ckpt_path = str(chat_registry.checkpoint_path(chat_id))
    session   = "co-sci-{}".format(chat_id[:8])
    script    = str(_AGENT_SCRIPT)

    # Launch without a goal so rental_agent prompts the user interactively.
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
        return chats

    chat_registry.add_chat(
        chat_id=chat_id,
        name=name,
        agent_script=script,
        tmux_session=session,
        checkpoint_file=ckpt_path,
    )

    # Attach immediately — the user enters their goal inside the agent.
    print()
    print("  Press Ctrl-B then D to detach and return to this menu.")
    print()
    subprocess.run(["tmux", "attach-session", "-t", session])
    return chat_registry.load_chats()


def _delete(chat: dict, chats_raw: list) -> list:
    """Kill session, remove checkpoint, and deregister the chat."""
    session = chat["tmux_session"]
    ckpt    = chat.get("checkpoint_file", "")
    name    = chat.get("name", chat["chat_id"][:8])

    confirm = input(
        "\n  Delete '{}' (kill session + remove files)? [y/N] ".format(name)
    ).strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return chats_raw

    # Kill tmux session
    subprocess.run(
        ["tmux", "kill-session", "-t", session],
        capture_output=True,
    )
    # Remove checkpoint file
    chat_registry.delete_checkpoint(ckpt)
    # Remove from registry
    chat_registry.remove_chat(chat["chat_id"])

    print(f"  Deleted '{name}'.")
    return chat_registry.load_chats()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _print_kill_reference() -> None:
    """Print background-daemon kill commands at startup."""
    print("=" * 72)
    print("  Background daemons (kill to pick up source-code changes):")
    print()
    print("  MCP server:  tmux kill-session -t ac_mcp_server")
    try:
        import importlib.util as _ilu
        import os as _os
        _hpc_path = str(Path(__file__).parent / "hpc_config.py")
        spec = _ilu.spec_from_file_location("_hpc_cfg", _hpc_path)
        _hpc = _ilu.module_from_spec(spec)
        spec.loader.exec_module(_hpc)
        from adaptive_computing.hpc.remote_manager import SESSION_NAME as _MGR_SESSION
        for machine in _hpc.machine_names:
            host = (_hpc.remote_hosts or {}).get(machine, "<login-node>")
            user = (_hpc.remote_usernames or {}).get(machine, "<user>")
            print("  Manager [{m}]:  ssh {u}@{h}  →  tmux kill-session -t {s}".format(
                m=machine, u=user, h=host, s=_MGR_SESSION))
    except Exception:
        print("  Manager:     ssh <login-node>  →  tmux kill-session -t manager_session")
        print("               (check hpc_config.py for the login node and username)")
    print()
    print("  Use [X] Total reset from the menu to kill everything at once.")
    print("  Use [D<N>] to delete individual agent sessions.")
    print("=" * 72)
    print()


def _total_reset(chats_raw: list) -> list:
    """Kill every background daemon: all agent sessions, MCP server, remote manager."""
    confirm = input(
        "\n  Total reset will DELETE ALL CHATS and kill all daemons\n"
        "  (agent sessions, MCP server, remote manager).\n"
        "  The experiments registry (registry.json + datasets/) is NOT deleted.\n"
        "  Proceed? [y/N] "
    ).strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return chats_raw

    killed = 0

    # Kill every registered agent session.
    for c in chats_raw:
        session = c.get("tmux_session", "")
        if session and _tmux_session_alive(session):
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
            print("  Killed agent session: {}".format(session))
            killed += 1
        chat_registry.delete_checkpoint(c.get("checkpoint_file", ""))
    chat_registry.save_chats([])

    # Kill MCP server.
    if subprocess.run(
        ["tmux", "has-session", "-t", "ac_mcp_server"], capture_output=True
    ).returncode == 0:
        subprocess.run(["tmux", "kill-session", "-t", "ac_mcp_server"], capture_output=True)
        print("  Killed MCP server (ac_mcp_server).")
        killed += 1
    else:
        print("  MCP server was not running.")

    # Kill remote manager(s) via SSH.
    try:
        import importlib.util as _ilu
        _hpc_path = str(Path(__file__).parent / "hpc_config.py")
        spec = _ilu.spec_from_file_location("_hpc_cfg", _hpc_path)
        _hpc = _ilu.module_from_spec(spec)
        spec.loader.exec_module(_hpc)
        from adaptive_computing.hpc.remote_manager import SESSION_NAME as _MGR_SESSION
        for machine in _hpc.machine_names:
            host = (_hpc.remote_hosts or {}).get(machine)
            user = (_hpc.remote_usernames or {}).get(machine)
            if not host or not user:
                continue
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 "{}@{}".format(user, host),
                 "tmux kill-session -t {} 2>/dev/null && echo KILLED || echo NOT_RUNNING".format(
                     _MGR_SESSION)],
                capture_output=True, text=True, timeout=20,
            )
            if "KILLED" in result.stdout:
                print("  Killed manager on {} ({}).".format(machine, host))
                killed += 1
            else:
                print("  Manager on {} was not running.".format(machine))
    except Exception as exc:
        print("  Could not reach remote manager: {}".format(exc))
        print("  Kill it manually: ssh <login-node> \"tmux kill-session -t manager_session\"")

    print("\n  Total reset complete — {} session(s) killed.".format(killed))
    print("  Restart co_scientist.py to begin fresh.\n")
    return []


def main() -> None:
    chats_raw = chat_registry.load_chats()
    _print_kill_reference()

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

        if upper == "X":
            chats_raw = _total_reset(chats_raw)
            continue

        # Delete: D<number>
        if upper.startswith("D") and upper[1:].isdigit():
            idx = int(upper[1:]) - 1
            if 0 <= idx < len(chats):
                chats_raw = _delete(chats[idx], chats_raw)
            else:
                print("  Invalid number.")
            continue

        # Attach: plain number
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(chats):
                _attach(chats[idx])
                # Refresh after returning from attach
                chats_raw = chat_registry.load_chats()
            else:
                print("  Invalid number.")
            continue

        print("  Unknown command: {}".format(raw))


if __name__ == "__main__":
    main()
