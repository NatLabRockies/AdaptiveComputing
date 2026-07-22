"""
chat_registry.py — Persistent store for co-scientist chat sessions.

Manages two artefacts on disk:
  chats.json        — index of all known chats (tmux session, script path, etc.)
  checkpoints/<id>_checkpoint.json — per-chat status snapshot written by
                                     the agent while it runs.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_AGENT_DIR = Path(__file__).parent
_CHATS_FILE = _AGENT_DIR / "chats.json"
_CHECKPOINTS_DIR = _AGENT_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# chats.json helpers
# ---------------------------------------------------------------------------

def load_chats() -> list:
    """Return the list of registered chats (may be empty)."""
    if not _CHATS_FILE.exists():
        return []
    try:
        with open(_CHATS_FILE) as f:
            return json.load(f).get("chats", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_chats(chats: list) -> None:
    """Overwrite chats.json with the given list."""
    _CHATS_FILE.write_text(json.dumps({"chats": chats}, indent=2))


def add_chat(
    chat_id: str,
    name: str,
    agent_script: str,
    tmux_session: str,
    checkpoint_file: str,
) -> None:
    """Append a new chat entry to chats.json."""
    chats = load_chats()
    chats.append(
        {
            "chat_id": chat_id,
            "name": name,
            "agent_script": agent_script,
            "tmux_session": tmux_session,
            "checkpoint_file": checkpoint_file,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_chats(chats)


def remove_chat(chat_id: str) -> None:
    """Remove a chat entry from chats.json."""
    chats = [c for c in load_chats() if c["chat_id"] != chat_id]
    save_chats(chats)


def get_chat(chat_id: str) -> dict:
    """Return the chat entry for *chat_id*, or {} if not found."""
    for c in load_chats():
        if c["chat_id"] == chat_id:
            return c
    return {}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(chat_id: str) -> Path:
    """Return the Path for a chat's checkpoint file (parent dir is created)."""
    _CHECKPOINTS_DIR.mkdir(exist_ok=True)
    return _CHECKPOINTS_DIR / f"{chat_id}_checkpoint.json"


def read_checkpoint(checkpoint_file: str) -> dict:
    """
    Return the checkpoint dict, or {} on any error.

    Checkpoint schema
    -----------------
    chat_id            : str
    name               : str  (human-readable label)
    status             : "active" | "waiting" | "completed" | "error"
    user_request       : str  (original user goal)
    n_pending          : int  (HPC jobs outstanding; 0 when not waiting)
    best_y             : float | null
    best_x             : dict | null
    latest_run_id      : str | null
    experiment_ids     : list[str]
    conversation_history: list  (same format as AgentState.conversation_history)
    graph_state        : dict | null  (serialised AgentState for possible resume)
    updated_at         : ISO-8601 timestamp
    """
    try:
        with open(checkpoint_file) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def write_checkpoint(checkpoint_file: str, data: dict) -> None:
    """Write *data* to *checkpoint_file*, creating parent dirs as needed."""
    path = Path(checkpoint_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        print(f"[chat_registry] Warning: could not write checkpoint: {exc}")


def delete_checkpoint(checkpoint_file: str) -> None:
    """Delete a checkpoint file, ignoring errors if it does not exist."""
    try:
        os.remove(checkpoint_file)
    except OSError:
        pass
