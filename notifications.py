from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
NOTIFICATIONS_FILE = BASE_DIR / "notificacoes.json"
NOTIFICATIONS_LOCK = threading.Lock()


def _ensure_file_exists() -> None:
    if NOTIFICATIONS_FILE.exists():
        return
    try:
        NOTIFICATIONS_FILE.write_text("{}\n", encoding="utf-8")
    except OSError:
        pass


def _read_states() -> dict[str, dict]:
    _ensure_file_exists()
    try:
        raw = json.loads(NOTIFICATIONS_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def _write_states(states: dict[str, dict]) -> None:
    try:
        NOTIFICATIONS_FILE.write_text(json.dumps(states, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _normalize_seen_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _empty_state() -> dict[str, object]:
    return {"seen_ids": [], "last_cleared_iso": ""}


def get_user_state(user_id: str) -> dict[str, object]:
    user_key = str(user_id or "").strip()
    if not user_key:
        return _empty_state()

    with NOTIFICATIONS_LOCK:
        states = _read_states()
        raw = states.get(user_key)
        if not isinstance(raw, dict):
            return _empty_state()
        return {
            "seen_ids": _normalize_seen_list(raw.get("seen_ids")),
            "last_cleared_iso": str(raw.get("last_cleared_iso") or "").strip(),
        }


def update_user_state(
    user_id: str,
    *,
    seen_ids: Iterable[str] | None = None,
    last_cleared_iso: str | None = None,
) -> dict[str, object]:
    user_key = str(user_id or "").strip()
    if not user_key:
        return _empty_state()

    with NOTIFICATIONS_LOCK:
        states = _read_states()
        raw = states.get(user_key) if isinstance(states.get(user_key), dict) else {}
        current_seen = set(_normalize_seen_list(raw.get("seen_ids")))
        if seen_ids is not None:
            for entry in seen_ids:
                normalized = str(entry or "").strip()
                if normalized:
                    current_seen.add(normalized)
        last_value = str(raw.get("last_cleared_iso") or "").strip()
        if last_cleared_iso is not None:
            last_value = str(last_cleared_iso or "").strip()
        next_state: dict[str, object] = {
            "seen_ids": sorted(current_seen),
            "last_cleared_iso": last_value,
        }
        states[user_key] = next_state
        _write_states(states)
        return next_state
