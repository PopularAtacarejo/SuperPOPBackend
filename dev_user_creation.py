"""Helper utilities for the developer-only user creation workflow."""

from __future__ import annotations

import secrets
import string
from typing import Iterable

DEFAULT_DEVELOPER_USER_ROLE = "Função pendente"
DEFAULT_DEV_PASSWORD_LENGTH = 10


def generate_developer_placeholder_password(length: int = DEFAULT_DEV_PASSWORD_LENGTH) -> str:
    """Generate a short random password that will be replaced later by the employee."""

    safe_length = max(4, length)
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(safe_length))


def prepare_developer_user_payload(
    payload: dict,
    funcao_default: str = DEFAULT_DEVELOPER_USER_ROLE,
    password_length: int = DEFAULT_DEV_PASSWORD_LENGTH,
) -> dict:
    """Ensure the minimal fields required for developer-created users."""

    if not isinstance(payload, dict):
        payload = {}

    normalized = dict(payload)
    if not normalized.get("funcao"):
        normalized["funcao"] = funcao_default
    if not normalized.get("senha"):
        normalized["senha"] = generate_developer_placeholder_password(password_length)
    normalized["email"] = str(normalized.get("email", "") or "").strip()
    normalized["pre_cadastro"] = True

    tags_raw = normalized.get("tags_acesso")
    if isinstance(tags_raw, Iterable):
        normalized["tags_acesso"] = [str(item).strip() for item in tags_raw if str(item).strip()]
    else:
        normalized["tags_acesso"] = []

    return normalized


__all__ = [
    "DEFAULT_DEVELOPER_USER_ROLE",
    "generate_developer_placeholder_password",
    "prepare_developer_user_payload",
]
