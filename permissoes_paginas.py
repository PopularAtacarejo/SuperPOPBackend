from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request


page_permissions_bp = Blueprint("permissoes_paginas", __name__)

BASE_DIR = Path(__file__).resolve().parent
PERMISSIONS_FILE = BASE_DIR / "PermissoesPaginas.json"
PERMISSIONS_LOCK = threading.Lock()

TAG_DEFINITIONS = [
    {"key": "usuario", "label": "Usuário padrão", "description": "Todos os usuários autenticados."},
    {"key": "admin", "label": "Administrador", "description": "Usuários com a tag admin."},
    {"key": "developer", "label": "Desenvolvedor", "description": "Usuários com a tag developer."},
]

PAGE_DEFINITIONS = [
    {"key": "superpop", "label": "Enviar Super POP", "path": "superpop.html", "category": "Geral"},
    {"key": "meus_superpops", "label": "Meus Super POPs", "path": "meus-superpops.html", "category": "Geral"},
    {"key": "aniversariantes", "label": "Aniversariantes", "path": "aniversariantes.html", "category": "Geral"},
    {"key": "dinamicas_pop", "label": "Dinâmicas POP", "path": "dinamicas-pop.html", "category": "Geral"},
    {"key": "rank", "label": "Ranking", "path": "rank.html", "category": "Geral"},
    {"key": "ganhadores", "label": "Ganhadores", "path": "ganhadores.html", "category": "Geral"},
    {"key": "sobre", "label": "Sobre", "path": "sobre.html", "category": "Geral"},
    {"key": "atualizacoes", "label": "Atualizações", "path": "atualizacoes.html", "category": "Geral"},
    {"key": "perfil", "label": "Meu perfil", "path": "perfil.html", "category": "Conta"},
    {"key": "analise", "label": "Análise de dados", "path": "analise.html", "category": "Gestão"},
    {"key": "usuarios", "label": "Usuários cadastrados", "path": "usuarios.html", "category": "Gestão"},
    {"key": "editar_usuarios", "label": "Editar usuários", "path": "editar-usuarios.html", "category": "Gestão"},
    {"key": "criar_usuarios", "label": "Criar usuários", "path": "criar-usuarios.html", "category": "Gestão"},
    {
        "key": "atualizacoes_editor",
        "label": "Gerenciar atualizações",
        "path": "atualizacoes-editor.html",
        "category": "Gestão",
    },
    {
        "key": "permissoes_paginas",
        "label": "Permissões de páginas",
        "path": "permissoes-paginas.html",
        "category": "Gestão",
        "developer_only": True,
    },
]

PAGE_KEYS = {item["key"] for item in PAGE_DEFINITIONS}
TAG_KEYS = {item["key"] for item in TAG_DEFINITIONS}
DEVELOPER_ONLY_PAGE_KEYS = {
    item["key"] for item in PAGE_DEFINITIONS if item.get("developer_only")
}

GENERAL_PAGE_KEYS = {
    "superpop",
    "meus_superpops",
    "aniversariantes",
    "dinamicas_pop",
    "rank",
    "ganhadores",
    "sobre",
    "atualizacoes",
    "perfil",
}

DEFAULT_PERMISSIONS = {
    "usuario": sorted(GENERAL_PAGE_KEYS),
    "admin": sorted(GENERAL_PAGE_KEYS | {"analise", "usuarios"}),
    "developer": sorted(PAGE_KEYS),
}


def _normalize_permissions(payload: object) -> dict[str, list[str]]:
    source = payload if isinstance(payload, dict) else {}
    normalized: dict[str, list[str]] = {}
    for tag in TAG_KEYS:
        raw_pages = source.get(tag, [])
        if not isinstance(raw_pages, list):
            raw_pages = []
        normalized[tag] = sorted(
            {
                str(page or "").strip()
                for page in raw_pages
                if str(page or "").strip() in PAGE_KEYS
            }
        )
        if tag != "developer":
            normalized[tag] = sorted(set(normalized[tag]) - DEVELOPER_ONLY_PAGE_KEYS)

    # A tela que administra a matriz nunca pode ser retirada do desenvolvedor.
    normalized["developer"] = sorted(set(normalized["developer"]) | {"permissoes_paginas"})
    return normalized


def _write_permissions(permissions: dict[str, list[str]]) -> None:
    PERMISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = PERMISSIONS_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(permissions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(PERMISSIONS_FILE)


def load_page_permissions() -> dict[str, list[str]]:
    with PERMISSIONS_LOCK:
        if not PERMISSIONS_FILE.exists():
            normalized = _normalize_permissions(DEFAULT_PERMISSIONS)
            _write_permissions(normalized)
            return normalized
        try:
            loaded = json.loads(PERMISSIONS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _normalize_permissions(DEFAULT_PERMISSIONS)
        return _normalize_permissions(loaded)


def save_page_permissions(payload: object) -> dict[str, list[str]]:
    normalized = _normalize_permissions(payload)
    with PERMISSIONS_LOCK:
        _write_permissions(normalized)
    return normalized


def effective_page_access(access_tags: list[str] | tuple[str, ...] | set[str]) -> dict[str, bool]:
    normalized_tags = {str(tag or "").strip().lower() for tag in access_tags if str(tag or "").strip()}
    effective_tags = {"usuario"} | normalized_tags
    permissions = load_page_permissions()
    allowed_pages: set[str] = set()
    for tag in effective_tags:
        allowed_pages.update(permissions.get(tag, []))
    return {page_key: page_key in allowed_pages for page_key in sorted(PAGE_KEYS)}


def can_access_page(access_tags: list[str] | tuple[str, ...] | set[str], page_key: str) -> bool:
    return bool(effective_page_access(access_tags).get(str(page_key or "").strip()))


def first_allowed_page(access_tags: list[str] | tuple[str, ...] | set[str]) -> str:
    access = effective_page_access(access_tags)
    for page in PAGE_DEFINITIONS:
        if access.get(page["key"]):
            return str(page["path"])
    return "perfil.html"


def _developer_context():
    from app import require_developer_only_api_context

    return require_developer_only_api_context()


@page_permissions_bp.get("/api/dev/page-permissions")
def get_page_permissions():
    _context, blocked = _developer_context()
    if blocked:
        return blocked
    return jsonify(
        {
            "ok": True,
            "tags": TAG_DEFINITIONS,
            "pages": PAGE_DEFINITIONS,
            "permissions": load_page_permissions(),
        }
    )


@page_permissions_bp.put("/api/dev/page-permissions")
def update_page_permissions():
    context, blocked = _developer_context()
    if blocked:
        return blocked
    payload = request.get_json(silent=True) or {}
    permissions = save_page_permissions(payload.get("permissions"))
    actor = (context or {}).get("usuario") or {}
    return jsonify(
        {
            "ok": True,
            "permissions": permissions,
            "updated_by": {
                "id": str(actor.get("id", "")),
                "nome": str(actor.get("nome", "")),
            },
        }
    )
