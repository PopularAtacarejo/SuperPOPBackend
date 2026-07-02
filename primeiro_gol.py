from __future__ import annotations

import json
import os
import shutil
import threading
import base64
import urllib.request
import urllib.parse
import uuid
from datetime import datetime
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request


primeiro_gol_bp = Blueprint("primeiro_gol", __name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.getenv("PRIMEIRO_GOL_DATA_FILE", BASE_DIR.parent / "primeiro_gol.json"))
LEGACY_DATA_FILE = BASE_DIR / "PrimeiroGol.json"
DATA_LOCK = threading.Lock()

EMPTY_STATE = {
    "jogadores": [],
    "palpites": [],
    "inicio_palpites_iso": "",
    "fim_palpites_iso": "",
    "descricao_premio": "",
    "regras": "",
}


def _empty_state() -> dict[str, Any]:
    return {
        "jogadores": [],
        "palpites": [],
        "inicio_palpites_iso": "",
        "fim_palpites_iso": "",
        "descricao_premio": "",
        "regras": "",
    }


def _normalize_state(loaded: Any) -> dict[str, Any]:
    if not isinstance(loaded, dict):
        return _empty_state()
    players = loaded.get("jogadores")
    predictions = loaded.get("palpites")
    return {
        "jogadores": [item for item in players if isinstance(item, dict)] if isinstance(players, list) else [],
        "palpites": [item for item in predictions if isinstance(item, dict)] if isinstance(predictions, list) else [],
        "inicio_palpites_iso": str(loaded.get("inicio_palpites_iso", "")).strip(),
        "fim_palpites_iso": str(loaded.get("fim_palpites_iso", "")).strip(),
        "descricao_premio": str(loaded.get("descricao_premio", "")).strip(),
        "regras": str(loaded.get("regras", "")).strip(),
    }


def _ensure_data_file() -> None:
    if DATA_FILE.exists():
        return
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LEGACY_DATA_FILE.exists():
        shutil.copy2(LEGACY_DATA_FILE, DATA_FILE)
        return
    DATA_FILE.write_text(json.dumps(EMPTY_STATE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_state_local(state: dict[str, Any]) -> None:
    _ensure_data_file()
    temporary = DATA_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(_normalize_state(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(DATA_FILE)


def _remote_data_url() -> str:
    configured = os.getenv("PRIMEIRO_GOL_SOURCE_URL", "").strip()
    if configured:
        return configured
    repo = os.getenv("GITHUB_REPO", "PopularAtacarejo/SuperPOP").strip()
    branch = os.getenv("GITHUB_BRANCH", "main").strip()
    repo_path = os.getenv("GITHUB_PRIMEIRO_GOL_FILE_PATH", "primeiro_gol.json").strip()
    if not repo or not branch or not repo_path:
        return ""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{repo_path}"


def _read_remote_state() -> dict[str, Any]:
    if os.getenv("PRIMEIRO_GOL_REMOTE_READ_ENABLED", "1").strip().lower() in {"0", "false", "no", "nao"}:
        return _empty_state()
    source_url = _remote_data_url()
    if not source_url:
        return _empty_state()
    try:
        request_obj = urllib.request.Request(
            source_url,
            headers={"User-Agent": "superpop-primeiro-gol"},
            method="GET",
        )
        with urllib.request.urlopen(request_obj, timeout=12) as response:
            loaded = json.loads(response.read().decode("utf-8"))
    except Exception:
        return _empty_state()
    return _normalize_state(loaded)


def _read_state() -> dict[str, Any]:
    _ensure_data_file()
    try:
        state = _normalize_state(json.loads(DATA_FILE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        state = _empty_state()
    if state["jogadores"] or state["palpites"]:
        return state
    remote_state = _read_remote_state()
    if remote_state["jogadores"] or remote_state["palpites"]:
        _write_state_local(remote_state)
        return remote_state
    return state


def _sync_state_to_github(state: dict[str, Any]) -> dict[str, Any]:
    try:
        from app import get_env
    except Exception as exc:  # pragma: no cover - best effort when imported outside Flask
        return {"synced": False, "reason": f"Sync GitHub indisponivel: {exc}"}

    token = get_env("GITHUB_TOKEN")
    if not token:
        return {"synced": False, "reason": "GITHUB_TOKEN nao configurado"}

    repo_path = get_env("GITHUB_PRIMEIRO_GOL_FILE_PATH", "primeiro_gol.json")
    if not repo_path:
        return {"synced": False, "reason": "Caminho do arquivo GitHub nao configurado"}

    repo = get_env("GITHUB_REPO", "PopularAtacarejo/SuperPOP")
    branch = get_env("GITHUB_BRANCH", "main")
    api_base = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    get_url = f"{api_base}?ref={urllib.parse.quote(branch)}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "superpop-primeiro-gol",
    }

    sha = None
    try:
        req_get = urllib.request.Request(get_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_get, timeout=20) as response:
            current = json.loads(response.read().decode("utf-8"))
            sha = current.get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return {"synced": False, "reason": f"GitHub GET falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub GET erro: {exc}"}

    normalized_state = _normalize_state(state)
    content = base64.b64encode(
        json.dumps(normalized_state, ensure_ascii=False, indent=2).encode("utf-8")
    ).decode("utf-8")
    utc_now = datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    payload = {
        "message": f"Atualiza {repo_path} ({utc_now})",
        "content": content,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        req_put = urllib.request.Request(
            api_base,
            data=json.dumps(payload).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req_put, timeout=30):
            return {
                "synced": True,
                "reason": "ok",
                "players_count": len(normalized_state["jogadores"]),
                "predictions_count": len(normalized_state["palpites"]),
            }
    except urllib.error.HTTPError as exc:
        return {"synced": False, "reason": f"GitHub PUT falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub PUT erro: {exc}"}


def _write_state(state: dict[str, Any]) -> dict[str, Any]:
    _write_state_local(state)
    return _sync_state_to_github(state)


def _auth_context() -> dict[str, Any] | None:
    from app import get_authenticated_user_context

    return get_authenticated_user_context()


def _now() -> datetime:
    from app import now_brazil

    return now_brazil()


def _now_iso() -> str:
    return _now().isoformat()


def _is_developer(context: dict[str, Any]) -> bool:
    permissions = context.get("permissoes") or {}
    return bool(permissions.get("edit_users"))


def _clean_text(value: object, maximum: int = 80) -> str:
    return " ".join(str(value or "").strip().split())[:maximum]


def _clean_multiline_text(value: object, maximum: int = 700) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [" ".join(line.strip().split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)[:maximum]


def _clean_image_url(value: object) -> tuple[str, str]:
    image_url = str(value or "").strip()
    if not image_url:
        return "", ""
    if len(image_url) > 1000:
        return "", "A URL da foto do jogador e muito longa."
    if not image_url.lower().startswith(("https://", "http://", "data:image/")):
        return "", "A foto do jogador precisa usar uma URL HTTP, HTTPS ou data:image."
    return image_url, ""


def _parse_datetime(date_value: object, time_value: object, label: str) -> tuple[datetime | None, str]:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip()
    if not date_text or not time_text:
        return None, f"Informe a data e o horario de {label}."
    try:
        parsed = datetime.fromisoformat(f"{date_text}T{time_text}")
    except ValueError:
        return None, f"Informe uma data e um horario validos para {label}."
    return parsed.replace(tzinfo=_now().tzinfo), ""


def _parse_state_datetime(state: dict[str, Any], key: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(state.get(key, "")).strip())
    except ValueError:
        return None


def _prediction_status(state: dict[str, Any]) -> str:
    start = _parse_state_datetime(state, "inicio_palpites_iso")
    end = _parse_state_datetime(state, "fim_palpites_iso")
    if not start or not end:
        return "nao_configurado"
    now = _now()
    if now < start:
        return "aguardando"
    if now >= end:
        return "encerrado"
    return "aberto"


def _normalize_period_payload(payload: dict[str, Any]) -> tuple[dict[str, str] | None, str]:
    start, start_error = _parse_datetime(
        payload.get("data_inicio_palpites"),
        payload.get("horario_inicio_palpites"),
        "inicio dos palpites",
    )
    end, end_error = _parse_datetime(
        payload.get("data_fim_palpites"),
        payload.get("horario_fim_palpites"),
        "encerramento dos palpites",
    )
    if start_error:
        return None, start_error
    if end_error:
        return None, end_error
    if not start or not end or end <= start:
        return None, "O fim dos palpites precisa ser posterior ao inicio."
    return {
        "inicio_palpites_iso": start.isoformat(),
        "fim_palpites_iso": end.isoformat(),
        "descricao_premio": _clean_multiline_text(payload.get("descricao_premio"), 500),
        "regras": _clean_multiline_text(payload.get("regras"), 900),
    }, ""


def _normalize_player_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    name = _clean_text(payload.get("nome"), 100)
    photo_url, photo_error = _clean_image_url(payload.get("foto_url"))
    if len(name) < 2:
        return None, "Informe o nome do jogador."
    if photo_error:
        return None, photo_error
    return {"nome": name, "foto_url": photo_url}, ""


def _find_player(players: list[dict[str, Any]], player_id: str) -> tuple[int, dict[str, Any] | None]:
    wanted = str(player_id or "").strip()
    for index, player in enumerate(players):
        if str(player.get("id", "")).strip() == wanted:
            return index, player
    return -1, None


def _find_prediction(predictions: list[dict[str, Any]], prediction_id: str) -> tuple[int, dict[str, Any] | None]:
    wanted = str(prediction_id or "").strip()
    for index, prediction in enumerate(predictions):
        if str(prediction.get("id", "")).strip() == wanted:
            return index, prediction
    return -1, None


def _players_map(players: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(player.get("id", "")).strip(): player for player in players if isinstance(player, dict)}


def _public_player(player: dict[str, Any], vote_counts: dict[str, int] | None = None, show_counts: bool = False) -> dict[str, Any]:
    player_id = str(player.get("id", "")).strip()
    return {
        "id": player_id,
        "nome": str(player.get("nome", "")),
        "foto_url": str(player.get("foto_url", "")),
        "created_at_iso": str(player.get("created_at_iso", "")),
        "updated_at_iso": str(player.get("updated_at_iso", "")),
        "total_palpites": int((vote_counts or {}).get(player_id, 0)) if show_counts else 0,
    }


def _public_prediction(
    prediction: dict[str, Any],
    players_map: dict[str, dict[str, Any]],
    employees_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    user_id = str(prediction.get("usuario_id", "")).strip()
    player_id = str(prediction.get("jogador_id", "")).strip()
    employee = employees_map.get(user_id) or {}
    player = players_map.get(player_id) or {}
    return {
        "id": str(prediction.get("id", "")),
        "usuario_id": user_id,
        "usuario_nome": str(employee.get("nome", "") or prediction.get("usuario_nome", "")),
        "usuario_foto": str(employee.get("foto_perfil_data_url", "") or prediction.get("usuario_foto", "")),
        "usuario_funcao": str(employee.get("funcao", "") or prediction.get("usuario_funcao", "")),
        "jogador_id": player_id,
        "jogador_nome": str(player.get("nome", "") or prediction.get("jogador_nome", "")),
        "jogador_foto": str(player.get("foto_url", "") or prediction.get("jogador_foto", "")),
        "enviado_em_iso": str(prediction.get("enviado_em_iso", "")),
    }


def _vote_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for prediction in predictions:
        player_id = str(prediction.get("jogador_id", "")).strip()
        if player_id:
            counts[player_id] = counts.get(player_id, 0) + 1
    return counts


def _public_state(state: dict[str, Any], user_id: str, developer: bool) -> dict[str, Any]:
    from app import read_employees

    players = state.get("jogadores")
    predictions = state.get("palpites")
    if not isinstance(players, list):
        players = []
    if not isinstance(predictions, list):
        predictions = []

    player_map = _players_map(players)
    employees_map = {str(emp.get("id", "")).strip(): emp for emp in read_employees() if isinstance(emp, dict)}
    own_prediction = next(
        (
            prediction
            for prediction in predictions
            if isinstance(prediction, dict) and str(prediction.get("usuario_id", "")).strip() == user_id
        ),
        None,
    )
    public_predictions = [
        _public_prediction(prediction, player_map, employees_map)
        for prediction in predictions
        if isinstance(prediction, dict)
    ]
    public_predictions.sort(key=lambda item: str(item.get("enviado_em_iso", "")), reverse=True)
    status = _prediction_status(state)
    revealed = developer or status == "encerrado"
    counts = _vote_counts(predictions) if revealed else {}
    return {
        "ok": True,
        "is_developer": developer,
        "jogadores": [_public_player(player, counts, revealed) for player in players if isinstance(player, dict)],
        "inicio_palpites_iso": str(state.get("inicio_palpites_iso", "")),
        "fim_palpites_iso": str(state.get("fim_palpites_iso", "")),
        "descricao_premio": str(state.get("descricao_premio", "")),
        "regras": str(state.get("regras", "")),
        "status_palpites": status,
        "palpite_aberto": status == "aberto",
        "escolhas_reveladas": revealed,
        "meu_palpite": _public_prediction(own_prediction, player_map, employees_map) if isinstance(own_prediction, dict) else None,
        "ja_enviou_palpite": isinstance(own_prediction, dict),
        "total_palpites": len(public_predictions) if revealed else 0,
        "palpites_enviados": public_predictions if revealed else [],
    }


def _send_first_goal_choice_email(prediction: dict[str, Any]) -> dict[str, Any]:
    user_id = str(prediction.get("usuario_id", "")).strip()
    if not user_id:
        return {"sent": False, "reason": "Palpite sem usuario vinculado."}

    try:
        from app import get_smtp_settings, mask_email_for_log, read_employees, send_email_with_fallback
    except Exception as exc:
        return {"sent": False, "reason": f"Envio de email indisponivel: {exc}"}

    employees = read_employees()
    employee = next(
        (
            item for item in employees
            if isinstance(item, dict) and str(item.get("id", "")).strip() == user_id
        ),
        {},
    )
    recipient_name = str(employee.get("nome", "") or prediction.get("usuario_nome", "")).strip() or "colaborador"
    recipient_email = str(employee.get("email", "")).strip().lower()
    if not recipient_email:
        return {"sent": False, "reason": "Colaborador sem email cadastrado."}

    smtp = get_smtp_settings()
    from_email = str(smtp.get("from_email", "")).strip()
    if not from_email:
        return {"sent": False, "reason": "Email remetente nao configurado."}

    player_name = str(prediction.get("jogador_nome", "")).strip() or "jogador selecionado"
    sent_at = str(prediction.get("enviado_em_iso", "")).strip()

    message = EmailMessage()
    from_name = str(smtp.get("from_name", "") or "SuperPop").strip()
    message["Subject"] = "Seu palpite do primeiro gol foi registrado"
    message["From"] = f"{from_name} <{from_email}>" if from_name else from_email
    message["To"] = recipient_email

    text_lines = [
        f"Ola, {recipient_name}.",
        "",
        "Seu palpite para quem fara o primeiro gol do Brasil foi registrado com sucesso.",
        f"Jogador escolhido: {player_name}",
    ]
    if sent_at:
        text_lines.append(f"Enviado em: {sent_at}")
    text_lines.extend(["", "Boa sorte!", "Equipe SuperPop"])
    text_content = "\n".join(text_lines)

    html_parts = [
        "<!DOCTYPE html><html><body style=\"font-family:Arial,sans-serif;color:#0f172a;line-height:1.5\">",
        f"<p>Ola, <strong>{escape(recipient_name)}</strong>.</p>",
        "<p>Seu palpite para quem fara o primeiro gol do Brasil foi registrado com sucesso.</p>",
        "<div style=\"padding:14px 16px;border:1px solid #e2e8f0;border-radius:12px;background:#f8fafc\">",
        f"<p style=\"margin:0\"><strong>Jogador escolhido:</strong> {escape(player_name)}</p>",
    ]
    if sent_at:
        html_parts.append(f"<p style=\"margin:8px 0 0\"><strong>Enviado em:</strong> {escape(sent_at)}</p>")
    html_parts.extend([
        "</div>",
        "<p>Boa sorte!<br>Equipe SuperPop</p>",
        "</body></html>",
    ])
    html_content = "".join(html_parts)

    message.set_content(text_content)
    message.add_alternative(html_content, subtype="html")
    sent, status = send_email_with_fallback(message, html_content, text_content)
    return {
        "sent": sent,
        "status": status,
        "to": mask_email_for_log(recipient_email),
    }


@primeiro_gol_bp.get("/api/primeiro-gol")
def get_first_goal_state():
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    user = context.get("usuario") or {}
    user_id = str(user.get("id", "")).strip()
    developer = _is_developer(context)
    with DATA_LOCK:
        state = _read_state()
    return jsonify(_public_state(state, user_id, developer))


@primeiro_gol_bp.put("/api/primeiro-gol/periodo")
def update_first_goal_period():
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem definir o periodo."}), 403

    normalized, error = _normalize_period_payload(request.get_json(silent=True) or {})
    if not normalized:
        return jsonify({"ok": False, "error": error}), 400

    with DATA_LOCK:
        state = _read_state()
        state.update(normalized)
        github_sync = _write_state(state)

    user = context.get("usuario") or {}
    return jsonify({
        **_public_state(state, str(user.get("id", "")).strip(), True),
        "github_sync": github_sync,
    })


@primeiro_gol_bp.post("/api/primeiro-gol/jogadores")
def create_player():
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem cadastrar jogadores."}), 403

    normalized, error = _normalize_player_payload(request.get_json(silent=True) or {})
    if not normalized:
        return jsonify({"ok": False, "error": error}), 400

    player = {
        "id": uuid.uuid4().hex,
        **normalized,
        "created_at_iso": _now_iso(),
    }
    with DATA_LOCK:
        state = _read_state()
        players = state.get("jogadores")
        if not isinstance(players, list):
            players = []
        players.append(player)
        state["jogadores"] = players
        github_sync = _write_state(state)

    return jsonify({"ok": True, "jogador": _public_player(player), "github_sync": github_sync}), 201


@primeiro_gol_bp.put("/api/primeiro-gol/jogadores/<player_id>")
def update_player(player_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem editar jogadores."}), 403

    normalized, error = _normalize_player_payload(request.get_json(silent=True) or {})
    if not normalized:
        return jsonify({"ok": False, "error": error}), 400

    with DATA_LOCK:
        state = _read_state()
        players = state.get("jogadores")
        if not isinstance(players, list):
            players = []
        index, player = _find_player(players, player_id)
        if not player:
            return jsonify({"ok": False, "error": "Jogador nao encontrado."}), 404
        player.update(normalized)
        player["updated_at_iso"] = _now_iso()
        players[index] = player
        state["jogadores"] = players
        github_sync = _write_state(state)

    return jsonify({"ok": True, "jogador": _public_player(player), "github_sync": github_sync})


@primeiro_gol_bp.delete("/api/primeiro-gol/jogadores/<player_id>")
def delete_player(player_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem excluir jogadores."}), 403

    with DATA_LOCK:
        state = _read_state()
        players = state.get("jogadores")
        predictions = state.get("palpites")
        if not isinstance(players, list):
            players = []
        if not isinstance(predictions, list):
            predictions = []
        index, player = _find_player(players, player_id)
        if not player:
            return jsonify({"ok": False, "error": "Jogador nao encontrado."}), 404
        players.pop(index)
        state["jogadores"] = players
        state["palpites"] = [
            prediction
            for prediction in predictions
            if isinstance(prediction, dict) and str(prediction.get("jogador_id", "")).strip() != str(player_id).strip()
        ]
        github_sync = _write_state(state)

    return jsonify({"ok": True, "github_sync": github_sync})


@primeiro_gol_bp.post("/api/primeiro-gol/palpite")
def save_first_goal_prediction():
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    payload = request.get_json(silent=True) or {}
    player_id = str(payload.get("jogador_id", "")).strip()
    if not player_id:
        return jsonify({"ok": False, "error": "Selecione um jogador."}), 400

    user = context.get("usuario") or {}
    user_id = str(user.get("id", "")).strip()
    if not user_id:
        return jsonify({"ok": False, "error": "Usuario sem identificacao valida."}), 400

    with DATA_LOCK:
        state = _read_state()
        prediction_status = _prediction_status(state)
        if prediction_status == "nao_configurado":
            return jsonify({"ok": False, "error": "O periodo para escolhas ainda nao foi configurado."}), 409
        if prediction_status == "aguardando":
            return jsonify({"ok": False, "error": "O periodo para escolhas ainda nao iniciou."}), 409
        if prediction_status == "encerrado":
            return jsonify({"ok": False, "error": "O periodo para escolhas ja foi encerrado."}), 409
        players = state.get("jogadores")
        predictions = state.get("palpites")
        if not isinstance(players, list):
            players = []
        if not isinstance(predictions, list):
            predictions = []
        _player_index, player = _find_player(players, player_id)
        if not player:
            return jsonify({"ok": False, "error": "Jogador nao encontrado."}), 404
        if any(isinstance(item, dict) and str(item.get("usuario_id", "")).strip() == user_id for item in predictions):
            return jsonify({"ok": False, "error": "Voce ja escolheu um jogador para o primeiro gol."}), 409

        prediction = {
            "id": uuid.uuid4().hex,
            "usuario_id": user_id,
            "usuario_nome": str(user.get("nome", "")).strip(),
            "usuario_foto": str(user.get("foto_perfil_data_url", "")).strip(),
            "usuario_funcao": str(user.get("funcao", "")).strip(),
            "jogador_id": str(player.get("id", "")).strip(),
            "jogador_nome": str(player.get("nome", "")).strip(),
            "jogador_foto": str(player.get("foto_url", "")).strip(),
            "enviado_em_iso": _now_iso(),
        }
        predictions.append(prediction)
        state["palpites"] = predictions
        github_sync = _write_state(state)

    choice_email = _send_first_goal_choice_email(prediction)

    return jsonify({
        "ok": True,
        "palpite": prediction,
        "github_sync": github_sync,
        "choice_email": choice_email,
    }), 201


@primeiro_gol_bp.delete("/api/primeiro-gol/palpites/<prediction_id>")
def delete_first_goal_prediction(prediction_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem excluir palpites."}), 403

    with DATA_LOCK:
        state = _read_state()
        predictions = state.get("palpites")
        if not isinstance(predictions, list):
            predictions = []
        index, prediction = _find_prediction(predictions, prediction_id)
        if not prediction:
            return jsonify({"ok": False, "error": "Palpite nao encontrado."}), 404
        predictions.pop(index)
        state["palpites"] = predictions
        github_sync = _write_state(state)

    return jsonify({"ok": True, "github_sync": github_sync})
