from __future__ import annotations

import json
import os
import shutil
import threading
import urllib.request
import uuid
from datetime import datetime
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request


dinamicas_pop_bp = Blueprint("dinamicas_pop", __name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.getenv("DINAMICAS_POP_DATA_FILE", BASE_DIR.parent / "dinamica.json"))
LEGACY_DATA_FILE = BASE_DIR / "DinamicasPOP.json"
DATA_LOCK = threading.Lock()


def _ensure_data_file() -> None:
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if LEGACY_DATA_FILE.exists():
            shutil.copy2(LEGACY_DATA_FILE, DATA_FILE)
            return
        DATA_FILE.write_text("[]\n", encoding="utf-8")


def _write_games_local(games: list[dict[str, Any]]) -> None:
    _ensure_data_file()
    temporary = DATA_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(games, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(DATA_FILE)


def _remote_data_url() -> str:
    configured = os.getenv("DINAMICAS_POP_SOURCE_URL", "").strip()
    if configured:
        return configured
    repo = os.getenv("GITHUB_REPO", "PopularAtacarejo/SuperPOP").strip()
    branch = os.getenv("GITHUB_BRANCH", "main").strip()
    repo_path = os.getenv("GITHUB_DINAMICAS_FILE_PATH", "dinamica.json").strip()
    if not repo or not branch or not repo_path:
        return ""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{repo_path}"


def _read_remote_games() -> list[dict[str, Any]]:
    if os.getenv("DINAMICAS_POP_REMOTE_READ_ENABLED", "1").strip().lower() in {"0", "false", "no", "nao"}:
        return []
    source_url = _remote_data_url()
    if not source_url:
        return []
    try:
        request_obj = urllib.request.Request(
            source_url,
            headers={"User-Agent": "superpop-dinamicas-pop"},
            method="GET",
        )
        with urllib.request.urlopen(request_obj, timeout=12) as response:
            loaded = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def _read_games() -> list[dict[str, Any]]:
    _ensure_data_file()
    try:
        loaded = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    local_games = [item for item in loaded if isinstance(item, dict)]
    if local_games:
        return local_games
    remote_games = _read_remote_games()
    if remote_games:
        _write_games_local(remote_games)
        return remote_games
    return []


def _sync_games_to_github(games: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from app import get_env, github_sync_json_file, github_sync_with_retry
    except Exception as exc:  # pragma: no cover - best effort when imported outside Flask
        return {"synced": False, "reason": f"Sync GitHub indisponivel: {exc}"}

    repo_path = get_env("GITHUB_DINAMICAS_FILE_PATH", "dinamica.json")
    return github_sync_with_retry(
        games,
        lambda records: github_sync_json_file(records, repo_path),
    )


def _write_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    _write_games_local(games)
    return _sync_games_to_github(games)


def _auth_context() -> dict[str, Any] | None:
    from app import get_authenticated_user_context

    return get_authenticated_user_context()


def _now() -> datetime:
    from app import now_brazil

    return now_brazil()


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
        return "", "A URL da imagem do time e muito longa."
    if not image_url.lower().startswith(("https://", "http://")):
        return "", "A imagem do time precisa usar uma URL HTTP ou HTTPS."
    return image_url, ""


def _parse_match_datetime(date_value: object, time_value: object) -> tuple[datetime | None, str]:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip()
    try:
        parsed = datetime.fromisoformat(f"{date_text}T{time_text}")
    except ValueError:
        return None, "Informe uma data e um horario validos."
    return parsed.replace(tzinfo=_now().tzinfo), ""


def _parse_optional_datetime(
    date_value: object,
    time_value: object,
    label: str,
) -> tuple[datetime | None, str]:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip()
    if not date_text and not time_text:
        return None, ""
    if not date_text or not time_text:
        return None, f"Informe a data e o horario de {label}."
    parsed, error = _parse_match_datetime(date_text, time_text)
    if error:
        return None, f"Informe uma data e um horario validos para {label}."
    return parsed, ""


def _parse_score(value: object) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 99 else None


def _score_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "casa"
    if away_score > home_score:
        return "visitante"
    return "empate"


def _format_match_datetime_for_email(game: dict[str, Any]) -> str:
    iso_value = str(game.get("inicio_iso", "")).strip()
    if iso_value:
        try:
            return datetime.fromisoformat(iso_value).strftime("%d/%m/%Y as %H:%M")
        except ValueError:
            pass

    date_value = str(game.get("data_jogo", "")).strip()
    time_value = str(game.get("horario_jogo", "")).strip()
    if date_value and time_value:
        return f"{date_value} as {time_value}"
    return date_value or time_value


def _html_paragraph(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    return escape(cleaned).replace("\n", "<br>")


def _send_prediction_winner_email(game: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    winner_user_id = str(prediction.get("usuario_id", "")).strip()
    if not winner_user_id:
        return {"sent": False, "reason": "Palpite sem usuario vinculado."}

    try:
        from app import get_smtp_settings, mask_email_for_log, read_employees, send_email_with_fallback
    except Exception as exc:
        return {"sent": False, "reason": f"Envio de email indisponivel: {exc}"}

    employees = read_employees()
    employee = next(
        (
            item for item in employees
            if isinstance(item, dict) and str(item.get("id", "")).strip() == winner_user_id
        ),
        {},
    )
    winner_name = str(employee.get("nome", "") or prediction.get("usuario_nome", "")).strip() or "colaborador"
    recipient_email = str(employee.get("email", "")).strip().lower()
    if not recipient_email:
        return {"sent": False, "reason": "Ganhador sem email cadastrado."}

    smtp = get_smtp_settings()
    from_email = str(smtp.get("from_email", "")).strip()
    if not from_email:
        return {"sent": False, "reason": "Email remetente nao configurado."}

    home_team = str(game.get("time_casa", "")).strip() or "Time da casa"
    away_team = str(game.get("time_visitante", "")).strip() or "Time visitante"
    match_label = f"{home_team} x {away_team}"
    match_datetime = _format_match_datetime_for_email(game)
    competition = str(game.get("competicao", "")).strip() or "Dinamica POP"
    prize = str(game.get("descricao_premio", "")).strip()
    rules = str(game.get("regras", "")).strip()
    home_score = str(prediction.get("gols_casa", "")).strip()
    away_score = str(prediction.get("gols_visitante", "")).strip()
    score_label = f"{home_score} x {away_score}" if home_score or away_score else ""

    message = EmailMessage()
    from_name = str(smtp.get("from_name", "") or "SuperPop").strip()
    message["Subject"] = "Voce ganhou na Dinamica POP!"
    message["From"] = f"{from_name} <{from_email}>" if from_name else from_email
    message["To"] = recipient_email

    text_lines = [
        f"Ola, {winner_name}.",
        "",
        f"Seu palpite foi selecionado como ganhador na {competition}.",
        f"Jogo: {match_label}",
    ]
    if match_datetime:
        text_lines.append(f"Data do jogo: {match_datetime}")
    if score_label:
        text_lines.append(f"Seu palpite: {score_label}")
    if prize:
        text_lines.extend(["", f"Premio: {prize}"])
    if rules:
        text_lines.extend(["", f"Regras: {rules}"])
    text_lines.extend(["", "Parabens!", "Equipe SuperPop"])
    text_content = "\n".join(text_lines)

    html_parts = [
        "<!DOCTYPE html><html><body style=\"font-family:Arial,sans-serif;color:#0f172a;line-height:1.5\">",
        f"<p>Ola, <strong>{escape(winner_name)}</strong>.</p>",
        f"<p>Seu palpite foi selecionado como ganhador na <strong>{escape(competition)}</strong>.</p>",
        "<div style=\"padding:14px 16px;border:1px solid #e2e8f0;border-radius:12px;background:#f8fafc\">",
        f"<p style=\"margin:0 0 8px\"><strong>Jogo:</strong> {escape(match_label)}</p>",
    ]
    if match_datetime:
        html_parts.append(f"<p style=\"margin:0 0 8px\"><strong>Data do jogo:</strong> {escape(match_datetime)}</p>")
    if score_label:
        html_parts.append(f"<p style=\"margin:0\"><strong>Seu palpite:</strong> {escape(score_label)}</p>")
    html_parts.append("</div>")
    if prize:
        html_parts.append(f"<p><strong>Premio:</strong><br>{_html_paragraph(prize)}</p>")
    if rules:
        html_parts.append(f"<p><strong>Regras:</strong><br>{_html_paragraph(rules)}</p>")
    html_parts.append("<p>Parabens!<br>Equipe SuperPop</p></body></html>")
    html_content = "".join(html_parts)

    message.set_content(text_content)
    message.add_alternative(html_content, subtype="html")
    sent, status = send_email_with_fallback(message, html_content, text_content)
    return {
        "sent": sent,
        "status": status,
        "to": mask_email_for_log(recipient_email),
    }


def _result_scores(game: dict[str, Any]) -> tuple[int | None, int | None]:
    result = game.get("resultado")
    if not isinstance(result, dict):
        return None, None
    return _parse_score(result.get("gols_casa")), _parse_score(result.get("gols_visitante"))


def _prediction_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id", "")),
        "usuario_id": str(item.get("usuario_id", "")),
        "usuario_nome": str(item.get("usuario_nome", "")),
        "usuario_foto": str(item.get("usuario_foto", "")),
        "usuario_funcao": str(item.get("usuario_funcao", "")),
        "gols_casa": item.get("gols_casa"),
        "gols_visitante": item.get("gols_visitante"),
        "enviado_em_iso": str(item.get("enviado_em_iso", "")),
        "alterado_em_iso": str(item.get("alterado_em_iso", "")),
    }


def _selected_winner_id(game: dict[str, Any]) -> str:
    return str(game.get("palpite_ganhador_id", "")).strip()


def _find_prediction_by_id(game: dict[str, Any], prediction_id: str) -> dict[str, Any] | None:
    wanted = str(prediction_id or "").strip()
    predictions = game.get("palpites")
    if not isinstance(predictions, list):
        return None
    for item in predictions:
        if isinstance(item, dict) and str(item.get("id", "")).strip() == wanted:
            return item
    return None


def _find_prediction_index_for_delete(
    predictions: list[Any],
    prediction_id: str,
    query_args: Any,
) -> int:
    wanted = str(prediction_id or "").strip()
    if wanted and wanted != "legacy":
        for index, item in enumerate(predictions):
            if isinstance(item, dict) and str(item.get("id", "")).strip() == wanted:
                return index

    wanted_user_id = str(query_args.get("usuario_id", "")).strip()
    wanted_sent_at = str(query_args.get("enviado_em_iso", "")).strip()
    wanted_home_score = _parse_score(query_args.get("gols_casa"))
    wanted_away_score = _parse_score(query_args.get("gols_visitante"))
    if not any([wanted_user_id, wanted_sent_at, wanted_home_score is not None, wanted_away_score is not None]):
        return -1

    candidates: list[int] = []
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            continue
        if wanted_user_id and str(item.get("usuario_id", "")).strip() != wanted_user_id:
            continue
        if wanted_sent_at and str(item.get("enviado_em_iso", "")).strip() != wanted_sent_at:
            continue
        if wanted_home_score is not None and _parse_score(item.get("gols_casa")) != wanted_home_score:
            continue
        if wanted_away_score is not None and _parse_score(item.get("gols_visitante")) != wanted_away_score:
            continue
        candidates.append(index)

    return candidates[0] if len(candidates) == 1 else -1


def _recalculate_game_winners(game: dict[str, Any]) -> None:
    home_result, away_result = _result_scores(game)
    if home_result is None or away_result is None:
        game["ganhadores"] = []
        game["acertos_resultado"] = []
        return

    result_outcome = _score_outcome(home_result, away_result)
    predictions = game.get("palpites")
    if not isinstance(predictions, list):
        predictions = []

    exact_winners: list[dict[str, Any]] = []
    outcome_hits: list[dict[str, Any]] = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        home_prediction = _parse_score(item.get("gols_casa"))
        away_prediction = _parse_score(item.get("gols_visitante"))
        if home_prediction is None or away_prediction is None:
            continue
        snapshot = _prediction_snapshot(item)
        if home_prediction == home_result and away_prediction == away_result:
            exact_winners.append(snapshot)
        if _score_outcome(home_prediction, away_prediction) == result_outcome:
            outcome_hits.append(snapshot)

    game["ganhadores"] = exact_winners
    game["acertos_resultado"] = outcome_hits


def _normalize_game_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    home_team = _clean_text(payload.get("time_casa"))
    away_team = _clean_text(payload.get("time_visitante"))
    competition = _clean_text(payload.get("competicao"), 100)
    prize_description = _clean_multiline_text(payload.get("descricao_premio"), 500)
    rules = _clean_multiline_text(payload.get("regras"), 900)
    home_image, home_image_error = _clean_image_url(payload.get("imagem_time_casa"))
    away_image, away_image_error = _clean_image_url(payload.get("imagem_time_visitante"))
    match_datetime, datetime_error = _parse_match_datetime(
        payload.get("data_jogo"),
        payload.get("horario_jogo"),
    )
    prediction_start, prediction_start_error = _parse_optional_datetime(
        payload.get("data_inicio_palpites"),
        payload.get("horario_inicio_palpites"),
        "inicio dos palpites",
    )
    prediction_end, prediction_end_error = _parse_optional_datetime(
        payload.get("data_fim_palpites"),
        payload.get("horario_fim_palpites"),
        "encerramento dos palpites",
    )

    if len(home_team) < 2 or len(away_team) < 2:
        return None, "Informe os dois times do jogo."
    if home_team.casefold() == away_team.casefold():
        return None, "Os times do jogo precisam ser diferentes."
    if home_image_error:
        return None, home_image_error
    if away_image_error:
        return None, away_image_error
    if datetime_error or not match_datetime:
        return None, datetime_error
    if prediction_start_error:
        return None, prediction_start_error
    if prediction_end_error:
        return None, prediction_end_error

    prediction_start = prediction_start or _now()
    prediction_end = prediction_end or match_datetime
    if prediction_end <= prediction_start:
        return None, "O fim dos palpites precisa ser posterior ao inicio."
    if prediction_end > match_datetime:
        return None, "O fim dos palpites nao pode ser posterior ao inicio do jogo."

    return {
        "time_casa": home_team,
        "time_visitante": away_team,
        "imagem_time_casa": home_image,
        "imagem_time_visitante": away_image,
        "competicao": competition,
        "descricao_premio": prize_description,
        "regras": rules,
        "data_jogo": match_datetime.strftime("%Y-%m-%d"),
        "horario_jogo": match_datetime.strftime("%H:%M"),
        "inicio_iso": match_datetime.isoformat(),
        "inicio_palpites_iso": prediction_start.isoformat(),
        "fim_palpites_iso": prediction_end.isoformat(),
    }, ""


def _find_game(games: list[dict[str, Any]], game_id: str) -> tuple[int, dict[str, Any] | None]:
    wanted = str(game_id or "").strip()
    for index, game in enumerate(games):
        if str(game.get("id", "")).strip() == wanted:
            return index, game
    return -1, None


def _parse_game_datetime(game: dict[str, Any], key: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(game.get(key, "")))
    except ValueError:
        return None


def _prediction_period(game: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    match_start = _parse_game_datetime(game, "inicio_iso")
    prediction_start = _parse_game_datetime(game, "inicio_palpites_iso")
    prediction_end = _parse_game_datetime(game, "fim_palpites_iso")
    return prediction_start, prediction_end or match_start


def _prediction_status(game: dict[str, Any]) -> str:
    prediction_start, prediction_end = _prediction_period(game)
    now = _now()
    if prediction_start and now < prediction_start:
        return "aguardando"
    if not prediction_end or now >= prediction_end:
        return "encerrado"
    return "aberto"


def _public_prediction(item: dict[str, Any], employees_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    user_id = str(item.get("usuario_id", ""))
    employee = employees_map.get(user_id) or {}
    user_photo = str(employee.get("foto_perfil_data_url", "") or item.get("usuario_foto", ""))
    user_role = str(employee.get("funcao", "") or item.get("usuario_funcao", ""))
    return {
        "id": str(item.get("id", "")),
        "usuario_id": user_id,
        "usuario_nome": str(item.get("usuario_nome", "")),
        "usuario_foto": user_photo,
        "usuario_funcao": user_role,
        "gols_casa": item.get("gols_casa"),
        "gols_visitante": item.get("gols_visitante"),
        "enviado_em_iso": str(item.get("enviado_em_iso", "")),
        "alterado_em_iso": str(item.get("alterado_em_iso", "")),
    }


def _public_selected_winner(game: dict[str, Any], employees_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    selected = _find_prediction_by_id(game, _selected_winner_id(game))
    if not selected:
        return None
    public = _public_prediction(selected, employees_map)
    public["selecionado_em_iso"] = str(game.get("palpite_ganhador_selecionado_em_iso", ""))
    selected_by = game.get("palpite_ganhador_selecionado_por")
    public["selecionado_por"] = selected_by if isinstance(selected_by, dict) else {}
    return public


def _public_result(game: dict[str, Any], employees_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = game.get("resultado")
    selected_winner = _public_selected_winner(game, employees_map)
    if not isinstance(result, dict):
        return {
            "definido": False,
            "gols_casa": None,
            "gols_visitante": None,
            "definido_em_iso": "",
            "definido_por": {},
            "ganhadores": [],
            "acertos_resultado": [],
            "total_ganhadores": 0,
            "total_acertos_resultado": 0,
            "ganhador_selecionado": selected_winner,
        }

    winners = game.get("ganhadores")
    if not isinstance(winners, list):
        winners = []
    outcome_hits = game.get("acertos_resultado")
    if not isinstance(outcome_hits, list):
        outcome_hits = []

    return {
        "definido": True,
        "gols_casa": result.get("gols_casa"),
        "gols_visitante": result.get("gols_visitante"),
        "definido_em_iso": str(result.get("definido_em_iso", "")),
        "definido_por": result.get("definido_por") if isinstance(result.get("definido_por"), dict) else {},
        "ganhadores": [
            _public_prediction(item, employees_map) for item in winners if isinstance(item, dict)
        ],
        "acertos_resultado": [
            _public_prediction(item, employees_map) for item in outcome_hits if isinstance(item, dict)
        ],
        "total_ganhadores": len([item for item in winners if isinstance(item, dict)]),
        "total_acertos_resultado": len([item for item in outcome_hits if isinstance(item, dict)]),
        "ganhador_selecionado": selected_winner,
    }


def _public_game(game: dict[str, Any], user_id: str, developer: bool, employees_map: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    if employees_map is None:
        from app import read_employees
        employees_map = {str(emp.get("id", "")).strip(): emp for emp in read_employees() if isinstance(emp, dict)}
    predictions = game.get("palpites")
    if not isinstance(predictions, list):
        predictions = []

    own_prediction = next(
        (
            item
            for item in predictions
            if isinstance(item, dict) and str(item.get("usuario_id", "")) == user_id
        ),
        None,
    )
    prediction_start, prediction_end = _prediction_period(game)
    prediction_status = _prediction_status(game)
    selected_winner_id = _selected_winner_id(game)
    public = {
        "id": str(game.get("id", "")),
        "time_casa": str(game.get("time_casa", "")),
        "time_visitante": str(game.get("time_visitante", "")),
        "imagem_time_casa": str(game.get("imagem_time_casa", "")),
        "imagem_time_visitante": str(game.get("imagem_time_visitante", "")),
        "competicao": str(game.get("competicao", "")),
        "descricao_premio": str(game.get("descricao_premio", "")),
        "regras": str(game.get("regras", "")),
        "data_jogo": str(game.get("data_jogo", "")),
        "horario_jogo": str(game.get("horario_jogo", "")),
        "inicio_iso": str(game.get("inicio_iso", "")),
        "inicio_palpites_iso": prediction_start.isoformat() if prediction_start else "",
        "fim_palpites_iso": prediction_end.isoformat() if prediction_end else "",
        "created_at_iso": str(game.get("created_at_iso", "")),
        "palpite_aberto": prediction_status == "aberto",
        "status_palpites": prediction_status,
        "total_palpites": len(predictions),
        "meu_palpite": own_prediction,
        "ja_enviou_palpite": bool(own_prediction),
        "palpite_ganhador_id": selected_winner_id,
        "resultado": _public_result(game, employees_map),
        "palpites_enviados": [
            {
                **_public_prediction(item, employees_map),
                "ganhador_selecionado": str(item.get("id", "")) == selected_winner_id,
            }
            for item in predictions if isinstance(item, dict)
        ],
    }
    if developer:
        public["palpites"] = [
            {
                **_public_prediction(item, employees_map),
                "ganhador_selecionado": str(item.get("id", "")) == selected_winner_id,
            }
            for item in predictions if isinstance(item, dict)
        ]
    return public


@dinamicas_pop_bp.get("/api/dinamicas-pop/jogos")
def list_games():
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    user = context.get("usuario") or {}
    user_id = str(user.get("id", "")).strip()
    developer = _is_developer(context)
    with DATA_LOCK:
        games = _read_games()

    games.sort(key=lambda item: str(item.get("inicio_iso", "")))
    
    from app import read_employees
    employees_map = {str(emp.get("id", "")).strip(): emp for emp in read_employees() if isinstance(emp, dict)}
    
    return jsonify(
        {
            "ok": True,
            "is_developer": developer,
            "jogos": [_public_game(game, user_id, developer, employees_map) for game in games],
        }
    )


@dinamicas_pop_bp.post("/api/dinamicas-pop/jogos")
def create_game():
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem cadastrar jogos."}), 403

    normalized, error = _normalize_game_payload(request.get_json(silent=True) or {})
    if not normalized:
        return jsonify({"ok": False, "error": error}), 400

    user = context.get("usuario") or {}
    game = {
        "id": uuid.uuid4().hex,
        **normalized,
        "created_at_iso": _now().isoformat(),
        "criado_por": {
            "id": str(user.get("id", "")),
            "nome": str(user.get("nome", "")),
        },
        "palpites": [],
    }
    with DATA_LOCK:
        games = _read_games()
        games.append(game)
        github_sync = _write_games(games)

    return jsonify({"ok": True, "jogo": _public_game(game, str(user.get("id", "")), True), "github_sync": github_sync}), 201


@dinamicas_pop_bp.put("/api/dinamicas-pop/jogos/<game_id>")
def update_game(game_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem editar jogos."}), 403

    normalized, error = _normalize_game_payload(request.get_json(silent=True) or {})
    if not normalized:
        return jsonify({"ok": False, "error": error}), 400

    with DATA_LOCK:
        games = _read_games()
        index, game = _find_game(games, game_id)
        if not game:
            return jsonify({"ok": False, "error": "Jogo nao encontrado."}), 404
        game.update(normalized)
        game["updated_at_iso"] = _now().isoformat()
        games[index] = game
        github_sync = _write_games(games)

    user_id = str((context.get("usuario") or {}).get("id", ""))
    return jsonify({"ok": True, "jogo": _public_game(game, user_id, True), "github_sync": github_sync})


@dinamicas_pop_bp.delete("/api/dinamicas-pop/jogos/<game_id>")
def delete_game(game_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem excluir jogos."}), 403

    with DATA_LOCK:
        games = _read_games()
        index, game = _find_game(games, game_id)
        if not game:
            return jsonify({"ok": False, "error": "Jogo nao encontrado."}), 404
        games.pop(index)
        github_sync = _write_games(games)
    return jsonify({"ok": True, "github_sync": github_sync})


@dinamicas_pop_bp.post("/api/dinamicas-pop/jogos/<game_id>/palpite")
def save_prediction(game_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    payload = request.get_json(silent=True) or {}
    home_score = _parse_score(payload.get("gols_casa"))
    away_score = _parse_score(payload.get("gols_visitante"))
    if home_score is None or away_score is None:
        return jsonify({"ok": False, "error": "Informe placares entre 0 e 99."}), 400

    user = context.get("usuario") or {}
    user_id = str(user.get("id", "")).strip()
    if not user_id:
        return jsonify({"ok": False, "error": "Usuario sem identificacao valida."}), 400

    with DATA_LOCK:
        games = _read_games()
        index, game = _find_game(games, game_id)
        if not game:
            return jsonify({"ok": False, "error": "Jogo nao encontrado."}), 404
        prediction_status = _prediction_status(game)
        if prediction_status == "aguardando":
            return jsonify({"ok": False, "error": "O periodo de palpites deste jogo ainda nao iniciou."}), 409
        if prediction_status == "encerrado":
            return jsonify({"ok": False, "error": "Os palpites deste jogo ja foram encerrados."}), 409

        predictions = game.get("palpites")
        if not isinstance(predictions, list):
            predictions = []
        if any(
            isinstance(item, dict) and str(item.get("usuario_id", "")) == user_id
            for item in predictions
        ):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Voce ja enviou seu palpite para este jogo. O palpite nao pode ser alterado.",
                    }
                ),
                409,
            )
        prediction = {
            "id": uuid.uuid4().hex,
            "usuario_id": user_id,
            "usuario_nome": str(user.get("nome", "")).strip(),
            "usuario_foto": str(user.get("foto_perfil_data_url", "")).strip(),
            "usuario_funcao": str(user.get("funcao", "")).strip(),
            "gols_casa": home_score,
            "gols_visitante": away_score,
            "enviado_em_iso": _now().isoformat(),
        }
        predictions.append(prediction)
        game["palpites"] = predictions
        _recalculate_game_winners(game)
        games[index] = game
        github_sync = _write_games(games)

    return jsonify({"ok": True, "palpite": prediction, "github_sync": github_sync})


@dinamicas_pop_bp.put("/api/dinamicas-pop/jogos/<game_id>/resultado")
def save_result(game_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem informar o resultado."}), 403

    payload = request.get_json(silent=True) or {}
    home_score = _parse_score(payload.get("gols_casa"))
    away_score = _parse_score(payload.get("gols_visitante"))
    if home_score is None or away_score is None:
        return jsonify({"ok": False, "error": "Informe o placar final entre 0 e 99."}), 400

    user = context.get("usuario") or {}
    user_id = str(user.get("id", "")).strip()
    with DATA_LOCK:
        games = _read_games()
        game_index, game = _find_game(games, game_id)
        if not game:
            return jsonify({"ok": False, "error": "Jogo nao encontrado."}), 404

        game["resultado"] = {
            "gols_casa": home_score,
            "gols_visitante": away_score,
            "definido_em_iso": _now().isoformat(),
            "definido_por": {
                "id": user_id,
                "nome": str(user.get("nome", "")).strip(),
            },
        }
        _recalculate_game_winners(game)
        games[game_index] = game
        github_sync = _write_games(games)

    return jsonify({"ok": True, "jogo": _public_game(game, user_id, True), "github_sync": github_sync})


@dinamicas_pop_bp.put("/api/dinamicas-pop/jogos/<game_id>/ganhador/<prediction_id>")
def select_prediction_winner(game_id: str, prediction_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem selecionar o ganhador."}), 403

    user = context.get("usuario") or {}
    user_id = str(user.get("id", "")).strip()
    with DATA_LOCK:
        games = _read_games()
        game_index, game = _find_game(games, game_id)
        if not game:
            return jsonify({"ok": False, "error": "Jogo nao encontrado."}), 404
        prediction = _find_prediction_by_id(game, prediction_id)
        if not prediction:
            return jsonify({"ok": False, "error": "Palpite nao encontrado."}), 404

        game["palpite_ganhador_id"] = str(prediction.get("id", "")).strip()
        game["palpite_ganhador_selecionado_em_iso"] = _now().isoformat()
        game["palpite_ganhador_selecionado_por"] = {
            "id": user_id,
            "nome": str(user.get("nome", "")).strip(),
        }
        games[game_index] = game
        github_sync = _write_games(games)
        email_game = dict(game)
        email_prediction = dict(prediction)

    winner_email = _send_prediction_winner_email(email_game, email_prediction)

    return jsonify(
        {
            "ok": True,
            "jogo": _public_game(game, user_id, True),
            "github_sync": github_sync,
            "winner_email": winner_email,
        }
    )


@dinamicas_pop_bp.put("/api/dinamicas-pop/jogos/<game_id>/palpites/<prediction_id>")
def update_prediction(game_id: str, prediction_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem alterar palpites."}), 403

    payload = request.get_json(silent=True) or {}
    home_score = _parse_score(payload.get("gols_casa"))
    away_score = _parse_score(payload.get("gols_visitante"))
    if home_score is None or away_score is None:
        return jsonify({"ok": False, "error": "Informe placares entre 0 e 99."}), 400

    with DATA_LOCK:
        games = _read_games()
        game_index, game = _find_game(games, game_id)
        if not game:
            return jsonify({"ok": False, "error": "Jogo nao encontrado."}), 404
        predictions = game.get("palpites")
        if not isinstance(predictions, list):
            predictions = []
        prediction_index = next(
            (
                index
                for index, item in enumerate(predictions)
                if isinstance(item, dict) and str(item.get("id", "")) == str(prediction_id)
            ),
            -1,
        )
        if prediction_index < 0:
            return jsonify({"ok": False, "error": "Palpite nao encontrado."}), 404
        prediction = predictions[prediction_index]
        prediction["gols_casa"] = home_score
        prediction["gols_visitante"] = away_score
        prediction["alterado_em_iso"] = _now().isoformat()
        predictions[prediction_index] = prediction
        game["palpites"] = predictions
        _recalculate_game_winners(game)
        games[game_index] = game
        github_sync = _write_games(games)
    return jsonify({"ok": True, "palpite": prediction, "github_sync": github_sync})


@dinamicas_pop_bp.delete("/api/dinamicas-pop/jogos/<game_id>/palpites/<prediction_id>")
def delete_prediction(game_id: str, prediction_id: str):
    context = _auth_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not _is_developer(context):
        return jsonify({"ok": False, "error": "Apenas desenvolvedores podem excluir palpites."}), 403

    with DATA_LOCK:
        games = _read_games()
        game_index, game = _find_game(games, game_id)
        if not game:
            return jsonify({"ok": False, "error": "Jogo nao encontrado."}), 404
        predictions = game.get("palpites")
        if not isinstance(predictions, list):
            predictions = []
        prediction_index = _find_prediction_index_for_delete(predictions, prediction_id, request.args)
        if prediction_index < 0:
            return jsonify({"ok": False, "error": "Palpite nao encontrado."}), 404
        deleted_prediction = predictions.pop(prediction_index)
        remaining = predictions
        game["palpites"] = remaining
        deleted_prediction_id = (
            str(deleted_prediction.get("id", "")).strip()
            if isinstance(deleted_prediction, dict)
            else str(prediction_id).strip()
        )
        if deleted_prediction_id and _selected_winner_id(game) == deleted_prediction_id:
            game.pop("palpite_ganhador_id", None)
            game.pop("palpite_ganhador_selecionado_em_iso", None)
            game.pop("palpite_ganhador_selecionado_por", None)
        _recalculate_game_winners(game)
        games[game_index] = game
        github_sync = _write_games(games)
    return jsonify({"ok": True, "github_sync": github_sync})
