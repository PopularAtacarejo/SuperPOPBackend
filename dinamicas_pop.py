from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request


dinamicas_pop_bp = Blueprint("dinamicas_pop", __name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "DinamicasPOP.json"
DATA_LOCK = threading.Lock()


def _ensure_data_file() -> None:
    if not DATA_FILE.exists():
        DATA_FILE.write_text("[]\n", encoding="utf-8")


def _read_games() -> list[dict[str, Any]]:
    _ensure_data_file()
    try:
        loaded = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def _write_games(games: list[dict[str, Any]]) -> None:
    _ensure_data_file()
    temporary = DATA_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(games, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(DATA_FILE)


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


def _normalize_game_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    home_team = _clean_text(payload.get("time_casa"))
    away_team = _clean_text(payload.get("time_visitante"))
    competition = _clean_text(payload.get("competicao"), 100)
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


def _public_prediction(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id", "")),
        "usuario_id": str(item.get("usuario_id", "")),
        "usuario_nome": str(item.get("usuario_nome", "")),
        "gols_casa": item.get("gols_casa"),
        "gols_visitante": item.get("gols_visitante"),
        "enviado_em_iso": str(item.get("enviado_em_iso", "")),
        "alterado_em_iso": str(item.get("alterado_em_iso", "")),
    }


def _public_game(game: dict[str, Any], user_id: str, developer: bool) -> dict[str, Any]:
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
    public = {
        "id": str(game.get("id", "")),
        "time_casa": str(game.get("time_casa", "")),
        "time_visitante": str(game.get("time_visitante", "")),
        "imagem_time_casa": str(game.get("imagem_time_casa", "")),
        "imagem_time_visitante": str(game.get("imagem_time_visitante", "")),
        "competicao": str(game.get("competicao", "")),
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
        "palpites_enviados": [
            _public_prediction(item) for item in predictions if isinstance(item, dict)
        ],
    }
    if developer:
        public["palpites"] = [
            _public_prediction(item) for item in predictions if isinstance(item, dict)
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
    return jsonify(
        {
            "ok": True,
            "is_developer": developer,
            "jogos": [_public_game(game, user_id, developer) for game in games],
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
        _write_games(games)

    return jsonify({"ok": True, "jogo": _public_game(game, str(user.get("id", "")), True)}), 201


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
        _write_games(games)

    user_id = str((context.get("usuario") or {}).get("id", ""))
    return jsonify({"ok": True, "jogo": _public_game(game, user_id, True)})


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
        _write_games(games)
    return jsonify({"ok": True})


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
            "gols_casa": home_score,
            "gols_visitante": away_score,
            "enviado_em_iso": _now().isoformat(),
        }
        predictions.append(prediction)
        game["palpites"] = predictions
        games[index] = game
        _write_games(games)

    return jsonify({"ok": True, "palpite": prediction})


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
        games[game_index] = game
        _write_games(games)
    return jsonify({"ok": True, "palpite": prediction})


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
        remaining = [
            item
            for item in predictions
            if not (isinstance(item, dict) and str(item.get("id", "")) == str(prediction_id))
        ]
        if len(remaining) == len(predictions):
            return jsonify({"ok": False, "error": "Palpite nao encontrado."}), 404
        game["palpites"] = remaining
        games[game_index] = game
        _write_games(games)
    return jsonify({"ok": True})
