from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from flask import Blueprint, jsonify, request

birthday_bp = Blueprint("aniversariantes", __name__)

MONTH_NAMES = {
    1: "janeiro",
    2: "fevereiro",
    3: "marco",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro",
}


def _parse_birth_iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except Exception:
        return None


def _calculate_age(birth: date, reference: date) -> int:
    age = reference.year - birth.year
    if (reference.month, reference.day) < (birth.month, birth.day):
        age -= 1
    return max(age, 0)


def _build_birth_entry(record: dict[str, Any], reference: date) -> dict | None:
    show = bool(record.get("mostrar_aniversario"))
    if not show:
        return None
    birth = _parse_birth_iso(str(record.get("data_nascimento_iso", "")))
    if not birth:
        return None
    return {
        "id": str(record.get("id", "")),
        "nome": str(record.get("nome", "")),
        "funcao": str(record.get("funcao", "")),
        "day": birth.day,
        "month": birth.month,
        "year": birth.year,
        "age": _calculate_age(birth, reference),
        "data_nascimento_iso": birth.isoformat(),
    }


def _load_visible_birthdays(reference: date) -> list[dict]:
    from app import read_employees

    records = read_employees()
    entries: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        entry = _build_birth_entry(record, reference)
        if entry and entry["month"] == reference.month:
            entries.append(entry)
    entries.sort(key=lambda item: (item["day"], item["nome"]))
    return entries


def _month_label(month: int) -> str:
    return MONTH_NAMES.get(month, str(month))


def _today_box(entries: list[dict], reference: datetime) -> dict:
    todays = [entry for entry in entries if entry["day"] == reference.day]
    if not todays:
        return {"available": False}
    expires = (reference + timedelta(hours=24)).isoformat()
    return {
        "available": True,
        "expires_at_iso": expires,
        "entries": todays,
        "message_hint": "Use emojis e deixe um parabéns especial!",
    }


def _now() -> datetime:
    from app import now_brazil

    return now_brazil()


def _get_authenticated_context():
    from app import get_authenticated_user_context

    return get_authenticated_user_context()


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _build_iso_date(day: int | None, month: int | None, year: int | None) -> tuple[str, str | None]:
    if day is None and month is None and year is None:
        return "", None
    if day is None or month is None or year is None:
        return "", "Informe dia, mês e ano completos."
    try:
        birth = date(year, month, day)
        return birth.isoformat(), None
    except ValueError:
        return "", "Data de nascimento inválida."


def _build_user_birthday_info(record: dict | None, reference: date | None = None) -> dict | None:
    if not record:
        return None
    birth = _parse_birth_iso(str(record.get("data_nascimento_iso", "")))
    mostrar = bool(record.get("mostrar_aniversario"))
    info: dict[str, Any] = {
        "id": str(record.get("id", "")),
        "nome": str(record.get("nome", "")),
        "mostrar": mostrar,
    }
    if birth:
        ref = reference or _now().date()
        info.update(
            {
                "day": birth.day,
                "month": birth.month,
                "year": birth.year,
                "age": _calculate_age(birth, ref),
            }
        )
    return info


@birthday_bp.route("/api/aniversariantes/mes")
def api_month_birthdays() -> Any:
    reference = _now().date()
    entries = _load_visible_birthdays(reference)
    return jsonify(
        {
            "ok": True,
            "month": reference.month,
            "month_label": _month_label(reference.month),
            "birthdays": entries,
        }
    )


@birthday_bp.route("/api/aniversariantes/hoje")
def api_today_birthdays() -> Any:
    now = _now()
    entries = _load_visible_birthdays(now.date())
    box = _today_box(entries, now)
    return jsonify({"ok": True, "box": box})


@birthday_bp.route("/api/aniversariantes/pessoal", methods=["GET", "POST"])
def api_personal_birthday() -> Any:
    context = _get_authenticated_context()
    if not context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    usuario = context.get("usuario") or {}
    employee_id = str(usuario.get("id", ""))
    if not employee_id:
        return jsonify({"ok": False, "error": "Usuario indefinido."}), 400

    if request.method == "GET":
        from app import read_employees

        records = read_employees()
        record = next((item for item in records if str(item.get("id", "")) == employee_id), None)
        return jsonify({"ok": True, "birthday": _build_user_birthday_info(record, _now().date())})

    payload = request.get_json(silent=True) or {}
    day = _parse_int(payload.get("dia"))
    month = _parse_int(payload.get("mes"))
    year = _parse_int(payload.get("ano"))
    mostrar = bool(payload.get("mostrar_aniversario"))
    iso, error = _build_iso_date(day, month, year)
    if error:
        return jsonify({"ok": False, "error": error}), 400

    from app import update_employee_record, build_employee_public_record

    def _apply(record: dict) -> dict:
        record["mostrar_aniversario"] = mostrar
        record["data_nascimento_iso"] = iso
        return record

    updated, github_sync = update_employee_record(employee_id, _apply)
    if not updated:
        return jsonify({"ok": False, "error": github_sync.get("reason", "Falha ao atualizar.")}), 500

    return jsonify(
        {
            "ok": True,
            "birthday": _build_user_birthday_info(updated, _now().date()),
            "funcionario": build_employee_public_record(updated),
        }
    )
