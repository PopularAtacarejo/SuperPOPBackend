import base64
import copy
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import smtplib
import ssl
import textwrap
import time
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import date, datetime, timedelta
from email.utils import getaddresses
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

from backup_manager import backup_employees, backup_logs
from dev_user_creation import prepare_developer_user_payload
from notifications import get_user_state as get_user_notification_state, update_user_state as update_user_notification_state

from flask import Flask, current_app, jsonify, redirect, request, send_from_directory, session, url_for
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont, ImageOps


BASE_DIR = Path(__file__).resolve().parent
CARDS_DIR = BASE_DIR / "generated" / "cards"
CARDS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_CACHE_DIR = BASE_DIR / "generated" / "templates"
TEMPLATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OAUTH_CACHE_DIR = BASE_DIR / "generated" / "oauth"
OAUTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = BASE_DIR / "Dados.json"
DATA_FILE_LOCK = threading.Lock()
RANK_REACTIONS_FILE = BASE_DIR / "RankReacoes.json"
RANK_REACTIONS_FILE_LOCK = threading.Lock()
SYSTEM_UPDATES_FILE = BASE_DIR / "Atualizacoes.json"
LEGACY_SYSTEM_UPDATES_FILE = BASE_DIR / "AtualizacoesSistema.json"
SYSTEM_UPDATES_FILE_LOCK = threading.Lock()
EMPLOYEES_FILE = BASE_DIR / "Funcioinarios.json"
EMPLOYEES_FILE_LOCK = threading.Lock()
BACKUP_DIR = BASE_DIR / "Backup"
EMPLOYEES_BACKUP_FILE = BACKUP_DIR / "Funcionarios-Backup.json"
MICROSOFT_OAUTH_TOKEN_FILE = OAUTH_CACHE_DIR / "microsoft_smtp_oauth.json"
PENDING_SEND_KEYS: set[str] = set()
ONLINE_USERS_LOCK = threading.Lock()
ONLINE_USERS_STATE: dict[str, dict] = {}
DEFAULT_TEMPLATE_PATHS = [
    BASE_DIR / "assets" / "Super-POP.png",
    BASE_DIR / "SuperPOP.png",
    Path(r"C:\Users\marke\OneDrive\Desktop\SuperPOP.png"),
]
DOTENV_PATH = BASE_DIR / ".env"
LAYOUT_FILE = BASE_DIR / "layout.json"
LAYOUT_CACHE_LOCK = threading.Lock()
LAYOUT_CACHE: dict[str, object] = {
    "source": "",
    "loaded_at": 0.0,
    "config": None,
}

DEFAULT_LAYOUT_CONFIG = {
    "template": {
        "base_size": {"width": 1059, "height": 662},
    },
    "text": {
        "collaborator_baseline": [278, 306],
        "recognized_baseline": [265, 345],
        "date_baseline": [672, 345],
        "collaborator_max_x": 474,
        "recognized_max_x": 530,
        "date_max_x": 938,
    },
    "checkbox": {
        "centers": {
            "acolhimento": [154, 686],
            "eficiencia": [390, 686],
            "cortesia": [596, 686],
            "resultado": [800, 686],
        },
        "center_y_offset": 240,
        "box_size": [29, 29],
        "line_width_scale": 4.0,
    },
    "message": {
        "origin": [110, 509],
        "max_width": 840,
        "max_lines": 3,
        "line_gap_base": 11,
    },
    "qr": {
        "base_size": 66,
        "x": 36,
        "bottom_margin": 46,
    },
}

DEFAULT_SYSTEM_UPDATES = [
    {
        "id": "seed-online-users-20260316",
        "data_referencia": "16/03/2026",
        "titulo": "Usuarios online no Super POP",
        "descricao": (
            "Adicionado menu de usuarios online em tempo quase real na tela de envio do "
            "Super POP, com lista de ativos e contador."
        ),
        "categoria": "Conectividade",
        "status": "Concluido",
        "autor": {"id": "", "nome": "Jeferson"},
        "created_at_iso": "2026-03-16T08:00:00-03:00",
    },
    {
        "id": "seed-rank-reactions-20260316",
        "data_referencia": "16/03/2026",
        "titulo": "Reacoes por emoji no ranking",
        "descricao": (
            "Usuarios agora podem reagir no ranking com emojis, alterar a reacao e "
            "visualizar quem reagiu em cada usuario."
        ),
        "categoria": "Ranking",
        "status": "Nova funcionalidade",
        "autor": {"id": "", "nome": "Jeferson"},
        "created_at_iso": "2026-03-16T08:10:00-03:00",
    },
    {
        "id": "seed-rank-rule-20260316",
        "data_referencia": "16/03/2026",
        "titulo": "Regra mensal do ranking ajustada",
        "descricao": (
            "No rank mensal, cada par remetente + destinatario conta apenas uma vez "
            "por mes, mesmo com varios envios."
        ),
        "categoria": "Regra de negocio",
        "status": "Concluido",
        "autor": {"id": "", "nome": "Jeferson"},
        "created_at_iso": "2026-03-16T08:20:00-03:00",
    },
    {
        "id": "seed-duplicate-day-20260316",
        "data_referencia": "16/03/2026",
        "titulo": "Bloqueio de envio duplicado no mesmo dia",
        "descricao": (
            "Mantida a regra de 1 Super POP por dia para o mesmo destinatario, com "
            "mensagem elegante para tentativas duplicadas."
        ),
        "categoria": "Protecao",
        "status": "Concluido",
        "autor": {"id": "", "nome": "Jeferson"},
        "created_at_iso": "2026-03-16T08:30:00-03:00",
    },
    {
        "id": "seed-register-robust-20260316",
        "data_referencia": "16/03/2026",
        "titulo": "Robustez no registro de Super POP",
        "descricao": (
            "Aplicados controles para reduzir falhas de contabilizacao e evitar "
            "concorrencia no envio."
        ),
        "categoria": "Confiabilidade",
        "status": "Concluido",
        "autor": {"id": "", "nome": "Jeferson"},
        "created_at_iso": "2026-03-16T08:40:00-03:00",
    },
    {
        "id": "seed-destination-fix-20260316",
        "data_referencia": "16/03/2026",
        "titulo": "Ajuste na selecao de destinatario",
        "descricao": (
            "Corrigido comportamento da tela para permitir alterar o destinatario "
            "apos a primeira selecao sem travamentos."
        ),
        "categoria": "Usabilidade",
        "status": "Concluido",
        "autor": {"id": "", "nome": "Jeferson"},
        "created_at_iso": "2026-03-16T08:50:00-03:00",
    },
]


def load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]

        # Preserve environment variables explicitly set by the OS/session.
        os.environ.setdefault(key, value)


load_dotenv_file(DOTENV_PATH)

if not DATA_FILE.exists():
    DATA_FILE.write_text("[]\n", encoding="utf-8")

if not RANK_REACTIONS_FILE.exists():
    RANK_REACTIONS_FILE.write_text("[]\n", encoding="utf-8")

if not SYSTEM_UPDATES_FILE.exists():
    if LEGACY_SYSTEM_UPDATES_FILE.exists():
        try:
            legacy_payload = LEGACY_SYSTEM_UPDATES_FILE.read_text(encoding="utf-8")
            loaded_legacy = json.loads(legacy_payload)
            if isinstance(loaded_legacy, list):
                SYSTEM_UPDATES_FILE.write_text(
                    json.dumps(loaded_legacy, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                raise ValueError("legacy_updates_not_list")
        except Exception:
            SYSTEM_UPDATES_FILE.write_text(
                json.dumps(DEFAULT_SYSTEM_UPDATES, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    else:
        SYSTEM_UPDATES_FILE.write_text(
            json.dumps(DEFAULT_SYSTEM_UPDATES, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

if not EMPLOYEES_FILE.exists():
    EMPLOYEES_FILE.write_text("[]\n", encoding="utf-8")

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
if not EMPLOYEES_BACKUP_FILE.exists():
    EMPLOYEES_BACKUP_FILE.write_text("[]\n", encoding="utf-8")

if not LAYOUT_FILE.exists():
    LAYOUT_FILE.write_text(
        json.dumps(DEFAULT_LAYOUT_CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def normalize_cors_origin(value: str) -> str:
    origin = str(value or "").strip()
    if not origin:
        return ""

    # Keep wildcard/regex-style patterns compatible with flask-cors.
    if origin == "*" or any(token in origin for token in ("*", "^", "$", "(", ")", "[", "]", "|")):
        return origin

    if "://" not in origin:
        return origin.rstrip("/")

    parsed = urllib.parse.urlsplit(origin)
    if not parsed.scheme or not parsed.netloc:
        return ""

    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def deep_merge_dict(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def normalize_layout_source_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""

    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return raw

    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if host == "github.com" and "/blob/" in path:
        parts = path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            owner = parts[0]
            repo = parts[1]
            branch = parts[3]
            file_path = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"

    return raw


def get_frontend_base_url() -> str:
    base = get_env("SUPERPOP_FRONTEND_URL", "https://popularatacarejo.github.io/SuperPOP")
    return re.sub(r"/+$", "", str(base or "").strip())


def build_frontend_url(path: str = "") -> str:
    base = get_frontend_base_url()
    clean_path = str(path or "").lstrip("/")
    if not base:
        return f"/{clean_path}" if clean_path else "/"
    return f"{base}/{clean_path}" if clean_path else base


def send_page_or_frontend(filename: str):
    local_file = BASE_DIR / filename
    if local_file.exists():
        return send_from_directory(BASE_DIR, filename)
    return redirect(build_frontend_url(filename))


def fetch_layout_config_from_url(url: str) -> dict | None:
    source_url = normalize_layout_source_url(url)
    if not source_url:
        return None

    timeout_seconds = max(3.0, to_number(get_env("LAYOUT_CONFIG_TIMEOUT_SECONDS", "12"), 12.0))
    request_obj = urllib.request.Request(
        source_url,
        headers={"User-Agent": "superpop-backend-layout-fetcher"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
            loaded = json.loads(payload)
    except Exception:
        return None

    return loaded if isinstance(loaded, dict) else None


def load_layout_config_remote_cached(url: str) -> dict | None:
    source_url = normalize_layout_source_url(url)
    if not source_url:
        return None

    cache_seconds = max(0.0, to_number(get_env("LAYOUT_CONFIG_CACHE_SECONDS", "120"), 120.0))
    now_ts = time.time()

    with LAYOUT_CACHE_LOCK:
        cached_source = str(LAYOUT_CACHE.get("source") or "")
        cached_loaded_at = to_number(LAYOUT_CACHE.get("loaded_at"), 0.0)
        cached_config = LAYOUT_CACHE.get("config")
        if (
            cached_source == source_url
            and isinstance(cached_config, dict)
            and (now_ts - cached_loaded_at) <= cache_seconds
        ):
            return copy.deepcopy(cached_config)

    loaded = fetch_layout_config_from_url(source_url)
    if not isinstance(loaded, dict):
        return None

    with LAYOUT_CACHE_LOCK:
        LAYOUT_CACHE["source"] = source_url
        LAYOUT_CACHE["loaded_at"] = now_ts
        LAYOUT_CACHE["config"] = copy.deepcopy(loaded)

    return loaded


def load_layout_config() -> dict:
    config = copy.deepcopy(DEFAULT_LAYOUT_CONFIG)
    remote_source = get_env("LAYOUT_CONFIG_URL")
    if remote_source:
        remote_loaded = load_layout_config_remote_cached(remote_source)
        if isinstance(remote_loaded, dict):
            return deep_merge_dict(config, remote_loaded)

    if LAYOUT_FILE.exists():
        try:
            loaded = json.loads(LAYOUT_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return deep_merge_dict(config, loaded)
        except Exception:
            pass

    return config


def to_pair(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return default
    return default


def to_number(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00ef\u00bf\u00bd")


def text_mojibake_score(text: str) -> int:
    if not text:
        return 0
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def repair_mojibake_text(value: object) -> str:
    text = str(value or "")
    if not text:
        return ""

    candidate = text
    for _ in range(2):
        if text_mojibake_score(candidate) <= 0:
            break
        try:
            repaired = candidate.encode("latin-1").decode("utf-8")
        except Exception:
            break
        if text_mojibake_score(repaired) > text_mojibake_score(candidate):
            break
        candidate = repaired

    return candidate


def to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = repair_mojibake_text(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on", "sim", "s"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "nao", "n\u00e3o"}:
        return False
    return default


def get_default_country_code() -> str:
    digits = re.sub(r"\D+", "", get_env("WHATSAPP_DEFAULT_COUNTRY_CODE", "55"))
    return digits or "55"


def get_default_area_code() -> str:
    digits = re.sub(r"\D+", "", get_env("WHATSAPP_DEFAULT_AREA_CODE", "82"))
    return digits or "82"


def get_whatsapp_send_mode() -> str:
    mode = get_env("WHATSAPP_SEND_MODE", "wa_me").lower()
    return mode if mode in {"wa_me", "webjs"} else "wa_me"


def get_card_auth_secret() -> str:
    secret = get_env("CARD_AUTH_SECRET")
    return secret or "superpop-auth-secret"


def build_card_auth_token(card_id: str) -> str:
    return hmac.new(
        get_card_auth_secret().encode("utf-8"),
        (card_id or "").encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_card_auth_url(card_id: str, token: str) -> str:
    base = get_env("PUBLIC_BASE_URL", "http://localhost:5000") or "http://localhost:5000"
    return (
        f"{base.rstrip('/')}/api/cards/verify/{urllib.parse.quote(card_id)}"
        f"?token={urllib.parse.quote(token)}"
    )


def generate_qr_code_image(content: str, size: int) -> Image.Image | None:
    if not content:
        return None
    try:
        import qrcode
    except Exception:
        return None

    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=1,
        )
        qr.add_data(content)
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        return qr_image.resize((size, size), Image.Resampling.NEAREST)
    except Exception:
        return None


def now_brazil() -> datetime:
    try:
        return datetime.now(ZoneInfo("America/Sao_Paulo"))
    except Exception:
        return datetime.now()


def normalize_whatsapp_number(raw: str) -> str:
    digits = normalize_whatsapp_digits(raw)
    return f"+{digits}" if digits else ""


def normalize_whatsapp_digits(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""

    digits = re.sub(r"\D+", "", value)
    if not digits:
        return ""

    # Keep explicit international format untouched (e.g. +5511..., +351...).
    if value.startswith("+") and len(digits) >= 10:
        return digits

    # Handle numbers provided as 00 + country code.
    if value.startswith("00") and len(digits) > 2:
        return digits[2:]

    digits = digits.lstrip("0")
    if not digits:
        return ""

    country_code = get_default_country_code()
    area_code = get_default_area_code()

    if digits.startswith(country_code):
        return digits

    if len(digits) in {10, 11}:
        return f"{country_code}{digits}"

    if len(digits) in {8, 9}:
        return f"{country_code}{area_code}{digits}"

    return f"{country_code}{area_code}{digits}"


def build_whatsapp_caption(payload: dict) -> str:
    collaborator = repair_mojibake_text(payload.get("colaborador", "")).strip() or "-"
    recognized_by = repair_mojibake_text(payload.get("reconhecido_por", "")).strip() or "-"
    values = payload.get("valores", [])
    values_clean = [repair_mojibake_text(item).strip() for item in values] if isinstance(values, list) else []
    values_text = ", ".join(item for item in values_clean if item) if values_clean else "-"
    date_text = payload.get("data", "") or now_brazil().strftime("%d/%m/%Y")
    message_text = repair_mojibake_text(payload.get("mensagem", "")).strip()
    sender_device_type = str(payload.get("sender_device_type", "")).strip().lower()
    use_emojis = sender_device_type == "mobile"

    title_line = "\U0001F389 *SuperPOP - Reconhecimento*" if use_emojis else "*SuperPOP - Reconhecimento*"
    to_line = (f"\U0001F464 *Para:* {collaborator}") if use_emojis else (f"*Para:* {collaborator}")
    from_line = (f"\U0001F64C *Enviado por:* {recognized_by}") if use_emojis else (f"*Enviado por:* {recognized_by}")
    values_line = (f"\u2B50 *Valores:* {values_text}") if use_emojis else (f"*Valores:* {values_text}")
    date_line = (f"\U0001F4C5 *Data:* {date_text}") if use_emojis else (f"*Data:* {date_text}")
    message_label = "\U0001F4AC *Mensagem:*" if use_emojis else "*Mensagem:*"
    cta_line = "Envie voc\u00ea tamb\u00e9m um Super POP! - https://popularatacarejo.github.io/SuperPOP/"

    lines = [
        title_line,
        "",
        to_line,
        from_line,
        values_line,
        date_line,
    ]
    if message_text:
        lines.extend(["", message_label, message_text])
    lines.extend(["", cta_line])

    return "\n".join(lines)


def post_json_request(url: str, payload: dict, headers: dict | None = None, timeout: float = 30.0) -> tuple[int, dict | None, str]:
    body = json.dumps(payload).encode("utf-8")
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    request_obj = urllib.request.Request(url, data=body, headers=merged_headers, method="POST")

    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return int(getattr(response, "status", 200) or 200), (parsed if isinstance(parsed, dict) else None), ""
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        parsed = None
        if detail:
            try:
                maybe_json = json.loads(detail)
                if isinstance(maybe_json, dict):
                    parsed = maybe_json
            except Exception:
                pass
        return exc.code, parsed, detail or str(exc.reason)
    except Exception as exc:
        return 0, None, str(exc)


def get_json_request(url: str, headers: dict | None = None, timeout: float = 20.0) -> tuple[int, dict | None, str]:
    request_obj = urllib.request.Request(url, headers=headers or {}, method="GET")

    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return int(getattr(response, "status", 200) or 200), (parsed if isinstance(parsed, dict) else None), ""
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        parsed = None
        if detail:
            try:
                maybe_json = json.loads(detail)
                if isinstance(maybe_json, dict):
                    parsed = maybe_json
            except Exception:
                pass
        return exc.code, parsed, detail or str(exc.reason)
    except Exception as exc:
        return 0, None, str(exc)


def send_image_via_whatsapp_webjs(destination: str, image_url: str, caption: str) -> dict:
    api_base = get_env("WHATSAPP_WEBJS_API_URL")
    if not api_base:
        return {
            "enabled": False,
            "ok": False,
            "error": "WHATSAPP_WEBJS_API_URL nao configurado.",
            "message_id": "",
            "to": destination or "",
            "provider": "whatsapp-web.js",
        }

    endpoint = f"{api_base.rstrip('/')}/send-image"
    api_token = get_env("WHATSAPP_WEBJS_API_TOKEN")
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    timeout_seconds = max(5.0, to_number(get_env("WHATSAPP_WEBJS_TIMEOUT_SECONDS", "45"), 45.0))
    retries = max(1, int(to_number(get_env("WHATSAPP_WEBJS_SEND_RETRIES", "2"), 2)))
    retry_delay = max(0.0, to_number(get_env("WHATSAPP_WEBJS_SEND_RETRY_DELAY_SECONDS", "1.2"), 1.2))

    status_code = 0
    response_payload: dict | None = None
    response_error = ""
    for attempt in range(1, retries + 1):
        status_code, response_payload, response_error = post_json_request(
            url=endpoint,
            payload={
                "to": destination,
                "image_url": image_url,
                "caption": caption,
                "filename": "superpop.png",
                "mime_type": "image/png",
            },
            headers=headers,
            timeout=timeout_seconds,
        )
        if status_code >= 200 and status_code < 300 and isinstance(response_payload, dict) and response_payload.get("ok"):
            return {
                "enabled": True,
                "ok": True,
                "error": "",
                "message_id": str(response_payload.get("message_id", "")).strip(),
                "to": str(response_payload.get("to", destination)).strip(),
                "provider": str(response_payload.get("provider", "whatsapp-web.js")).strip() or "whatsapp-web.js",
            }
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    backend_error = ""
    if isinstance(response_payload, dict):
        backend_error = str(response_payload.get("error", "")).strip()
    if not backend_error:
        backend_error = response_error or f"Falha no servico whatsapp-web.js (HTTP {status_code})."

    return {
        "enabled": True,
        "ok": False,
        "error": backend_error,
        "message_id": "",
        "to": destination or "",
        "provider": "whatsapp-web.js",
    }


def resolve_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def normalize_payload(payload: dict) -> dict:
    values = payload.get("valores", [])
    if not isinstance(values, list):
        values = []

    raw_num_colaborador = str(payload.get("numero_colaborador", "")).strip()
    raw_num_reconhecido = str(payload.get("numero_reconhecido_por", "")).strip()
    raw_to = str(payload.get("to", "")).strip()

    normalized_num_colaborador = normalize_whatsapp_digits(raw_num_colaborador)
    normalized_num_reconhecido = normalize_whatsapp_digits(raw_num_reconhecido)
    normalized_to = normalize_whatsapp_digits(raw_to)
    raw_sender_device_type = str(payload.get("sender_device_type", "")).strip().lower()
    normalized_sender_device_type = ""
    if raw_sender_device_type in {"mobile", "celular", "phone", "smartphone"}:
        normalized_sender_device_type = "mobile"
    elif raw_sender_device_type in {"desktop", "pc", "computador", "web"}:
        normalized_sender_device_type = "desktop"

    return {
        "colaborador": repair_mojibake_text(payload.get("colaborador", "")).strip(),
        "numero_colaborador": normalized_num_colaborador or raw_num_colaborador,
        "funcao_colaborador": repair_mojibake_text(payload.get("funcao_colaborador", "")).strip(),
        "reconhecido_por": repair_mojibake_text(payload.get("reconhecido_por", "")).strip(),
        "numero_reconhecido_por": normalized_num_reconhecido or raw_num_reconhecido,
        "funcao_reconhecido_por": repair_mojibake_text(payload.get("funcao_reconhecido_por", "")).strip(),
        "valores": [
            repair_mojibake_text(v).strip()
            for v in values
            if repair_mojibake_text(v).strip()
        ],
        "mensagem": repair_mojibake_text(payload.get("mensagem", "")).strip(),
        "data": str(payload.get("data", "")).strip(),
        "to": normalized_to or normalized_num_colaborador or raw_to,
        "format": str(payload.get("format", "image")).strip().lower(),
        "send_mode": str(payload.get("send_mode", "")).strip().lower(),
        "sender_device_type": normalized_sender_device_type,
    }


def infer_sender_device_type_from_user_agent(user_agent: str) -> str:
    ua = str(user_agent or "").strip().lower()
    if not ua:
        return "desktop"
    mobile_tokens = (
        "android",
        "iphone",
        "ipad",
        "ipod",
        "mobile",
        "windows phone",
        "opera mini",
        "blackberry",
    )
    return "mobile" if any(token in ua for token in mobile_tokens) else "desktop"


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    width: int,
    fill: str,
    line_spacing: int = 8,
) -> int:
    if not text:
        return y

    approx_chars = max(10, int(width / max(7, font.size if hasattr(font, "size") else 14)))
    lines = textwrap.wrap(text, width=approx_chars)
    cursor_y = y

    for line in lines:
        draw.text((x, cursor_y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, cursor_y), line, font=font)
        line_h = (bbox[3] - bbox[1]) if bbox else 18
        cursor_y += line_h + line_spacing

    return cursor_y


def trim_text_with_ellipsis(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    value = (text or "").strip()
    if not value:
        return ""

    if draw.textlength(value, font=font) <= max_width:
        return value

    ellipsis = "..."
    if draw.textlength(ellipsis, font=font) > max_width:
        return ""

    low, high = 0, len(value)
    best = ellipsis
    while low <= high:
        mid = (low + high) // 2
        candidate = value[:mid].rstrip() + ellipsis
        if draw.textlength(candidate, font=font) <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def wrap_text_by_pixel_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    content = (text or "").replace("\r\n", "\n").strip()
    if not content:
        return []

    lines: list[str] = []
    truncated = False

    def push_line(value: str) -> bool:
        nonlocal truncated
        if max_lines > 0 and len(lines) >= max_lines:
            truncated = True
            return False
        lines.append(value)
        return True

    for paragraph in content.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        current = ""
        tokens = paragraph.split()
        for token in tokens:
            candidate = f"{current} {token}".strip() if current else token
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
                continue

            if current:
                if not push_line(current):
                    break
                current = token
            else:
                current = token

            while current and draw.textlength(current, font=font) > max_width:
                low, high = 1, len(current)
                best_chunk_len = 1
                while low <= high:
                    mid = (low + high) // 2
                    chunk = current[:mid]
                    if draw.textlength(chunk, font=font) <= max_width:
                        best_chunk_len = mid
                        low = mid + 1
                    else:
                        high = mid - 1

                if not push_line(current[:best_chunk_len]):
                    break
                current = current[best_chunk_len:].lstrip()

            if max_lines > 0 and len(lines) >= max_lines and current:
                truncated = True
                break

        if max_lines > 0 and len(lines) >= max_lines:
            truncated = True
            break

        if current:
            if not push_line(current):
                break

    if truncated and lines:
        lines[-1] = trim_text_with_ellipsis(draw, lines[-1], font, max_width)

    return lines


def create_card_image_default(data: dict, image_path: Path, auth_qr_text: str = "") -> None:
    width, height = 1400, 860
    image = Image.new("RGB", (width, height), "#F8F9FA")
    draw = ImageDraw.Draw(image)

    draw.ellipse((-220, -220, 520, 520), fill="#FFE5E8")
    draw.ellipse((980, -200, 1620, 440), fill="#FFF2CC")
    draw.rectangle((0, 0, width, 14), fill="#E63946")

    card_x1, card_y1, card_x2, card_y2 = 90, 90, 1310, 770
    draw.rounded_rectangle((card_x1, card_y1, card_x2, card_y2), radius=30, fill="#FFFFFF", outline="#E5E7EB", width=3)
    draw.rectangle((card_x1, card_y1, card_x1 + 14, card_y2), fill="#E63946")

    title_font = resolve_font(52, bold=True)
    subtitle_font = resolve_font(26, bold=True)
    label_font = resolve_font(24, bold=True)
    value_font = resolve_font(26)
    body_font = resolve_font(28)

    draw.text((140, 130), "Cartao SuperPop", font=title_font, fill="#E63946")
    draw.text((140, 200), "Reconhecimento", font=subtitle_font, fill="#374151")

    colaborador_view = repair_mojibake_text(data["colaborador"]).strip() or "-"
    remetente_view = repair_mojibake_text(data["reconhecido_por"]).strip() or "-"

    y = 260
    fields = [
        ("Para", colaborador_view),
        ("Numero para", data["numero_colaborador"] or "-"),
        ("Enviado por", remetente_view),
        ("Numero de quem envia", data["numero_reconhecido_por"] or "-"),
        (
            "Valores",
            ", ".join(
                value for value in [repair_mojibake_text(v).strip() for v in data["valores"]]
                if value
            ) if data["valores"] else "-"
        ),
        ("Data", data["data"] or now_brazil().strftime("%d/%m/%Y")),
    ]

    for label, value in fields:
        draw.text((140, y), f"{label}:", font=label_font, fill="#111827")
        draw.text((420, y), value, font=value_font, fill="#1F2937")
        y += 50

    draw.text((140, y + 6), "Mensagem:", font=label_font, fill="#111827")
    message_box_x1, message_box_y1, message_box_x2, message_box_y2 = 140, y + 48, 1260, 710
    draw.rounded_rectangle((message_box_x1, message_box_y1, message_box_x2, message_box_y2), radius=18, fill="#F9FAFB", outline="#E5E7EB", width=2)
    draw_wrapped_text(
        draw=draw,
        text=repair_mojibake_text(data["mensagem"]).strip() or "-",
        font=body_font,
        x=168,
        y=message_box_y1 + 20,
        width=(message_box_x2 - message_box_x1 - 56),
        fill="#111827",
        line_spacing=6,
    )

    if auth_qr_text:
        qr_size = 120
        qr_image = generate_qr_code_image(auth_qr_text, qr_size)
        if qr_image:
            image.paste(qr_image, (120, 620))

    draw.text((880, 732), "Valorizando pessoas, construindo historias.", font=resolve_font(20, bold=True), fill="#E63946")
    image.save(image_path, format="PNG", optimize=True)


def create_card_image_from_template(data: dict, image_path: Path, template_path: Path, auth_qr_text: str = "") -> None:
    image = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    layout = load_layout_config()
    template_layout = layout.get("template", {})
    text_layout = layout.get("text", {})
    checkbox_layout = layout.get("checkbox", {})
    message_layout = layout.get("message", {})
    qr_layout = layout.get("qr", {})

    w, h = image.size
    base_size = template_layout.get("base_size", {})
    base_w = max(1, int(to_number(base_size.get("width"), 1059)))
    base_h = max(1, int(to_number(base_size.get("height"), 662)))
    sx = w / base_w
    sy = h / base_h

    def px(x: float) -> int:
        return int(round(x * sx))

    def py(y: float) -> int:
        return int(round(y * sy))

    def clean_value(value: str, max_len: int) -> str:
        value = repair_mojibake_text(value).strip()
        if not value:
            return "-"
        if len(value) <= max_len:
            return value
        return value[: max_len - 3].rstrip() + "..."

    def fit_text_to_width(value: str, font: ImageFont.ImageFont, max_width: int) -> str:
        text = (value or "").strip() or "-"
        if draw.textlength(text, font=font) <= max_width:
            return text

        ellipsis = "..."
        low, high = 0, len(text)
        best = "-"
        while low <= high:
            mid = (low + high) // 2
            candidate = text[:mid].rstrip() + ellipsis
            if draw.textlength(candidate, font=font) <= max_width:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best

    def normalize_value_key(text: str) -> str:
        plain = unicodedata.normalize("NFD", text or "")
        plain = "".join(ch for ch in plain if unicodedata.category(ch) != "Mn")
        return plain.lower().strip()

    line_font = resolve_font(max(20, int(30 * ((sx + sy) / 2))), bold=True)
    name_font = resolve_font(max(18, int(24 * ((sx + sy) / 2))), bold=True)
    message_font = resolve_font(max(18, int(24 * ((sx + sy) / 2))))
    collaborator_base = to_pair(text_layout.get("collaborator_baseline"), (278.0, 306.0))
    recognized_base = to_pair(text_layout.get("recognized_baseline"), (265.0, 345.0))
    date_base = to_pair(text_layout.get("date_baseline"), (672.0, 345.0))
    collaborator_line_x = px(collaborator_base[0])
    collaborator_line_y = py(collaborator_base[1])
    recognized_line_x = px(recognized_base[0])
    recognized_line_y = py(recognized_base[1])
    date_line_x = px(date_base[0])
    date_line_y = py(date_base[1])

    collaborator_view = clean_value(data["colaborador"], 40)
    recognized_view = clean_value(data["reconhecido_por"], 40)
    date_view = clean_value(data["data"] or now_brazil().strftime("%d/%m/%Y"), 16)

    collaborator_max_x = to_number(text_layout.get("collaborator_max_x"), 474.0)
    recognized_max_x = to_number(text_layout.get("recognized_max_x"), 530.0)
    date_max_x = to_number(text_layout.get("date_max_x"), 938.0)
    collaborator_view = fit_text_to_width(collaborator_view, name_font, px(max(1.0, collaborator_max_x - collaborator_base[0])))
    recognized_view = fit_text_to_width(recognized_view, name_font, px(max(1.0, recognized_max_x - recognized_base[0])))
    date_view = fit_text_to_width(date_view, line_font, px(max(1.0, date_max_x - date_base[0])))

    try:
        draw.text((collaborator_line_x, collaborator_line_y), collaborator_view, font=name_font, fill="#1f2937", anchor="ls")
        draw.text((recognized_line_x, recognized_line_y), recognized_view, font=name_font, fill="#1f2937", anchor="ls")
        draw.text((date_line_x, date_line_y), date_view, font=line_font, fill="#1f2937", anchor="ls")
    except Exception:
        name_ascent = int(getattr(name_font, "size", 24) * 0.75)
        date_ascent = int(getattr(line_font, "size", 30) * 0.75)
        draw.text((collaborator_line_x, collaborator_line_y - name_ascent), collaborator_view, font=name_font, fill="#1f2937")
        draw.text((recognized_line_x, recognized_line_y - name_ascent), recognized_view, font=name_font, fill="#1f2937")
        draw.text((date_line_x, date_line_y - date_ascent), date_view, font=line_font, fill="#1f2937")

    selected = {normalize_value_key(repair_mojibake_text(v)) for v in data["valores"]}
    checkbox_centers = checkbox_layout.get("centers", {})
    checkbox_center_y_offset = to_number(checkbox_layout.get("center_y_offset"), 240.0)
    checkbox_box_size = to_pair(checkbox_layout.get("box_size"), (29.0, 29.0))
    checkbox_line_width_scale = to_number(checkbox_layout.get("line_width_scale"), 4.0)

    for key, center in checkbox_centers.items():
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue

        center_x_base = to_number(center[0], 0.0)
        center_y_base = to_number(center[1], 0.0)
        if key in selected:
            center_y_template = center_y_base - checkbox_center_y_offset if center_y_base > base_h else center_y_base
            center_x = px(center_x_base)
            center_y = py(center_y_template)

            box_w = max(px(checkbox_box_size[0]), 22)
            box_h = max(py(checkbox_box_size[1]), 22)
            box_left = center_x - int(round(box_w / 2))
            box_top = center_y - int(round(box_h / 2))
            check_width = max(3, int(round(checkbox_line_width_scale * ((sx + sy) / 2))))
            p1 = (
                box_left + int(round(box_w * 0.24)),
                box_top + int(round(box_h * 0.56)),
            )
            p2 = (
                box_left + int(round(box_w * 0.44)),
                box_top + int(round(box_h * 0.76)),
            )
            p3 = (
                box_left + int(round(box_w * 0.78)),
                box_top + int(round(box_h * 0.30)),
            )
            draw.line([p1, p2], fill="#16a34a", width=check_width)
            draw.line([p2, p3], fill="#16a34a", width=check_width)

    message_text = repair_mojibake_text(data["mensagem"]).strip()
    if message_text:
        message_origin = to_pair(message_layout.get("origin"), (110.0, 509.0))
        message_x = px(message_origin[0])
        message_y = py(message_origin[1])
        message_width = px(to_number(message_layout.get("max_width"), 840.0))
        wrapped = wrap_text_by_pixel_width(
            draw=draw,
            text=message_text,
            font=message_font,
            max_width=message_width,
            max_lines=int(to_number(message_layout.get("max_lines"), 3)),
        )
        line_gap = max(8, py(to_number(message_layout.get("line_gap_base"), 11.0)))
        cursor_y = message_y
        for line in wrapped:
            draw.text((message_x, cursor_y), line, font=message_font, fill="#1f2937")
            bbox = draw.textbbox((message_x, cursor_y), line, font=message_font)
            line_h = (bbox[3] - bbox[1]) if bbox else py(30)
            cursor_y += line_h + line_gap

    if auth_qr_text:
        qr_size_base = to_number(qr_layout.get("base_size"), 66.0)
        qr_x_base = to_number(qr_layout.get("x"), 36.0)
        qr_bottom_margin_base = to_number(qr_layout.get("bottom_margin"), 46.0)
        qr_size = max(px(qr_size_base), py(qr_size_base))
        qr_image = generate_qr_code_image(auth_qr_text, qr_size)
        if qr_image:
            qr_x = px(qr_x_base)
            qr_y = h - qr_size - py(qr_bottom_margin_base)
            image.paste(qr_image, (qr_x, qr_y))

    image.save(image_path, format="PNG", optimize=True)


def is_http_url(value: str) -> bool:
    value = (value or "").strip().lower()
    return value.startswith("http://") or value.startswith("https://")


def download_template_from_url(url: str) -> Path | None:
    template_url = (url or "").strip()
    if not template_url:
        return None

    cache_name = f"template-{hashlib.sha256(template_url.encode('utf-8')).hexdigest()[:24]}.png"
    cached_path = TEMPLATE_CACHE_DIR / cache_name
    if cached_path.exists():
        return cached_path

    request_obj = urllib.request.Request(
        template_url,
        headers={"User-Agent": "superpop-backend-template-fetcher"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=30) as response:
            image_bytes = response.read()
    except Exception:
        return None

    try:
        with Image.open(io.BytesIO(image_bytes)) as downloaded:
            downloaded.convert("RGB").save(cached_path, format="PNG", optimize=True)
    except Exception:
        return None

    return cached_path


def resolve_card_template_path() -> Path | None:
    template_env = get_env("CARD_TEMPLATE_PATH")

    if template_env:
        if is_http_url(template_env):
            downloaded_path = download_template_from_url(template_env)
            if downloaded_path and downloaded_path.exists():
                return downloaded_path
        else:
            local_template = Path(template_env)
            if not local_template.is_absolute():
                local_template = BASE_DIR / local_template
            if local_template.exists():
                return local_template

    for default_path in DEFAULT_TEMPLATE_PATHS:
        if default_path.exists():
            return default_path

    cached_templates = sorted(TEMPLATE_CACHE_DIR.glob("template-*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cached_templates:
        return cached_templates[0]

    return None


def create_card_image(data: dict, image_path: Path, auth_qr_text: str = "") -> None:
    template_path = resolve_card_template_path()
    if template_path:
        create_card_image_from_template(data, image_path, template_path, auth_qr_text=auth_qr_text)
        return

    create_card_image_default(data, image_path, auth_qr_text=auth_qr_text)


def build_media_url(app: Flask, filename: str) -> str:
    base = get_env("PUBLIC_BASE_URL")
    media_path = f"/media/{filename}"
    if base:
        return f"{base.rstrip('/')}{media_path}"
    return url_for("serve_media", filename=filename, _external=True)


def upload_image_to_imgbb(image_path: Path) -> dict:
    api_key = get_env("IMGBB_API_KEY")
    if not api_key:
        return {"ok": False, "url": "", "delete_url": "", "error": "IMGBB_API_KEY nao configurado"}

    try:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": "", "delete_url": "", "error": f"Falha ao ler imagem: {exc}"}

    form = urllib.parse.urlencode(
        {
            "key": api_key,
            "image": image_b64,
            "name": image_path.stem,
        }
    ).encode("utf-8")

    request_obj = urllib.request.Request(
        "https://api.imgbb.com/1/upload",
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=60) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "url": "", "delete_url": "", "error": f"ImgBB HTTP {exc.code}: {detail or exc.reason}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": "", "delete_url": "", "error": f"Falha no upload ImgBB: {exc}"}

    if not payload.get("success"):
        return {"ok": False, "url": "", "delete_url": "", "error": f"ImgBB retornou erro: {payload}"}

    data = payload.get("data", {}) or {}
    public_url = str(data.get("url") or data.get("display_url") or "").strip()
    delete_url = str(data.get("delete_url") or "").strip()
    if not public_url:
        return {"ok": False, "url": "", "delete_url": delete_url, "error": "ImgBB nao retornou URL publica"}

    return {"ok": True, "url": public_url, "delete_url": delete_url, "error": ""}


def read_logs() -> list:
    if not DATA_FILE.exists():
        return []
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def build_daily_send_key(sender_number: str, destination_number: str, day_value: str) -> str:
    sender = str(sender_number or "").strip()
    destination = str(destination_number or "").strip()
    day = str(day_value or "").strip()
    if not sender or not destination or not day:
        return ""
    return f"{day}|{sender}|{destination}"


def find_duplicate_send_same_day(logs: list, sender_number: str, destination_number: str, day_value: str) -> dict | None:
    if not sender_number or not destination_number or not day_value:
        return None

    for record in reversed(logs):
        if not isinstance(record, dict):
            continue

        day = str(record.get("dia", "")).strip()
        if day != day_value:
            continue

        remetente = record.get("remetente", {}) or {}
        destinatario = record.get("destinatario", {}) or {}

        sender_saved = normalize_whatsapp_number(
            str(remetente.get("numero_normalizado") or remetente.get("numero") or "")
        )
        destination_saved = normalize_whatsapp_number(
            str(destinatario.get("numero_normalizado") or destinatario.get("numero") or "")
        )

        if sender_saved == sender_number and destination_saved == destination_number:
            return record

    return None


def log_record_key(record: dict) -> str:
    if not isinstance(record, dict):
        return ""

    record_id = str(record.get("id", "")).strip()
    if record_id:
        return f"id:{record_id}"

    card_id = str(record.get("card_id", "")).strip()
    if card_id:
        return f"card:{card_id}"

    iso_value = str(record.get("data_hora_iso", "")).strip()
    if iso_value:
        return f"iso:{iso_value}"

    try:
        # Last-resort stable key for older records without id/card_id.
        return "raw:" + json.dumps(record, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return "raw:" + str(record)


def merge_log_lists(*sources: list) -> list:
    merged: list = []
    seen: set[str] = set()

    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            key = log_record_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)

    return merged


def write_logs(logs: list) -> None:
    backup_logs()
    temp_file = DATA_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(DATA_FILE)


def github_sync_logs(logs: list) -> dict:
    token = get_env("GITHUB_TOKEN")
    if not token:
        return {"synced": False, "reason": "GITHUB_TOKEN nao configurado"}

    repo = get_env("GITHUB_REPO", "PopularAtacarejo/SuperPOP")
    file_path = get_env("GITHUB_FILE_PATH", "Dados.json")
    branch = get_env("GITHUB_BRANCH", "main")
    api_base = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    get_url = f"{api_base}?ref={urllib.parse.quote(branch)}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "superpop-backend",
    }

    sha = None
    remote_logs: list = []
    try:
        req_get = urllib.request.Request(get_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_get, timeout=20) as resp:
            current = json.loads(resp.read().decode("utf-8"))
            sha = current.get("sha")
            encoded_content = str(current.get("content") or "").strip()
            if encoded_content:
                try:
                    decoded_content = base64.b64decode(encoded_content).decode("utf-8")
                    loaded_remote = json.loads(decoded_content)
                    if isinstance(loaded_remote, list):
                        remote_logs = loaded_remote
                except Exception:  # noqa: BLE001
                    remote_logs = []
            if not remote_logs:
                download_url = str(current.get("download_url") or "").strip()
                if download_url:
                    try:
                        req_download = urllib.request.Request(download_url, headers=headers, method="GET")
                        with urllib.request.urlopen(req_download, timeout=20) as download_resp:
                            download_payload = json.loads(download_resp.read().decode("utf-8"))
                            if isinstance(download_payload, list):
                                remote_logs = download_payload
                    except Exception:  # noqa: BLE001
                        remote_logs = []
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return {"synced": False, "reason": f"GitHub GET falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub GET erro: {exc}"}

    # Keep local-first precedence so in-place updates (e.g. reactions) are not
    # overwritten by stale remote copies with the same id/card_id.
    merged_logs = merge_log_lists(logs, remote_logs)
    content = base64.b64encode(json.dumps(merged_logs, ensure_ascii=False, indent=2).encode("utf-8")).decode("utf-8")
    utc_now = datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    payload = {
        "message": f"Atualiza Dados.json ({utc_now})",
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
                "merged_logs": merged_logs,
                "remote_count": len(remote_logs),
                "sent_count": len(logs),
                "merged_count": len(merged_logs),
            }
    except urllib.error.HTTPError as exc:
        return {"synced": False, "reason": f"GitHub PUT falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub PUT erro: {exc}"}


def github_sync_logs_with_retry(logs: list) -> dict:
    retries = max(1, int(to_number(get_env("GITHUB_SYNC_RETRIES", "3"), 3)))
    retry_delay = max(0.0, to_number(get_env("GITHUB_SYNC_RETRY_DELAY_SECONDS", "1.0"), 1.0))
    last_result = {"synced": False, "reason": "Sync nao executado."}

    for attempt in range(1, retries + 1):
        result = github_sync_logs(logs)
        result["attempt"] = attempt
        result["max_attempts"] = retries
        if result.get("synced"):
            return result
        last_result = result
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    return last_result


def is_github_sync_required() -> bool:
    return to_bool(get_env("GITHUB_SYNC_REQUIRED", "1"), True)


def append_send_log(record: dict) -> dict:
    with DATA_FILE_LOCK:
        logs = merge_log_lists(read_logs(), [record])
        write_logs(logs)
        github_sync = github_sync_logs_with_retry(logs)
        merged_logs = github_sync.get("merged_logs")
        if isinstance(merged_logs, list):
            write_logs(merged_logs)
            github_sync.pop("merged_logs", None)
    return github_sync


def read_system_updates() -> list:
    if not SYSTEM_UPDATES_FILE.exists():
        return []
    try:
        data = json.loads(SYSTEM_UPDATES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def write_system_updates(records: list) -> None:
    temp_file = SYSTEM_UPDATES_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(SYSTEM_UPDATES_FILE)


def system_update_record_key(record: dict) -> str:
    if not isinstance(record, dict):
        return ""

    record_id = str(record.get("id", "")).strip()
    if record_id:
        return f"id:{record_id}"

    created_at_iso = str(record.get("created_at_iso", "")).strip()
    if created_at_iso:
        return f"iso:{created_at_iso}"

    try:
        return "raw:" + json.dumps(record, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return "raw:" + str(record)


def merge_system_update_lists(*sources: list) -> list:
    merged: list = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            key = system_update_record_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def github_sync_system_updates(records: list) -> dict:
    token = get_env("GITHUB_TOKEN")
    if not token:
        return {"synced": False, "reason": "GITHUB_TOKEN nao configurado"}

    repo = get_env("GITHUB_REPO", "PopularAtacarejo/SuperPOP")
    file_path = get_env("GITHUB_SYSTEM_UPDATES_FILE_PATH", "Atualizacoes.json")
    branch = get_env("GITHUB_BRANCH", "main")
    api_base = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    get_url = f"{api_base}?ref={urllib.parse.quote(branch)}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "superpop-backend",
    }

    sha = None
    remote_records: list = []
    try:
        req_get = urllib.request.Request(get_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_get, timeout=20) as resp:
            current = json.loads(resp.read().decode("utf-8"))
            sha = current.get("sha")
            encoded_content = str(current.get("content") or "").strip()
            if encoded_content:
                try:
                    decoded_content = base64.b64decode(encoded_content).decode("utf-8")
                    loaded_remote = json.loads(decoded_content)
                    if isinstance(loaded_remote, list):
                        remote_records = loaded_remote
                except Exception:  # noqa: BLE001
                    remote_records = []
            if not remote_records:
                download_url = str(current.get("download_url") or "").strip()
                if download_url:
                    try:
                        req_download = urllib.request.Request(download_url, headers=headers, method="GET")
                        with urllib.request.urlopen(req_download, timeout=20) as download_resp:
                            download_payload = json.loads(download_resp.read().decode("utf-8"))
                            if isinstance(download_payload, list):
                                remote_records = download_payload
                    except Exception:  # noqa: BLE001
                        remote_records = []
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return {"synced": False, "reason": f"GitHub GET falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub GET erro: {exc}"}

    # Keep local-first precedence so in-place edits are not overwritten by stale
    # remote copies with the same id.
    merged_records = merge_system_update_lists(records, remote_records)
    content = base64.b64encode(json.dumps(merged_records, ensure_ascii=False, indent=2).encode("utf-8")).decode("utf-8")
    utc_now = datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    payload = {
        "message": f"Atualiza Atualizacoes.json ({utc_now})",
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
                "merged_records": merged_records,
                "remote_count": len(remote_records),
                "sent_count": len(records),
                "merged_count": len(merged_records),
            }
    except urllib.error.HTTPError as exc:
        return {"synced": False, "reason": f"GitHub PUT falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub PUT erro: {exc}"}


def github_sync_system_updates_with_retry(records: list) -> dict:
    retries = max(1, int(to_number(get_env("GITHUB_SYNC_RETRIES", "3"), 3)))
    retry_delay = max(0.0, to_number(get_env("GITHUB_SYNC_RETRY_DELAY_SECONDS", "1.0"), 1.0))
    last_result = {"synced": False, "reason": "Sync nao executado."}

    for attempt in range(1, retries + 1):
        result = github_sync_system_updates(records)
        result["attempt"] = attempt
        result["max_attempts"] = retries
        if result.get("synced"):
            return result
        last_result = result
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    return last_result


def normalize_system_update_date(date_value: str) -> str:
    raw = str(date_value or "").strip()
    if raw:
        try:
            parsed = datetime.strptime(raw, "%d/%m/%Y")
            return parsed.strftime("%d/%m/%Y")
        except ValueError:
            pass
    return now_brazil().strftime("%d/%m/%Y")


def parse_system_update_sort_timestamp(record: dict) -> float:
    if not isinstance(record, dict):
        return 0.0

    date_value = str(record.get("data_referencia", "")).strip()
    created_at_iso = str(record.get("created_at_iso", "")).strip()
    timestamp = 0.0
    if date_value:
        try:
            timestamp = datetime.strptime(date_value, "%d/%m/%Y").timestamp()
        except ValueError:
            timestamp = 0.0
    if created_at_iso:
        try:
            timestamp = max(timestamp, datetime.fromisoformat(created_at_iso).timestamp())
        except ValueError:
            pass
    return timestamp


def normalize_system_update_write_payload(payload: dict) -> tuple[dict, str]:
    safe_payload = payload if isinstance(payload, dict) else {}
    titulo = str(safe_payload.get("titulo", "")).strip()
    descricao = str(safe_payload.get("descricao", "")).strip()
    categoria = str(safe_payload.get("categoria", "")).strip() or "Atualizacao"
    status = str(safe_payload.get("status", "")).strip() or "Concluido"
    data_referencia = normalize_system_update_date(str(safe_payload.get("data_referencia", "")).strip())

    if len(titulo) < 4:
        return {}, "Titulo invalido. Informe pelo menos 4 caracteres."
    if len(descricao) < 8:
        return {}, "Descricao invalida. Informe pelo menos 8 caracteres."

    return {
        "titulo": titulo[:140],
        "descricao": descricao[:4000],
        "categoria": categoria[:60],
        "status": status[:60],
        "data_referencia": data_referencia,
    }, ""


def find_system_update_index_by_id(records: list, update_id: str) -> int:
    wanted_id = str(update_id or "").strip()
    if not wanted_id:
        return -1
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        if str(item.get("id", "")).strip() == wanted_id:
            return index
    return -1


def save_system_update(payload: dict, actor: dict) -> tuple[dict, str]:
    normalized_payload, payload_error = normalize_system_update_write_payload(payload)
    if payload_error:
        return {}, payload_error

    actor_user = actor if isinstance(actor, dict) else {}
    actor_id = str(actor_user.get("id", "")).strip()
    actor_nome = str(actor_user.get("nome", "")).strip() or "Desenvolvedor"
    created_at_iso = now_brazil().isoformat()

    record = {
        "id": uuid.uuid4().hex,
        "data_referencia": normalized_payload["data_referencia"],
        "titulo": normalized_payload["titulo"],
        "descricao": normalized_payload["descricao"],
        "categoria": normalized_payload["categoria"],
        "status": normalized_payload["status"],
        "autor": {
            "id": actor_id,
            "nome": actor_nome,
        },
        "created_at_iso": created_at_iso,
    }

    with SYSTEM_UPDATES_FILE_LOCK:
        records = read_system_updates()
        refreshed_records, _refresh_error = refresh_local_system_updates_from_remote(records)
        if isinstance(refreshed_records, list):
            records = refreshed_records

        records = merge_system_update_lists(records, [record])
        write_system_updates(records)
        github_sync = github_sync_system_updates_with_retry(records)
        merged_records = github_sync.get("merged_records")
        if isinstance(merged_records, list):
            write_system_updates(merged_records)
            github_sync.pop("merged_records", None)
            records = merged_records

    return {"record": record, "records": records, "github_sync": github_sync}, ""


def update_system_update(update_id: str, payload: dict, actor: dict) -> tuple[dict, str, int]:
    wanted_id = str(update_id or "").strip()
    if not wanted_id:
        return {}, "Atualizacao nao identificada.", 400

    normalized_payload, payload_error = normalize_system_update_write_payload(payload)
    if payload_error:
        return {}, payload_error, 400

    actor_user = actor if isinstance(actor, dict) else {}
    actor_id = str(actor_user.get("id", "")).strip()
    actor_nome = str(actor_user.get("nome", "")).strip() or "Desenvolvedor"
    now_iso = now_brazil().isoformat()

    with SYSTEM_UPDATES_FILE_LOCK:
        records = read_system_updates()
        refreshed_records, _refresh_error = refresh_local_system_updates_from_remote(records)
        if isinstance(refreshed_records, list):
            records = refreshed_records

        target_index = find_system_update_index_by_id(records, wanted_id)
        if target_index < 0:
            return {}, "Atualizacao nao encontrada.", 404

        current = records[target_index] if isinstance(records[target_index], dict) else {}
        updated_record = {
            "id": wanted_id,
            "data_referencia": normalized_payload["data_referencia"],
            "titulo": normalized_payload["titulo"],
            "descricao": normalized_payload["descricao"],
            "categoria": normalized_payload["categoria"],
            "status": normalized_payload["status"],
            "autor": {
                "id": actor_id,
                "nome": actor_nome,
            },
            "created_at_iso": str(current.get("created_at_iso", "")).strip() or now_iso,
            "updated_at_iso": now_iso,
        }
        records[target_index] = updated_record
        write_system_updates(records)

        github_sync = github_sync_system_updates_with_retry(records)
        merged_records = github_sync.get("merged_records")
        if isinstance(merged_records, list):
            write_system_updates(merged_records)
            github_sync.pop("merged_records", None)
            records = merged_records

        target_index = find_system_update_index_by_id(records, wanted_id)
        if target_index >= 0 and isinstance(records[target_index], dict):
            updated_record = records[target_index]

    return {"record": updated_record, "records": records, "github_sync": github_sync}, "", 200


def delete_system_update(update_id: str) -> tuple[dict, str, int]:
    wanted_id = str(update_id or "").strip()
    if not wanted_id:
        return {}, "Atualizacao nao identificada.", 400

    with SYSTEM_UPDATES_FILE_LOCK:
        records = read_system_updates()
        refreshed_records, _refresh_error = refresh_local_system_updates_from_remote(records)
        if isinstance(refreshed_records, list):
            records = refreshed_records

        target_index = find_system_update_index_by_id(records, wanted_id)
        if target_index < 0:
            return {}, "Atualizacao nao encontrada.", 404

        deleted_record = records.pop(target_index)
        write_system_updates(records)

        github_sync = github_sync_system_updates_with_retry(records)
        merged_records = github_sync.get("merged_records")
        if isinstance(merged_records, list):
            write_system_updates(merged_records)
            github_sync.pop("merged_records", None)
            records = merged_records

    return {"record": deleted_record, "records": records, "github_sync": github_sync}, "", 200


def build_system_updates_payload(records: list) -> dict:
    items: list[dict] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        author = item.get("autor", {}) or {}
        items.append(
            {
                "id": str(item.get("id", "")).strip(),
                "data_referencia": normalize_system_update_date(str(item.get("data_referencia", "")).strip()),
                "titulo": str(item.get("titulo", "")).strip(),
                "descricao": str(item.get("descricao", "")).strip(),
                "categoria": str(item.get("categoria", "")).strip() or "Atualizacao",
                "status": str(item.get("status", "")).strip() or "Concluido",
                "autor": {
                    "id": str(author.get("id", "")).strip(),
                    "nome": str(author.get("nome", "")).strip() or "Desenvolvedor",
                },
                "created_at_iso": str(item.get("created_at_iso", "")).strip(),
                "updated_at_iso": str(item.get("updated_at_iso", "")).strip(),
            }
        )

    items.sort(key=parse_system_update_sort_timestamp, reverse=True)
    return {"ok": True, "items": items, "total": len(items)}


RANK_REACTION_ALLOWED_EMOJIS = {"\U0001F44F", "\U0001F525", "\U0001F4AF", "\U0001F389", "\U0001F680", "\u2764\ufe0f", "\U0001F64C", "\u2b50"}
RANK_REACTION_RANK_KINDS = {"received", "sent"}
SUPERPOP_REACTION_ALLOWED_EMOJIS = {"\U0001F44F", "\U0001F525", "\U0001F4AF", "\U0001F389", "\U0001F680", "\u2764\ufe0f", "\U0001F64C", "\u2b50"}


def read_rank_reactions() -> list:
    if not RANK_REACTIONS_FILE.exists():
        return []
    try:
        data = json.loads(RANK_REACTIONS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def write_rank_reactions(records: list) -> None:
    temp_file = RANK_REACTIONS_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(RANK_REACTIONS_FILE)


def rank_reaction_record_key(record: dict) -> str:
    if not isinstance(record, dict):
        return ""

    record_id = str(record.get("id", "")).strip()
    if record_id:
        return f"id:{record_id}"

    month_key = str(record.get("month_key", "")).strip()
    rank_kind = str(record.get("rank_kind", "")).strip().lower()
    target_name_key = str(record.get("target_name_key", "")).strip()
    reactor = record.get("reactor", {}) or {}
    reactor_id = str(reactor.get("id", "")).strip()
    reactor_name = str(reactor.get("nome", "")).strip()
    reactor_name_key = normalize_name_key(reactor_name)
    if month_key and rank_kind and target_name_key and (reactor_id or reactor_name_key):
        return f"cmp:{month_key}|{rank_kind}|{target_name_key}|{reactor_id or reactor_name_key}"

    reacted_at_iso = str(record.get("reacted_at_iso", "")).strip()
    if reacted_at_iso:
        return f"iso:{reacted_at_iso}"

    try:
        return "raw:" + json.dumps(record, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return "raw:" + str(record)


def merge_rank_reaction_lists(*sources: list) -> list:
    merged: list = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            key = rank_reaction_record_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def github_sync_rank_reactions(records: list) -> dict:
    token = get_env("GITHUB_TOKEN")
    if not token:
        return {"synced": False, "reason": "GITHUB_TOKEN nao configurado"}

    repo = get_env("GITHUB_REPO", "PopularAtacarejo/SuperPOP")
    file_path = get_env("GITHUB_RANK_REACTIONS_FILE_PATH", "RankReacoes.json")
    branch = get_env("GITHUB_BRANCH", "main")
    api_base = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    get_url = f"{api_base}?ref={urllib.parse.quote(branch)}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "superpop-backend",
    }

    sha = None
    remote_records: list = []
    try:
        req_get = urllib.request.Request(get_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_get, timeout=20) as resp:
            current = json.loads(resp.read().decode("utf-8"))
            sha = current.get("sha")
            encoded_content = str(current.get("content") or "").strip()
            if encoded_content:
                try:
                    decoded_content = base64.b64decode(encoded_content).decode("utf-8")
                    loaded_remote = json.loads(decoded_content)
                    if isinstance(loaded_remote, list):
                        remote_records = loaded_remote
                except Exception:  # noqa: BLE001
                    remote_records = []
            if not remote_records:
                download_url = str(current.get("download_url") or "").strip()
                if download_url:
                    try:
                        req_download = urllib.request.Request(download_url, headers=headers, method="GET")
                        with urllib.request.urlopen(req_download, timeout=20) as download_resp:
                            download_payload = json.loads(download_resp.read().decode("utf-8"))
                            if isinstance(download_payload, list):
                                remote_records = download_payload
                    except Exception:  # noqa: BLE001
                        remote_records = []
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return {"synced": False, "reason": f"GitHub GET falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub GET erro: {exc}"}

    merged_records = merge_rank_reaction_lists(remote_records, records)
    content = base64.b64encode(json.dumps(merged_records, ensure_ascii=False, indent=2).encode("utf-8")).decode("utf-8")
    utc_now = datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    payload = {
        "message": f"Atualiza RankReacoes.json ({utc_now})",
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
                "merged_records": merged_records,
                "remote_count": len(remote_records),
                "sent_count": len(records),
                "merged_count": len(merged_records),
            }
    except urllib.error.HTTPError as exc:
        return {"synced": False, "reason": f"GitHub PUT falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub PUT erro: {exc}"}


def github_sync_rank_reactions_with_retry(records: list) -> dict:
    retries = max(1, int(to_number(get_env("GITHUB_SYNC_RETRIES", "3"), 3)))
    retry_delay = max(0.0, to_number(get_env("GITHUB_SYNC_RETRY_DELAY_SECONDS", "1.0"), 1.0))
    last_result = {"synced": False, "reason": "Sync nao executado."}

    for attempt in range(1, retries + 1):
        result = github_sync_rank_reactions(records)
        result["attempt"] = attempt
        result["max_attempts"] = retries
        if result.get("synced"):
            return result
        last_result = result
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    return last_result


def save_rank_reaction(
    month_key: str,
    rank_kind: str,
    target_name: str,
    emoji: str,
    reactor_id: str,
    reactor_nome: str,
) -> tuple[dict, str]:
    clean_month = str(month_key or "").strip()
    clean_kind = str(rank_kind or "").strip().lower()
    clean_target_name = str(target_name or "").strip()
    clean_target_name_key = normalize_name_key(clean_target_name)
    clean_emoji = str(emoji or "").strip()
    clean_reactor_id = str(reactor_id or "").strip()
    clean_reactor_nome = str(reactor_nome or "").strip()

    if not clean_month or not re.fullmatch(r"(\d{4})-(\d{2})", clean_month):
        return {}, "Mes invalido. Use YYYY-MM."
    if clean_kind not in RANK_REACTION_RANK_KINDS:
        return {}, "Tipo de ranking invalido."
    if not clean_target_name or not clean_target_name_key:
        return {}, "Nome de destino invalido."
    if clean_emoji and clean_emoji not in RANK_REACTION_ALLOWED_EMOJIS:
        return {}, "Emoji invalido para reacao."
    if not clean_reactor_id and not normalize_name_key(clean_reactor_nome):
        return {}, "Usuario sem identificacao valida para reagir."

    with RANK_REACTIONS_FILE_LOCK:
        records = read_rank_reactions()
        existing_index = -1
        for index, item in enumerate(records):
            if not isinstance(item, dict):
                continue
            item_month = str(item.get("month_key", "")).strip()
            item_kind = str(item.get("rank_kind", "")).strip().lower()
            item_target = str(item.get("target_name_key", "")).strip()
            reactor = item.get("reactor", {}) or {}
            item_reactor_id = str(reactor.get("id", "")).strip()
            item_reactor_nome = str(reactor.get("nome", "")).strip()
            item_reactor_key = normalize_name_key(item_reactor_nome)
            current_reactor_key = normalize_name_key(clean_reactor_nome)
            if item_month != clean_month or item_kind != clean_kind or item_target != clean_target_name_key:
                continue
            same_reactor = False
            if clean_reactor_id and item_reactor_id:
                same_reactor = clean_reactor_id == item_reactor_id
            elif current_reactor_key and item_reactor_key:
                same_reactor = current_reactor_key == item_reactor_key
            if same_reactor:
                existing_index = index
                break

        if clean_emoji:
            now_iso = now_brazil().isoformat()
            record = {
                "id": records[existing_index].get("id") if existing_index >= 0 else uuid.uuid4().hex,
                "month_key": clean_month,
                "rank_kind": clean_kind,
                "target_name": clean_target_name,
                "target_name_key": clean_target_name_key,
                "emoji": clean_emoji,
                "reactor": {
                    "id": clean_reactor_id,
                    "nome": clean_reactor_nome,
                },
                "reacted_at_iso": now_iso,
            }
            if existing_index >= 0:
                records[existing_index] = record
            else:
                records.append(record)
        elif existing_index >= 0:
            records.pop(existing_index)

        write_rank_reactions(records)
        github_sync = github_sync_rank_reactions_with_retry(records)
        merged_records = github_sync.get("merged_records")
        if isinstance(merged_records, list):
            write_rank_reactions(merged_records)
            github_sync.pop("merged_records", None)
            records = merged_records

    return {"records": records, "github_sync": github_sync}, ""


def build_rank_reactions_payload(records: list, month_key: str, viewer_user_id: str) -> dict:
    grouped: dict[str, dict] = {}
    clean_month = str(month_key or "").strip()
    clean_viewer_id = str(viewer_user_id or "").strip()

    for item in records:
        if not isinstance(item, dict):
            continue
        item_month = str(item.get("month_key", "")).strip()
        if item_month != clean_month:
            continue

        rank_kind = str(item.get("rank_kind", "")).strip().lower()
        if rank_kind not in RANK_REACTION_RANK_KINDS:
            continue

        target_name = str(item.get("target_name", "")).strip()
        target_name_key = str(item.get("target_name_key", "")).strip() or normalize_name_key(target_name)
        emoji = str(item.get("emoji", "")).strip()
        reactor = item.get("reactor", {}) or {}
        reactor_id = str(reactor.get("id", "")).strip()
        reactor_nome = str(reactor.get("nome", "")).strip()
        reacted_at_iso = str(item.get("reacted_at_iso", "")).strip()
        if not target_name_key or not emoji:
            continue

        group_key = f"{rank_kind}|{target_name_key}"
        if group_key not in grouped:
            grouped[group_key] = {
                "rank_kind": rank_kind,
                "target_name": target_name or target_name_key,
                "target_name_key": target_name_key,
                "total_reacoes": 0,
                "my_reaction": "",
                "reacoes": [],
            }
        group = grouped[group_key]
        group["total_reacoes"] += 1
        if target_name:
            group["target_name"] = target_name
        reaction_item = {
            "emoji": emoji,
            "reactor_id": reactor_id,
            "reactor_nome": reactor_nome or "Usuario",
            "reacted_at_iso": reacted_at_iso,
        }
        group["reacoes"].append(reaction_item)
        if clean_viewer_id and reactor_id and reactor_id == clean_viewer_id:
            group["my_reaction"] = emoji

    for group in grouped.values():
        emoji_map: dict[str, dict] = {}
        for reaction in group["reacoes"]:
            emoji = reaction["emoji"]
            current = emoji_map.get(emoji)
            if not current:
                current = {"emoji": emoji, "total": 0, "reatores": []}
                emoji_map[emoji] = current
            current["total"] += 1
            current["reatores"].append(str(reaction.get("reactor_nome", "Usuario")))
        por_emoji = sorted(
            emoji_map.values(),
            key=lambda item: (-int(item.get("total", 0)), str(item.get("emoji", ""))),
        )
        for item in por_emoji:
            names = list(dict.fromkeys([str(name or "Usuario").strip() or "Usuario" for name in item.get("reatores", [])]))
            item["reatores"] = names
        group["por_emoji"] = por_emoji
        group["reacoes"] = sorted(
            group["reacoes"],
            key=lambda item: (str(item.get("reactor_nome", "")).lower(), str(item.get("emoji", ""))),
        )

    return {
        "ok": True,
        "month": clean_month,
        "items": sorted(
            grouped.values(),
            key=lambda item: (str(item.get("rank_kind", "")), str(item.get("target_name", "")).lower()),
        ),
    }


EMPLOYEE_PHONE_PATTERN = re.compile(r"^\(\d{2}\)\s9\s\d{4}\s-\s\d{4}$")
EMPLOYEE_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PROFILE_IMAGE_DATA_URL_PATTERN = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,([a-z0-9+/=\s]+)$", re.IGNORECASE)
ACCESS_TAG_ALIASES = {
    "administrador": "admin",
    "administradora": "admin",
    "admin": "admin",
    "developer": "developer",
    "desenvolvedor": "developer",
    "desenvolvedora": "developer",
    "dev": "developer",
}
KNOWN_ACCESS_TAGS = {"admin", "developer"}
ANALYTICS_ACCESS_TAGS = {"admin", "developer"}
MANAGE_USERS_ACCESS_TAGS = {"admin", "developer"}
DEVELOPER_ONLY_ACCESS_TAGS = {"developer"}
ANALYTICS_ADMIN_ROLE_PATTERN = re.compile(r"\b(admin|administrador|administradora)\b", re.IGNORECASE)
PROFILE_IMAGE_MAX_BYTES = 4 * 1024 * 1024
PROFILE_IMAGE_OUTPUT_SIZE = 256


def normalize_employee_phone_digits(value: str) -> str:
    return re.sub(r"\D", "", str(value or ""))


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_access_tag(value: object) -> str:
    normalized = normalize_name_key(str(value or ""))
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return ACCESS_TAG_ALIASES.get(normalized, normalized)


def parse_access_tags(value: object) -> list[str]:
    raw_items: list[object]
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif isinstance(value, str):
        raw_items = re.split(r"[,\n;|/]+", value)
    elif value is None:
        raw_items = []
    else:
        raw_items = [value]

    normalized_tags: list[str] = []
    for item in raw_items:
        normalized = normalize_access_tag(item)
        if normalized and normalized not in normalized_tags:
            normalized_tags.append(normalized)
    return normalized_tags


def normalize_selected_access_tags(value: object) -> list[str]:
    normalized_tags = parse_access_tags(value)
    selected: list[str] = []
    for item in normalized_tags:
        if item not in KNOWN_ACCESS_TAGS:
            continue
        if item == "developer":
            return ["developer"]
        if item not in selected:
            selected.append(item)
    return selected


def infer_access_tags_from_role(role_name: str) -> list[str]:
    role_value = str(role_name or "").strip()
    inferred: list[str] = []
    if ANALYTICS_ADMIN_ROLE_PATTERN.search(role_value):
        inferred.append("admin")
    return inferred


def normalize_profile_image_data_url(value: object) -> tuple[str, str]:
    raw_value = str(value or "").strip()
    if not raw_value:
        return "", ""

    match = PROFILE_IMAGE_DATA_URL_PATTERN.fullmatch(raw_value)
    if not match:
        return "", "Formato de imagem invalido. Envie PNG, JPG ou WEBP."

    mime_type = str(match.group(1) or "").lower()
    encoded = re.sub(r"\s+", "", str(match.group(2) or ""))
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception:
        return "", "Imagem de perfil invalida."

    if not image_bytes or len(image_bytes) > PROFILE_IMAGE_MAX_BYTES:
        return "", "A imagem de perfil excede o limite permitido."

    if not mime_type.startswith("image/"):
        return "", "Tipo de arquivo de imagem nao permitido."

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened_image:
            prepared = ImageOps.exif_transpose(opened_image).convert("RGB")
            fitted = ImageOps.fit(
                prepared,
                (PROFILE_IMAGE_OUTPUT_SIZE, PROFILE_IMAGE_OUTPUT_SIZE),
                method=Image.Resampling.LANCZOS,
            )
            output = io.BytesIO()
            try:
                fitted.save(output, format="JPEG", quality=84, optimize=True)
            except Exception:
                output = io.BytesIO()
                fitted.save(output, format="JPEG", quality=84)
    except Exception:
        return "", "Nao foi possivel processar a imagem de perfil."

    encoded_output = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_output}", ""


def extract_employee_access_tags(record: dict) -> list[str]:
    if not isinstance(record, dict):
        return []

    tags: list[str] = []
    for key in ("tags_acesso", "tags", "tag", "acessos", "role", "perfil"):
        for item in parse_access_tags(record.get(key)):
            if item not in tags:
                tags.append(item)

    for item in infer_access_tags_from_role(record.get("funcao", "")):
        if item not in tags:
            tags.append(item)

    return tags


def build_user_permissions(access_tags: list[str]) -> dict:
    normalized_tags = {normalize_access_tag(item) for item in access_tags if item}
    from permissoes_paginas import effective_page_access

    page_access = effective_page_access(normalized_tags)
    return {
        "analytics": bool(normalized_tags.intersection(ANALYTICS_ACCESS_TAGS)),
        "manage_users": bool(normalized_tags.intersection(MANAGE_USERS_ACCESS_TAGS)),
        "edit_users": bool(normalized_tags.intersection(DEVELOPER_ONLY_ACCESS_TAGS)),
        "page_access": page_access,
    }


def normalize_profile_update_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        payload = {}

    return {
        "nome": normalize_spaces(payload.get("nome", "")),
        "email": str(payload.get("email", "") or "").strip().lower(),
        "numero_celular": normalize_spaces(payload.get("numero_celular", "")),
        "foto_perfil_data_url": str(payload.get("foto_perfil_data_url", "") or "").strip(),
        "remover_foto": to_bool(payload.get("remover_foto"), False),
        "data_nascimento": str(
            payload.get("data_nascimento")
            or payload.get("data_nascimento_iso")
            or ""
        ).strip(),
        "mostrar_aniversario": to_bool(payload.get("mostrar_aniversario"), False),
    }


def normalize_employee_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        payload = {}
    return {
        "nome": normalize_spaces(payload.get("nome", "")),
        "funcao": normalize_spaces(payload.get("funcao", "")),
        "numero_celular": normalize_spaces(payload.get("numero_celular", "")),
        "email": str(payload.get("email", "") or "").strip().lower(),
        "senha": str(payload.get("senha", "") or ""),
        "tags_acesso": normalize_selected_access_tags(
            payload.get("tags_acesso", payload.get("nivel_acesso"))
        ),
        "data_nascimento": str(
            payload.get("data_nascimento")
            or payload.get("data_nascimento_iso")
            or ""
        ).strip(),
        "mostrar_aniversario": to_bool(payload.get("mostrar_aniversario"), False),
    }


def normalize_employee_edit_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        payload = {}
    return {
        "nome": normalize_spaces(payload.get("nome", "")),
        "funcao": normalize_spaces(payload.get("funcao", "")),
        "numero_celular": normalize_spaces(payload.get("numero_celular", "")),
        "email": str(payload.get("email", "") or "").strip().lower(),
        "senha": str(payload.get("senha", "") or ""),
        "tags_acesso": normalize_selected_access_tags(
            payload.get("tags_acesso", payload.get("nivel_acesso"))
        ),
    }


def parse_birth_date_iso(value: object) -> tuple[str, str | None]:
    raw = str(value or "").strip()
    if not raw:
        return "", None
    parsed = None
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(raw, date_format).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return "", "Data de nascimento inválida. Use o formato DD/MM/AAAA."
    today = now_brazil().date()
    if parsed > today:
        return "", "Data de nascimento não pode ser no futuro."
    if parsed.year < 1900:
        return "", "Informe uma data de nascimento a partir de 1900."
    return parsed.isoformat(), None


def validate_employee_payload(payload: dict) -> tuple[bool, str]:
    nome = payload.get("nome", "")
    funcao = payload.get("funcao", "")
    numero_celular = payload.get("numero_celular", "")
    email = payload.get("email", "")
    senha = payload.get("senha", "")

    if len(nome) < 3:
        return False, "Informe um nome valido com pelo menos 3 caracteres."
    if len(funcao) < 2:
        return False, "Informe uma funcao valida."
    if not EMPLOYEE_PHONE_PATTERN.fullmatch(numero_celular):
        return False, "Numero de celular invalido. Use o formato (xx) 9 0000 - 0000."

    phone_digits = normalize_employee_phone_digits(numero_celular)
    if len(phone_digits) != 11 or phone_digits[2] != "9":
        return False, "Numero de celular invalido."

    if email and not EMPLOYEE_EMAIL_PATTERN.fullmatch(email):
        return False, "Email invalido."
    birth_iso, birth_error = parse_birth_date_iso(payload.get("data_nascimento"))
    if birth_error:
        return False, birth_error
    if len(senha) < 6:
        return False, "A senha deve ter pelo menos 6 caracteres."

    return True, ""


def validate_employee_edit_payload(payload: dict) -> tuple[bool, str]:
    nome = payload.get("nome", "")
    funcao = payload.get("funcao", "")
    numero_celular = payload.get("numero_celular", "")
    email = payload.get("email", "")
    senha = str(payload.get("senha", "") or "")

    if len(nome) < 3:
        return False, "Informe um nome valido com pelo menos 3 caracteres."
    if len(funcao) < 2:
        return False, "Informe uma funcao valida."
    if not EMPLOYEE_PHONE_PATTERN.fullmatch(numero_celular):
        return False, "Numero de celular invalido. Use o formato (xx) 9 0000 - 0000."

    phone_digits = normalize_employee_phone_digits(numero_celular)
    if len(phone_digits) != 11 or phone_digits[2] != "9":
        return False, "Numero de celular invalido."

    if email and not EMPLOYEE_EMAIL_PATTERN.fullmatch(email):
        return False, "Email invalido."
    if senha and len(senha) < 6:
        return False, "A senha deve ter pelo menos 6 caracteres."

    return True, ""


def read_employees() -> list:
    if not EMPLOYEES_FILE.exists():
        return []
    try:
        data = json.loads(EMPLOYEES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def write_json_atomic(target: Path, payload: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_file = target.with_suffix(".tmp")
    temp_file.write_text(payload, encoding="utf-8")
    try:
        temp_file.replace(target)
    finally:
        if temp_file.exists():
            temp_file.unlink()


def write_employees_backup(records: list) -> None:
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    write_json_atomic(EMPLOYEES_BACKUP_FILE, payload)


def write_employees(records: list) -> None:
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    temp_file = EMPLOYEES_FILE.with_suffix(".tmp")
    temp_file.write_text(payload, encoding="utf-8")
    backup_path = None
    try:
        backup_path = backup_employees()
    except Exception:
        backup_path = None

    try:
        temp_file.replace(EMPLOYEES_FILE)
    except Exception:
        if backup_path and backup_path.exists():
            try:
                shutil.copy2(backup_path, EMPLOYEES_FILE)
            except Exception:
                pass
        raise
    finally:
        if temp_file.exists():
            temp_file.unlink()
    write_employees_backup(records)


def employee_record_key(record: dict) -> str:
    if not isinstance(record, dict):
        return ""

    record_id = str(record.get("id", "")).strip()
    if record_id:
        return f"id:{record_id}"

    phone_value = normalize_employee_phone_digits(record.get("numero_normalizado") or record.get("numero_celular") or "")
    if phone_value:
        return f"phone:{phone_value}"

    email_value = str(record.get("email", "") or "").strip().lower()
    if email_value:
        return f"email:{email_value}"

    try:
        return "raw:" + json.dumps(record, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return "raw:" + str(record)


def merge_employee_lists(*sources: list) -> list:
    merged: list = []
    seen: set[str] = set()

    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            key = employee_record_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)

    return merged


def github_sync_json_file(records: list, repo_path: str) -> dict:
    token = get_env("GITHUB_TOKEN")
    if not token:
        return {"synced": False, "reason": "GITHUB_TOKEN nao configurado"}

    normalized_path = str(repo_path or "").strip()
    if not normalized_path:
        return {"synced": False, "reason": "Caminho do arquivo GitHub nao configurado"}

    repo = get_env("GITHUB_REPO", "PopularAtacarejo/SuperPOP")
    branch = get_env("GITHUB_BRANCH", "main")
    api_base = f"https://api.github.com/repos/{repo}/contents/{normalized_path}"
    get_url = f"{api_base}?ref={urllib.parse.quote(branch)}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "superpop-backend",
    }

    sha = None
    remote_records: list = []
    try:
        req_get = urllib.request.Request(get_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_get, timeout=20) as resp:
            current = json.loads(resp.read().decode("utf-8"))
            sha = current.get("sha")
            encoded_content = str(current.get("content") or "").strip()
            if encoded_content:
                try:
                    decoded_content = base64.b64decode(encoded_content).decode("utf-8")
                    loaded_remote = json.loads(decoded_content)
                    if isinstance(loaded_remote, list):
                        remote_records = loaded_remote
                except Exception:  # noqa: BLE001
                    remote_records = []
            if not remote_records:
                download_url = str(current.get("download_url") or "").strip()
                if download_url:
                    try:
                        req_download = urllib.request.Request(download_url, headers=headers, method="GET")
                        with urllib.request.urlopen(req_download, timeout=20) as download_resp:
                            download_payload = json.loads(download_resp.read().decode("utf-8"))
                            if isinstance(download_payload, list):
                                remote_records = download_payload
                    except Exception:  # noqa: BLE001
                        remote_records = []
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return {"synced": False, "reason": f"GitHub GET falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub GET erro: {exc}"}

    synced_records = [item for item in records if isinstance(item, dict)]
    content = base64.b64encode(json.dumps(synced_records, ensure_ascii=False, indent=2).encode("utf-8")).decode("utf-8")
    utc_now = datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    payload = {
        "message": f"Atualiza {normalized_path} ({utc_now})",
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
                "merged_records": synced_records,
                "remote_count": len(remote_records),
                "sent_count": len(records),
                "merged_count": len(synced_records),
            }
    except urllib.error.HTTPError as exc:
        return {"synced": False, "reason": f"GitHub PUT falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub PUT erro: {exc}"}


def github_sync_employees(records: list) -> dict:
    file_path = get_env("GITHUB_EMPLOYEES_FILE_PATH", "Funcioinarios.json")
    return github_sync_json_file(records, file_path)


def github_sync_employees_backup(records: list) -> dict:
    backup_path = get_env("GITHUB_EMPLOYEES_BACKUP_FILE_PATH", "Backup/Funcionarios-Backup.json")
    return github_sync_json_file(records, backup_path)


def github_sync_with_retry(records: list, sync_function) -> dict:
    retries = max(1, int(to_number(get_env("GITHUB_SYNC_RETRIES", "3"), 3)))
    retry_delay = max(0.0, to_number(get_env("GITHUB_SYNC_RETRY_DELAY_SECONDS", "1.0"), 1.0))
    last_result = {"synced": False, "reason": "Sync nao executado."}

    for attempt in range(1, retries + 1):
        result = sync_function(records)
        result["attempt"] = attempt
        result["max_attempts"] = retries
        if result.get("synced"):
            return result
        last_result = result
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    return last_result


def github_sync_employees_with_retry(records: list) -> dict:
    return github_sync_with_retry(records, github_sync_employees)


def github_sync_employees_backup_with_retry(records: list) -> dict:
    return github_sync_with_retry(records, github_sync_employees_backup)


def append_employee_record(record: dict) -> dict:
    with EMPLOYEES_FILE_LOCK:
        records = merge_employee_lists(read_employees(), [record])
        write_employees(records)
        github_sync = github_sync_employees_with_retry(records)
        merged_records = github_sync.get("merged_records")
        final_records = records
        if isinstance(merged_records, list):
            write_employees(merged_records)
            final_records = merged_records
            github_sync.pop("merged_records", None)
        github_sync_employees_backup_with_retry(final_records)
    return github_sync


def save_employee_records(records: list) -> tuple[list, dict]:
    write_employees(records)
    github_sync = github_sync_employees_with_retry(records)
    merged_records = github_sync.get("merged_records")
    final_records = records
    if isinstance(merged_records, list):
        write_employees(merged_records)
        final_records = merged_records
        github_sync.pop("merged_records", None)
    github_sync_employees_backup_with_retry(final_records)
    return final_records, github_sync


def update_employee_record(employee_id: str, updater) -> tuple[dict | None, dict]:
    wanted_id = str(employee_id or "").strip()
    if not wanted_id:
        return None, {"synced": False, "reason": "Funcionario nao identificado."}

    with EMPLOYEES_FILE_LOCK:
        records = read_employees()
        updated_employee = None
        for index, item in enumerate(records):
            if str(item.get("id", "")).strip() != wanted_id:
                continue
            current = copy.deepcopy(item)
            updated = updater(current)
            if not isinstance(updated, dict):
                return None, {"synced": False, "reason": "Atualizacao invalida."}
            records[index] = updated
            updated_employee = updated
            break

        if not updated_employee:
            return None, {"synced": False, "reason": "Funcionario nao encontrado."}

        persisted_records, github_sync = save_employee_records(records)
        persisted_employee = find_employee_by_id(persisted_records, wanted_id) or updated_employee
        return persisted_employee, github_sync


def update_employee_record_with_records(employee_id: str, updater) -> tuple[dict | None, dict]:
    wanted_id = str(employee_id or "").strip()
    if not wanted_id:
        return None, {"synced": False, "reason": "Funcionario nao identificado."}

    with EMPLOYEES_FILE_LOCK:
        records = read_employees()
        updated_employee = None

        def apply_update() -> bool:
            nonlocal updated_employee, records
            for index, item in enumerate(records):
                if str(item.get("id", "")).strip() != wanted_id:
                    continue
                current = copy.deepcopy(item)
                updated = updater(current, records)
                if not isinstance(updated, dict):
                    raise ValueError("Atualizacao invalida.")
                records[index] = updated
                updated_employee = updated
                return True
            return False

        try:
            found = apply_update()
        except ValueError as exc:
            return None, {"synced": False, "reason": str(exc) or "Atualizacao invalida."}

        if not found:
            refreshed_records, _refresh_error = refresh_local_employees_from_remote(records)
            records = refreshed_records
            try:
                found = apply_update()
            except ValueError as exc:
                return None, {"synced": False, "reason": str(exc) or "Atualizacao invalida."}

        if not found or not updated_employee:
            return None, {"synced": False, "reason": "Funcionario nao encontrado."}

        persisted_records, github_sync = save_employee_records(records)
        persisted_employee = find_employee_by_id(persisted_records, wanted_id) or updated_employee
        return persisted_employee, github_sync


def delete_employee_record(employee_id: str) -> tuple[dict | None, dict]:
    wanted_id = str(employee_id or "").strip()
    if not wanted_id:
        return None, {"synced": False, "reason": "Funcionario nao identificado."}

    with EMPLOYEES_FILE_LOCK:
        records = read_employees()

        def split_records(source_records: list) -> tuple[list, dict | None]:
            kept: list = []
            removed: dict | None = None
            for item in source_records:
                if not isinstance(item, dict):
                    kept.append(item)
                    continue
                if removed is None and str(item.get("id", "")).strip() == wanted_id:
                    removed = copy.deepcopy(item)
                    continue
                kept.append(item)
            return kept, removed

        remaining_records, deleted_employee = split_records(records)
        if not deleted_employee:
            refreshed_records, _refresh_error = refresh_local_employees_from_remote(records)
            remaining_records, deleted_employee = split_records(refreshed_records)

        if not deleted_employee:
            return None, {"synced": False, "reason": "Funcionario nao encontrado."}

        _persisted_records, github_sync = save_employee_records(remaining_records)
        return deleted_employee, github_sync


def build_password_hash(password: str) -> tuple[str, str, int]:
    iterations = max(120000, int(to_number(get_env("PASSWORD_HASH_ITERATIONS", "180000"), 180000)))
    salt_bytes = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
    return salt_bytes.hex(), digest.hex(), iterations


def get_password_reset_secret() -> str:
    return get_env("PASSWORD_RESET_SECRET") or get_env("FLASK_SECRET_KEY", "superpop-dev-secret")


def get_password_reset_expiration_minutes() -> int:
    return max(5, int(to_number(get_env("PASSWORD_RESET_EXPIRATION_MINUTES", "30"), 30)))


def build_password_reset_token(employee: dict) -> str:
    employee_id = str(employee.get("id", "")).strip()
    email = str(employee.get("email", "")).strip().lower()
    expires_at = int(time.time()) + (get_password_reset_expiration_minutes() * 60)
    payload_json = json.dumps(
        {"employee_id": employee_id, "email": email, "exp": expires_at},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("ascii").rstrip("=")
    signature = hmac.new(
        get_password_reset_secret().encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload_b64}.{signature}"


def parse_password_reset_token(token: str) -> tuple[dict | None, str]:
    token_value = str(token or "").strip()
    if "." not in token_value:
        return None, "Token invalido."

    payload_b64, signature = token_value.rsplit(".", 1)
    expected_signature = hmac.new(
        get_password_reset_secret().encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        return None, "Token invalido."

    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload_raw = base64.urlsafe_b64decode((payload_b64 + padding).encode("ascii")).decode("utf-8")
        payload = json.loads(payload_raw)
    except Exception:
        return None, "Token invalido."

    if not isinstance(payload, dict):
        return None, "Token invalido."

    employee_id = str(payload.get("employee_id", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    expires_at = int(to_number(payload.get("exp"), 0))
    if not employee_id or not email or expires_at <= 0:
        return None, "Token invalido."
    if expires_at < int(time.time()):
        return None, "Token expirado."

    return {"employee_id": employee_id, "email": email, "exp": expires_at}, ""


def build_password_reset_url(token: str) -> str:
    clean_token = urllib.parse.quote(str(token or "").strip(), safe="")
    return build_frontend_url(f"login.html?reset_token={clean_token}")


def get_public_base_url() -> str:
    base = get_env("PUBLIC_BASE_URL", "http://localhost:5000") or "http://localhost:5000"
    return str(base).rstrip("/")


def build_public_backend_url(path: str = "") -> str:
    clean_path = str(path or "").lstrip("/")
    base = get_public_base_url()
    return f"{base}/{clean_path}" if clean_path else base


def get_smtp_settings() -> dict:
    port_default = 465 if to_bool(get_env("SMTP_USE_SSL", "0"), False) else 587
    return {
        "host": get_env("SMTP_HOST"),
        "port": max(1, int(to_number(get_env("SMTP_PORT", str(port_default)), port_default))),
        "username": get_env("SMTP_USERNAME"),
        "password": get_env("SMTP_PASSWORD"),
        "from_email": get_env("SMTP_FROM_EMAIL"),
        "from_name": get_env("SMTP_FROM_NAME", "SuperPop"),
        "use_tls": to_bool(get_env("SMTP_USE_TLS", "1"), True),
        "use_ssl": to_bool(get_env("SMTP_USE_SSL", "0"), False),
    }


def get_brevo_api_settings() -> dict:
    return {
        "api_key": get_env("BREVO_API_KEY"),
        "base_url": get_env("BREVO_API_BASE_URL", "https://api.brevo.com/v3").rstrip("/"),
        "sandbox": to_bool(get_env("BREVO_SANDBOX_MODE", "0"), False),
    }


def get_microsoft_oauth_settings() -> dict:
    redirect_default = build_public_backend_url("api/system/microsoft-oauth/callback")
    return {
        "tenant": get_env("MICROSOFT_OAUTH_TENANT", "consumers") or "consumers",
        "client_id": get_env("MICROSOFT_OAUTH_CLIENT_ID"),
        "client_secret": get_env("MICROSOFT_OAUTH_CLIENT_SECRET"),
        "redirect_uri": get_env("MICROSOFT_OAUTH_REDIRECT_URI", redirect_default) or redirect_default,
        "scope": get_env("MICROSOFT_OAUTH_SCOPE", "https://outlook.office.com/SMTP.Send offline_access"),
    }


def load_microsoft_oauth_token_store() -> dict:
    if not MICROSOFT_OAUTH_TOKEN_FILE.exists():
        return {}
    try:
        loaded = json.loads(MICROSOFT_OAUTH_TOKEN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_microsoft_oauth_token_store(payload: dict) -> None:
    MICROSOFT_OAUTH_TOKEN_FILE.write_text(
        json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def get_microsoft_refresh_token() -> str:
    from_env = get_env("MICROSOFT_OAUTH_REFRESH_TOKEN")
    if from_env:
        return from_env
    stored = load_microsoft_oauth_token_store()
    return str(stored.get("refresh_token", "") or "").strip()


def build_microsoft_authorize_url(state_token: str) -> str:
    settings = get_microsoft_oauth_settings()
    authorize_base = f"https://login.microsoftonline.com/{urllib.parse.quote(settings['tenant'])}/oauth2/v2.0/authorize"
    query = urllib.parse.urlencode(
        {
            "client_id": settings["client_id"],
            "response_type": "code",
            "redirect_uri": settings["redirect_uri"],
            "response_mode": "query",
            "scope": settings["scope"],
            "state": state_token,
            "prompt": "select_account",
        }
    )
    return f"{authorize_base}?{query}"


def request_microsoft_oauth_token(form_payload: dict) -> tuple[dict | None, str]:
    settings = get_microsoft_oauth_settings()
    token_url = f"https://login.microsoftonline.com/{urllib.parse.quote(settings['tenant'])}/oauth2/v2.0/token"
    payload = {key: str(value) for key, value in form_payload.items() if value is not None and value != ""}
    encoded = urllib.parse.urlencode(payload).encode("utf-8")
    request_obj = urllib.request.Request(
        token_url,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=30) as response:
            raw = response.read().decode("utf-8")
            loaded = json.loads(raw)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
            loaded = json.loads(body)
            description = str(loaded.get("error_description") or loaded.get("error") or body).strip()
        except Exception:
            description = str(exc)
        return None, description
    except Exception as exc:
        return None, str(exc)

    if not isinstance(loaded, dict):
        return None, "Resposta invalida do Microsoft OAuth."

    if loaded.get("error"):
        return None, str(loaded.get("error_description") or loaded.get("error") or "Falha no Microsoft OAuth.").strip()

    return loaded, ""


def exchange_microsoft_code_for_refresh_token(code: str) -> tuple[dict | None, str]:
    settings = get_microsoft_oauth_settings()
    payload = {
        "client_id": settings["client_id"],
        "scope": settings["scope"],
        "code": str(code or "").strip(),
        "redirect_uri": settings["redirect_uri"],
        "grant_type": "authorization_code",
    }
    if settings["client_secret"]:
        payload["client_secret"] = settings["client_secret"]
    return request_microsoft_oauth_token(payload)


def refresh_microsoft_smtp_access_token() -> tuple[str, str]:
    settings = get_microsoft_oauth_settings()
    refresh_token = get_microsoft_refresh_token()
    if not settings["client_id"]:
        return "", "MICROSOFT_OAUTH_CLIENT_ID nao configurado."
    if not refresh_token:
        return "", "Refresh token Microsoft OAuth nao configurado."

    payload = {
        "client_id": settings["client_id"],
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": settings["scope"],
    }
    if settings["client_secret"]:
        payload["client_secret"] = settings["client_secret"]

    token_response, token_error = request_microsoft_oauth_token(payload)
    if not token_response:
        return "", token_error or "Falha ao atualizar token Microsoft OAuth."

    new_refresh_token = str(token_response.get("refresh_token", "") or "").strip()
    if new_refresh_token:
        stored = load_microsoft_oauth_token_store()
        stored.update(
            {
                "refresh_token": new_refresh_token,
                "updated_at_iso": now_brazil().isoformat(),
                "scope": str(token_response.get("scope", settings["scope"]) or settings["scope"]),
                "token_type": str(token_response.get("token_type", "Bearer") or "Bearer"),
            }
        )
        save_microsoft_oauth_token_store(stored)

    access_token = str(token_response.get("access_token", "") or "").strip()
    if not access_token:
        return "", "Microsoft OAuth nao retornou access token."

    return access_token, ""


def is_microsoft_oauth_ready() -> bool:
    settings = get_microsoft_oauth_settings()
    return bool(settings["client_id"] and settings["redirect_uri"] and get_microsoft_refresh_token())


def build_smtp_xoauth2_string(username: str, access_token: str) -> str:
    raw = f"user={username}\x01auth=Bearer {access_token}\x01\x01"
    return base64.b64encode(raw.encode("utf-8")).decode("ascii")


def is_smtp_configured() -> bool:
    smtp = get_smtp_settings()
    return bool(smtp["host"] and smtp["port"] and smtp["from_email"])


def is_brevo_api_configured() -> bool:
    settings = get_brevo_api_settings()
    smtp = get_smtp_settings()
    return bool(settings["api_key"] and smtp["from_email"])


def get_email_delivery_mode() -> str:
    if is_brevo_api_configured():
        return "brevo_api"
    if is_microsoft_oauth_ready():
        return "microsoft_oauth"
    return "basic_smtp"


def mask_email_for_log(email: str) -> str:
    normalized = str(email or "").strip().lower()
    if not normalized or "@" not in normalized:
        return normalized or "-"
    local_part, domain = normalized.split("@", 1)
    if len(local_part) <= 2:
        masked_local = local_part[:1] + "*"
    else:
        masked_local = local_part[:2] + "*" * max(1, len(local_part) - 2)
    return f"{masked_local}@{domain}"


def send_email_via_brevo_api(message: EmailMessage, html_content: str, text_content: str) -> tuple[bool, str]:
    settings = get_brevo_api_settings()
    smtp = get_smtp_settings()
    if not is_brevo_api_configured():
        return False, "Brevo API nao configurada."

    recipients: list[dict[str, str]] = []
    for _display_name, email_addr in getaddresses(message.get_all("To", [])):
        recipients.append({"email": email_addr})
    if not recipients:
        to_header = str(message.get("To", "") or "").strip()
        if to_header:
            recipients.append({"email": to_header})
    if not recipients:
        return False, "Destinatario do email nao informado."

    payload = {
        "sender": {
            "name": str(smtp["from_name"] or "SuperPop").strip() or "SuperPop",
            "email": str(smtp["from_email"]).strip(),
        },
        "to": recipients,
        "subject": str(message.get("Subject", "") or "").strip(),
        "htmlContent": html_content,
        "textContent": text_content,
    }
    headers = {
        "accept": "application/json",
        "api-key": settings["api_key"],
        "content-type": "application/json",
    }
    if settings["sandbox"]:
        payload["headers"] = {"X-Sib-Sandbox": "drop"}

    request_obj = urllib.request.Request(
        f"{settings['base_url']}/smtp/email",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=30) as response:
            status = int(response.status)
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = str(exc)
        return False, f"Falha ao enviar email pela Brevo API ({exc.code}): {body}"
    except Exception as exc:
        return False, f"Falha ao enviar email pela Brevo API: {exc}"

    if status not in {200, 201, 202}:
        return False, f"Falha ao enviar email pela Brevo API (HTTP {status}): {body}"

    try:
        parsed = json.loads(body) if body else {}
    except Exception:
        parsed = {}

    message_id = ""
    if isinstance(parsed, dict):
        message_id = str(parsed.get("messageId", "") or parsed.get("message_id", "") or "").strip()

    return True, (f"Brevo accepted message_id={message_id}" if message_id else "Brevo accepted")


def send_email_via_basic_smtp(message: EmailMessage) -> tuple[bool, str]:
    smtp = get_smtp_settings()
    if not is_smtp_configured():
        return False, "SMTP nao configurado."

    try:
        if smtp["use_ssl"]:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(str(smtp["host"]), int(smtp["port"]), context=context, timeout=25) as server:
                if smtp["username"] and smtp["password"]:
                    server.login(str(smtp["username"]), str(smtp["password"]))
                server.send_message(message)
        else:
            with smtplib.SMTP(str(smtp["host"]), int(smtp["port"]), timeout=25) as server:
                server.ehlo()
                if smtp["use_tls"]:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
                    server.ehlo()
                if smtp["username"] and smtp["password"]:
                    server.login(str(smtp["username"]), str(smtp["password"]))
                server.send_message(message)
    except Exception as exc:
        return False, f"Falha ao enviar email: {exc}"

    return True, "ok"


def send_email_via_microsoft_oauth(message: EmailMessage) -> tuple[bool, str]:
    smtp = get_smtp_settings()
    if not smtp["host"] or not smtp["port"] or not smtp["from_email"]:
        return False, "SMTP nao configurado."

    access_token, token_error = refresh_microsoft_smtp_access_token()
    if not access_token:
        return False, token_error

    auth_string = build_smtp_xoauth2_string(str(smtp["from_email"]), access_token)

    try:
        with smtplib.SMTP(str(smtp["host"]), int(smtp["port"]), timeout=25) as server:
            server.ehlo()
            if smtp["use_tls"]:
                context = ssl.create_default_context()
                server.starttls(context=context)
                server.ehlo()
            code, response = server.docmd("AUTH", "XOAUTH2 " + auth_string)
            if int(code) != 235:
                return False, f"Falha ao autenticar no SMTP Microsoft OAuth ({code}): {response!r}"
            server.send_message(message)
    except Exception as exc:
        return False, f"Falha ao enviar email: {exc}"

    return True, "ok"


def send_email_with_fallback(message: EmailMessage, html_content: str, text_content: str) -> tuple[bool, str]:
    delivery_attempts = []

    if is_brevo_api_configured():
        delivery_attempts.append(("brevo_api", lambda: send_email_via_brevo_api(message, html_content, text_content)))

    if is_microsoft_oauth_ready():
        delivery_attempts.append(("microsoft_oauth", lambda: send_email_via_microsoft_oauth(message)))

    delivery_attempts.append(("basic_smtp", lambda: send_email_via_basic_smtp(message)))

    errors: list[str] = []
    for provider_name, attempt in delivery_attempts:
        sent, provider_status = attempt()
        if sent:
            if errors:
                return True, f"{provider_name} ok after fallback ({provider_status})"
            return True, provider_status
        errors.append(f"{provider_name}: {provider_status}")

    return False, " | ".join(errors)


def send_password_reset_email(employee: dict, reset_url: str) -> tuple[bool, str]:
    smtp = get_smtp_settings()
    if not is_smtp_configured():
        return False, "SMTP nao configurado."

    employee_name = str(employee.get("nome", "")).strip() or "colaborador"
    recipient_email = str(employee.get("email", "")).strip().lower()
    if not recipient_email:
        return False, "Funcionario sem email cadastrado."

    message = EmailMessage()
    from_name = str(smtp["from_name"] or "SuperPop").strip()
    message["Subject"] = "Recuperacao de senha - SuperPop"
    message["From"] = f"{from_name} <{smtp['from_email']}>" if from_name else str(smtp["from_email"])
    message["To"] = recipient_email
    text_content = "\n".join(
        [
            f"Ola, {employee_name}.",
            "",
            "Recebemos um pedido para redefinir sua senha no SuperPop.",
            "Use o link abaixo para cadastrar uma nova senha:",
            reset_url,
            "",
            f"Esse link expira em {get_password_reset_expiration_minutes()} minutos.",
            "Se voce nao solicitou essa alteracao, ignore este email.",
        ]
    )
    html_content = (
        "<!DOCTYPE html><html><body style=\"font-family:Arial,sans-serif;color:#0f172a\">"
        f"<p>Ola, <strong>{employee_name}</strong>.</p>"
        "<p>Recebemos um pedido para redefinir sua senha no SuperPop.</p>"
        f"<p><a href=\"{reset_url}\" style=\"display:inline-block;padding:12px 18px;background:#E63946;color:#ffffff;text-decoration:none;border-radius:10px;font-weight:bold\">Redefinir senha</a></p>"
        f"<p>Ou use este link: <br><a href=\"{reset_url}\">{reset_url}</a></p>"
        f"<p>Esse link expira em {get_password_reset_expiration_minutes()} minutos.</p>"
        "<p>Se voce nao solicitou essa alteracao, ignore este email.</p>"
        "</body></html>"
    )
    message.set_content(text_content)
    message.add_alternative(html_content, subtype="html")

    return send_email_with_fallback(message, html_content, text_content)


def build_employee_record(payload: dict, created_iso: str) -> dict:
    salt_hex, hash_hex, iterations = build_password_hash(payload.get("senha", ""))
    phone_digits = normalize_employee_phone_digits(payload.get("numero_celular", ""))
    return {
        "id": uuid.uuid4().hex,
        "nome": repair_mojibake_text(payload.get("nome", "")).strip(),
        "funcao": repair_mojibake_text(payload.get("funcao", "")).strip(),
        "numero_celular": payload.get("numero_celular", ""),
        "numero_normalizado": phone_digits,
        "email": payload.get("email", ""),
        "tags_acesso": list(payload.get("tags_acesso", [])),
        "senha": {
            "algoritmo": "pbkdf2_sha256",
            "salt": salt_hex,
            "hash": hash_hex,
            "iteracoes": iterations,
        },
        "foto_perfil_data_url": "",
        "data_cadastro_iso": created_iso,
        "data_nascimento_iso": str(payload.get("data_nascimento_iso", "")).strip(),
        "mostrar_aniversario": bool(payload.get("mostrar_aniversario")),
        "pre_cadastro": bool(payload.get("pre_cadastro")),
    }


def build_employee_public_record(record: dict) -> dict:
    tags_acesso = extract_employee_access_tags(record)
    return {
        "id": str(record.get("id", "")).strip(),
        "nome": repair_mojibake_text(record.get("nome", "")).strip(),
        "funcao": repair_mojibake_text(record.get("funcao", "")).strip(),
        "numero_celular": str(record.get("numero_celular", "")).strip(),
        "email": str(record.get("email", "")).strip(),
        "foto_perfil_data_url": str(record.get("foto_perfil_data_url", "")).strip(),
        "data_cadastro_iso": str(record.get("data_cadastro_iso", "")).strip(),
        "data_nascimento_iso": str(record.get("data_nascimento_iso", "")).strip(),
        "mostrar_aniversario": bool(record.get("mostrar_aniversario")),
        "pre_cadastro": bool(record.get("pre_cadastro")),
        "tags_acesso": tags_acesso,
        "permissoes": build_user_permissions(tags_acesso),
    }


def find_duplicate_employee(records: list, phone_digits: str, email: str) -> tuple[bool, str]:
    email_normalized = str(email or "").strip().lower()
    for item in records:
        if not isinstance(item, dict):
            continue
        existing_phone = normalize_employee_phone_digits(item.get("numero_normalizado") or item.get("numero_celular") or "")
        if phone_digits and existing_phone and existing_phone == phone_digits:
            return True, "Ja existe cadastro com esse numero de celular."
        existing_email = str(item.get("email", "") or "").strip().lower()
        if email_normalized and existing_email and existing_email == email_normalized:
            return True, "Ja existe cadastro com esse email."
    return False, ""


def find_duplicate_employee_for_update(records: list, employee_id: str, phone_digits: str, email: str) -> tuple[bool, str]:
    wanted_id = str(employee_id or "").strip()
    email_normalized = str(email or "").strip().lower()
    for item in records:
        if not isinstance(item, dict):
            continue
        if str(item.get("id", "")).strip() == wanted_id:
            continue
        existing_phone = normalize_employee_phone_digits(item.get("numero_normalizado") or item.get("numero_celular") or "")
        if phone_digits and existing_phone and existing_phone == phone_digits:
            return True, "Ja existe outro cadastro com esse numero de celular."
        existing_email = str(item.get("email", "") or "").strip().lower()
        if email_normalized and existing_email and existing_email == email_normalized:
            return True, "Ja existe outro cadastro com esse email."
    return False, ""


def find_employee_by_phone(records: list, phone_digits: str) -> dict | None:
    for item in records:
        if not isinstance(item, dict):
            continue
        existing_phone = normalize_employee_phone_digits(item.get("numero_normalizado") or item.get("numero_celular") or "")
        if phone_digits and existing_phone == phone_digits:
            return item
    return None


def find_employee_by_email(records: list, email: str) -> dict | None:
    wanted_email = str(email or "").strip().lower()
    if not wanted_email:
        return None
    for item in records:
        if not isinstance(item, dict):
            continue
        existing_email = str(item.get("email", "") or "").strip().lower()
        if existing_email and existing_email == wanted_email:
            return item
    return None


def find_employee_by_id(records: list, employee_id: str) -> dict | None:
    wanted_id = str(employee_id or "").strip()
    if not wanted_id:
        return None
    for item in records:
        if not isinstance(item, dict):
            continue
        if str(item.get("id", "")).strip() == wanted_id:
            return item
    return None


def find_employee_by_phone_or_name(records: list, phone_value: str = "", name_value: str = "") -> dict | None:
    wanted_phone = normalize_whatsapp_number(str(phone_value or "").strip())
    wanted_name_key = normalize_name_key(str(name_value or "").strip())
    fallback_by_name = None

    for item in records:
        if not isinstance(item, dict):
            continue

        public_record = build_employee_public_record(item)
        employee_phone = normalize_whatsapp_number(public_record.get("numero_celular", ""))
        if wanted_phone and employee_phone and employee_phone == wanted_phone:
            return item

        employee_name_key = normalize_name_key(public_record.get("nome", ""))
        if not fallback_by_name and wanted_name_key and employee_name_key and employee_name_key == wanted_name_key:
            fallback_by_name = item

    return fallback_by_name


def enrich_superpop_payload_with_employees(payload: dict, auth_user_id: str, employees: list) -> dict:
    if not isinstance(payload, dict):
        return normalize_payload({})

    enriched = dict(payload)
    sender_employee = None
    if auth_user_id:
        sender_employee = find_employee_by_id(employees, auth_user_id)
    if not sender_employee:
        sender_employee = find_employee_by_phone_or_name(
            employees,
            phone_value=enriched.get("numero_reconhecido_por", ""),
            name_value=enriched.get("reconhecido_por", ""),
        )
    if sender_employee:
        sender_public = build_employee_public_record(sender_employee)
        sender_name = str(sender_public.get("nome", "")).strip()
        sender_role = str(sender_public.get("funcao", "")).strip()
        sender_phone = str(sender_public.get("numero_celular", "")).strip()
        if sender_name:
            enriched["reconhecido_por"] = sender_name
        if sender_role:
            enriched["funcao_reconhecido_por"] = sender_role
        if sender_phone:
            enriched["numero_reconhecido_por"] = sender_phone

    receiver_employee = find_employee_by_phone_or_name(
        employees,
        phone_value=enriched.get("to", "") or enriched.get("numero_colaborador", ""),
        name_value=enriched.get("colaborador", ""),
    )
    if receiver_employee:
        receiver_public = build_employee_public_record(receiver_employee)
        receiver_name = str(receiver_public.get("nome", "")).strip()
        receiver_role = str(receiver_public.get("funcao", "")).strip()
        receiver_phone = str(receiver_public.get("numero_celular", "")).strip()
        if receiver_name:
            enriched["colaborador"] = receiver_name
        if receiver_role:
            enriched["funcao_colaborador"] = receiver_role
        if receiver_phone:
            enriched["numero_colaborador"] = receiver_phone
            enriched["to"] = receiver_phone

    return normalize_payload(enriched)


def validate_superpop_register_payload(payload: dict) -> tuple[bool, str]:
    collaborator_name = str(payload.get("colaborador", "")).strip()
    sender_name = str(payload.get("reconhecido_por", "")).strip()
    destination_number = normalize_whatsapp_number(str(payload.get("to", "") or payload.get("numero_colaborador", "")).strip())
    sender_number = normalize_whatsapp_number(str(payload.get("numero_reconhecido_por", "")).strip())
    values = payload.get("valores", [])
    message = str(payload.get("mensagem", "")).strip()

    missing: list[str] = []
    if not collaborator_name or collaborator_name == "-":
        missing.append("nome do colaborador")
    if not destination_number:
        missing.append("numero do colaborador")
    if not sender_name or sender_name == "-":
        missing.append("nome de quem envia")
    if not sender_number:
        missing.append("numero de quem envia")
    if not isinstance(values, list) or not any(str(item).strip() for item in values):
        missing.append("ao menos 1 valor")
    if not message:
        missing.append("mensagem")

    if missing:
        return False, "Dados obrigatorios ausentes para registrar o SuperPOP: " + ", ".join(missing) + "."

    return True, ""


def verify_employee_password(record: dict, password: str) -> bool:
    stored = record.get("senha")
    plain_password = str(password or "")
    if not plain_password:
        return False

    if isinstance(stored, dict):
        algo = str(stored.get("algoritmo", "")).strip().lower()
        salt_hex = str(stored.get("salt", "")).strip()
        hash_hex = str(stored.get("hash", "")).strip().lower()
        iterations = max(1, int(to_number(stored.get("iteracoes"), 180000)))
        if algo == "pbkdf2_sha256" and salt_hex and hash_hex:
            try:
                salt_bytes = bytes.fromhex(salt_hex)
            except ValueError:
                return False
            computed = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt_bytes, iterations).hex().lower()
            return hmac.compare_digest(computed, hash_hex)
        return False

    # Backward compatibility if old records have plain string password.
    if isinstance(stored, str):
        return hmac.compare_digest(stored, plain_password)

    return False


def make_log_record(
    payload: dict,
    card_id: str,
    auth_qr_url: str,
    local_date: str,
    local_time: str,
    local_iso: str,
    destination: str,
    sender_number: str,
    send_status: str,
    send_error: str,
    message_sid: str,
    format_selected: str,
    image_url: str,
    pdf_url: str,
    media_url: str,
    uploaded_image_url: str,
    upload_status: str,
    upload_error: str,
) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "card_id": card_id,
        "dia": local_date,
        "horario": local_time,
        "data_hora_iso": local_iso,
        "destinatario": {
            "nome": repair_mojibake_text(payload["colaborador"]).strip() or "-",
            "numero": payload["numero_colaborador"] or "-",
            "numero_normalizado": destination or "-",
            "funcao": repair_mojibake_text(payload["funcao_colaborador"]).strip() or "-",
        },
        "remetente": {
            "nome": repair_mojibake_text(payload["reconhecido_por"]).strip() or "-",
            "numero": payload["numero_reconhecido_por"] or "-",
            "numero_normalizado": sender_number or "-",
            "funcao": repair_mojibake_text(payload["funcao_reconhecido_por"]).strip() or "-",
        },
        "opcoes_marcadas": payload["valores"],
        "mensagem": repair_mojibake_text(payload["mensagem"]).strip() or "-",
        "whatsapp": {
            "status": send_status,
            "to": destination or "-",
            "message_sid": message_sid or "",
            "error": send_error,
            "format": format_selected,
        },
        "arquivos": {
            "image_url": image_url,
            "uploaded_image_url": uploaded_image_url or "",
            "pdf_url": pdf_url,
            "enviado_url": media_url,
            "auth_qr_url": auth_qr_url,
            "upload_status": upload_status,
            "upload_error": upload_error,
        },
    }


RANK_DATA_SOURCE_DEFAULT = "https://github.com/PopularAtacarejo/SuperPOP/blob/main/Dados.json"
MY_SUPERPOPS_SOURCE_URL = "https://github.com/PopularAtacarejo/SuperPOP/blob/main/Dados.json"
EMPLOYEES_DATA_SOURCE_DEFAULT = "https://github.com/PopularAtacarejo/SuperPOP/blob/main/Funcioinarios.json"
RANK_REACTIONS_SOURCE_DEFAULT = "https://github.com/PopularAtacarejo/SuperPOP/blob/main/RankReacoes.json"
SYSTEM_UPDATES_SOURCE_DEFAULT = "https://github.com/PopularAtacarejo/SuperPOP/blob/main/Atualizacoes.json"


def normalize_name_key(name: str) -> str:
    plain = unicodedata.normalize("NFD", name or "")
    plain = "".join(ch for ch in plain if unicodedata.category(ch) != "Mn")
    plain = re.sub(r"\s+", " ", plain).strip().lower()
    return plain


def parse_log_timestamp(record: dict) -> float:
    if not isinstance(record, dict):
        return 0.0

    iso_value = str(record.get("data_hora_iso", "")).strip()
    if iso_value:
        try:
            return datetime.fromisoformat(iso_value).timestamp()
        except ValueError:
            pass

    dia_value = str(record.get("dia", "")).strip()
    hora_value = str(record.get("horario", "")).strip() or "00:00:00"
    if dia_value:
        try:
            return datetime.strptime(f"{dia_value} {hora_value}", "%d/%m/%Y %H:%M:%S").timestamp()
        except ValueError:
            pass

    return 0.0


def _normalize_notification_destination_number(record: dict) -> str:
    destinatario = (record.get("destinatario") or {}) if isinstance(record.get("destinatario"), dict) else {}
    candidate = str(destinatario.get("numero_normalizado") or destinatario.get("numero") or "").strip()
    normalized = normalize_whatsapp_number(candidate)
    return normalized or ""


def _record_matches_notification_user(record: dict, user_number: str, user_name_key: str) -> bool:
    if not isinstance(record, dict):
        return False
    destin_number = _normalize_notification_destination_number(record)
    if destin_number and user_number and destin_number == user_number:
        return True
    destinatario = (record.get("destinatario") or {}) if isinstance(record.get("destinatario"), dict) else {}
    dest_name_key = normalize_name_key(str(destinatario.get("nome", "")))
    if dest_name_key and user_name_key and dest_name_key == user_name_key:
        return True
    return False


def _parse_iso_timestamp(value: object) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except Exception:
        return None


def _is_notification_record_viewed(record: dict, seen_ids: set[str], cleared_timestamp: float | None) -> bool:
    record_id = str(record.get("id", "")).strip()
    if record_id and record_id in seen_ids:
        return True
    if cleared_timestamp is None:
        return False
    record_ts = parse_log_timestamp(record)
    if record_ts and record_ts <= cleared_timestamp:
        return True
    return False


def resolve_rank_data_source_url() -> tuple[str, str]:
    configured = get_env("RANK_DATA_SOURCE_URL", RANK_DATA_SOURCE_DEFAULT)
    resolved = normalize_layout_source_url(configured)
    return configured, resolved


def fetch_rank_logs_remote(source_url: str) -> tuple[list, str]:
    if not source_url:
        return [], "URL da fonte de rank nao foi configurada."

    timeout_seconds = max(5.0, to_number(get_env("RANK_SOURCE_TIMEOUT_SECONDS", "20"), 20.0))
    request_obj = urllib.request.Request(
        source_url,
        headers={"User-Agent": "superpop-backend-rank-fetcher", "Accept": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
            loaded = json.loads(payload) if payload else []
    except urllib.error.HTTPError as exc:
        return [], f"Falha ao ler fonte remota (HTTP {exc.code})."
    except json.JSONDecodeError:
        return [], "Fonte remota retornou JSON invalido."
    except Exception as exc:  # noqa: BLE001
        return [], f"Falha ao ler fonte remota: {exc}"

    if not isinstance(loaded, list):
        return [], "Fonte remota nao retornou uma lista de registros."

    return loaded, ""


def resolve_rank_reactions_source_url() -> tuple[str, str]:
    configured = get_env("RANK_REACTIONS_SOURCE_URL", RANK_REACTIONS_SOURCE_DEFAULT)
    resolved = normalize_layout_source_url(configured)
    return configured, resolved


def fetch_rank_reactions_remote(source_url: str) -> tuple[list, str]:
    if not source_url:
        return [], "URL da fonte de reacoes nao foi configurada."

    timeout_seconds = max(5.0, to_number(get_env("RANK_REACTIONS_SOURCE_TIMEOUT_SECONDS", "20"), 20.0))
    request_obj = urllib.request.Request(
        source_url,
        headers={"User-Agent": "superpop-backend-rank-reactions-fetcher", "Accept": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
            loaded = json.loads(payload) if payload else []
    except urllib.error.HTTPError as exc:
        return [], f"Falha ao ler fonte remota de reacoes (HTTP {exc.code})."
    except json.JSONDecodeError:
        return [], "Fonte remota de reacoes retornou JSON invalido."
    except Exception as exc:  # noqa: BLE001
        return [], f"Falha ao ler fonte remota de reacoes: {exc}"

    if not isinstance(loaded, list):
        return [], "Fonte remota de reacoes nao retornou uma lista."

    return loaded, ""


def resolve_system_updates_source_url() -> tuple[str, str]:
    configured = get_env("SYSTEM_UPDATES_SOURCE_URL", SYSTEM_UPDATES_SOURCE_DEFAULT)
    resolved = normalize_layout_source_url(configured)
    return configured, resolved


def fetch_system_updates_remote(source_url: str) -> tuple[list, str]:
    if not source_url:
        return [], "URL da fonte de atualizacoes nao foi configurada."

    timeout_seconds = max(5.0, to_number(get_env("SYSTEM_UPDATES_SOURCE_TIMEOUT_SECONDS", "20"), 20.0))
    request_obj = urllib.request.Request(
        source_url,
        headers={"User-Agent": "superpop-backend-system-updates-fetcher", "Accept": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
            loaded = json.loads(payload) if payload else []
    except urllib.error.HTTPError as exc:
        return [], f"Falha ao ler fonte remota de atualizacoes (HTTP {exc.code})."
    except json.JSONDecodeError:
        return [], "Fonte remota de atualizacoes retornou JSON invalido."
    except Exception as exc:  # noqa: BLE001
        return [], f"Falha ao ler fonte remota de atualizacoes: {exc}"

    if not isinstance(loaded, list):
        return [], "Fonte remota de atualizacoes nao retornou uma lista."

    return loaded, ""


def resolve_employees_data_source_url() -> tuple[str, str]:
    configured = get_env("EMPLOYEES_DATA_SOURCE_URL", EMPLOYEES_DATA_SOURCE_DEFAULT)
    resolved = normalize_layout_source_url(configured)
    return configured, resolved


def fetch_employees_remote(source_url: str) -> tuple[list, str]:
    if not source_url:
        return [], "URL da fonte de funcionarios nao foi configurada."

    timeout_seconds = max(5.0, to_number(get_env("EMPLOYEES_SOURCE_TIMEOUT_SECONDS", "20"), 20.0))
    request_obj = urllib.request.Request(
        source_url,
        headers={"User-Agent": "superpop-backend-employees-fetcher", "Accept": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
            loaded = json.loads(payload) if payload else []
    except urllib.error.HTTPError as exc:
        return [], f"Falha ao ler fonte remota de funcionarios (HTTP {exc.code})."
    except json.JSONDecodeError:
        return [], "Fonte remota de funcionarios retornou JSON invalido."
    except Exception as exc:  # noqa: BLE001
        return [], f"Falha ao ler fonte remota de funcionarios: {exc}"

    if isinstance(loaded, dict) and isinstance(loaded.get("funcionarios"), list):
        loaded = loaded.get("funcionarios")

    if not isinstance(loaded, list):
        return [], "Fonte remota de funcionarios nao retornou uma lista."

    records = [item for item in loaded if isinstance(item, dict)]
    return records, ""


def refresh_local_employees_from_remote(existing_records: list) -> tuple[list, str]:
    configured_source, resolved_source = resolve_employees_data_source_url()
    remote_records, error = fetch_employees_remote(resolved_source)
    if error:
        return existing_records, error

    merged_records = merge_employee_lists(existing_records, remote_records)
    if merged_records != existing_records:
        write_employees(merged_records)
    return merged_records, ""


def load_employee_records_for_write_validation(existing_records: list) -> tuple[list, str]:
    records, error = refresh_local_employees_from_remote(existing_records)
    if error and is_github_sync_required():
        return records, "Nao foi possivel validar a base de usuarios no GitHub. Tente novamente em instantes."
    return records, ""


def pick_actor_name(record: dict, actor: str) -> str:
    if not isinstance(record, dict):
        return ""

    if actor == "recebeu":
        destinatario = record.get("destinatario", {}) or {}
        nome = str(destinatario.get("nome", "")).strip()
        if nome:
            return nome
        return str(record.get("colaborador", "")).strip()

    remetente = record.get("remetente", {}) or {}
    nome = str(remetente.get("nome", "")).strip()
    if nome:
        return nome
    return str(record.get("reconhecido_por", "")).strip()


def build_actor_rank(logs: list, actor: str) -> list:
    grouped: dict[str, dict] = {}
    for record in logs:
        if not isinstance(record, dict):
            continue

        nome = pick_actor_name(record, actor)
        if not nome or nome == "-":
            continue

        key = normalize_name_key(nome)
        if not key:
            continue

        opcoes = record.get("opcoes_marcadas", [])
        total_valores = len(opcoes) if isinstance(opcoes, list) else 0
        timestamp = parse_log_timestamp(record)
        dia_value = str(record.get("dia", "")).strip()
        hora_value = str(record.get("horario", "")).strip()

        current = grouped.get(key)
        if not current:
            grouped[key] = {
                "nome": nome,
                "total_superpop": 1,
                "total_valores": total_valores,
                "ultima_data": dia_value,
                "ultimo_horario": hora_value,
                "_latest_ts": timestamp,
            }
            continue

        current["total_superpop"] += 1
        current["total_valores"] += total_valores
        if timestamp >= current["_latest_ts"]:
            current["_latest_ts"] = timestamp
            current["ultima_data"] = dia_value
            current["ultimo_horario"] = hora_value
            current["nome"] = nome

    ranking = sorted(
        grouped.values(),
        key=lambda item: (
            -int(item.get("total_superpop", 0)),
            -int(item.get("total_valores", 0)),
            str(item.get("nome", "")).lower(),
        ),
    )

    for index, item in enumerate(ranking, start=1):
        item["posicao"] = index
        item.pop("_latest_ts", None)

    return ranking


def build_rank_payload(logs: list, source_configured: str, source_resolved: str) -> dict:
    received_rank = build_actor_rank(logs, actor="recebeu")
    sent_rank = build_actor_rank(logs, actor="enviou")
    total_superpop = sum(int(item.get("total_superpop", 0)) for item in received_rank)

    return {
        "ok": True,
        "gerado_em": now_brazil().isoformat(),
        "fonte": {
            "url_configurada": source_configured,
            "url_resolvida": source_resolved,
        },
        "resumo": {
            "total_registros": len(logs),
            "total_superpop": total_superpop,
            "colaboradores_que_receberam": len(received_rank),
            "colaboradores_que_enviaram": len(sent_rank),
        },
        "rankings": {
            "mais_receberam": received_rank,
            "mais_enviaram": sent_rank,
        },
    }


PT_BR_MONTH_NAMES = (
    "janeiro",
    "fevereiro",
    "marco",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)


def format_month_label(month_key: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(month_key or "").strip())
    if not match:
        return str(month_key or "").strip()
    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        return str(month_key or "").strip()
    return f"{PT_BR_MONTH_NAMES[month - 1]}/{year}"


def extract_month_key_from_log(record: dict) -> str:
    if not isinstance(record, dict):
        return ""

    dia_value = str(record.get("dia", "")).strip()
    dia_match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", dia_value)
    if dia_match:
        year = int(dia_match.group(3))
        month = int(dia_match.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"

    iso_value = str(record.get("data_hora_iso", "")).strip()
    if iso_value:
        iso_candidate = iso_value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso_candidate)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(ZoneInfo("America/Sao_Paulo"))
            return f"{parsed.year:04d}-{parsed.month:02d}"
        except ValueError:
            pass

    timestamp = parse_log_timestamp(record)
    if timestamp > 0:
        parsed = datetime.fromtimestamp(timestamp, tz=ZoneInfo("America/Sao_Paulo"))
        return f"{parsed.year:04d}-{parsed.month:02d}"

    return ""


def normalize_actor_number(value: object) -> str:
    return normalize_whatsapp_number(str(value or "").strip())


def log_matches_user(record: dict, user_name_key: str, user_number: str) -> tuple[bool, bool]:
    if not isinstance(record, dict):
        return False, False

    remetente = record.get("remetente", {}) or {}
    destinatario = record.get("destinatario", {}) or {}

    sender_name_key = normalize_name_key(str(remetente.get("nome") or record.get("reconhecido_por") or ""))
    receiver_name_key = normalize_name_key(str(destinatario.get("nome") or record.get("colaborador") or ""))
    sender_number = normalize_actor_number(
        remetente.get("numero_normalizado") or remetente.get("numero") or record.get("numero_reconhecido_por")
    )
    receiver_number = normalize_actor_number(
        destinatario.get("numero_normalizado") or destinatario.get("numero") or record.get("numero_colaborador")
    )

    sender_match = False
    receiver_match = False

    if user_number:
        sender_match = bool(sender_number and sender_number == user_number)
        receiver_match = bool(receiver_number and receiver_number == user_number)

    if user_name_key:
        if not sender_match:
            sender_match = bool(sender_name_key and sender_name_key == user_name_key)
        if not receiver_match:
            receiver_match = bool(receiver_name_key and receiver_name_key == user_name_key)

    return sender_match, receiver_match


def refresh_local_logs_from_remote(existing_logs: list) -> tuple[list, str]:
    _configured_source, resolved_source = resolve_rank_data_source_url()
    remote_logs, error = fetch_rank_logs_remote(resolved_source)
    if error:
        return existing_logs, error

    merged_logs = merge_log_lists(existing_logs, remote_logs)
    if merged_logs != existing_logs:
        write_logs(merged_logs)
    return merged_logs, ""


def load_logs_for_history_view() -> tuple[list, dict]:
    preferred_configured = MY_SUPERPOPS_SOURCE_URL
    preferred_resolved = normalize_layout_source_url(preferred_configured)
    configured_source, resolved_source = resolve_rank_data_source_url()
    local_logs = read_logs()

    candidates: list[tuple[str, str]] = [(preferred_configured, preferred_resolved)]
    if resolved_source != preferred_resolved:
        candidates.append((configured_source, resolved_source))

    remote_errors: list[str] = []
    for configured, resolved in candidates:
        remote_logs, error = fetch_rank_logs_remote(resolved)
        if not error and isinstance(remote_logs, list):
            merged_logs = merge_log_lists(local_logs, remote_logs)
            return merged_logs, {
                "tipo": "remoto",
                "url_configurada": configured,
                "url_resolvida": resolved,
                "remoto_total": len(remote_logs),
                "local_total": len(local_logs),
                "merged_total": len(merged_logs),
            }
        remote_errors.append(error or f"Falha ao ler fonte remota: {resolved}")

    return local_logs, {
        "tipo": "local",
        "url_configurada": preferred_configured,
        "url_resolvida": preferred_resolved,
        "local_total": len(local_logs),
        "erro_remoto": " | ".join(item for item in remote_errors if item),
    }


def normalize_log_recipient_reaction(record: dict) -> dict:
    raw = record.get("reacao_destinatario", {}) if isinstance(record, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    reactor = raw.get("reactor", {}) or {}
    if not isinstance(reactor, dict):
        reactor = {}
    return {
        "emoji": str(raw.get("emoji", "")).strip(),
        "updated_at_iso": str(raw.get("updated_at_iso", "")).strip(),
        "reactor": {
            "id": str(reactor.get("id", "")).strip(),
            "nome": str(reactor.get("nome", "")).strip() or "Usuario",
        },
    }


def find_log_record_index(records: list, card_id: str, log_id: str) -> int:
    clean_card_id = str(card_id or "").strip()
    clean_log_id = str(log_id or "").strip()
    if not clean_card_id and not clean_log_id:
        return -1

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        if clean_card_id and str(record.get("card_id", "")).strip() == clean_card_id:
            return index
        if clean_log_id and str(record.get("id", "")).strip() == clean_log_id:
            return index
    return -1


def load_rank_reactions_for_view() -> tuple[list, dict]:
    configured_source, resolved_source = resolve_rank_reactions_source_url()
    remote_records, error = fetch_rank_reactions_remote(resolved_source)
    if not error and isinstance(remote_records, list):
        return remote_records, {
            "tipo": "remoto",
            "url_configurada": configured_source,
            "url_resolvida": resolved_source,
            "remoto_total": len(remote_records),
        }

    local_records = read_rank_reactions()
    return local_records, {
        "tipo": "local",
        "url_configurada": configured_source,
        "url_resolvida": resolved_source,
        "local_total": len(local_records),
        "erro_remoto": error,
    }


def refresh_local_system_updates_from_remote(existing_records: list) -> tuple[list, str]:
    configured_source, resolved_source = resolve_system_updates_source_url()
    remote_records, error = fetch_system_updates_remote(resolved_source)
    if error:
        return existing_records, error

    merged_records = merge_system_update_lists(existing_records, remote_records)
    if merged_records != existing_records:
        write_system_updates(merged_records)
    return merged_records, ""


def load_system_updates_for_view() -> tuple[list, dict]:
    configured_source, resolved_source = resolve_system_updates_source_url()
    remote_records, error = fetch_system_updates_remote(resolved_source)
    local_records = read_system_updates()
    if not error and isinstance(remote_records, list):
        merged_records = merge_system_update_lists(local_records, remote_records)
        return merged_records, {
            "tipo": "remoto",
            "url_configurada": configured_source,
            "url_resolvida": resolved_source,
            "remoto_total": len(remote_records),
            "local_total": len(local_records),
            "merged_total": len(merged_records),
        }

    return local_records, {
        "tipo": "local",
        "url_configurada": configured_source,
        "url_resolvida": resolved_source,
        "local_total": len(local_records),
        "erro_remoto": error,
    }


def load_employees_for_admin_view() -> tuple[list, dict]:
    configured_source, resolved_source = resolve_employees_data_source_url()
    remote_records, error = fetch_employees_remote(resolved_source)

    if not error and isinstance(remote_records, list):
        return remote_records, {
            "tipo": "remoto",
            "url_configurada": configured_source,
            "url_resolvida": resolved_source,
            "remoto_total": len(remote_records),
        }

    local_records = read_employees()
    return local_records, {
        "tipo": "local",
        "url_configurada": configured_source,
        "url_resolvida": resolved_source,
        "local_total": len(local_records),
        "erro_remoto": error,
    }


def parse_log_datetime_local(record: dict) -> datetime | None:
    if not isinstance(record, dict):
        return None

    iso_value = str(record.get("data_hora_iso", "")).strip()
    if iso_value:
        iso_candidate = iso_value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso_candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=ZoneInfo("America/Sao_Paulo"))
            return parsed.astimezone(ZoneInfo("America/Sao_Paulo"))
        except ValueError:
            pass

    dia_value = str(record.get("dia", "")).strip()
    hora_value = str(record.get("horario", "")).strip() or "00:00:00"
    if dia_value:
        try:
            parsed = datetime.strptime(f"{dia_value} {hora_value}", "%d/%m/%Y %H:%M:%S")
            return parsed.replace(tzinfo=ZoneInfo("America/Sao_Paulo"))
        except ValueError:
            return None

    return None


def build_period_counter() -> dict:
    return {
        "enviados": 0,
        "recebidos": 0,
        "chaves_enviadas": 0,
        "chaves_recebidas": 0,
    }


def resolve_employee_match_key(actor: dict, employee_keys_by_number: dict, employee_keys_by_name: dict) -> str | None:
    if not isinstance(actor, dict):
        return None

    actor_number = normalize_actor_number(actor.get("numero_normalizado") or actor.get("numero"))
    if actor_number and actor_number in employee_keys_by_number:
        return employee_keys_by_number[actor_number]

    actor_name_key = normalize_name_key(str(actor.get("nome", "")).strip())
    if actor_name_key and actor_name_key in employee_keys_by_name:
        return employee_keys_by_name[actor_name_key]

    return None


def update_period_bucket(bucket: dict, bucket_name: str, role: str, keys_count: int) -> None:
    if bucket_name not in bucket:
        bucket[bucket_name] = build_period_counter()

    current = bucket[bucket_name]
    if role == "sent":
        current["enviados"] += 1
        current["chaves_enviadas"] += keys_count
    else:
        current["recebidos"] += 1
        current["chaves_recebidas"] += keys_count


def build_analytics_log_item(record: dict, timestamp: datetime | None = None) -> dict:
    if not isinstance(record, dict):
        return {}

    destinatario = record.get("destinatario", {}) or {}
    remetente = record.get("remetente", {}) or {}
    arquivos = record.get("arquivos", {}) or {}
    valores = record.get("opcoes_marcadas", [])
    if not isinstance(valores, list):
        valores = []
    valores_limpos = [repair_mojibake_text(item).strip() for item in valores if repair_mojibake_text(item).strip()]

    mensagem_value = repair_mojibake_text(record.get("mensagem", "")).strip()
    if mensagem_value == "-":
        mensagem_value = ""

    uploaded_image = str(arquivos.get("uploaded_image_url", "")).strip()
    local_image = str(arquivos.get("image_url", "")).strip()
    best_image_url = uploaded_image or local_image

    timestamp_value = parse_log_datetime_local(record)
    if timestamp and (not timestamp_value or timestamp > timestamp_value):
        timestamp_value = timestamp

    return {
        "id": str(record.get("id", "")).strip(),
        "card_id": str(record.get("card_id", "")).strip(),
        "dia": str(record.get("dia", "")).strip(),
        "horario": str(record.get("horario", "")).strip(),
        "data_hora_iso": str(record.get("data_hora_iso", "")).strip(),
        "ordem_ts": timestamp_value.timestamp() if timestamp_value else 0.0,
        "remetente": {
            "nome": repair_mojibake_text(remetente.get("nome", "")).strip(),
            "numero": str(remetente.get("numero", "")).strip(),
            "funcao": repair_mojibake_text(remetente.get("funcao", "")).strip(),
        },
        "destinatario": {
            "nome": repair_mojibake_text(destinatario.get("nome", "")).strip(),
            "numero": str(destinatario.get("numero", "")).strip(),
            "funcao": repair_mojibake_text(destinatario.get("funcao", "")).strip(),
        },
        "valores": valores_limpos,
        "total_chaves": len(valores_limpos),
        "mensagem": mensagem_value,
        "arquivos": {
            "image_url": local_image,
            "uploaded_image_url": uploaded_image,
            "best_image_url": best_image_url,
            "auth_qr_url": str(arquivos.get("auth_qr_url", "")).strip(),
        },
    }


def build_analytics_payload(logs: list, employees: list, logs_source: dict, employees_source: dict) -> dict:
    now_value = now_brazil()
    day_start = now_value.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = day_start - timedelta(days=day_start.weekday())
    month_start = day_start.replace(day=1)

    users_map: dict[str, dict] = {}
    employee_keys_by_number: dict[str, str] = {}
    employee_keys_by_name: dict[str, str] = {}

    for employee in employees:
        if not isinstance(employee, dict):
            continue

        public_record = build_employee_public_record(employee)
        user_id = public_record.get("id") or employee_record_key(employee)
        if not user_id:
            continue

        user_state = {
            "usuario": public_record,
            "totais": build_period_counter(),
            "periodos": {
                "dia": build_period_counter(),
                "semana": build_period_counter(),
                "mes": build_period_counter(),
            },
            "ultimos_eventos": {
                "ultimo_envio_iso": "",
                "ultimo_recebimento_iso": "",
            },
        }
        users_map[user_id] = user_state

        number_key = normalize_whatsapp_number(public_record.get("numero_celular", ""))
        if number_key:
            employee_keys_by_number[number_key] = user_id

        name_key = normalize_name_key(public_record.get("nome", ""))
        if name_key:
            employee_keys_by_name[name_key] = user_id

    rankings_sent: dict[str, dict] = {}
    rankings_received: dict[str, dict] = {}
    key_sender_rank: dict[str, dict] = {}
    collaborator_key_rank: dict[str, dict] = {}
    global_counters = {
        "totais": {
            "usuarios": len(users_map),
            "superpops": 0,
            "chaves_marcadas": 0,
        },
        "periodos": {
            "dia": build_period_counter(),
            "semana": build_period_counter(),
            "mes": build_period_counter(),
        },
    }
    monthly_series: dict[str, dict] = {}
    detailed_logs: list[dict] = []

    for record in logs:
        if not isinstance(record, dict):
            continue

        timestamp = parse_log_datetime_local(record)
        detailed_item = build_analytics_log_item(record, timestamp=timestamp)
        if detailed_item:
            detailed_logs.append(detailed_item)
        keys_count = len(record.get("opcoes_marcadas", [])) if isinstance(record.get("opcoes_marcadas"), list) else 0
        sender = record.get("remetente", {}) or {}
        receiver = record.get("destinatario", {}) or {}
        sender_key = resolve_employee_match_key(sender, employee_keys_by_number, employee_keys_by_name)
        receiver_key = resolve_employee_match_key(receiver, employee_keys_by_number, employee_keys_by_name)
        sender_name = str(sender.get("nome", "")).strip() or "-"
        receiver_name = str(receiver.get("nome", "")).strip() or "-"
        month_key = extract_month_key_from_log(record)

        global_counters["totais"]["superpops"] += 1
        global_counters["totais"]["chaves_marcadas"] += keys_count

        if month_key:
            if month_key not in monthly_series:
                monthly_series[month_key] = {
                    "chave": month_key,
                    "label": format_month_label(month_key),
                    "superpops": 0,
                    "chaves_marcadas": 0,
                }
            monthly_series[month_key]["superpops"] += 1
            monthly_series[month_key]["chaves_marcadas"] += keys_count

        in_day = bool(timestamp and timestamp >= day_start)
        in_week = bool(timestamp and timestamp >= week_start)
        in_month = bool(timestamp and timestamp >= month_start)

        def apply_periods(target_bucket: dict, role: str) -> None:
            if in_day:
                update_period_bucket(target_bucket, "dia", role, keys_count)
            if in_week:
                update_period_bucket(target_bucket, "semana", role, keys_count)
            if in_month:
                update_period_bucket(target_bucket, "mes", role, keys_count)

        apply_periods(global_counters["periodos"], "sent")
        apply_periods(global_counters["periodos"], "received")

        if sender_key and sender_key in users_map:
            sender_state = users_map[sender_key]
            sender_state["totais"]["enviados"] += 1
            sender_state["totais"]["chaves_enviadas"] += keys_count
            apply_periods(sender_state["periodos"], "sent")
            if timestamp:
                latest_iso = timestamp.isoformat()
                if latest_iso > str(sender_state["ultimos_eventos"]["ultimo_envio_iso"] or ""):
                    sender_state["ultimos_eventos"]["ultimo_envio_iso"] = latest_iso

        if receiver_key and receiver_key in users_map:
            receiver_state = users_map[receiver_key]
            receiver_state["totais"]["recebidos"] += 1
            receiver_state["totais"]["chaves_recebidas"] += keys_count
            apply_periods(receiver_state["periodos"], "received")
            if timestamp:
                latest_iso = timestamp.isoformat()
                if latest_iso > str(receiver_state["ultimos_eventos"]["ultimo_recebimento_iso"] or ""):
                    receiver_state["ultimos_eventos"]["ultimo_recebimento_iso"] = latest_iso

        if sender_name and sender_name != "-":
            rank_sent = rankings_sent.setdefault(sender_name, {"nome": sender_name, "total_superpops": 0, "total_chaves": 0})
            rank_sent["total_superpops"] += 1
            rank_sent["total_chaves"] += keys_count
            key_rank = key_sender_rank.setdefault(sender_name, {"nome": sender_name, "total_superpops": 0, "total_chaves": 0})
            key_rank["total_superpops"] += 1
            key_rank["total_chaves"] += keys_count

        if receiver_name and receiver_name != "-":
            rank_received = rankings_received.setdefault(receiver_name, {"nome": receiver_name, "total_superpops": 0, "total_chaves": 0})
            rank_received["total_superpops"] += 1
            rank_received["total_chaves"] += keys_count
            collaborator_rank = collaborator_key_rank.setdefault(receiver_name, {"nome": receiver_name, "total_superpops": 0, "total_chaves": 0})
            collaborator_rank["total_superpops"] += 1
            collaborator_rank["total_chaves"] += keys_count

    def serialize_rank(items: dict[str, dict]) -> list[dict]:
        ranking = sorted(
            items.values(),
            key=lambda item: (-int(item.get("total_chaves", 0)), -int(item.get("total_superpops", 0)), str(item.get("nome", "")).lower()),
        )
        for index, item in enumerate(ranking, start=1):
            total_superpops = max(1, int(item.get("total_superpops", 0)))
            item["posicao"] = index
            item["media_chaves_por_superpop"] = round(float(item.get("total_chaves", 0)) / total_superpops, 2)
        return ranking

    users_serialized = sorted(
        users_map.values(),
        key=lambda item: str((item.get("usuario") or {}).get("nome", "")).lower(),
    )
    detailed_logs_sorted = sorted(
        detailed_logs,
        key=lambda item: (float(item.get("ordem_ts", 0.0)), str(item.get("data_hora_iso", "")).strip()),
        reverse=True,
    )
    for item in detailed_logs_sorted:
        item.pop("ordem_ts", None)

    return {
        "ok": True,
        "gerado_em": now_value.isoformat(),
        "fonte": {
            "logs": logs_source,
            "usuarios": employees_source,
        },
        "resumo": global_counters,
        "ranking": {
            "mais_enviaram": serialize_rank(rankings_sent),
            "mais_receberam": serialize_rank(rankings_received),
            "mais_chaves_marcadas": serialize_rank(key_sender_rank),
            "colaboradores_com_mais_chaves": serialize_rank(collaborator_key_rank),
        },
        "historico_mensal": [monthly_series[key] for key in sorted(monthly_series.keys(), reverse=True)],
        "usuarios": users_serialized,
        "registros": detailed_logs_sorted,
    }


def build_user_log_item(record: dict, role: str) -> dict:
    destinatario = record.get("destinatario", {}) or {}
    remetente = record.get("remetente", {}) or {}
    arquivos = record.get("arquivos", {}) or {}
    whatsapp = record.get("whatsapp", {}) or {}

    other_actor = destinatario if role == "sent" else remetente
    valores = record.get("opcoes_marcadas", [])
    if not isinstance(valores, list):
        valores = []

    mensagem_value = repair_mojibake_text(record.get("mensagem", "")).strip()
    if mensagem_value == "-":
        mensagem_value = ""
    reacao_destinatario = normalize_log_recipient_reaction(record)

    return {
        "id": str(record.get("id", "")).strip(),
        "card_id": str(record.get("card_id", "")).strip(),
        "dia": str(record.get("dia", "")).strip(),
        "horario": str(record.get("horario", "")).strip(),
        "data_hora_iso": str(record.get("data_hora_iso", "")).strip(),
        "papel": role,
        "outra_pessoa": {
            "nome": repair_mojibake_text(other_actor.get("nome", "")).strip(),
            "numero": str(other_actor.get("numero", "")).strip(),
            "funcao": repair_mojibake_text(other_actor.get("funcao", "")).strip(),
        },
        "valores": [repair_mojibake_text(item).strip() for item in valores if repair_mojibake_text(item).strip()],
        "mensagem": mensagem_value,
        "whatsapp": {
            "status": str(whatsapp.get("status", "")).strip(),
            "to": str(whatsapp.get("to", "")).strip(),
            "error": str(whatsapp.get("error", "")).strip(),
        },
        "arquivos": {
            "image_url": str(arquivos.get("image_url", "")).strip(),
            "uploaded_image_url": str(arquivos.get("uploaded_image_url", "")).strip(),
            "auth_qr_url": str(arquivos.get("auth_qr_url", "")).strip(),
        },
        "reacao_destinatario": reacao_destinatario,
    }


def build_admin_users_payload(employees: list, employees_source: dict) -> dict:
    users: list[dict] = []
    for employee in employees:
        if not isinstance(employee, dict):
            continue
        public_record = build_employee_public_record(employee)
        users.append(
            {
                "id": public_record.get("id", ""),
                "nome": public_record.get("nome", ""),
                "telefone": public_record.get("numero_celular", ""),
                "funcao": public_record.get("funcao", ""),
                "email": public_record.get("email", ""),
                "data_cadastro_iso": public_record.get("data_cadastro_iso", ""),
                "tags_acesso": list(public_record.get("tags_acesso", [])),
            }
        )

    users.sort(key=lambda item: str(item.get("nome", "")).lower())
    return {
        "ok": True,
        "gerado_em": now_brazil().isoformat(),
        "fonte": {"usuarios": employees_source},
        "usuarios": users,
        "resumo": {"total_usuarios": len(users)},
    }


app = Flask(__name__)
app.secret_key = get_env("FLASK_SECRET_KEY", "superpop-dev-secret")
session_hours = max(1.0, to_number(get_env("AUTH_SESSION_HOURS", "24"), 24.0))
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=session_hours)
app.config["SESSION_COOKIE_HTTPONLY"] = True
session_cookie_samesite_raw = get_env("SESSION_COOKIE_SAMESITE", "None").strip().lower()
if session_cookie_samesite_raw not in {"lax", "strict", "none"}:
    session_cookie_samesite_raw = "none"
session_cookie_samesite = "None" if session_cookie_samesite_raw == "none" else session_cookie_samesite_raw.capitalize()
session_cookie_secure_default = session_cookie_samesite_raw == "none"
app.config["SESSION_COOKIE_SAMESITE"] = session_cookie_samesite
app.config["SESSION_COOKIE_SECURE"] = to_bool(
    get_env("SESSION_COOKIE_SECURE", "1" if session_cookie_secure_default else "0"),
    session_cookie_secure_default,
)
app.config["SESSION_COOKIE_PARTITIONED"] = to_bool(
    get_env("SESSION_COOKIE_PARTITIONED", "1" if session_cookie_secure_default else "0"),
    session_cookie_secure_default,
)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True
cors_origins_raw = get_env(
    "CORS_ALLOWED_ORIGINS",
    "https://popularatacarejo.github.io,https://superpopbackend.onrender.com,http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5000,http://localhost:5000",
)
cors_allowed_origins: list[str] = []
for raw_origin in cors_origins_raw.split(","):
    normalized_origin = normalize_cors_origin(raw_origin)
    if normalized_origin and normalized_origin not in cors_allowed_origins:
        cors_allowed_origins.append(normalized_origin)
frontend_origin = normalize_cors_origin(get_frontend_base_url())
if frontend_origin and frontend_origin not in cors_allowed_origins:
    cors_allowed_origins.append(frontend_origin)
if not cors_allowed_origins:
    cors_allowed_origins = ["https://superpopbackend.onrender.com"]
CORS(
    app,
    supports_credentials=True,
    resources={
        r"/api/*": {"origins": cors_allowed_origins},
        r"/health": {"origins": cors_allowed_origins},
        r"/Dados.json": {"origins": cors_allowed_origins},
        r"/Funcioinarios.json": {"origins": cors_allowed_origins},
        r"/FuncoesSupermercado.json": {"origins": cors_allowed_origins},
    },
)

from aniversariantes import birthday_bp
app.register_blueprint(birthday_bp)
from dinamicas_pop import dinamicas_pop_bp
app.register_blueprint(dinamicas_pop_bp)
from permissoes_paginas import page_permissions_bp
app.register_blueprint(page_permissions_bp)


def is_user_logged_in() -> bool:
    return bool(session.get("auth_user_id"))


def get_authenticated_user_context() -> dict | None:
    if not is_user_logged_in():
        return None

    auth_user_id = str(session.get("auth_user_id", "")).strip()
    auth_user_nome = repair_mojibake_text(session.get("auth_user_nome", "")).strip()
    auth_user_funcao = repair_mojibake_text(session.get("auth_user_funcao", "")).strip()
    auth_user_numero = str(session.get("auth_user_numero", "")).strip()
    auth_user_email = str(session.get("auth_user_email", "")).strip()

    employee = None
    if auth_user_id:
        with EMPLOYEES_FILE_LOCK:
            records = read_employees()
            employee = find_employee_by_id(records, auth_user_id)
            if not employee:
                refreshed_records, _refresh_error = refresh_local_employees_from_remote(records)
                employee = find_employee_by_id(refreshed_records, auth_user_id)

    if employee:
        public_employee = build_employee_public_record(employee)
        auth_user_nome = public_employee.get("nome", "")
        auth_user_funcao = public_employee.get("funcao", "")
        auth_user_numero = public_employee.get("numero_celular", "")
        auth_user_email = public_employee.get("email", "")
        session["auth_user_nome"] = auth_user_nome
        session["auth_user_funcao"] = auth_user_funcao
        session["auth_user_numero"] = auth_user_numero
        session["auth_user_email"] = auth_user_email
    else:
        fallback_record = {
            "id": auth_user_id,
            "nome": auth_user_nome,
            "funcao": auth_user_funcao,
            "numero_celular": auth_user_numero,
            "email": auth_user_email,
        }
        public_employee = build_employee_public_record(fallback_record)

    tags_acesso = list(public_employee.get("tags_acesso", []))
    permissoes = dict(public_employee.get("permissoes", {}))

    return {
        "employee": employee,
        "usuario": {
            **public_employee,
            "login_at": str(session.get("auth_login_at", "")).strip(),
        },
        "permissoes": permissoes,
        "tags_acesso": tags_acesso,
    }


def require_admin_api_context() -> tuple[dict | None, object | None]:
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return None, (jsonify({"ok": False, "error": "Nao autenticado."}), 401)

    normalized_tags = {normalize_access_tag(item) for item in auth_context.get("tags_acesso", []) if item}
    if not normalized_tags.intersection(ANALYTICS_ACCESS_TAGS):
        return None, (jsonify({"ok": False, "error": "Sem permissao."}), 403)

    return auth_context, None


def require_admin_only_api_context() -> tuple[dict | None, object | None]:
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return None, (jsonify({"ok": False, "error": "Nao autenticado."}), 401)

    normalized_tags = {normalize_access_tag(item) for item in auth_context.get("tags_acesso", []) if item}
    if not normalized_tags.intersection(MANAGE_USERS_ACCESS_TAGS):
        return None, (jsonify({"ok": False, "error": "Acesso restrito a administradores e desenvolvedores."}), 403)

    return auth_context, None


def require_developer_only_api_context() -> tuple[dict | None, object | None]:
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return None, (jsonify({"ok": False, "error": "Nao autenticado."}), 401)

    normalized_tags = {normalize_access_tag(item) for item in auth_context.get("tags_acesso", []) if item}
    if not normalized_tags.intersection(DEVELOPER_ONLY_ACCESS_TAGS):
        return None, (jsonify({"ok": False, "error": "Acesso restrito a desenvolvedores."}), 403)

    return auth_context, None


ONLINE_USERS_TTL_SECONDS = max(45.0, to_number(get_env("ONLINE_USERS_TTL_SECONDS", "120"), 120.0))


def build_online_user_key(user: dict) -> str:
    user_id = str((user or {}).get("id", "") or "").strip()
    if user_id:
        return f"id:{user_id}"

    user_email = str((user or {}).get("email", "") or "").strip().lower()
    if user_email:
        return f"email:{user_email}"

    user_phone = normalize_employee_phone_digits(str((user or {}).get("numero_celular", "") or ""))
    if user_phone:
        return f"phone:{user_phone}"

    user_name_key = normalize_name_key(str((user or {}).get("nome", "") or ""))
    if user_name_key:
        return f"name:{user_name_key}"

    return ""


def prune_online_users_locked(now_ts: float | None = None) -> None:
    now_value = time.time() if now_ts is None else now_ts
    stale_before = now_value - ONLINE_USERS_TTL_SECONDS
    stale_keys: list[str] = []
    for key, record in ONLINE_USERS_STATE.items():
        if not isinstance(record, dict):
            stale_keys.append(key)
            continue
        last_seen_ts = to_number(record.get("last_seen_ts"), 0.0)
        if last_seen_ts <= stale_before:
            stale_keys.append(key)
    for key in stale_keys:
        ONLINE_USERS_STATE.pop(key, None)


def upsert_online_user(user: dict) -> int:
    if not isinstance(user, dict):
        return 0

    key = build_online_user_key(user)
    if not key:
        return 0

    now_ts = time.time()
    online_user_entry = {
        "id": str(user.get("id", "") or "").strip(),
        "nome": str(user.get("nome", "") or "").strip(),
        "funcao": str(user.get("funcao", "") or "").strip(),
        "foto_perfil_data_url": str(user.get("foto_perfil_data_url", "") or "").strip(),
        "last_seen_iso": now_brazil().isoformat(),
        "last_seen_ts": now_ts,
    }

    with ONLINE_USERS_LOCK:
        prune_online_users_locked(now_ts)
        ONLINE_USERS_STATE[key] = online_user_entry
        return len(ONLINE_USERS_STATE)


def remove_online_user(user: dict) -> int:
    key = build_online_user_key(user if isinstance(user, dict) else {})
    with ONLINE_USERS_LOCK:
        if key:
            ONLINE_USERS_STATE.pop(key, None)
        prune_online_users_locked(time.time())
        return len(ONLINE_USERS_STATE)


def list_online_users() -> list:
    with ONLINE_USERS_LOCK:
        prune_online_users_locked(time.time())
        records = [dict(item) for item in ONLINE_USERS_STATE.values() if isinstance(item, dict)]

    records.sort(
        key=lambda item: (
            normalize_name_key(str(item.get("nome", "") or "")),
            -to_number(item.get("last_seen_ts"), 0.0),
        )
    )
    return records


def require_login_redirect():
    if not is_user_logged_in():
        return redirect(url_for("serve_login_page"))
    return None


def require_page_redirect(page_key: str):
    blocked = require_login_redirect()
    if blocked:
        return blocked

    from permissoes_paginas import can_access_page, first_allowed_page

    auth_context = get_authenticated_user_context()
    tags = list((auth_context or {}).get("tags_acesso", []))
    if not can_access_page(tags, page_key):
        return redirect("/" + first_allowed_page(tags))
    return None


def require_analytics_redirect():
    blocked = require_login_redirect()
    if blocked:
        return blocked

    auth_context = get_authenticated_user_context()
    if not auth_context or not auth_context.get("permissoes", {}).get("analytics"):
        return redirect(url_for("serve_superpop_file"))
    return None


def require_admin_only_redirect():
    blocked = require_login_redirect()
    if blocked:
        return blocked

    auth_context = get_authenticated_user_context()
    if not auth_context or not auth_context.get("permissoes", {}).get("manage_users"):
        return redirect(url_for("serve_superpop_file"))
    return None


def require_developer_only_redirect():
    blocked = require_login_redirect()
    if blocked:
        return blocked

    auth_context = get_authenticated_user_context()
    if not auth_context or not auth_context.get("permissoes", {}).get("edit_users"):
        return redirect(url_for("serve_superpop_file"))
    return None


@app.get("/")
def serve_superpop_home():
    if is_user_logged_in():
        return redirect(url_for("serve_superpop_file"))
    return redirect(url_for("serve_login_page"))


@app.get("/superpop.html")
def serve_superpop_file():
    blocked = require_page_redirect("superpop")
    if blocked:
        return blocked
    return send_page_or_frontend("superpop.html")


@app.get("/rank")
@app.get("/rank.html")
def serve_rank_page():
    blocked = require_page_redirect("rank")
    if blocked:
        return blocked
    return send_page_or_frontend("rank.html")


@app.get("/ganhadores")
@app.get("/ganhadores.html")
def serve_month_winners_page():
    blocked = require_page_redirect("ganhadores")
    if blocked:
        return blocked
    return send_page_or_frontend("ganhadores.html")


@app.get("/meus-superpops")
@app.get("/meus-superpops.html")
def serve_my_superpops_page():
    blocked = require_page_redirect("meus_superpops")
    if blocked:
        return blocked
    return send_page_or_frontend("meus-superpops.html")


@app.get("/aniversariantes")
@app.get("/aniversariantes.html")
def serve_birthdays_page():
    blocked = require_page_redirect("aniversariantes")
    if blocked:
        return blocked
    return send_page_or_frontend("aniversariantes.html")


@app.get("/dinamicas-pop")
@app.get("/dinamicas-pop.html")
def serve_pop_dynamics_page():
    blocked = require_page_redirect("dinamicas_pop")
    if blocked:
        return blocked
    return send_page_or_frontend("dinamicas-pop.html")


@app.get("/sobre")
@app.get("/sobre.html")
def serve_about_page():
    blocked = require_page_redirect("sobre")
    if blocked:
        return blocked
    return send_page_or_frontend("sobre.html")


@app.get("/atualizacoes")
@app.get("/atualizacoes.html")
def serve_updates_page():
    blocked = require_page_redirect("atualizacoes")
    if blocked:
        return blocked
    return send_page_or_frontend("atualizacoes.html")


@app.get("/atualizacoes-editor")
@app.get("/atualizacoes-editor.html")
def serve_updates_editor_page():
    blocked = require_page_redirect("atualizacoes_editor")
    if blocked:
        return blocked
    return send_page_or_frontend("atualizacoes-editor.html")


@app.get("/perfil")
@app.get("/perfil.html")
def serve_profile_page():
    blocked = require_page_redirect("perfil")
    if blocked:
        return blocked
    return send_page_or_frontend("perfil.html")


@app.get("/analise")
@app.get("/analise.html")
def serve_analytics_page():
    blocked = require_page_redirect("analise")
    if blocked:
        return blocked
    return send_page_or_frontend("analise.html")


@app.get("/usuarios")
@app.get("/usuarios.html")
def serve_admin_users_page():
    blocked = require_page_redirect("usuarios")
    if blocked:
        return blocked
    return send_page_or_frontend("usuarios.html")


@app.get("/editar-usuarios")
@app.get("/editar-usuarios.html")
def serve_edit_users_page():
    blocked = require_page_redirect("editar_usuarios")
    if blocked:
        return blocked
    return send_page_or_frontend("editar-usuarios.html")



@app.get("/criar-usuarios")
@app.get("/criar-usuarios.html")
def serve_create_users_page():
    blocked = require_page_redirect("criar_usuarios")
    if blocked:
        return blocked
    return send_page_or_frontend("criar-usuarios.html")


@app.get("/permissoes-paginas")
@app.get("/permissoes-paginas.html")
def serve_page_permissions_page():
    blocked = require_developer_only_redirect()
    if blocked:
        return blocked
    return send_page_or_frontend("permissoes-paginas.html")


@app.get("/cadastro")
@app.get("/cadastro.html")
def serve_register_page():
    if is_user_logged_in():
        return redirect(url_for("serve_superpop_file"))
    return send_page_or_frontend("cadastro.html")


@app.get("/login")
@app.get("/login.html")
@app.get("/index")
@app.get("/index.html")
def serve_login_page():
    if is_user_logged_in():
        return redirect(url_for("serve_superpop_file"))
    return send_page_or_frontend("index.html")


@app.get("/acesso")
@app.get("/acesso.html")
def serve_access_page():
    return redirect(url_for("serve_login_page"))


@app.get("/Dados.json")
def serve_dados_file():
    return send_from_directory(BASE_DIR, "Dados.json")


@app.get("/FuncoesSupermercado.json")
def serve_funcoes_supermercado_file():
    return send_from_directory(BASE_DIR, "FuncoesSupermercado.json")


@app.get("/Funcioinarios.json")
def serve_funcioinarios_file():
    return send_from_directory(BASE_DIR, "Funcioinarios.json")


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "superpop-backend"})


@app.get("/media/<path:filename>")
def serve_media(filename: str):
    return send_from_directory(CARDS_DIR, filename)


@app.get("/api/whatsapp-webjs/status")
def whatsapp_webjs_status():
    api_base = get_env("WHATSAPP_WEBJS_API_URL")
    if not api_base:
        return jsonify({"ok": False, "enabled": False, "error": "WHATSAPP_WEBJS_API_URL nao configurado."}), 400

    token = get_env("WHATSAPP_WEBJS_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout_seconds = max(5.0, to_number(get_env("WHATSAPP_WEBJS_TIMEOUT_SECONDS", "45"), 45.0))
    status_code, payload, error = get_json_request(
        url=f"{api_base.rstrip('/')}/session/status",
        headers=headers,
        timeout=timeout_seconds,
    )
    if status_code >= 200 and status_code < 300 and isinstance(payload, dict):
        return jsonify({"ok": True, "enabled": True, "status": payload})

    return (
        jsonify(
            {
                "ok": False,
                "enabled": True,
                "error": (payload or {}).get("error", "") if isinstance(payload, dict) else "",
                "detail": error,
                "status_code": status_code,
            }
        ),
        502,
    )


@app.get("/api/whatsapp-webjs/qr")
def whatsapp_webjs_qr():
    api_base = get_env("WHATSAPP_WEBJS_API_URL")
    if not api_base:
        return jsonify({"ok": False, "enabled": False, "error": "WHATSAPP_WEBJS_API_URL nao configurado."}), 400

    token = get_env("WHATSAPP_WEBJS_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout_seconds = max(5.0, to_number(get_env("WHATSAPP_WEBJS_TIMEOUT_SECONDS", "45"), 45.0))
    status_code, payload, error = get_json_request(
        url=f"{api_base.rstrip('/')}/session/qr",
        headers=headers,
        timeout=timeout_seconds,
    )
    if status_code >= 200 and status_code < 300 and isinstance(payload, dict):
        return jsonify({"ok": True, "enabled": True, "qr": payload})

    return (
        jsonify(
            {
                "ok": False,
                "enabled": True,
                "error": (payload or {}).get("error", "") if isinstance(payload, dict) else "",
                "detail": error,
                "status_code": status_code,
            }
        ),
        502,
    )


@app.get("/api/cards/verify/<card_id>")
def verify_card(card_id: str):
    token = str(request.args.get("token", "")).strip()
    if not token:
        return jsonify({"ok": False, "auth_valid": False, "error": "Token de autenticacao ausente."}), 400

    expected_token = build_card_auth_token(card_id)
    if not hmac.compare_digest(token, expected_token):
        return jsonify({"ok": False, "auth_valid": False, "error": "Token de autenticacao invalido."}), 401

    for record in reversed(read_logs()):
        if str(record.get("card_id", "")).strip() == card_id:
            return jsonify({"ok": True, "auth_valid": True, "card_id": card_id, "registro": record})

    return jsonify({"ok": False, "auth_valid": False, "error": "Cartao nao encontrado nos registros."}), 404


@app.get("/api/rank")
def api_rank():
    logs, source_info = load_logs_for_history_view()
    payload = build_rank_payload(
        logs,
        str(source_info.get("url_configurada", MY_SUPERPOPS_SOURCE_URL)),
        str(source_info.get("url_resolvida", normalize_layout_source_url(MY_SUPERPOPS_SOURCE_URL))),
    )
    payload["fonte"] = source_info
    return jsonify(payload)


@app.get("/api/rank/reactions")
def api_rank_reactions():
    month_param = str(request.args.get("month", "")).strip()
    rank_kind_param = str(request.args.get("rank_kind", "")).strip().lower()
    month_key = month_param or now_brazil().strftime("%Y-%m")

    month_match = re.fullmatch(r"(\d{4})-(\d{2})", month_key)
    if not month_match:
        return jsonify({"ok": False, "error": "Parametro month invalido. Use YYYY-MM."}), 400
    month_number = int(month_match.group(2))
    if month_number < 1 or month_number > 12:
        return jsonify({"ok": False, "error": "Parametro month invalido. Mes fora do intervalo."}), 400

    if rank_kind_param and rank_kind_param not in RANK_REACTION_RANK_KINDS:
        return jsonify({"ok": False, "error": "Parametro rank_kind invalido."}), 400

    auth_context = get_authenticated_user_context()
    viewer = (auth_context or {}).get("usuario", {}) if isinstance(auth_context, dict) else {}
    viewer_id = str((viewer or {}).get("id", "")).strip()

    records, source_info = load_rank_reactions_for_view()
    payload = build_rank_reactions_payload(records, month_key=month_key, viewer_user_id=viewer_id)
    if rank_kind_param:
        payload["items"] = [item for item in payload.get("items", []) if str(item.get("rank_kind", "")) == rank_kind_param]
    payload["source"] = source_info
    payload["viewer"] = {
        "id": viewer_id,
        "nome": str((viewer or {}).get("nome", "")).strip(),
    }
    return jsonify(payload)


@app.post("/api/rank/reactions")
def api_rank_reactions_save():
    if not is_user_logged_in():
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    viewer = auth_context.get("usuario", {}) or {}
    viewer_id = str(viewer.get("id", "")).strip()
    viewer_name = str(viewer.get("nome", "")).strip()
    if not viewer_id and not viewer_name:
        return jsonify({"ok": False, "error": "Usuario autenticado sem identificacao valida."}), 400

    payload = request.get_json(silent=True) or {}
    month_key = str(payload.get("month", "")).strip() or now_brazil().strftime("%Y-%m")
    rank_kind = str(payload.get("rank_kind", "")).strip().lower()
    target_name = str(payload.get("target_name", "")).strip()
    emoji = str(payload.get("emoji", "")).strip()

    save_result, save_error = save_rank_reaction(
        month_key=month_key,
        rank_kind=rank_kind,
        target_name=target_name,
        emoji=emoji,
        reactor_id=viewer_id,
        reactor_nome=viewer_name,
    )
    if save_error:
        return jsonify({"ok": False, "error": save_error}), 400

    github_sync = save_result.get("github_sync", {}) if isinstance(save_result, dict) else {}
    github_required = is_github_sync_required()
    github_synced = bool(github_sync.get("synced"))
    if github_required and not github_synced:
        reason = str(github_sync.get("reason", "")).strip()
        error_message = "Nao foi possivel salvar a reacao no GitHub."
        if reason:
            error_message = f"{error_message} Motivo: {reason}"
        return jsonify({"ok": False, "error": error_message, "github_sync": github_sync}), 503

    records = save_result.get("records", []) if isinstance(save_result, dict) else []
    response_payload = build_rank_reactions_payload(records, month_key=month_key, viewer_user_id=viewer_id)
    response_payload["viewer"] = {"id": viewer_id, "nome": viewer_name}
    response_payload["github_sync"] = github_sync
    return jsonify(response_payload)


@app.get("/api/system-updates")
def api_system_updates():
    if not is_user_logged_in():
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    records, source_info = load_system_updates_for_view()
    payload = build_system_updates_payload(records)
    payload["source"] = source_info
    return jsonify(payload)


@app.post("/api/system-updates")
def api_system_updates_save():
    auth_context, blocked = require_developer_only_api_context()
    if blocked:
        return blocked

    payload = request.get_json(silent=True) or {}
    save_result, save_error = save_system_update(payload, auth_context.get("usuario", {}) if auth_context else {})
    if save_error:
        return jsonify({"ok": False, "error": save_error}), 400

    github_sync = save_result.get("github_sync", {}) if isinstance(save_result, dict) else {}
    github_required = is_github_sync_required()
    github_synced = bool(github_sync.get("synced"))
    if github_required and not github_synced:
        reason = str(github_sync.get("reason", "")).strip()
        error_message = "Nao foi possivel salvar a atualizacao no GitHub."
        if reason:
            error_message = f"{error_message} Motivo: {reason}"
        return jsonify({"ok": False, "error": error_message, "github_sync": github_sync}), 503

    records = save_result.get("records", []) if isinstance(save_result, dict) else []
    response_payload = build_system_updates_payload(records)
    response_payload["github_sync"] = github_sync
    response_payload["saved"] = save_result.get("record", {}) if isinstance(save_result, dict) else {}
    response_payload["source"] = {"tipo": "local", "local_total": len(records)}
    return jsonify(response_payload)


@app.put("/api/system-updates/<update_id>")
def api_system_updates_update(update_id: str):
    auth_context, blocked = require_developer_only_api_context()
    if blocked:
        return blocked

    payload = request.get_json(silent=True) or {}
    update_result, update_error, status_code = update_system_update(
        update_id=update_id,
        payload=payload,
        actor=auth_context.get("usuario", {}) if auth_context else {},
    )
    if update_error:
        return jsonify({"ok": False, "error": update_error}), status_code

    github_sync = update_result.get("github_sync", {}) if isinstance(update_result, dict) else {}
    github_required = is_github_sync_required()
    github_synced = bool(github_sync.get("synced"))
    if github_required and not github_synced:
        reason = str(github_sync.get("reason", "")).strip()
        error_message = "Nao foi possivel atualizar a atualizacao no GitHub."
        if reason:
            error_message = f"{error_message} Motivo: {reason}"
        return jsonify({"ok": False, "error": error_message, "github_sync": github_sync}), 503

    records = update_result.get("records", []) if isinstance(update_result, dict) else []
    response_payload = build_system_updates_payload(records)
    response_payload["github_sync"] = github_sync
    response_payload["updated"] = update_result.get("record", {}) if isinstance(update_result, dict) else {}
    response_payload["source"] = {"tipo": "local", "local_total": len(records)}
    return jsonify(response_payload)


@app.delete("/api/system-updates/<update_id>")
def api_system_updates_delete(update_id: str):
    _auth_context, blocked = require_developer_only_api_context()
    if blocked:
        return blocked

    delete_result, delete_error, status_code = delete_system_update(update_id=update_id)
    if delete_error:
        return jsonify({"ok": False, "error": delete_error}), status_code

    github_sync = delete_result.get("github_sync", {}) if isinstance(delete_result, dict) else {}
    github_required = is_github_sync_required()
    github_synced = bool(github_sync.get("synced"))
    if github_required and not github_synced:
        reason = str(github_sync.get("reason", "")).strip()
        error_message = "Nao foi possivel excluir a atualizacao no GitHub."
        if reason:
            error_message = f"{error_message} Motivo: {reason}"
        return jsonify({"ok": False, "error": error_message, "github_sync": github_sync}), 503

    records = delete_result.get("records", []) if isinstance(delete_result, dict) else []
    response_payload = build_system_updates_payload(records)
    response_payload["github_sync"] = github_sync
    response_payload["deleted"] = delete_result.get("record", {}) if isinstance(delete_result, dict) else {}
    response_payload["source"] = {"tipo": "local", "local_total": len(records)}
    return jsonify(response_payload)


@app.get("/api/admin/analytics")
def api_admin_analytics():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    if not auth_context.get("permissoes", {}).get("analytics"):
        return jsonify({"ok": False, "error": "Acesso restrito a administradores e desenvolvedores."}), 403

    logs, logs_source = load_logs_for_history_view()
    employees, employees_source = load_employees_for_admin_view()
    return jsonify(build_analytics_payload(logs, employees, logs_source, employees_source))


@app.get("/api/admin/users")
def api_admin_users():
    _auth_context, blocked = require_admin_only_api_context()
    if blocked:
        return blocked

    employees, employees_source = load_employees_for_admin_view()
    return jsonify(build_admin_users_payload(employees, employees_source))


@app.get("/api/dev/users")
def api_dev_users():
    _auth_context, blocked = require_developer_only_api_context()
    if blocked:
        return blocked

    employees, employees_source = load_employees_for_admin_view()
    return jsonify(build_admin_users_payload(employees, employees_source))



@app.post("/api/dev/users")
def api_dev_create_user():
    _auth_context, blocked = require_developer_only_api_context()
    if blocked:
        return blocked

    raw_payload = request.get_json(silent=True) or {}
    payload = normalize_employee_payload(raw_payload)
    payload = prepare_developer_user_payload(payload)

    valid, validation_error = validate_employee_payload(payload)
    if not valid:
        return jsonify({"ok": False, "error": validation_error}), 400

    birth_iso, birth_error = parse_birth_date_iso(payload.get("data_nascimento"))
    if birth_error:
        return jsonify({"ok": False, "error": birth_error}), 400
    payload["data_nascimento_iso"] = birth_iso

    phone_digits = normalize_employee_phone_digits(payload.get("numero_celular", ""))

    with EMPLOYEES_FILE_LOCK:
        try:
            backup_employees()
        except Exception as exc:  # pragma: no cover - best-effort
            current_app.logger.warning(
                "Falha ao criar backup de funcionarios antes do cadastro dev: %s", exc
            )
        existing_records = read_employees()
        existing_records, source_error = load_employee_records_for_write_validation(existing_records)
        if source_error:
            return jsonify({"ok": False, "error": source_error}), 503
        duplicated, duplicate_error = find_duplicate_employee(
            existing_records,
            phone_digits,
            payload.get("email", ""),
        )
        if duplicated:
            return jsonify({"ok": False, "error": duplicate_error, "duplicate": True}), 409

    created_iso = now_brazil().isoformat()
    employee_record = build_employee_record(payload, created_iso)
    github_sync = append_employee_record(employee_record)
    github_synced = bool(github_sync.get("synced"))
    github_required = is_github_sync_required()
    if github_required and not github_synced:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Falha ao sincronizar Funcioinarios.json com o GitHub. Cadastro salvo apenas localmente.",
                    "saved_local": True,
                    "github_sync": github_sync,
                    "usuario": build_employee_public_record(employee_record),
                }
            ),
            503,
        )

    return jsonify(
        {
            "ok": True,
            "usuario": build_employee_public_record(employee_record),
            "github_sync": github_sync,
        }
    )


@app.put("/api/dev/users/<employee_id>")
def api_dev_update_user(employee_id: str):
    _auth_context, blocked = require_developer_only_api_context()
    if blocked:
        return blocked

    payload = normalize_employee_edit_payload(request.get_json(silent=True) or {})
    valid, validation_error = validate_employee_edit_payload(payload)
    if not valid:
        return jsonify({"ok": False, "error": validation_error}), 400

    phone_digits = normalize_employee_phone_digits(payload.get("numero_celular", ""))
    email = str(payload.get("email", "") or "").strip().lower()
    new_password = str(payload.get("senha", "") or "")

    def apply_updates(current: dict, records: list) -> dict:
        duplicated, duplicate_error = find_duplicate_employee_for_update(records, employee_id, phone_digits, email)
        if duplicated:
            raise ValueError(duplicate_error)

        current["nome"] = payload["nome"]
        current["funcao"] = payload["funcao"]
        current["numero_celular"] = payload["numero_celular"]
        current["numero_normalizado"] = phone_digits
        current["email"] = email
        current["tags_acesso"] = list(payload.get("tags_acesso", []))
        if new_password:
            salt_hex, hash_hex, iterations = build_password_hash(new_password)
            current["senha"] = {
                "algoritmo": "pbkdf2_sha256",
                "salt": salt_hex,
                "hash": hash_hex,
                "iteracoes": iterations,
            }
        return current

    updated_employee, github_sync = update_employee_record_with_records(employee_id, apply_updates)
    if not updated_employee:
        reason = str(github_sync.get("reason", "") or "").strip()
        if reason == "Funcionario nao encontrado.":
            return jsonify({"ok": False, "error": reason}), 404
        if reason in {"Ja existe outro cadastro com esse numero de celular.", "Ja existe outro cadastro com esse email."}:
            return jsonify({"ok": False, "error": reason}), 409
        if reason:
            return jsonify({"ok": False, "error": reason}), 400
        return jsonify({"ok": False, "error": "Nao foi possivel atualizar o usuario."}), 500

    public_employee = build_employee_public_record(updated_employee)
    if is_github_sync_required() and not github_sync.get("synced"):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Usuario atualizado localmente, mas nao foi possivel sincronizar Funcioinarios.json com o GitHub. A alteracao ainda pode nao aparecer na base principal.",
                    "saved_local": True,
                    "github_sync": github_sync,
                    "usuario": public_employee,
                }
            ),
            503,
        )

    response_payload = {
        "ok": True,
        "usuario": {
            "id": public_employee.get("id", ""),
            "nome": public_employee.get("nome", ""),
            "telefone": public_employee.get("numero_celular", ""),
            "funcao": public_employee.get("funcao", ""),
            "email": public_employee.get("email", ""),
            "data_cadastro_iso": public_employee.get("data_cadastro_iso", ""),
            "tags_acesso": list(public_employee.get("tags_acesso", [])),
        },
    }
    if not github_sync.get("synced"):
        response_payload["warning"] = (
            "Usuario atualizado localmente, mas houve falha ao sincronizar Funcioinarios.json com o GitHub."
        )
    return jsonify(response_payload)


@app.delete("/api/dev/users/<employee_id>")
def api_dev_delete_user(employee_id: str):
    auth_context, blocked = require_developer_only_api_context()
    if blocked:
        return blocked

    wanted_id = str(employee_id or "").strip()
    current_user_id = str(((auth_context or {}).get("usuario") or {}).get("id", "")).strip()
    if current_user_id and wanted_id and current_user_id == wanted_id:
        return jsonify({"ok": False, "error": "Voce nao pode excluir a propria conta."}), 400

    deleted_employee, github_sync = delete_employee_record(wanted_id)
    if not deleted_employee:
        reason = str(github_sync.get("reason", "") or "").strip()
        if reason == "Funcionario nao encontrado.":
            return jsonify({"ok": False, "error": reason}), 404
        if reason:
            return jsonify({"ok": False, "error": reason}), 400
        return jsonify({"ok": False, "error": "Nao foi possivel excluir o usuario."}), 500

    public_employee = build_employee_public_record(deleted_employee)
    if is_github_sync_required() and not github_sync.get("synced"):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Usuario excluido localmente, mas nao foi possivel sincronizar Funcioinarios.json com o GitHub. Como o GitHub e a base principal, esse usuario ainda pode bloquear um novo cadastro.",
                    "saved_local": True,
                    "github_sync": github_sync,
                    "usuario_excluido": public_employee,
                }
            ),
            503,
        )

    response_payload = {
        "ok": True,
        "usuario_excluido": {
            "id": public_employee.get("id", ""),
            "nome": public_employee.get("nome", ""),
            "telefone": public_employee.get("numero_celular", ""),
            "funcao": public_employee.get("funcao", ""),
            "email": public_employee.get("email", ""),
            "tags_acesso": list(public_employee.get("tags_acesso", [])),
        },
    }
    if not github_sync.get("synced"):
        response_payload["warning"] = (
            "Usuario excluido localmente, mas houve falha ao sincronizar Funcioinarios.json com o GitHub."
        )
    return jsonify(response_payload)


@app.get("/api/me/profile")
def api_my_profile():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    return jsonify({"ok": True, "usuario": auth_context["usuario"]})


@app.put("/api/me/profile")
def api_update_my_profile():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    employee = auth_context.get("employee")
    if not isinstance(employee, dict):
        return jsonify({"ok": False, "error": "Usuario nao encontrado."}), 404

    payload = normalize_profile_update_payload(request.get_json(silent=True) or {})
    nome = payload.get("nome", "")
    email = payload.get("email", "")
    numero_celular = payload.get("numero_celular", "") or str(employee.get("numero_celular", "")).strip()
    remover_foto = bool(payload.get("remover_foto"))
    foto_perfil_data_url = payload.get("foto_perfil_data_url", "")

    if len(nome) < 3:
        return jsonify({"ok": False, "error": "Informe um nome valido com pelo menos 3 caracteres."}), 400
    if email and not EMPLOYEE_EMAIL_PATTERN.fullmatch(email):
        return jsonify({"ok": False, "error": "Email invalido."}), 400
    phone_digits = normalize_employee_phone_digits(numero_celular)
    if len(phone_digits) != 11 or phone_digits[2] != "9":
        return jsonify({"ok": False, "error": "Numero de celular invalido."}), 400
    if not EMPLOYEE_PHONE_PATTERN.fullmatch(numero_celular):
        numero_celular = (
            f"({phone_digits[0:2]}) {phone_digits[2]} {phone_digits[3:7]} - {phone_digits[7:11]}"
        )

    birth_iso, birth_error = parse_birth_date_iso(payload.get("data_nascimento"))
    if birth_error:
        return jsonify({"ok": False, "error": birth_error}), 400

    normalized_photo = ""
    if not remover_foto and foto_perfil_data_url:
        normalized_photo, photo_error = normalize_profile_image_data_url(foto_perfil_data_url)
        if photo_error:
            return jsonify({"ok": False, "error": photo_error}), 400

    employee_id = str(employee.get("id", "")).strip()

    def apply_updates(current: dict, records: list) -> dict:
        duplicated, duplicate_error = find_duplicate_employee_for_update(records, employee_id, phone_digits, email)
        if duplicated:
            raise ValueError(duplicate_error)

        current["nome"] = nome
        current["email"] = email
        current["numero_celular"] = numero_celular
        current["numero_normalizado"] = phone_digits
        current["data_nascimento_iso"] = birth_iso
        current["mostrar_aniversario"] = to_bool(payload.get("mostrar_aniversario"), False)
        if remover_foto:
            current["foto_perfil_data_url"] = ""
        elif normalized_photo:
            current["foto_perfil_data_url"] = normalized_photo
        return current

    updated_employee, github_sync = update_employee_record_with_records(employee_id, apply_updates)
    if not updated_employee:
        reason = str(github_sync.get("reason", "") or "").strip()
        if reason in {"Ja existe outro cadastro com esse numero de celular.", "Ja existe outro cadastro com esse email."}:
            return jsonify({"ok": False, "error": reason}), 409
        return jsonify({"ok": False, "error": "Nao foi possivel atualizar o perfil."}), 500

    public_employee = build_employee_public_record(updated_employee)
    session["auth_user_nome"] = public_employee.get("nome", "")
    session["auth_user_email"] = public_employee.get("email", "")
    session["auth_user_numero"] = public_employee.get("numero_celular", "")

    return jsonify(
        {
            "ok": True,
            "usuario": {
                **public_employee,
                "login_at": str(session.get("auth_login_at", "")).strip(),
            },
            "github_sync": github_sync,
        }
    )


@app.post("/api/me/password")
def api_update_my_password():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    employee = auth_context.get("employee")
    if not isinstance(employee, dict):
        return jsonify({"ok": False, "error": "Usuario nao encontrado."}), 404

    payload = request.get_json(silent=True) or {}
    senha_atual = str(payload.get("senha_atual", "") or "")
    nova_senha = str(payload.get("nova_senha", "") or "")
    confirmar_senha = str(payload.get("confirmar_senha", "") or "")

    if not senha_atual:
        return jsonify({"ok": False, "error": "Informe a senha atual."}), 400
    if len(nova_senha) < 6:
        return jsonify({"ok": False, "error": "A nova senha deve ter pelo menos 6 caracteres."}), 400
    if confirmar_senha and nova_senha != confirmar_senha:
        return jsonify({"ok": False, "error": "A confirmacao da senha nao confere."}), 400
    if not verify_employee_password(employee, senha_atual):
        return jsonify({"ok": False, "error": "Senha atual invalida."}), 401
    if verify_employee_password(employee, nova_senha):
        return jsonify({"ok": False, "error": "A nova senha deve ser diferente da atual."}), 400

    salt_hex, hash_hex, iterations = build_password_hash(nova_senha)
    employee_id = str(employee.get("id", "")).strip()

    def apply_password(current: dict) -> dict:
        current["senha"] = {
            "algoritmo": "pbkdf2_sha256",
            "salt": salt_hex,
            "hash": hash_hex,
            "iteracoes": iterations,
        }
        return current

    updated_employee, github_sync = update_employee_record(employee_id, apply_password)
    if not updated_employee:
        return jsonify({"ok": False, "error": "Nao foi possivel atualizar a senha."}), 500

    return jsonify({"ok": True, "github_sync": github_sync})


@app.get("/api/me/superpops")
def api_my_superpops():
    if not is_user_logged_in():
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    month_param = str(request.args.get("month", "")).strip()
    if month_param:
        month_match = re.fullmatch(r"(\d{4})-(\d{2})", month_param)
        if not month_match:
            return jsonify({"ok": False, "error": "Parametro month invalido. Use YYYY-MM."}), 400
        month_number = int(month_match.group(2))
        if month_number < 1 or month_number > 12:
            return jsonify({"ok": False, "error": "Parametro month invalido. Mes fora do intervalo."}), 400

    auth_user_id = str(session.get("auth_user_id", "")).strip()
    auth_user_nome = repair_mojibake_text(session.get("auth_user_nome", "")).strip()
    auth_user_funcao = repair_mojibake_text(session.get("auth_user_funcao", "")).strip()
    auth_user_numero = str(session.get("auth_user_numero", "")).strip()

    if auth_user_id and (not auth_user_nome or not auth_user_numero):
        with EMPLOYEES_FILE_LOCK:
            employee = find_employee_by_id(read_employees(), auth_user_id)
        if employee:
            public_employee = build_employee_public_record(employee)
            auth_user_nome = public_employee.get("nome", "")
            auth_user_funcao = public_employee.get("funcao", "")
            auth_user_numero = public_employee.get("numero_celular", "")
            session["auth_user_nome"] = auth_user_nome
            session["auth_user_funcao"] = auth_user_funcao
            session["auth_user_numero"] = auth_user_numero

    if not auth_user_nome and not auth_user_numero:
        return jsonify({"ok": False, "error": "Sessao invalida. Faca login novamente."}), 401

    logs, source_info = load_logs_for_history_view()
    current_month_key = now_brazil().strftime("%Y-%m")
    selected_month_key = month_param or current_month_key
    user_name_key = normalize_name_key(auth_user_nome)
    user_number = normalize_whatsapp_number(auth_user_numero)

    month_counters: dict[str, dict] = {}
    sent_records: list[dict] = []
    received_records: list[dict] = []

    for record in logs:
        if not isinstance(record, dict):
            continue

        sender_match, receiver_match = log_matches_user(record, user_name_key, user_number)
        if not sender_match and not receiver_match:
            continue

        month_key = extract_month_key_from_log(record)
        if not month_key:
            continue

        if month_key not in month_counters:
            month_counters[month_key] = {
                "chave": month_key,
                "label": format_month_label(month_key),
                "total_registros": 0,
                "enviados": 0,
                "recebidos": 0,
            }

        month_counters[month_key]["total_registros"] += 1
        if sender_match:
            month_counters[month_key]["enviados"] += 1
        if receiver_match:
            month_counters[month_key]["recebidos"] += 1

        if month_key == selected_month_key:
            if sender_match:
                sent_records.append(record)
            if receiver_match:
                received_records.append(record)

    if current_month_key not in month_counters:
        month_counters[current_month_key] = {
            "chave": current_month_key,
            "label": format_month_label(current_month_key),
            "total_registros": 0,
            "enviados": 0,
            "recebidos": 0,
        }

    if selected_month_key not in month_counters:
        month_counters[selected_month_key] = {
            "chave": selected_month_key,
            "label": format_month_label(selected_month_key),
            "total_registros": 0,
            "enviados": 0,
            "recebidos": 0,
        }

    sent_records.sort(key=parse_log_timestamp, reverse=True)
    received_records.sort(key=parse_log_timestamp, reverse=True)
    sent_items = [build_user_log_item(record, role="sent") for record in sent_records]
    received_items = [build_user_log_item(record, role="received") for record in received_records]
    months_available = [month_counters[key] for key in sorted(month_counters.keys(), reverse=True)]

    return jsonify(
        {
            "ok": True,
            "usuario": {
                "id": auth_user_id,
                "nome": auth_user_nome,
                "funcao": auth_user_funcao,
                "numero_celular": auth_user_numero,
            },
            "mes_atual": current_month_key,
            "mes_selecionado": selected_month_key,
            "meses_disponiveis": months_available,
            "resumo_mes": month_counters[selected_month_key],
            "enviados": sent_items,
            "recebidos": received_items,
            "fonte": source_info,
        }
    )


def _collect_today_superpop_notifications(
    auth_user_id: str,
    auth_user_name: str,
    auth_user_number: str,
    reference_date: date,
) -> list[dict]:
    today_key = reference_date.strftime("%d/%m/%Y")
    normalized_number = normalize_whatsapp_number(auth_user_number)
    name_key = normalize_name_key(auth_user_name)
    state = get_user_notification_state(auth_user_id)
    seen_ids = {str(item).strip() for item in state.get("seen_ids", []) if str(item or "").strip()}
    cleared_ts = _parse_iso_timestamp(state.get("last_cleared_iso"))
    entries: list[dict] = []

    with DATA_FILE_LOCK:
        records = read_logs()

    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("dia", "")).strip() != today_key:
            continue
        if not _record_matches_notification_user(record, normalized_number, name_key):
            continue
        entry: dict = {
            "id": str(record.get("id", "")).strip(),
            "card_id": str(record.get("card_id", "")).strip(),
            "dia": str(record.get("dia", "")).strip(),
            "horario": str(record.get("horario", "")).strip(),
            "data_hora_iso": str(record.get("data_hora_iso", "")).strip(),
            "mensagem": repair_mojibake_text(record.get("mensagem", "")).strip(),
            "valores": [],
            "remetente": {},
        }
        opcoes = record.get("opcoes_marcadas", [])
        if isinstance(opcoes, list):
            for item in opcoes:
                cleaned = repair_mojibake_text(item).strip()
                if cleaned:
                    entry["valores"].append(cleaned)
        remetente = record.get("remetente") or {}
        if isinstance(remetente, dict):
            entry["remetente"] = {
                "nome": repair_mojibake_text(remetente.get("nome", "")).strip(),
                "funcao": repair_mojibake_text(remetente.get("funcao", "")).strip(),
            }
        entry["visualizado"] = _is_notification_record_viewed(record, seen_ids, cleared_ts)
        entries.append(entry)

    entries.sort(key=parse_log_timestamp, reverse=True)
    return entries


def _notification_event_is_viewed(
    notification_id: str,
    event_iso: object,
    seen_ids: set[str],
    cleared_timestamp: float | None,
) -> bool:
    safe_id = str(notification_id or "").strip()
    if safe_id and safe_id in seen_ids:
        return True
    if cleared_timestamp is None:
        return False
    event_ts = _parse_iso_timestamp(event_iso)
    return bool(event_ts is not None and event_ts <= cleared_timestamp)


def _dynamic_notification_timestamp(entry: dict) -> float:
    parsed = _parse_iso_timestamp(entry.get("data_hora_iso"))
    return parsed if parsed is not None else 0.0


def _collect_dinamicas_pop_winner_notifications(auth_user_id: str) -> list[dict]:
    user_id = str(auth_user_id or "").strip()
    if not user_id:
        return []

    state = get_user_notification_state(user_id)
    seen_ids = {str(item).strip() for item in state.get("seen_ids", []) if str(item or "").strip()}
    cleared_ts = _parse_iso_timestamp(state.get("last_cleared_iso"))

    try:
        from dinamicas_pop import DATA_LOCK as DINAMICAS_LOCK
        from dinamicas_pop import _read_games as read_dinamicas_games
    except Exception:
        return []

    with DINAMICAS_LOCK:
        games = read_dinamicas_games()

    entries: list[dict] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        winner_prediction_id = str(game.get("palpite_ganhador_id", "")).strip()
        if not winner_prediction_id:
            continue
        predictions = game.get("palpites")
        if not isinstance(predictions, list):
            continue
        winner_prediction = next(
            (
                item
                for item in predictions
                if isinstance(item, dict)
                and str(item.get("id", "")).strip() == winner_prediction_id
                and str(item.get("usuario_id", "")).strip() == user_id
            ),
            None,
        )
        if not winner_prediction:
            continue

        selected_at = str(game.get("palpite_ganhador_selecionado_em_iso", "")).strip()
        notification_id = "dinamica_pop:" + ":".join(
            [
                str(game.get("id", "")).strip(),
                winner_prediction_id,
                selected_at or str(game.get("updated_at_iso", "")).strip() or str(game.get("created_at_iso", "")).strip(),
            ]
        )
        home_team = str(game.get("time_casa", "")).strip() or "Time da casa"
        away_team = str(game.get("time_visitante", "")).strip() or "Time visitante"
        home_score = str(winner_prediction.get("gols_casa", "")).strip()
        away_score = str(winner_prediction.get("gols_visitante", "")).strip()
        prize = str(game.get("descricao_premio", "")).strip()
        competition = str(game.get("competicao", "")).strip() or "Dinâmicas POP"
        match_label = f"{home_team} x {away_team}"
        score_label = f"{home_score} x {away_score}" if home_score or away_score else ""
        whatsapp_text = (
            "Ganhei na Dinâmica POP! "
            f"Meu palpite em {match_label}"
            + (f" foi {score_label}." if score_label else ".")
            + (f" Prêmio: {prize}." if prize else "")
        )

        entries.append(
            {
                "id": notification_id,
                "tipo": "dinamica_pop_ganhador",
                "titulo": "Você ganhou na Dinâmica POP!",
                "mensagem": (
                    f"Seu palpite foi selecionado como ganhador em {match_label}."
                    + (f" Prêmio: {prize}" if prize else "")
                ),
                "data_hora_iso": selected_at,
                "visualizado": _notification_event_is_viewed(notification_id, selected_at, seen_ids, cleared_ts),
                "competicao": competition,
                "premio": prize,
                "regras": str(game.get("regras", "")).strip(),
                "whatsapp_text": whatsapp_text,
                "jogo": {
                    "id": str(game.get("id", "")).strip(),
                    "time_casa": home_team,
                    "time_visitante": away_team,
                    "data_jogo": str(game.get("data_jogo", "")).strip(),
                    "horario_jogo": str(game.get("horario_jogo", "")).strip(),
                },
                "palpite": {
                    "id": winner_prediction_id,
                    "gols_casa": winner_prediction.get("gols_casa"),
                    "gols_visitante": winner_prediction.get("gols_visitante"),
                },
            }
        )

    entries.sort(key=_dynamic_notification_timestamp, reverse=True)
    return entries[:20]


@app.get("/api/me/notifications/superpops")
def api_superpop_notifications():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    usuario = auth_context.get("usuario") or {}
    auth_user_id = str(usuario.get("id", "")).strip()
    auth_user_nome = str(usuario.get("nome", "")).strip()
    auth_user_numero = str(usuario.get("numero_celular", "")).strip()
    today = now_brazil().date()
    notifications = _collect_today_superpop_notifications(
        auth_user_id,
        auth_user_nome,
        auth_user_numero,
        today,
    )
    notifications.extend(_collect_dinamicas_pop_winner_notifications(auth_user_id))
    notifications.sort(
        key=lambda item: _dynamic_notification_timestamp(item) or parse_log_timestamp(item),
        reverse=True,
    )
    unread = sum(1 for item in notifications if not item.get("visualizado"))
    return jsonify(
        {
            "ok": True,
            "total": len(notifications),
            "unread": unread,
            "notifications": notifications,
            "today": today.strftime("%d/%m/%Y"),
        }
    )


@app.post("/api/me/notifications/superpops/mark")
def mark_superpop_notification():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    usuario = auth_context.get("usuario") or {}
    auth_user_id = str(usuario.get("id", "")).strip()
    payload = request.get_json(silent=True) or {}
    log_id = str(payload.get("log_id") or payload.get("id") or "").strip()
    if not log_id:
        return jsonify({"ok": False, "error": "log_id obrigatorio."}), 400

    update_user_notification_state(auth_user_id, seen_ids=[log_id])
    return jsonify({"ok": True})


@app.post("/api/me/notifications/superpops/mark-all")
def mark_all_superpop_notifications():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    usuario = auth_context.get("usuario") or {}
    auth_user_id = str(usuario.get("id", "")).strip()
    marker_time = now_brazil().isoformat()
    update_user_notification_state(auth_user_id, last_cleared_iso=marker_time)
    return jsonify({"ok": True, "marked_at": marker_time})


def save_my_superpop_reaction(
    card_id: str,
    log_id: str,
    emoji: str,
    reactor_id: str,
    reactor_nome: str,
    reactor_numero: str,
) -> tuple[dict, str, int]:
    clean_card_id = str(card_id or "").strip()
    clean_log_id = str(log_id or "").strip()
    clean_emoji = str(emoji or "").strip()
    clean_reactor_id = str(reactor_id or "").strip()
    clean_reactor_nome = str(reactor_nome or "").strip()
    clean_reactor_numero = normalize_whatsapp_number(str(reactor_numero or "").strip())
    clean_reactor_name_key = normalize_name_key(clean_reactor_nome)

    if not clean_card_id and not clean_log_id:
        return {}, "Informe o card_id do Super POP.", 400
    if clean_emoji and clean_emoji not in SUPERPOP_REACTION_ALLOWED_EMOJIS:
        return {}, "Emoji invalido para reacao.", 400
    if not clean_reactor_id and not clean_reactor_name_key and not clean_reactor_numero:
        return {}, "Usuario sem identificacao valida para reagir.", 400

    with DATA_FILE_LOCK:
        records = read_logs()
        refreshed_records, _refresh_error = refresh_local_logs_from_remote(records)
        if isinstance(refreshed_records, list):
            records = refreshed_records

        target_index = find_log_record_index(records, clean_card_id, clean_log_id)
        if target_index < 0:
            return {}, "Super POP nao encontrado.", 404

        target_record = records[target_index]
        _sender_match, receiver_match = log_matches_user(target_record, clean_reactor_name_key, clean_reactor_numero)
        if not receiver_match:
            return {}, "Apenas o destinatario deste Super POP pode reagir.", 403

        if clean_emoji:
            target_record["reacao_destinatario"] = {
                "emoji": clean_emoji,
                "updated_at_iso": now_brazil().isoformat(),
                "reactor": {
                    "id": clean_reactor_id,
                    "nome": clean_reactor_nome or "Usuario",
                },
            }
        else:
            target_record.pop("reacao_destinatario", None)

        records[target_index] = target_record
        write_logs(records)

        github_sync = github_sync_logs_with_retry(records)
        merged_logs = github_sync.get("merged_logs")
        if isinstance(merged_logs, list):
            write_logs(merged_logs)
            github_sync.pop("merged_logs", None)
            records = merged_logs

        latest_index = find_log_record_index(records, clean_card_id, clean_log_id)
        if latest_index >= 0:
            target_record = records[latest_index]

    return {"records": records, "record": target_record, "github_sync": github_sync}, "", 200


@app.post("/api/me/superpops/reaction")
def api_my_superpops_reaction():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401
    auth_user = auth_context.get("usuario", {}) if isinstance(auth_context, dict) else {}

    payload = request.get_json(silent=True) or {}
    card_id = str(payload.get("card_id", "")).strip()
    log_id = str(payload.get("log_id", "")).strip()
    emoji = str(payload.get("emoji", "")).strip()

    save_result, save_error, status_code = save_my_superpop_reaction(
        card_id=card_id,
        log_id=log_id,
        emoji=emoji,
        reactor_id=str(auth_user.get("id", "")).strip(),
        reactor_nome=str(auth_user.get("nome", "")).strip(),
        reactor_numero=str(auth_user.get("numero_celular", "")).strip(),
    )
    if save_error:
        return jsonify({"ok": False, "error": save_error}), status_code

    github_sync = save_result.get("github_sync", {}) if isinstance(save_result, dict) else {}
    github_required = is_github_sync_required()
    github_synced = bool(github_sync.get("synced"))
    if github_required and not github_synced:
        reason = str(github_sync.get("reason", "")).strip()
        error_message = "Nao foi possivel salvar a reacao no GitHub."
        if reason:
            error_message = f"{error_message} Motivo: {reason}"
        return jsonify({"ok": False, "error": error_message, "github_sync": github_sync}), 503

    saved_record = save_result.get("record", {}) if isinstance(save_result, dict) else {}
    response_item = build_user_log_item(saved_record if isinstance(saved_record, dict) else {}, role="received")
    reaction_payload = response_item.get("reacao_destinatario", {}) if isinstance(response_item, dict) else {}
    reaction_emoji = str((reaction_payload or {}).get("emoji", "")).strip()

    return jsonify(
        {
            "ok": True,
            "item": response_item,
            "reacao_destinatario": reaction_payload,
            "reaction_removed": not bool(reaction_emoji),
            "github_sync": github_sync,
        }
    )


@app.post("/api/funcionarios/register")
def register_employee():
    payload = normalize_employee_payload(request.get_json(silent=True) or {})
    valid, validation_error = validate_employee_payload(payload)
    if not valid:
        return jsonify({"ok": False, "error": validation_error}), 400

    phone_digits = normalize_employee_phone_digits(payload.get("numero_celular", ""))
    pre_cadastro_id = ""
    with EMPLOYEES_FILE_LOCK:
        try:
            backup_employees()
        except Exception as exc:  # pragma: no cover - best-effort
            current_app.logger.warning(
                "Falha ao criar backup de funcionarios antes do cadastro: %s", exc
            )
        existing_records = read_employees()
        existing_records, source_error = load_employee_records_for_write_validation(existing_records)
        if source_error:
            return jsonify({"ok": False, "error": source_error}), 503
        pre_cadastro_candidate = find_employee_by_phone(existing_records, phone_digits)
        if isinstance(pre_cadastro_candidate, dict) and pre_cadastro_candidate.get("pre_cadastro"):
            pre_cadastro_id = str(pre_cadastro_candidate.get("id", "")).strip()
        else:
            duplicated, duplicate_error = find_duplicate_employee(
                existing_records,
                phone_digits,
                payload.get("email", ""),
            )
            if duplicated:
                return jsonify({"ok": False, "error": duplicate_error, "duplicate": True}), 409

    birth_iso, birth_error = parse_birth_date_iso(payload.get("data_nascimento"))
    if birth_error:
        return jsonify({"ok": False, "error": birth_error}), 400
    payload["data_nascimento_iso"] = birth_iso
    payload["pre_cadastro"] = False

    def update_pre_cadastro_record(employee_id: str) -> tuple[dict | None, dict]:
        def apply_updates(current: dict, records: list) -> dict:
            if not isinstance(current, dict):
                raise ValueError("Funcionario invalido.")
            current["nome"] = payload["nome"]
            current["funcao"] = payload["funcao"]
            current["numero_celular"] = payload["numero_celular"]
            current["numero_normalizado"] = phone_digits
            current["email"] = payload.get("email", "")
            current["data_nascimento_iso"] = payload.get("data_nascimento_iso", "")
            current["mostrar_aniversario"] = to_bool(payload.get("mostrar_aniversario"), False)
            current["tags_acesso"] = list(payload.get("tags_acesso", []))
            current["pre_cadastro"] = False
            salt_hex, hash_hex, iterations = build_password_hash(payload.get("senha", ""))
            current["senha"] = {
                "algoritmo": "pbkdf2_sha256",
                "salt": salt_hex,
                "hash": hash_hex,
                "iteracoes": iterations,
            }
            return current

        return update_employee_record_with_records(employee_id, apply_updates)

    if pre_cadastro_id:
        updated_employee, github_sync = update_pre_cadastro_record(pre_cadastro_id)
        if not updated_employee:
            reason = str(github_sync.get("reason", "") or "").strip()
            if reason:
                return jsonify({"ok": False, "error": reason}), 400
            return jsonify({"ok": False, "error": "Nao foi possivel atualizar o cadastro."}), 500

        github_synced = bool(github_sync.get("synced"))
        github_required = is_github_sync_required()
        if github_required and not github_synced:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Cadastro atualizado localmente, mas houve falha ao sincronizar Funcioinarios.json com o GitHub.",
                        "saved_local": True,
                        "github_sync": github_sync,
                        "funcionario": build_employee_public_record(updated_employee),
                    }
                ),
                503,
            )

        return jsonify(
            {
                "ok": True,
                "funcionario": build_employee_public_record(updated_employee),
                "github_sync": github_sync,
                "pre_cadastro": True,
            }
        )

    created_iso = now_brazil().isoformat()
    employee_record = build_employee_record(payload, created_iso)
    github_sync = append_employee_record(employee_record)
    github_synced = bool(github_sync.get("synced"))
    github_required = is_github_sync_required()

    if github_required and not github_synced:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Falha ao sincronizar Funcioinarios.json com o GitHub. Cadastro salvo apenas localmente.",
                    "saved_local": True,
                    "github_sync": github_sync,
                    "funcionario": build_employee_public_record(employee_record),
                }
            ),
            503,
        )

    return jsonify(
        {
            "ok": True,
            "funcionario": build_employee_public_record(employee_record),
            "github_sync": github_sync,
        }
    )


@app.post("/api/auth/login")
def login_employee():
    payload = request.get_json(silent=True) or {}
    login_raw = str(payload.get("numero_celular", "") or payload.get("login", "") or "").strip()
    password_raw = str(payload.get("senha", "") or "")
    keep_connected = to_bool(payload.get("manter_conectado"), True)
    login_email = login_raw.lower()
    phone_digits = normalize_employee_phone_digits(login_raw)
    login_is_email = bool(login_email and "@" in login_email)

    if login_is_email:
        if not EMPLOYEE_EMAIL_PATTERN.fullmatch(login_email):
            return jsonify({"ok": False, "error": "Email invalido."}), 400
    elif len(phone_digits) != 11 or (phone_digits and phone_digits[2] != "9"):
        return jsonify({"ok": False, "error": "Numero de celular invalido."}), 400
    if not password_raw:
        return jsonify({"ok": False, "error": "Senha obrigatoria."}), 400

    with EMPLOYEES_FILE_LOCK:
        records = read_employees()
        employee = find_employee_by_email(records, login_email) if login_is_email else find_employee_by_phone(records, phone_digits)
        if not employee:
            refreshed_records, _refresh_error = refresh_local_employees_from_remote(records)
            records = refreshed_records
            employee = find_employee_by_email(records, login_email) if login_is_email else find_employee_by_phone(records, phone_digits)

    if not employee:
        return jsonify({"ok": False, "error": "Usuario ou senha invalidos."}), 401

    if not verify_employee_password(employee, password_raw):
        return jsonify({"ok": False, "error": "Usuario ou senha invalidos."}), 401

    public_employee = build_employee_public_record(employee)
    session.permanent = keep_connected
    session["auth_user_id"] = public_employee.get("id", "")
    session["auth_user_nome"] = public_employee.get("nome", "")
    session["auth_user_funcao"] = public_employee.get("funcao", "")
    session["auth_user_numero"] = public_employee.get("numero_celular", "")
    session["auth_user_email"] = public_employee.get("email", "")
    session["auth_login_at"] = now_brazil().isoformat()

    return jsonify(
        {
            "ok": True,
            "funcionario": public_employee,
        }
    )


@app.post("/api/auth/password-reset/request")
def request_password_reset():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "") or "").strip().lower()
    if not EMPLOYEE_EMAIL_PATTERN.fullmatch(email):
        return jsonify({"ok": False, "error": "Email invalido."}), 400
    if not is_smtp_configured():
        return jsonify({"ok": False, "error": "Recuperacao por email indisponivel. Configure o SMTP no backend."}), 503

    with EMPLOYEES_FILE_LOCK:
        records = read_employees()
        employee = find_employee_by_email(records, email)
        if not employee:
            refreshed_records, _refresh_error = refresh_local_employees_from_remote(records)
            employee = find_employee_by_email(refreshed_records, email)

    masked_email = mask_email_for_log(email)
    if isinstance(employee, dict):
        delivery_mode = get_email_delivery_mode()
        current_app.logger.info(
            "password_reset_request matched email=%s employee_id=%s delivery_mode=%s",
            masked_email,
            str(employee.get("id", "") or "-"),
            delivery_mode,
        )
        token = build_password_reset_token(employee)
        reset_url = build_password_reset_url(token)
        sent, send_error = send_password_reset_email(employee, reset_url)
        if not sent:
            current_app.logger.warning(
                "password_reset_request failed email=%s employee_id=%s delivery_mode=%s error=%s",
                masked_email,
                str(employee.get("id", "") or "-"),
                delivery_mode,
                str(send_error or "unknown"),
            )
            return jsonify({"ok": False, "error": send_error}), 503
        current_app.logger.info(
            "password_reset_request accepted email=%s employee_id=%s delivery_mode=%s provider_status=%s",
            masked_email,
            str(employee.get("id", "") or "-"),
            delivery_mode,
            str(send_error or "ok"),
        )
    else:
        current_app.logger.info("password_reset_request no_match email=%s", masked_email)

    return jsonify(
        {
            "ok": True,
            "message": "Se existir uma conta com esse email, enviamos o link de recuperacao.",
        }
    )


@app.post("/api/auth/password-reset/confirm")
def confirm_password_reset():
    payload = request.get_json(silent=True) or {}
    token = str(payload.get("token", "") or "").strip()
    nova_senha = str(payload.get("nova_senha", "") or "")
    confirmar_senha = str(payload.get("confirmar_senha", "") or "")

    if not token:
        return jsonify({"ok": False, "error": "Token obrigatorio."}), 400
    if len(nova_senha) < 6:
        return jsonify({"ok": False, "error": "A nova senha deve ter pelo menos 6 caracteres."}), 400
    if confirmar_senha != nova_senha:
        return jsonify({"ok": False, "error": "A confirmacao da senha nao confere."}), 400

    token_payload, token_error = parse_password_reset_token(token)
    if not token_payload:
        return jsonify({"ok": False, "error": token_error or "Token invalido."}), 400

    employee_id = str(token_payload.get("employee_id", "")).strip()
    token_email = str(token_payload.get("email", "")).strip().lower()

    with EMPLOYEES_FILE_LOCK:
        employee = find_employee_by_id(read_employees(), employee_id)

    if not employee:
        return jsonify({"ok": False, "error": "Usuario nao encontrado."}), 404

    current_email = str(employee.get("email", "")).strip().lower()
    if not current_email or current_email != token_email:
        return jsonify({"ok": False, "error": "Token invalido para este usuario."}), 400
    if verify_employee_password(employee, nova_senha):
        return jsonify({"ok": False, "error": "A nova senha deve ser diferente da atual."}), 400

    salt_hex, hash_hex, iterations = build_password_hash(nova_senha)

    def apply_password(current: dict) -> dict:
        current["senha"] = {
            "algoritmo": "pbkdf2_sha256",
            "salt": salt_hex,
            "hash": hash_hex,
            "iteracoes": iterations,
        }
        return current

    updated_employee, github_sync = update_employee_record(employee_id, apply_password)
    if not updated_employee:
        return jsonify({"ok": False, "error": "Nao foi possivel redefinir a senha."}), 500

    return jsonify({"ok": True, "message": "Senha redefinida com sucesso.", "github_sync": github_sync})


@app.get("/api/system/microsoft-oauth/status")
def microsoft_oauth_status():
    _auth_context, blocked = require_admin_api_context()
    if blocked:
        return blocked

    settings = get_microsoft_oauth_settings()
    stored = load_microsoft_oauth_token_store()
    refresh_token = get_microsoft_refresh_token()
    return jsonify(
        {
            "ok": True,
            "provider": "microsoft_oauth",
            "configured": bool(settings["client_id"] and settings["redirect_uri"]),
            "connected": bool(refresh_token),
            "tenant": settings["tenant"],
            "redirect_uri": settings["redirect_uri"],
            "scope": settings["scope"],
            "has_client_secret": bool(settings["client_secret"]),
            "token_file": str(MICROSOFT_OAUTH_TOKEN_FILE),
            "updated_at_iso": str(stored.get("updated_at_iso", "") or "").strip(),
            "start_url": build_public_backend_url("api/system/microsoft-oauth/start") if settings["client_id"] else "",
        }
    )


@app.get("/api/system/microsoft-oauth/start")
def microsoft_oauth_start():
    _auth_context, blocked = require_admin_api_context()
    if blocked:
        return blocked

    settings = get_microsoft_oauth_settings()
    if not settings["client_id"]:
        return jsonify({"ok": False, "error": "MICROSOFT_OAUTH_CLIENT_ID nao configurado."}), 400

    state_token = uuid.uuid4().hex
    session["microsoft_oauth_state"] = state_token
    return redirect(build_microsoft_authorize_url(state_token))


@app.get("/api/system/microsoft-oauth/callback")
def microsoft_oauth_callback():
    returned_state = str(request.args.get("state", "") or "").strip()
    expected_state = str(session.get("microsoft_oauth_state", "") or "").strip()
    if not returned_state or not expected_state or not hmac.compare_digest(returned_state, expected_state):
        return jsonify({"ok": False, "error": "Estado OAuth invalido."}), 400

    auth_error = str(request.args.get("error", "") or "").strip()
    if auth_error:
        description = str(request.args.get("error_description", "") or auth_error).strip()
        return jsonify({"ok": False, "error": description}), 400

    code = str(request.args.get("code", "") or "").strip()
    if not code:
        return jsonify({"ok": False, "error": "Codigo OAuth ausente."}), 400

    token_response, token_error = exchange_microsoft_code_for_refresh_token(code)
    session.pop("microsoft_oauth_state", None)
    if not token_response:
        return jsonify({"ok": False, "error": token_error or "Falha ao concluir Microsoft OAuth."}), 400

    refresh_token = str(token_response.get("refresh_token", "") or "").strip()
    if not refresh_token:
        return jsonify({"ok": False, "error": "Microsoft OAuth nao retornou refresh token."}), 400

    save_microsoft_oauth_token_store(
        {
            "refresh_token": refresh_token,
            "updated_at_iso": now_brazil().isoformat(),
            "scope": str(token_response.get("scope", "") or "").strip(),
            "token_type": str(token_response.get("token_type", "Bearer") or "Bearer"),
        }
    )
    return redirect(build_frontend_url("superpop.html?microsoft_oauth=connected"))


@app.post("/api/system/microsoft-oauth/disconnect")
def microsoft_oauth_disconnect():
    _auth_context, blocked = require_admin_api_context()
    if blocked:
        return blocked

    if MICROSOFT_OAUTH_TOKEN_FILE.exists():
        try:
            MICROSOFT_OAUTH_TOKEN_FILE.unlink()
        except Exception as exc:
            return jsonify({"ok": False, "error": f"Nao foi possivel remover o token local: {exc}"}), 500

    return jsonify({"ok": True})


@app.get("/api/auth/me")
def auth_me():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    upsert_online_user(auth_context["usuario"])

    return jsonify(
        {
            "ok": True,
            "usuario": auth_context["usuario"],
            "permissoes": auth_context["permissoes"],
        }
    )


@app.post("/api/presence/heartbeat")
def presence_heartbeat():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    online_count = upsert_online_user(auth_context["usuario"])
    return jsonify({"ok": True, "online_count": online_count})


@app.get("/api/presence/online")
def presence_online():
    auth_context = get_authenticated_user_context()
    if not auth_context:
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    upsert_online_user(auth_context["usuario"])
    online_users = list_online_users()
    return jsonify(
        {
            "ok": True,
            "online_count": len(online_users),
            "online_users": online_users,
        }
    )


@app.post("/api/auth/logout")
def auth_logout():
    auth_context = get_authenticated_user_context()
    if auth_context:
        remove_online_user(auth_context.get("usuario", {}))
    session.clear()
    return jsonify({"ok": True})


@app.post("/api/cards/generate")
def generate_card():
    payload = normalize_payload(request.get_json(silent=True) or {})
    card_id = uuid.uuid4().hex[:10]
    auth_token = build_card_auth_token(card_id)
    auth_qr_url = build_card_auth_url(card_id, auth_token)
    image_name = f"card-{card_id}.png"
    image_path = CARDS_DIR / image_name

    create_card_image(payload, image_path, auth_qr_text=auth_qr_url)

    return jsonify(
        {
            "ok": True,
            "card_id": card_id,
            "auth_qr_url": auth_qr_url,
            "image_url": build_media_url(app, image_name),
            "image_file": image_name,
        }
    )


@app.post("/api/logs/register")
def register_log():
    if not is_user_logged_in():
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    payload = normalize_payload(request.get_json(silent=True) or {})
    if not payload.get("sender_device_type"):
        payload["sender_device_type"] = infer_sender_device_type_from_user_agent(
            request.headers.get("User-Agent", "")
        )
    auth_user_id = str(session.get("auth_user_id", "")).strip()
    auth_user_nome = repair_mojibake_text(session.get("auth_user_nome", "")).strip()
    auth_user_funcao = repair_mojibake_text(session.get("auth_user_funcao", "")).strip()
    auth_user_numero = str(session.get("auth_user_numero", "")).strip()

    if auth_user_id and (not auth_user_nome or not auth_user_numero):
        with EMPLOYEES_FILE_LOCK:
            employee = find_employee_by_id(read_employees(), auth_user_id)
        if employee:
            public_employee = build_employee_public_record(employee)
            auth_user_nome = public_employee.get("nome", "")
            auth_user_funcao = public_employee.get("funcao", "")
            auth_user_numero = public_employee.get("numero_celular", "")
            session["auth_user_nome"] = auth_user_nome
            session["auth_user_funcao"] = auth_user_funcao
            session["auth_user_numero"] = auth_user_numero

    if not auth_user_nome or not auth_user_numero:
        return jsonify({"ok": False, "error": "Sessao invalida. Faca login novamente."}), 401

    with EMPLOYEES_FILE_LOCK:
        employee_records = read_employees()

    payload["reconhecido_por"] = auth_user_nome
    payload["funcao_reconhecido_por"] = auth_user_funcao
    payload["numero_reconhecido_por"] = auth_user_numero
    payload = normalize_payload(payload)
    payload = enrich_superpop_payload_with_employees(
        payload=payload,
        auth_user_id=auth_user_id,
        employees=employee_records,
    )

    payload_valid, payload_error = validate_superpop_register_payload(payload)
    if not payload_valid:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": payload_error,
                    "prevent_manual_send": True,
                }
            ),
            400,
        )

    local_now = now_brazil()
    local_date = payload["data"] or local_now.strftime("%d/%m/%Y")
    local_time = local_now.strftime("%H:%M:%S")
    local_iso = local_now.isoformat()
    destination = normalize_whatsapp_number(payload["to"] or payload["numero_colaborador"])
    sender_number = normalize_whatsapp_number(payload["numero_reconhecido_por"])
    send_day_key = build_daily_send_key(sender_number, destination, local_date)
    send_day_reserved = False

    with DATA_FILE_LOCK:
        try:
            backup_logs()
        except Exception as exc:  # pragma: no cover - best-effort
            current_app.logger.warning(
                "Falha ao criar backup de logs antes de registrar um SuperPOP: %s", exc
            )
        existing_logs = read_logs()
        duplicate_record = find_duplicate_send_same_day(
            logs=existing_logs,
            sender_number=sender_number,
            destination_number=destination,
            day_value=local_date,
        )
        if duplicate_record:
            duplicate_name = (
                str((duplicate_record.get("destinatario", {}) or {}).get("nome", "")).strip()
                or payload.get("colaborador")
                or "esse colaborador"
            )
            duplicate_time = str(duplicate_record.get("horario", "")).strip()
            duplicate_date = str(duplicate_record.get("dia", "")).strip() or local_date
            duplicate_hint = f"{duplicate_date}" + (f" as {duplicate_time}" if duplicate_time else "")
            return (
                jsonify(
                    {
                        "ok": False,
                        "duplicate_send": True,
                        "error": (
                            f"Hoje voce ja enviou um SuperPOP para {duplicate_name} ({duplicate_hint}). "
                            "Para manter a regra de reconhecimento, cada pessoa pode receber apenas 1 SuperPOP seu por dia. "
                            "Tente novamente amanha."
                        ),
                        "duplicate": {
                            "dia": duplicate_date,
                            "horario": duplicate_time,
                            "card_id": str(duplicate_record.get("card_id", "")).strip(),
                        },
                    }
                ),
                409,
            )

        if send_day_key:
            if send_day_key in PENDING_SEND_KEYS:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "duplicate_send": True,
                            "error": (
                                "Seu envio para essa pessoa ainda esta em processamento. "
                                "Aguarde alguns instantes para evitar duplicidade e tente novamente."
                            ),
                        }
                    ),
                    409,
                )
            PENDING_SEND_KEYS.add(send_day_key)
            send_day_reserved = True

    try:
        card_id = uuid.uuid4().hex[:10]
        auth_token = build_card_auth_token(card_id)
        auth_qr_url = build_card_auth_url(card_id, auth_token)
        image_name = f"card-{card_id}.png"
        image_path = CARDS_DIR / image_name

        create_card_image(payload, image_path, auth_qr_text=auth_qr_url)

        format_selected = "image"
        media_url = build_media_url(app, image_name)
        image_url = build_media_url(app, image_name)
        pdf_url = ""
        imgbb_result = upload_image_to_imgbb(image_path)
        uploaded_image_url = imgbb_result["url"] if imgbb_result.get("ok") else ""
        upload_status = "success" if uploaded_image_url else "error"
        upload_error = imgbb_result.get("error", "")
        send_mode = payload.get("send_mode") or get_whatsapp_send_mode()
        if send_mode not in {"wa_me", "webjs"}:
            send_mode = get_whatsapp_send_mode()
        send_status = "wa_me"
        send_error = ""
        message_sid = ""
        webjs_result = {
            "enabled": False,
            "ok": False,
            "error": "",
            "message_id": "",
            "to": destination or "",
            "provider": "whatsapp-web.js",
        }

        if send_mode == "webjs" and uploaded_image_url and destination:
            caption = build_whatsapp_caption(payload)
            webjs_result = send_image_via_whatsapp_webjs(
                destination=destination,
                image_url=uploaded_image_url,
                caption=caption,
            )
            if webjs_result.get("ok"):
                send_status = "webjs_sent"
                message_sid = str(webjs_result.get("message_id", "")).strip()
            else:
                send_status = "webjs_error"
                send_error = str(webjs_result.get("error", "")).strip()
        elif send_mode == "webjs" and not destination:
            send_status = "webjs_error"
            send_error = "Numero de destino invalido para envio direto."

        log_record = make_log_record(
            payload=payload,
            card_id=card_id,
            auth_qr_url=auth_qr_url,
            local_date=local_date,
            local_time=local_time,
            local_iso=local_iso,
            destination=destination,
            sender_number=sender_number,
            send_status=send_status,
            send_error=send_error,
            message_sid=message_sid,
            format_selected=format_selected,
            image_url=image_url,
            pdf_url=pdf_url,
            media_url=media_url,
            uploaded_image_url=uploaded_image_url,
            upload_status=upload_status,
            upload_error=upload_error,
        )
        github_sync = append_send_log(log_record)
        github_synced = bool(github_sync.get("synced"))
        github_required = is_github_sync_required()
        if github_required and not github_synced:
            sync_reason = str(github_sync.get("reason", "")).strip()
            sync_error = "Nao foi possivel sincronizar o SuperPOP no GitHub. O envio nao foi confirmado."
            if sync_reason:
                sync_error = f"{sync_error} Motivo: {sync_reason}"
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": sync_error,
                        "prevent_manual_send": True,
                        "log_saved": False,
                        "log_saved_local": True,
                        "github_sync": github_sync,
                        "delivery": {
                            "mode": send_mode,
                            "method": "webjs" if str(send_status).startswith("webjs") else "wa_me",
                            "ok": send_status == "webjs_sent",
                            "status": send_status,
                            "error": send_error,
                            "to": destination,
                        },
                        "webjs": webjs_result,
                    }
                ),
                503,
            )

        upload_warning = ""
        upload_fallback_used = False
        if not uploaded_image_url:
            uploaded_image_url = image_url
            upload_fallback_used = True
            upload_warning = upload_error or "Nao foi possivel fazer upload no ImgBB. Usando link direto do servidor."

        delivery_method = "webjs" if str(send_status).startswith("webjs") else "wa_me"
        delivery_error = send_error
        if upload_warning:
            delivery_error = "; ".join(part for part in [delivery_error, upload_warning] if part)

        return jsonify(
            {
                "ok": True,
                "card_id": card_id,
                "auth_qr_url": auth_qr_url,
                "image_url": image_url,
                "uploaded_image_url": uploaded_image_url,
                "delete_image_url": imgbb_result.get("delete_url", ""),
                "pdf_url": pdf_url,
                "media_url": media_url,
                "log_saved": True,
                "log_saved_local": True,
                "github_sync": github_sync,
                "upload_warning": upload_warning,
                "upload_fallback_used": upload_fallback_used,
                "delivery": {
                    "mode": send_mode,
                    "method": delivery_method,
                    "ok": send_status == "webjs_sent",
                    "status": send_status,
                    "error": delivery_error,
                    "to": destination,
                },
                "webjs": webjs_result,
            }
        )
    finally:
        if send_day_reserved and send_day_key:
            with DATA_FILE_LOCK:
                PENDING_SEND_KEYS.discard(send_day_key)


if __name__ == "__main__":
    port = int(get_env("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

