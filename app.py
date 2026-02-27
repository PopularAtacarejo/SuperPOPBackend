import base64
import copy
import hashlib
import hmac
import io
import json
import os
import re
import textwrap
import time
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
CARDS_DIR = BASE_DIR / "generated" / "cards"
CARDS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_CACHE_DIR = BASE_DIR / "generated" / "templates"
TEMPLATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = BASE_DIR / "Dados.json"
DATA_FILE_LOCK = threading.Lock()
EMPLOYEES_FILE = BASE_DIR / "Funcioinarios.json"
EMPLOYEES_FILE_LOCK = threading.Lock()
PENDING_SEND_KEYS: set[str] = set()
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

if not EMPLOYEES_FILE.exists():
    EMPLOYEES_FILE.write_text("[]\n", encoding="utf-8")

if not LAYOUT_FILE.exists():
    LAYOUT_FILE.write_text(
        json.dumps(DEFAULT_LAYOUT_CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


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


def to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on", "sim", "s"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "nao", "não"}:
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
    collaborator = payload.get("colaborador", "") or "-"
    recognized_by = payload.get("reconhecido_por", "") or "-"
    values = payload.get("valores", [])
    values_text = ", ".join(values) if isinstance(values, list) and values else "-"
    date_text = payload.get("data", "") or now_brazil().strftime("%d/%m/%Y")
    message_text = (payload.get("mensagem", "") or "").strip()

    lines = [
        "🎉 *SuperPOP - Reconhecimento*",
        "",
        f"👤 *Para:* {collaborator}",
        f"🙌 *Enviado por:* {recognized_by}",
        f"⭐ *Valores:* {values_text}",
        f"📅 *Data:* {date_text}",
    ]
    if message_text:
        lines.extend(["", "💬 *Mensagem:*", message_text])

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

    return {
        "colaborador": str(payload.get("colaborador", "")).strip(),
        "numero_colaborador": normalized_num_colaborador or raw_num_colaborador,
        "funcao_colaborador": str(payload.get("funcao_colaborador", "")).strip(),
        "reconhecido_por": str(payload.get("reconhecido_por", "")).strip(),
        "numero_reconhecido_por": normalized_num_reconhecido or raw_num_reconhecido,
        "funcao_reconhecido_por": str(payload.get("funcao_reconhecido_por", "")).strip(),
        "valores": [str(v).strip() for v in values if str(v).strip()],
        "mensagem": str(payload.get("mensagem", "")).strip(),
        "data": str(payload.get("data", "")).strip(),
        "to": normalized_to or normalized_num_colaborador or raw_to,
        "format": str(payload.get("format", "image")).strip().lower(),
        "send_mode": str(payload.get("send_mode", "")).strip().lower(),
    }


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

    colaborador_view = data["colaborador"] or "-"
    remetente_view = data["reconhecido_por"] or "-"

    y = 260
    fields = [
        ("Para", colaborador_view),
        ("Numero para", data["numero_colaborador"] or "-"),
        ("Enviado por", remetente_view),
        ("Numero de quem envia", data["numero_reconhecido_por"] or "-"),
        ("Valores", ", ".join(data["valores"]) if data["valores"] else "-"),
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
        text=data["mensagem"] or "-",
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
        value = (value or "").strip()
        if not value:
            return "-"
        if len(value) <= max_len:
            return value
        return value[: max_len - 1].rstrip() + "…"

    def fit_text_to_width(value: str, font: ImageFont.ImageFont, max_width: int) -> str:
        text = (value or "").strip() or "-"
        if draw.textlength(text, font=font) <= max_width:
            return text

        ellipsis = "…"
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

    selected = {normalize_value_key(v) for v in data["valores"]}
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

    message_text = (data["mensagem"] or "").strip()
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

    merged_logs = merge_log_lists(remote_logs, logs)
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


EMPLOYEE_PHONE_PATTERN = re.compile(r"^\(\d{2}\)\s9\s\d{4}\s-\s\d{4}$")
EMPLOYEE_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def normalize_employee_phone_digits(value: str) -> str:
    return re.sub(r"\D", "", str(value or ""))


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_employee_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        payload = {}
    return {
        "nome": normalize_spaces(payload.get("nome", "")),
        "funcao": normalize_spaces(payload.get("funcao", "")),
        "numero_celular": normalize_spaces(payload.get("numero_celular", "")),
        "email": str(payload.get("email", "") or "").strip().lower(),
        "senha": str(payload.get("senha", "") or ""),
    }


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
    if len(senha) < 6:
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


def write_employees(records: list) -> None:
    temp_file = EMPLOYEES_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(EMPLOYEES_FILE)


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


def github_sync_employees(records: list) -> dict:
    token = get_env("GITHUB_TOKEN")
    if not token:
        return {"synced": False, "reason": "GITHUB_TOKEN nao configurado"}

    repo = get_env("GITHUB_REPO", "PopularAtacarejo/SuperPOP")
    file_path = get_env("GITHUB_EMPLOYEES_FILE_PATH", "Funcioinarios.json")
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

    merged_records = merge_employee_lists(remote_records, records)
    content = base64.b64encode(json.dumps(merged_records, ensure_ascii=False, indent=2).encode("utf-8")).decode("utf-8")
    utc_now = datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    payload = {
        "message": f"Atualiza Funcioinarios.json ({utc_now})",
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


def github_sync_employees_with_retry(records: list) -> dict:
    retries = max(1, int(to_number(get_env("GITHUB_SYNC_RETRIES", "3"), 3)))
    retry_delay = max(0.0, to_number(get_env("GITHUB_SYNC_RETRY_DELAY_SECONDS", "1.0"), 1.0))
    last_result = {"synced": False, "reason": "Sync nao executado."}

    for attempt in range(1, retries + 1):
        result = github_sync_employees(records)
        result["attempt"] = attempt
        result["max_attempts"] = retries
        if result.get("synced"):
            return result
        last_result = result
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    return last_result


def append_employee_record(record: dict) -> dict:
    with EMPLOYEES_FILE_LOCK:
        records = merge_employee_lists(read_employees(), [record])
        write_employees(records)
        github_sync = github_sync_employees_with_retry(records)
        merged_records = github_sync.get("merged_records")
        if isinstance(merged_records, list):
            write_employees(merged_records)
            github_sync.pop("merged_records", None)
    return github_sync


def build_password_hash(password: str) -> tuple[str, str, int]:
    iterations = max(120000, int(to_number(get_env("PASSWORD_HASH_ITERATIONS", "180000"), 180000)))
    salt_bytes = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
    return salt_bytes.hex(), digest.hex(), iterations


def build_employee_record(payload: dict, created_iso: str) -> dict:
    salt_hex, hash_hex, iterations = build_password_hash(payload.get("senha", ""))
    phone_digits = normalize_employee_phone_digits(payload.get("numero_celular", ""))
    return {
        "id": uuid.uuid4().hex,
        "nome": payload.get("nome", ""),
        "funcao": payload.get("funcao", ""),
        "numero_celular": payload.get("numero_celular", ""),
        "numero_normalizado": phone_digits,
        "email": payload.get("email", ""),
        "senha": {
            "algoritmo": "pbkdf2_sha256",
            "salt": salt_hex,
            "hash": hash_hex,
            "iteracoes": iterations,
        },
        "data_cadastro_iso": created_iso,
    }


def build_employee_public_record(record: dict) -> dict:
    return {
        "id": str(record.get("id", "")).strip(),
        "nome": str(record.get("nome", "")).strip(),
        "funcao": str(record.get("funcao", "")).strip(),
        "numero_celular": str(record.get("numero_celular", "")).strip(),
        "email": str(record.get("email", "")).strip(),
        "data_cadastro_iso": str(record.get("data_cadastro_iso", "")).strip(),
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


def find_employee_by_phone(records: list, phone_digits: str) -> dict | None:
    for item in records:
        if not isinstance(item, dict):
            continue
        existing_phone = normalize_employee_phone_digits(item.get("numero_normalizado") or item.get("numero_celular") or "")
        if phone_digits and existing_phone == phone_digits:
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
            "nome": payload["colaborador"] or "-",
            "numero": payload["numero_colaborador"] or "-",
            "numero_normalizado": destination or "-",
            "funcao": payload["funcao_colaborador"] or "-",
        },
        "remetente": {
            "nome": payload["reconhecido_por"] or "-",
            "numero": payload["numero_reconhecido_por"] or "-",
            "numero_normalizado": sender_number or "-",
            "funcao": payload["funcao_reconhecido_por"] or "-",
        },
        "opcoes_marcadas": payload["valores"],
        "mensagem": payload["mensagem"] or "-",
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


def load_logs_for_history_view() -> tuple[list, dict]:
    preferred_configured = MY_SUPERPOPS_SOURCE_URL
    preferred_resolved = normalize_layout_source_url(preferred_configured)
    configured_source, resolved_source = resolve_rank_data_source_url()

    candidates: list[tuple[str, str]] = [(preferred_configured, preferred_resolved)]
    if resolved_source != preferred_resolved:
        candidates.append((configured_source, resolved_source))

    remote_errors: list[str] = []
    for configured, resolved in candidates:
        remote_logs, error = fetch_rank_logs_remote(resolved)
        if not error and isinstance(remote_logs, list):
            return remote_logs, {
                "tipo": "remoto",
                "url_configurada": configured,
                "url_resolvida": resolved,
            }
        remote_errors.append(error or f"Falha ao ler fonte remota: {resolved}")

    return read_logs(), {
        "tipo": "local",
        "url_configurada": preferred_configured,
        "url_resolvida": preferred_resolved,
        "erro_remoto": " | ".join(item for item in remote_errors if item),
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

    mensagem_value = str(record.get("mensagem", "")).strip()
    if mensagem_value == "-":
        mensagem_value = ""

    return {
        "id": str(record.get("id", "")).strip(),
        "card_id": str(record.get("card_id", "")).strip(),
        "dia": str(record.get("dia", "")).strip(),
        "horario": str(record.get("horario", "")).strip(),
        "data_hora_iso": str(record.get("data_hora_iso", "")).strip(),
        "papel": role,
        "outra_pessoa": {
            "nome": str(other_actor.get("nome", "")).strip(),
            "numero": str(other_actor.get("numero", "")).strip(),
            "funcao": str(other_actor.get("funcao", "")).strip(),
        },
        "valores": [str(item).strip() for item in valores if str(item).strip()],
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
    }


app = Flask(__name__)
app.secret_key = get_env("FLASK_SECRET_KEY", "superpop-dev-secret")
session_hours = max(1.0, to_number(get_env("AUTH_SESSION_HOURS", "24"), 24.0))
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=session_hours)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = to_bool(get_env("SESSION_COOKIE_SECURE", "0"), False)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True
CORS(app)


def is_user_logged_in() -> bool:
    return bool(session.get("auth_user_id"))


def require_login_redirect():
    if not is_user_logged_in():
        return redirect(url_for("serve_login_page"))
    return None


@app.get("/")
def serve_superpop_home():
    if is_user_logged_in():
        return redirect(url_for("serve_superpop_file"))
    return redirect(url_for("serve_login_page"))


@app.get("/superpop.html")
def serve_superpop_file():
    blocked = require_login_redirect()
    if blocked:
        return blocked
    return send_from_directory(BASE_DIR, "superpop.html")


@app.get("/rank")
@app.get("/rank.html")
def serve_rank_page():
    blocked = require_login_redirect()
    if blocked:
        return blocked
    return send_from_directory(BASE_DIR, "rank.html")


@app.get("/ganhadores")
@app.get("/ganhadores.html")
def serve_month_winners_page():
    blocked = require_login_redirect()
    if blocked:
        return blocked
    return send_from_directory(BASE_DIR, "ganhadores.html")


@app.get("/meus-superpops")
@app.get("/meus-superpops.html")
def serve_my_superpops_page():
    blocked = require_login_redirect()
    if blocked:
        return blocked
    return send_from_directory(BASE_DIR, "meus-superpops.html")


@app.get("/cadastro")
@app.get("/cadastro.html")
def serve_register_page():
    if is_user_logged_in():
        return redirect(url_for("serve_superpop_file"))
    return send_from_directory(BASE_DIR, "cadastro.html")


@app.get("/login")
@app.get("/login.html")
def serve_login_page():
    if is_user_logged_in():
        return redirect(url_for("serve_superpop_file"))
    return send_from_directory(BASE_DIR, "login.html")


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
    configured_source, resolved_source = resolve_rank_data_source_url()
    logs, error = fetch_rank_logs_remote(resolved_source)
    if error:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": error,
                    "fonte": {
                        "url_configurada": configured_source,
                        "url_resolvida": resolved_source,
                    },
                }
            ),
            502,
        )
    return jsonify(build_rank_payload(logs, configured_source, resolved_source))


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
    auth_user_nome = str(session.get("auth_user_nome", "")).strip()
    auth_user_funcao = str(session.get("auth_user_funcao", "")).strip()
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


@app.post("/api/funcionarios/register")
def register_employee():
    payload = normalize_employee_payload(request.get_json(silent=True) or {})
    valid, validation_error = validate_employee_payload(payload)
    if not valid:
        return jsonify({"ok": False, "error": validation_error}), 400

    phone_digits = normalize_employee_phone_digits(payload.get("numero_celular", ""))
    with EMPLOYEES_FILE_LOCK:
        existing_records = read_employees()
        duplicated, duplicate_error = find_duplicate_employee(existing_records, phone_digits, payload.get("email", ""))
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
    phone_raw = str(payload.get("numero_celular", "") or "").strip()
    password_raw = str(payload.get("senha", "") or "")
    keep_connected = to_bool(payload.get("manter_conectado"), True)
    phone_digits = normalize_employee_phone_digits(phone_raw)

    if len(phone_digits) != 11 or (phone_digits and phone_digits[2] != "9"):
        return jsonify({"ok": False, "error": "Numero de celular invalido."}), 400
    if not password_raw:
        return jsonify({"ok": False, "error": "Senha obrigatoria."}), 400

    with EMPLOYEES_FILE_LOCK:
        records = read_employees()
        employee = find_employee_by_phone(records, phone_digits)

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
    session["auth_login_at"] = now_brazil().isoformat()

    return jsonify(
        {
            "ok": True,
            "funcionario": public_employee,
        }
    )


@app.get("/api/auth/me")
def auth_me():
    if not is_user_logged_in():
        return jsonify({"ok": False, "error": "Nao autenticado."}), 401

    auth_user_id = str(session.get("auth_user_id", "")).strip()
    auth_user_nome = str(session.get("auth_user_nome", "")).strip()
    auth_user_funcao = str(session.get("auth_user_funcao", "")).strip()
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

    return jsonify(
        {
            "ok": True,
            "usuario": {
                "id": auth_user_id,
                "nome": auth_user_nome,
                "funcao": auth_user_funcao,
                "numero_celular": auth_user_numero,
                "login_at": str(session.get("auth_login_at", "")).strip(),
            },
        }
    )


@app.post("/api/auth/logout")
def auth_logout():
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
    auth_user_id = str(session.get("auth_user_id", "")).strip()
    auth_user_nome = str(session.get("auth_user_nome", "")).strip()
    auth_user_funcao = str(session.get("auth_user_funcao", "")).strip()
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

    payload["reconhecido_por"] = auth_user_nome
    payload["funcao_reconhecido_por"] = auth_user_funcao
    payload["numero_reconhecido_por"] = auth_user_numero
    payload = normalize_payload(payload)

    local_now = now_brazil()
    local_date = payload["data"] or local_now.strftime("%d/%m/%Y")
    local_time = local_now.strftime("%H:%M:%S")
    local_iso = local_now.isoformat()
    destination = normalize_whatsapp_number(payload["to"] or payload["numero_colaborador"])
    sender_number = normalize_whatsapp_number(payload["numero_reconhecido_por"])
    send_day_key = build_daily_send_key(sender_number, destination, local_date)
    send_day_reserved = False

    with DATA_FILE_LOCK:
        existing_logs = read_logs()
        duplicate_record = find_duplicate_send_same_day(
            logs=existing_logs,
            sender_number=sender_number,
            destination_number=destination,
            day_value=local_date,
        )
        if duplicate_record:
            duplicate_name = str((duplicate_record.get("destinatario", {}) or {}).get("nome", "")).strip() or payload.get("colaborador") or "esse colaborador"
            duplicate_time = str(duplicate_record.get("horario", "")).strip()
            duplicate_date = str(duplicate_record.get("dia", "")).strip() or local_date
            duplicate_hint = f" em {duplicate_date}" + (f" às {duplicate_time}" if duplicate_time else "")
            return (
                jsonify(
                    {
                        "ok": False,
                        "duplicate_send": True,
                        "error": f"Você já enviou SuperPOP para {duplicate_name}{duplicate_hint}.",
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
                            "error": "Você já enviou SuperPOP para essa pessoa hoje. Aguarde o envio atual finalizar.",
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
            return (
                jsonify(
                    {
                        "ok": False,
                        "card_id": card_id,
                        "auth_qr_url": auth_qr_url,
                        "error": "Falha ao sincronizar Dados.json com o GitHub. Registro nao confirmado.",
                        "image_url": image_url,
                        "uploaded_image_url": uploaded_image_url,
                        "pdf_url": pdf_url,
                        "media_url": media_url,
                        "log_saved": False,
                        "log_saved_local": True,
                        "github_sync": github_sync,
                        "delivery": {
                            "mode": send_mode,
                            "method": "webjs" if send_mode == "webjs" else "wa_me",
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

        if not uploaded_image_url:
            return (
                jsonify(
                    {
                        "ok": False,
                        "card_id": card_id,
                        "auth_qr_url": auth_qr_url,
                        "error": upload_error or "Nao foi possivel fazer upload da imagem no ImgBB.",
                        "image_url": image_url,
                        "pdf_url": pdf_url,
                        "uploaded_image_url": "",
                        "log_saved": True,
                        "github_sync": github_sync,
                        "delivery": {
                            "mode": send_mode,
                            "method": "webjs" if send_mode == "webjs" else "wa_me",
                            "ok": False,
                            "status": send_status,
                            "error": upload_error or "Nao foi possivel fazer upload da imagem no ImgBB.",
                            "to": destination,
                        },
                        "webjs": webjs_result,
                    }
                ),
                400,
            )

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
                "github_sync": github_sync,
                "delivery": {
                    "mode": send_mode,
                    "method": "webjs" if send_mode == "webjs" else "wa_me",
                    "ok": send_status == "webjs_sent",
                    "status": send_status,
                    "error": send_error,
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
