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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
CARDS_DIR = BASE_DIR / "generated" / "cards"
CARDS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_CACHE_DIR = BASE_DIR / "generated" / "templates"
TEMPLATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = BASE_DIR / "Dados.json"
DATA_FILE_LOCK = threading.Lock()
DEFAULT_TEMPLATE_PATH = Path(r"C:\Users\marke\OneDrive\Desktop\SuperPOP.png")
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


def get_default_country_code() -> str:
    digits = re.sub(r"\D+", "", get_env("WHATSAPP_DEFAULT_COUNTRY_CODE", "55"))
    return digits or "55"


def get_default_area_code() -> str:
    digits = re.sub(r"\D+", "", get_env("WHATSAPP_DEFAULT_AREA_CODE", "82"))
    return digits or "82"


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
    collaborator_view = fit_text_to_width(collaborator_view, line_font, px(max(1.0, collaborator_max_x - collaborator_base[0])))
    recognized_view = fit_text_to_width(recognized_view, line_font, px(max(1.0, recognized_max_x - recognized_base[0])))
    date_view = fit_text_to_width(date_view, line_font, px(max(1.0, date_max_x - date_base[0])))

    try:
        draw.text((collaborator_line_x, collaborator_line_y), collaborator_view, font=line_font, fill="#1f2937", anchor="ls")
        draw.text((recognized_line_x, recognized_line_y), recognized_view, font=line_font, fill="#1f2937", anchor="ls")
        draw.text((date_line_x, date_line_y), date_view, font=line_font, fill="#1f2937", anchor="ls")
    except Exception:
        ascent = int(getattr(line_font, "size", 30) * 0.75)
        draw.text((collaborator_line_x, collaborator_line_y - ascent), collaborator_view, font=line_font, fill="#1f2937")
        draw.text((recognized_line_x, recognized_line_y - ascent), recognized_view, font=line_font, fill="#1f2937")
        draw.text((date_line_x, date_line_y - ascent), date_view, font=line_font, fill="#1f2937")

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
            if local_template.exists():
                return local_template

    if DEFAULT_TEMPLATE_PATH.exists():
        return DEFAULT_TEMPLATE_PATH

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
    try:
        req_get = urllib.request.Request(get_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_get, timeout=20) as resp:
            current = json.loads(resp.read().decode("utf-8"))
            sha = current.get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return {"synced": False, "reason": f"GitHub GET falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub GET erro: {exc}"}

    content = base64.b64encode(json.dumps(logs, ensure_ascii=False, indent=2).encode("utf-8")).decode("utf-8")
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
            return {"synced": True, "reason": "ok"}
    except urllib.error.HTTPError as exc:
        return {"synced": False, "reason": f"GitHub PUT falhou ({exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": f"GitHub PUT erro: {exc}"}


def append_send_log(record: dict) -> dict:
    with DATA_FILE_LOCK:
        logs = read_logs()
        logs.append(record)
        write_logs(logs)
        github_sync = github_sync_logs(logs)
    return github_sync


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


app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "superpop-backend"})


@app.get("/media/<path:filename>")
def serve_media(filename: str):
    return send_from_directory(CARDS_DIR, filename)


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
    payload = normalize_payload(request.get_json(silent=True) or {})
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

    destination = normalize_whatsapp_number(payload["to"] or payload["numero_colaborador"])
    sender_number = normalize_whatsapp_number(payload["numero_reconhecido_por"])

    local_now = now_brazil()
    local_date = payload["data"] or local_now.strftime("%d/%m/%Y")
    local_time = local_now.strftime("%H:%M:%S")
    local_iso = local_now.isoformat()

    log_record = make_log_record(
        payload=payload,
        card_id=card_id,
        auth_qr_url=auth_qr_url,
        local_date=local_date,
        local_time=local_time,
        local_iso=local_iso,
        destination=destination,
        sender_number=sender_number,
        send_status="wa_me",
        send_error="",
        message_sid="",
        format_selected=format_selected,
        image_url=image_url,
        pdf_url=pdf_url,
        media_url=media_url,
        uploaded_image_url=uploaded_image_url,
        upload_status=upload_status,
        upload_error=upload_error,
    )
    github_sync = append_send_log(log_record)

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
        }
    )


if __name__ == "__main__":
    port = int(get_env("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
