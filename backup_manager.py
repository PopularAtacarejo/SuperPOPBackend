from __future__ import annotations

import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("superpop-backup")

BASE_DIR = Path(__file__).resolve().parent
BACKUP_ROOT = BASE_DIR / "backups"
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

DATA_FILE = BASE_DIR / "Dados.json"
EMPLOYEES_FILE = BASE_DIR / "Funcioinarios.json"

BACKUP_DIRS = {
    "logs": BACKUP_ROOT / "dados",
    "employees": BACKUP_ROOT / "funcionarios",
}


def _read_default_retention() -> int:
    raw_value = os.getenv("BACKUP_RETENTION_DAYS", "30")
    try:
        parsed = int(raw_value)
        return max(0, parsed)
    except Exception:
        return 30


DEFAULT_RETENTION_DAYS = _read_default_retention()


def _now() -> datetime:
    return datetime.now().astimezone()


def _format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M%S")


def _snapshot(source: Path, target_dir: Path, *, when: datetime | None = None) -> Path | None:
    if not source.exists():
        logger.debug("Skipping backup because %s is missing", source)
        return None

    timestamp = when or _now()
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix or ".bak"
    destination = target_dir / f"{source.stem}-{_format_timestamp(timestamp)}{suffix}"

    try:
        shutil.copy2(source, destination)
        logger.debug("Created backup %s", destination)
        return destination
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Failed to back up %s to %s: %s", source, destination, exc)
        return None


def backup_logs(*, when: datetime | None = None) -> Path | None:
    return _snapshot(DATA_FILE, BACKUP_DIRS["logs"], when=when)


def backup_employees(*, when: datetime | None = None) -> Path | None:
    return _snapshot(EMPLOYEES_FILE, BACKUP_DIRS["employees"], when=when)


def backup_all(
    *,
    include_logs: bool = True,
    include_employees: bool = True,
    when: datetime | None = None,
) -> dict[str, Path | None]:
    timestamp = when or _now()
    results: dict[str, Path | None] = {}

    if include_logs:
        results["logs"] = backup_logs(when=timestamp)
    if include_employees:
        results["employees"] = backup_employees(when=timestamp)

    return results


def prune_backups(retention_days: int = DEFAULT_RETENTION_DAYS) -> dict[str, int]:
    retention = max(0, retention_days)
    if retention == 0:
        return {"removed": 0}

    cutoff = datetime.now().timestamp() - retention * 86400
    removed = 0

    for entry in list(BACKUP_ROOT.rglob("*")):
        if not entry.is_file():
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            logger.debug("Unable to prune backup %s", entry)
            continue

    directories = sorted(
        (entry for entry in BACKUP_ROOT.rglob("*") if entry.is_dir()),
        reverse=True,
    )
    for directory in directories:
        if directory == BACKUP_ROOT:
            continue
        try:
            next(directory.iterdir())
        except StopIteration:
            try:
                directory.rmdir()
            except OSError:
                logger.debug("Could not remove empty backup directory %s", directory)

    return {"removed": removed}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cria cópias de segurança dos principais arquivos JSON do SuperPOP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-logs",
        action="store_true",
        help="Não copia o arquivo Dados.json",
    )
    parser.add_argument(
        "--skip-employees",
        action="store_true",
        help="Não copia o arquivo Funcioinarios.json",
    )
    parser.add_argument(
        "--retention-days",
        "-r",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help="Número de dias para manter os arquivos de backup",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Não remover arquivos antigos após o backup",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    include_logs = not args.skip_logs
    include_employees = not args.skip_employees
    timestamp = _now()

    backups: dict[str, Path | None] = {}
    if include_logs or include_employees:
        backups = backup_all(
            include_logs=include_logs,
            include_employees=include_employees,
            when=timestamp,
        )

    pruned: dict[str, int] | None = None
    if not args.no_prune:
        pruned = prune_backups(args.retention_days)

    print("Backup realizado em", timestamp.isoformat())
    if backups:
        for key, path in backups.items():
            label = str(path) if path else "ignorado (arquivo ausente)"
            print(f" - {key}: {label}")
    else:
        print(" - Nenhum backup executado (ambos os arquivos ignorados).")

    if pruned is not None:
        removed = pruned.get("removed", 0)
        print(f"Prune: removidos {removed} arquivos com mais de {max(0, args.retention_days)} dias.")
    else:
        print("Prune: etapa ignorada (--no-prune).")


if __name__ == "__main__":
    main()
