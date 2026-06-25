from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from backup_manager import backup_all

LOGGER = logging.getLogger("superpop-daily-backup")


def _perform_backup() -> None:
    timestamp = datetime.now().isoformat()
    LOGGER.info("Iniciando backup manual (%s)", timestamp)
    results = backup_all()
    for key, path in results.items():
        if path:
            LOGGER.info("  • %s -> %s", key, path)
        else:
            LOGGER.info("  • %s -> ignorado (arquivo ausente)", key)


def _seconds_until(hour: int, minute: int, tz: ZoneInfo) -> float:
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _wait_until_next_run(hour: int, minute: int, tz: ZoneInfo) -> None:
    delay = _seconds_until(hour, minute, tz)
    next_run = datetime.now(tz) + timedelta(seconds=delay)
    LOGGER.info("Próximo backup agendado para %s", next_run.strftime("%Y-%m-%d %H:%M:%S"))
    time.sleep(delay)


def _run_scheduler(hour: int, minute: int, tz: ZoneInfo) -> None:
    LOGGER.info("Agendador iniciado (%s horário local)", tz)
    while True:
        try:
            _wait_until_next_run(hour, minute, tz)
            _perform_backup()
        except KeyboardInterrupt:
            LOGGER.info("Agendador interrompido pelo usuário")
            raise
        except Exception:
            LOGGER.exception("Erro durante o backup agendado")
            time.sleep(30)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa os backups do SuperPOP todos os dias às 20h")
    parser.add_argument(
        "--timezone",
        default="America/Sao_Paulo",
        help="Identificador da zona (padrão: America/Sao_Paulo)",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=20,
        help="Hora do dia (0-23) para executar o backup",
    )
    parser.add_argument(
        "--minute",
        type=int,
        default=0,
        help="Minuto da hora para executar o backup",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Executa um backup imediato e encerra",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not 0 <= args.hour <= 23:
        LOGGER.error("A hora precisa estar entre 0 e 23")
        sys.exit(1)
    if not 0 <= args.minute <= 59:
        LOGGER.error("Os minutos precisam estar entre 0 e 59")
        sys.exit(1)

    try:
        tz = ZoneInfo(args.timezone)
    except Exception:
        LOGGER.error("Zona horária inválida: %s", args.timezone)
        sys.exit(1)

    if args.run_once:
        _perform_backup()
        return

    def _handle_exit(signum, frame):
        LOGGER.info("Sinal %s recebido, encerrando agendador", signum)
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_exit)

    try:
        _run_scheduler(args.hour, args.minute, tz)
    except KeyboardInterrupt:
        LOGGER.info("Agendador finalizado")


if __name__ == "__main__":
    main()
