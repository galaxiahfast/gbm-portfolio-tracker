"""Optional Windows DPAPI vault for unattended read-only market-data tasks.

`setup` prompts invisibly and writes only Windows current-user encrypted bytes.
`run` decrypts into this process's environment, then invokes a research script.
Nothing here accesses Alpaca's trading API, portfolio.db, or production jobs.
"""
from __future__ import annotations

import argparse
import ctypes
import getpass
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prospective_archive_common import ARCHIVE


VAULT = ARCHIVE / "secrets" / "market_data_current_user.dpapi"
FIELDS = ("FINNHUB_API_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY")


def _dpapi(value: bytes, *, decrypt=False) -> bytes:
    if os.name != "nt":
        raise RuntimeError("La bóveda DPAPI solo funciona en Windows.")
    class Blob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    source, destination = Blob(len(value), buffer), Blob()
    function = getattr(ctypes.windll.crypt32,
                       "CryptUnprotectData" if decrypt else "CryptProtectData")
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                         ctypes.POINTER(Blob)]
    function.restype = ctypes.c_int
    if not function(ctypes.byref(source), None, None, None, None, 1,
                    ctypes.byref(destination)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree(destination.pbData)


def setup(*, replace=False):
    if VAULT.exists() and not replace:
        raise RuntimeError("Ya hay una bóveda. Usa `setup --replace` para rotar sus claves.")
    secrets = {name: getpass.getpass(f"{name} (oculta): ").strip() for name in FIELDS}
    if any(not value for value in secrets.values()):
        raise ValueError("Las tres claves son obligatorias; no se escribió la bóveda.")
    ciphertext = _dpapi(json.dumps(secrets, sort_keys=True).encode("utf-8"))
    VAULT.parent.mkdir(parents=True, exist_ok=True)
    temporary = VAULT.with_suffix(".dpapi.tmp")
    temporary.write_bytes(ciphertext)
    os.replace(temporary, VAULT)
    del secrets
    print("Bóveda DPAPI creada para el usuario actual; ninguna clave se imprimió.")


def _load():
    if not VAULT.is_file():
        raise RuntimeError("Falta bóveda DPAPI. Ejecuta `setup` con las claves vigentes.")
    secrets = json.loads(_dpapi(VAULT.read_bytes(), decrypt=True).decode("utf-8"))
    if set(secrets) != set(FIELDS) or any(not isinstance(secrets[name], str) or not secrets[name]
                                          for name in FIELDS):
        raise ValueError("Bóveda prospectiva incompleta.")
    return secrets


def run(task: str):
    missing = [name for name in FIELDS if not os.environ.get(name)]
    if missing:
        secrets = _load()
        for name in missing:
            os.environ[name] = secrets[name]
        del secrets
    if task == "collector":
        from scripts.collect_prospective_snapshots import main
    elif task == "labeler":
        from scripts.label_prospective_snapshots import main
    elif task == "coverage":
        from scripts.prospective_coverage import main
    elif task == "backfill":
        from scripts.backfill_prospective_gaps import main
    else:
        raise ValueError("Tarea prospectiva desconocida.")
    try:
        main([])
    except Exception as exc:
        # pythonw is intentionally windowless; retain a safe failure marker.
        folder = ARCHIVE / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "scheduler.log").open("a", encoding="utf-8") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} {task} FAILED {type(exc).__name__}\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup_cmd = sub.add_parser("setup")
    setup_cmd.add_argument("--replace", action="store_true")
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("task", choices=("collector", "labeler", "coverage", "backfill"))
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup(replace=args.replace)
    else:
        try:
            run(args.task)
        except Exception as exc:
            # Covers failures before a collector/labeler logger exists, e.g.
            # Task Scheduler cannot decrypt DPAPI in the selected logon mode.
            folder = ARCHIVE / "logs"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "scheduler.log").open("a", encoding="utf-8") as log:
                log.write(f"{datetime.now(timezone.utc).isoformat()} {args.task} STARTUP_FAILED "
                          f"{type(exc).__name__} vault_exists={VAULT.is_file()}\n")
            raise


if __name__ == "__main__":
    main()
