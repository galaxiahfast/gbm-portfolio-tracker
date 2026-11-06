"""Restore an encrypted Git backup into an EMPTY local data directory.

Requires the same GBM_BACKUP_KEY used to create the backup. Existing database
or receipt files are never overwritten, so a local ledger cannot be replaced
silently by an older Git copy.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
import zipfile

from scripts.github_backup import decode_key, decrypt_bytes, decrypt_receipts


ROOT = Path(__file__).resolve().parents[1]


def restore(repo: Path, data_dir: Path, key: bytes) -> dict:
    manifest = json.loads((repo / "backups" / "manifest.json").read_text(encoding="utf-8"))
    db_cipher = (repo / "backups" / "portfolio.db.aesgcm").read_bytes()
    receipts_cipher = (repo / "backups" / "receipts.aesgcm").read_bytes()
    if (hashlib.sha256(db_cipher).hexdigest() != manifest["ciphertext_sha256"] or
            hashlib.sha256(receipts_cipher).hexdigest() != manifest["receipts_ciphertext_sha256"]):
        raise ValueError("El manifiesto no coincide con los archivos cifrados.")
    database_bytes = decrypt_bytes(db_cipher, key)
    receipts_bytes = decrypt_receipts(receipts_cipher, key)
    if not database_bytes.startswith(b"SQLite format 3\0"):
        raise ValueError("El respaldo descifrado no es SQLite.")
    with tempfile.TemporaryDirectory(prefix="gbm-restore-verify-") as temp:
        probe = Path(temp) / "probe.db"
        probe.write_bytes(database_bytes)
        connection = sqlite3.connect(str(probe))
        try:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("El respaldo SQLite no pasó integrity_check.")
        finally:
            connection.close()
    pending = []
    with zipfile.ZipFile(io.BytesIO(receipts_bytes)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != manifest["receipts_count"] or len(names) != len(set(names)):
            raise ValueError("Cantidad o nombres de comprobantes no coinciden.")
        for name in names:
            relative = PurePosixPath(name)
            if (relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts)
                    or ":" in name or "\\" in name):
                raise ValueError("Ruta de comprobante no segura.")
            destination = data_dir / "receipts" / Path(*relative.parts)
            contents = archive.read(name)
            if destination.exists():
                if not destination.is_file() or destination.read_bytes() != contents:
                    raise FileExistsError(f"Comprobante local distinto: {destination.name}")
            else:
                pending.append((destination, contents))
    db_target = data_dir / "portfolio.db"
    db_created = False
    if not db_target.exists():
        db_target.parent.mkdir(parents=True, exist_ok=True)
        db_target.write_bytes(database_bytes)
        db_created = True
    for destination, contents in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            output.write(contents)
    return {"database_created": db_created, "database_existing_preserved": not db_created,
            "receipts_created": len(pending), "receipts_total": len(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    encoded = os.environ.get("GBM_BACKUP_KEY", "")
    if not encoded:
        raise SystemExit("Falta GBM_BACKUP_KEY; no se restauró ningún archivo.")
    print(json.dumps(restore(args.repo.resolve(), args.data_dir.resolve(), decode_key(encoded)),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
