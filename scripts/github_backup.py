"""Respaldo seguro y reproducible del libro SQLite para GitHub.

La base nunca se agrega a Git en claro. Se crea una instantánea consistente con
la API de backup de SQLite, se cifra mediante AES-256-GCM y se exporta aparte el
esquema SQL sin registros. La clave solo se lee de ``GBM_BACKUP_KEY``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MAGIC = b"GBM-SQLITE-AESGCM-v1\0"
AAD = b"gbm-portfolio-tracker/sqlite-backup/v1"
TRACKED_BACKUP_FILES = (
    "backups/portfolio.db.aesgcm",
    "backups/manifest.json",
    "database/schema.sql",
)


def decode_key(encoded: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("GBM_BACKUP_KEY no es Base64 URL-safe válida.") from exc
    if len(key) != 32:
        raise ValueError("GBM_BACKUP_KEY debe representar exactamente 32 bytes.")
    return key


def sqlite_snapshot(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"No existe la base SQLite: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    origin = sqlite3.connect(source)
    target = sqlite3.connect(destination)
    try:
        origin.execute("PRAGMA wal_checkpoint(PASSIVE)")
        origin.backup(target)
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("La instantánea SQLite no superó integrity_check.")
    finally:
        target.close()
        origin.close()


def encrypt_bytes(plaintext: bytes, key: bytes) -> bytes:
    nonce = os.urandom(12)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, plaintext, AAD)


def decrypt_bytes(payload: bytes, key: bytes) -> bytes:
    if not payload.startswith(MAGIC) or len(payload) <= len(MAGIC) + 12:
        raise ValueError("Formato de respaldo cifrado no reconocido.")
    offset = len(MAGIC)
    nonce = payload[offset : offset + 12]
    return AESGCM(key).decrypt(nonce, payload[offset + 12 :], AAD)


def export_schema(database: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
            ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END, name
            """
        ).fetchall()
    header = "-- Esquema generado automáticamente; no contiene datos.\n"
    body = "\n\n".join(str(row[0]).rstrip(";") + ";" for row in rows)
    destination.write_text(header + body + "\n", encoding="utf-8")


def build_backup(database: Path, repo_root: Path, key: bytes) -> dict[str, object]:
    encrypted_path = repo_root / "backups" / "portfolio.db.aesgcm"
    manifest_path = repo_root / "backups" / "manifest.json"
    schema_path = repo_root / "database" / "schema.sql"
    encrypted_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gbm-sqlite-") as temp_dir:
        snapshot = Path(temp_dir) / "portfolio.snapshot.db"
        sqlite_snapshot(database, snapshot)
        plaintext = snapshot.read_bytes()
        encrypted = encrypt_bytes(plaintext, key)
    temporary = encrypted_path.with_suffix(".tmp")
    temporary.write_bytes(encrypted)
    temporary.replace(encrypted_path)
    export_schema(database, schema_path)
    with sqlite3.connect(database) as connection:
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        migration = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()[0]
    manifest: dict[str, object] = {
        "format": "GBM-SQLITE-AESGCM-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ciphertext_sha256": hashlib.sha256(encrypted).hexdigest(),
        "encrypted_bytes": len(encrypted),
        "schema_migration": int(migration),
        "sqlite_user_version": schema_version,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_git(repo_root: Path, *, commit: bool, push: bool) -> None:
    if not (repo_root / ".git").exists():
        raise RuntimeError("La carpeta indicada no es un repositorio Git.")
    subprocess.run(
        ["git", "add", "--", *TRACKED_BACKUP_FILES],
        cwd=repo_root,
        check=True,
    )
    if commit:
        subprocess.run(
            ["git", "commit", "-m", "backup: actualizar respaldo SQLite cifrado"],
            cwd=repo_root,
            check=True,
        )
    if push:
        if not commit:
            raise ValueError("--push requiere --commit para evitar subir cambios ambiguos.")
        subprocess.run(["git", "push"], cwd=repo_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/portfolio.db"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--stage", action="store_true", help="Prepara los tres archivos seguros en Git.")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    encoded_key = os.environ.get("GBM_BACKUP_KEY", "")
    if not encoded_key:
        raise SystemExit("Falta GBM_BACKUP_KEY; el respaldo no se generó.")
    repo_root = args.repo.resolve()
    manifest = build_backup(args.database.resolve(), repo_root, decode_key(encoded_key))
    if args.stage or args.commit or args.push:
        run_git(repo_root, commit=args.commit, push=args.push)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
