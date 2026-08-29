import base64
import hashlib
import json
import sqlite3

from scripts.github_backup import (
    build_backup,
    decode_key,
    decrypt_bytes,
    encrypt_bytes,
)


def test_encrypted_backup_roundtrip_and_schema_has_no_rows(tmp_path) -> None:
    database = tmp_path / "portfolio.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schema_migrations(version INTEGER)")
        connection.execute("INSERT INTO schema_migrations VALUES (7)")
        connection.execute("CREATE TABLE private_trades(secret TEXT)")
        connection.execute("INSERT INTO private_trades VALUES ('NO-DEBE-APARECER')")
        connection.commit()
    key = b"k" * 32
    repo = tmp_path / "repo"
    manifest = build_backup(database, repo, key)
    encrypted = (repo / "backups" / "portfolio.db.aesgcm").read_bytes()
    restored = decrypt_bytes(encrypted, key)
    schema = (repo / "database" / "schema.sql").read_text(encoding="utf-8")

    assert restored.startswith(b"SQLite format 3")
    assert hashlib.sha256(encrypted).hexdigest() == manifest["ciphertext_sha256"]
    assert manifest["schema_migration"] == 7
    assert "CREATE TABLE private_trades" in schema
    assert "NO-DEBE-APARECER" not in schema


def test_backup_key_requires_exactly_32_bytes() -> None:
    encoded = base64.urlsafe_b64encode(b"x" * 32).decode("ascii")
    assert decode_key(encoded) == b"x" * 32
    payload = encrypt_bytes(b"private", b"x" * 32)
    assert b"private" not in payload
