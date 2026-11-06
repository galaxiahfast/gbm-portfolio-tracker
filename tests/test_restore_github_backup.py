import sqlite3

import pytest

from scripts.github_backup import build_backup
from scripts.restore_github_backup import restore


def test_encrypted_backup_restores_database_and_receipts_without_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    database = source / "portfolio.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schema_migrations(version INTEGER)")
        connection.execute("INSERT INTO schema_migrations VALUES (1)")
        connection.execute("CREATE TABLE trades(value TEXT)")
        connection.execute("INSERT INTO trades VALUES ('private')")
    receipt = source / "receipts" / "SMCI" / "ticket.jpg"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"private-receipt")

    key = b"k" * 32
    repo = tmp_path / "repo"
    manifest = build_backup(database, repo, key)
    assert manifest["receipts_count"] == 1
    assert b"private-receipt" not in (repo / "backups" / "receipts.aesgcm").read_bytes()

    destination = tmp_path / "other-computer"
    result = restore(repo, destination, key)
    assert result == {"database_created": True, "database_existing_preserved": False,
                      "receipts_created": 1, "receipts_total": 1}
    with sqlite3.connect(destination / "portfolio.db") as connection:
        assert connection.execute("SELECT value FROM trades").fetchone()[0] == "private"
    assert (destination / "receipts" / "SMCI" / "ticket.jpg").read_bytes() == b"private-receipt"

    assert restore(repo, destination, key)["database_existing_preserved"] is True
    (destination / "receipts" / "SMCI" / "ticket.jpg").write_bytes(b"different-local-file")
    with pytest.raises(FileExistsError):
        restore(repo, destination, key)
    assert (destination / "receipts" / "SMCI" / "ticket.jpg").read_bytes() == b"different-local-file"
