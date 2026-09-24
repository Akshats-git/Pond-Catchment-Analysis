"""Saved analyses (HLD FR-9): keep a result, list them, open one again.

SQLite, because it is in the standard library, needs no server, and one file is all a
village-scale planning tool will ever need; the HLD's container 4 is this file on a
volume. The store keeps the *whole* response, so reopening a site draws exactly what was
analysed, and a small summary beside it so the list never has to parse a megabyte of
GeoJSON per row.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from app.config import StoreConfig, settings
from app.providers.elevation import REPO_ROOT

__all__ = ["Store", "store", "summarise"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ponds (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    notes      TEXT NOT NULL DEFAULT '',
    created    REAL NOT NULL,
    source     TEXT NOT NULL,
    summary    TEXT NOT NULL,
    response   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ponds_created ON ponds (created DESC);
"""


def summarise(response: dict) -> dict:
    """The handful of numbers a list of saved sites shows."""
    site = response["recommended_site"]
    area = response.get("area") or {}
    return {
        "location": site["location"],
        "catchment_ha": site["catchment"]["area_ha"],
        "confidence": site["catchment"].get("confidence"),
        "annual_runoff_m3": site["runoff"]["annual_runoff_m3"],
        "capacity_m3": site["storage"]["capacity_m3"],
        "fill_ratio": site["runoff"].get("fill_ratio"),
        "sites": 1 + len(response.get("alternative_sites") or []),
        "selection_bbox": area.get("selection_bbox") or response["input"].get("bbox"),
        "selection": area.get("geometry"),
        "filename": response["input"].get("filename"),
    }


class Store:
    def __init__(self, config: StoreConfig | None = None, path: str | Path | None = None) -> None:
        self.config = config or settings.store
        raw = Path(path or self.config.path)
        self.path = raw if raw.is_absolute() else REPO_ROOT / raw
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        db.row_factory = sqlite3.Row
        return db

    def save(self, name: str, response: dict, notes: str = "") -> dict:
        record = {
            "id": uuid.uuid4().hex[:12],
            "name": name.strip()[: self.config.max_name_length] or "Untitled site",
            "notes": notes[:2000],
            "created": time.time(),
            "source": response["input"].get("source", "contour_file"),
            "summary": summarise(response),
        }
        with self._lock, self._connect() as db:
            count = db.execute("SELECT COUNT(*) FROM ponds").fetchone()[0]
            if count >= self.config.max_saved:
                # Oldest first: a planning tool that refuses to save is worse than one that
                # forgets the analysis nobody has opened in the longest time.
                db.execute("DELETE FROM ponds WHERE id IN (SELECT id FROM ponds ORDER BY created LIMIT ?)",
                           (count - self.config.max_saved + 1,))
            db.execute(
                "INSERT INTO ponds (id, name, notes, created, source, summary, response) VALUES (?,?,?,?,?,?,?)",
                (record["id"], record["name"], record["notes"], record["created"], record["source"],
                 json.dumps(record["summary"]), json.dumps(response)),
            )
        return record

    def list(self, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        with self._connect() as db:
            total = db.execute("SELECT COUNT(*) FROM ponds").fetchone()[0]
            rows = db.execute(
                "SELECT id, name, notes, created, source, summary FROM ponds ORDER BY created DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row(r) for r in rows], total

    def get(self, pond_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM ponds WHERE id = ?", (pond_id,)).fetchone()
        if row is None:
            return None
        record = self._row(row)
        record["response"] = json.loads(row["response"])
        return record

    def delete(self, pond_id: str) -> bool:
        with self._lock, self._connect() as db:
            return db.execute("DELETE FROM ponds WHERE id = ?", (pond_id,)).rowcount > 0

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "name": row["name"], "notes": row["notes"], "created": row["created"],
            "source": row["source"], "summary": json.loads(row["summary"]),
        }


_STORE: Store | None = None


def store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


def reset_store(new: Store | None = None) -> Store | None:
    global _STORE
    _STORE = new
    return _STORE
