"""Canonical asset registry – the single source of truth for every image.

The registry is a JSON-backed manifest mapping ``relative_path`` to an
``AssetRecord``.  It lives inside the vault workspace and is the only
place that answers "what images do we know about?"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class AssetRecord(BaseModel):
    """Everything we know about one canonical image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    asset_id: str                    # sha256[:20]
    relative_path: str               # relative to scan root
    source_path: str                 # absolute path on disk (string for JSON)
    sha256: str
    size_bytes: int
    modified_ns: int
    width: int
    height: int
    mime_type: Optional[str] = None

    # Enrichment (populated lazily by later pipeline stages)
    exif_ts: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    detection_count: int = 0
    scene_indexed: bool = False
    region_indexed: bool = False


class AssetRegistry:
    """Persistent, JSON-backed asset manifest."""

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, AssetRecord] = {}
        self._load()

    # ── Queries ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, relative_path: str) -> bool:
        return relative_path in self._records

    def get(self, relative_path: str) -> Optional[AssetRecord]:
        return self._records.get(relative_path)

    def get_by_hash(self, sha256: str) -> Optional[AssetRecord]:
        for r in self._records.values():
            if r.sha256 == sha256:
                return r
        return None

    def all(self) -> Iterator[AssetRecord]:
        yield from self._records.values()

    # ── Mutations ────────────────────────────────────────────────────────

    def upsert(self, record: AssetRecord) -> None:
        self._records[record.relative_path] = record

    def remove(self, relative_path: str) -> None:
        self._records.pop(relative_path, None)

    def save(self) -> None:
        """Atomically persist the registry to disk."""
        payload = {
            "records": [r.model_dump(mode="json") for r in self._records.values()]
        }
        tmp = NamedTemporaryFile(
            "w", delete=False, encoding="utf-8",
            dir=self._path.parent, suffix=".tmp",
        )
        try:
            json.dump(payload, tmp, indent=2)
            tmp.close()
            Path(tmp.name).replace(self._path)
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise

    # ── Internal ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            blob = json.loads(self._path.read_text(encoding="utf-8"))
            rows = blob.get("records", []) if isinstance(blob, dict) else blob
            for row in rows:
                rec = AssetRecord.model_validate(row)
                self._records[rec.relative_path] = rec
        except Exception as exc:
            logger.warning("Could not load registry %s: %s", self._path, exc)
