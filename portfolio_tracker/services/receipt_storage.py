"""Conservacion auditable de imagenes sin guardar blobs pesados en SQLite."""

from __future__ import annotations

import hashlib
import io
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import MAX_RECEIPT_BYTES, PROJECT_ROOT, RECEIPTS_DIR, ensure_data_directories


@dataclass(frozen=True, slots=True)
class StoredReceipt:
    sha256: str
    original_filename: str
    mime_type: str
    original_path: str
    thumbnail_path: str
    width: int
    height: int
    byte_size: int


class ReceiptStorage:
    ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

    def save(self, content: bytes, original_filename: str) -> StoredReceipt:
        ensure_data_directories()
        if not content:
            raise ValueError("La imagen esta vacia.")
        if len(content) > MAX_RECEIPT_BYTES:
            raise ValueError("La imagen excede el limite de 12 MB.")

        try:
            with Image.open(io.BytesIO(content)) as source:
                source.verify()
            with Image.open(io.BytesIO(content)) as source:
                image_format = (source.format or "").upper()
                if image_format not in self.ALLOWED_FORMATS:
                    raise ValueError("Formato no permitido. Usa JPG, PNG o WEBP.")
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                width, height = oriented.size
                if width * height > 40_000_000:
                    raise ValueError("La imagen tiene demasiados pixeles.")

                digest = hashlib.sha256(content).hexdigest()
                bucket = RECEIPTS_DIR / digest[:2]
                bucket.mkdir(parents=True, exist_ok=True)
                extension = self.ALLOWED_FORMATS[image_format]
                original_path = bucket / f"{digest}{extension}"
                thumbnail_path = bucket / f"{digest}.thumb.webp"

                if not original_path.exists():
                    original_path.write_bytes(content)
                if not thumbnail_path.exists():
                    thumbnail = oriented.copy()
                    thumbnail.thumbnail((720, 1280), Image.Resampling.LANCZOS)
                    thumbnail.save(thumbnail_path, "WEBP", quality=82, method=6)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("El archivo no es una imagen valida.") from exc

        mime_type = mimetypes.guess_type(original_path.name)[0] or "application/octet-stream"
        return StoredReceipt(
            sha256=digest,
            original_filename=Path(original_filename).name,
            mime_type=mime_type,
            original_path=self._portable_path(original_path),
            thumbnail_path=self._portable_path(thumbnail_path),
            width=width,
            height=height,
            byte_size=len(content),
        )

    @staticmethod
    def _portable_path(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

