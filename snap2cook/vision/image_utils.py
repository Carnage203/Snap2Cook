"""Image validation, EXIF stripping, resizing and hashing."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from snap2cook.exceptions import InvalidImageError

MAX_BYTES = 5 * 1024 * 1024
MAX_SIDE = 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True)
class PreparedImage:
    """A cleaned JPEG ready for the vision API; ``sha256`` is the cache key."""

    data: bytes
    sha256: str
    width: int
    height: int


class ImageValidator:
    """Validates uploads with Pillow (not by extension) and normalises them."""

    def prepare(self, raw: bytes) -> PreparedImage:
        """Return a resized, EXIF-free JPEG or raise InvalidImageError."""
        if not raw:
            raise InvalidImageError("Empty upload")
        if len(raw) > MAX_BYTES:
            raise InvalidImageError(f"Image too large: {len(raw)} bytes", size=len(raw))
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise InvalidImageError(f"Unsupported format {probe.format}", format=probe.format)
                probe.verify()
            with Image.open(io.BytesIO(raw)) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
            raise InvalidImageError(f"Corrupt or unreadable image: {type(exc).__name__}") from exc
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)  # re-encoding drops EXIF
        data = buffer.getvalue()
        return PreparedImage(data, hashlib.sha256(data).hexdigest(), img.width, img.height)
