"""Photo upload handling: JPEG/PNG only, verified with Pillow, ALL metadata stripped (EXIF incl.
GPS, ICC, text chunks) by rebuilding the image from raw pixels, resized to max 1280 px, stored
under a random file name."""

from __future__ import annotations

import io
import re
import secrets
from pathlib import Path

from PIL import Image, ImageOps

MAX_PHOTO_BYTES = 5 * 1024 * 1024
MAX_SIDE = 1280
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs
NAME_RE = re.compile(r"^[0-9a-f]{32}\.(jpg|png)$")
PRIVACY_REMINDER = ("Thank you. Please never photograph people's faces or vehicle number "
                    "plates; photo metadata (EXIF, GPS) has been removed.")
_MAGIC = {b"\xff\xd8\xff": "JPEG", b"\x89PNG\r\n\x1a\n": "PNG"}


class PhotoError(ValueError):
    pass


def sniff(data: bytes) -> str | None:
    for magic, fmt in _MAGIC.items():
        if data.startswith(magic):
            return fmt
    return None


def clean_photo(data: bytes) -> tuple[bytes, str]:
    """Return (re-encoded bytes without any metadata, extension)."""
    if len(data) > MAX_PHOTO_BYTES:
        raise PhotoError("photo larger than 5 MB")
    fmt = sniff(data)
    if fmt is None:
        raise PhotoError("only JPEG or PNG photos are accepted")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            if im.format != fmt:
                raise PhotoError("file content does not match a JPEG/PNG image")
            im = ImageOps.exif_transpose(im)  # keep the visual orientation, then drop EXIF
            mode = "RGB" if fmt == "JPEG" else ("RGBA" if im.mode in ("RGBA", "LA", "P")
                                                 else "RGB")
            im = im.convert(mode)
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            clean = Image.frombytes(mode, im.size, im.tobytes())  # pixels only, no info dict
    except PhotoError:
        raise
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError) as e:
        raise PhotoError(f"invalid image: {e}") from None
    out = io.BytesIO()
    if fmt == "JPEG":
        clean.save(out, "JPEG", quality=85)
        return out.getvalue(), "jpg"
    clean.save(out, "PNG", optimize=True)
    return out.getvalue(), "png"


def save_photo(data: bytes, photos_dir: str | Path) -> str:
    """Clean and store; returns the random file name."""
    clean, ext = clean_photo(data)
    d = Path(photos_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = f"{secrets.token_hex(16)}.{ext}"
    (d / name).write_bytes(clean)
    return name


def photo_path(photos_dir: str | Path, name: str) -> Path | None:
    """Resolve a stored photo name safely (strict pattern, no path tricks)."""
    if not NAME_RE.match(name or ""):
        return None
    base = Path(photos_dir).resolve()
    p = (base / name).resolve()
    return p if p.parent == base and p.is_file() else None
