"""Offline QR code for the deployed demo URL (optional dev dependency `segno`).

    pip install -e '.[qr]'
    python scripts/make_qr.py https://<your-app>.onrender.com/ --out docs/qr.png   # or .svg
"""

from __future__ import annotations

import argparse
import sys
from urllib.parse import urlparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write a QR code for a URL (offline, no web service)")
    p.add_argument("url")
    p.add_argument("--out", default="docs/qr.png", help=".png or .svg")
    p.add_argument("--scale", type=int, default=10, help="pixels per module (PNG)")
    a = p.parse_args(argv)
    u = urlparse(a.url)
    if u.scheme not in ("http", "https") or not u.netloc:
        print(f"error: not an http(s) URL: {a.url!r}", file=sys.stderr)
        return 2
    try:
        import segno
    except ImportError:
        print("error: segno is not installed; run `pip install -e '.[qr]'`", file=sys.stderr)
        return 1
    segno.make(a.url, error="m").save(a.out, scale=a.scale, border=4)
    print(f"QR code for {a.url} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
