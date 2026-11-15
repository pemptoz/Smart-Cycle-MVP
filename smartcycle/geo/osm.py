"""OpenStreetMap network download (Overpass API) -> segmented GeoJSON.

Two layers in one file, told apart by `is_cycleway`:
- cycleways (`true`): highway=cycleway; highway in (path, footway, track, pedestrian) with
  bicycle in (designated, yes); any highway with cycleway / cycleway:left / :right / :both whose
  value is not no / none / separate;
- ordinary roads (`false`): residential, tertiary, secondary, primary, unclassified, service, so
  rides on roads without a cycle lane can still be matched.

Ways are split at junction nodes (nodes shared by several ways), then into 50–100 m segments
with stable ids `seg_<wayid>_<k>`. Data © OpenStreetMap contributors (ODbL).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from smartcycle.geo.network import EARTH_M_PER_DEG, polyline_length_m, segmentise

OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter")
USER_AGENT = "smartcycle-mvp/0.1 (student project, road-quality research)"
ROAD_HIGHWAYS = ("residential", "tertiary", "secondary", "primary", "unclassified", "service")
PATH_HIGHWAYS = ("path", "footway", "track", "pedestrian")
NO_VALUES = {"no", "none", "separate"}
CYCLEWAY_KEYS = ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")


class OverpassError(RuntimeError):
    pass


def overpass_query(bbox: tuple[float, float, float, float], timeout_s: int = 90) -> str:
    """bbox = (south, west, north, east)."""
    b = ",".join(f"{v:.6f}" for v in bbox)
    paths = "|".join(PATH_HIGHWAYS)
    roads = "|".join(ROAD_HIGHWAYS)
    cyc = "".join(f'  way["highway"]["{k}"]["{k}"!~"^(no|none|separate)$"]({b});\n'
                  for k in CYCLEWAY_KEYS)
    return (f"[out:json][timeout:{timeout_s}];\n(\n"
            f'  way["highway"="cycleway"]({b});\n'
            f'  way["highway"~"^({paths})$"]["bicycle"~"^(designated|yes)$"]({b});\n'
            f"{cyc}"
            f'  way["highway"~"^({roads})$"]({b});\n'
            ");\nout body;\n>;\nout skel qt;\n")


def is_cycleway(tags: dict) -> bool:
    hw = tags.get("highway", "")
    if hw == "cycleway":
        return True
    if hw in PATH_HIGHWAYS and tags.get("bicycle") in ("designated", "yes"):
        return True
    return any(k in tags and tags[k] not in NO_VALUES for k in CYCLEWAY_KEYS)


def keep_way(tags: dict) -> bool:
    return is_cycleway(tags) or tags.get("highway") in ROAD_HIGHWAYS


def fetch_overpass(query: str, timeout_s: int = 90, retries: int = 2) -> dict:
    """POST the query; up to `retries` retries over the mirrors. Raises OverpassError."""
    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for attempt in range(retries + 1):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s + 15) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last = e
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    raise OverpassError(f"Overpass API unreachable after {retries + 1} attempts: {last}")


def overpass_to_segments(osm: dict) -> dict:
    """Overpass JSON (ways with node refs + nodes) -> segmented GeoJSON FeatureCollection."""
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in osm.get("elements", [])
             if e.get("type") == "node"}
    ways = [e for e in osm.get("elements", [])
            if e.get("type") == "way" and keep_way(e.get("tags", {}))]
    use: dict[int, int] = {}
    for w in ways:
        for nid in set(w.get("nodes", [])):
            use[nid] = use.get(nid, 0) + 1
    feats = []
    for w in ways:
        tags = w.get("tags", {})
        refs = [n for n in w.get("nodes", []) if n in nodes]
        if len(refs) < 2:
            continue
        # split at junctions (inner nodes shared with another way)
        pieces, cur = [], [refs[0]]
        for nid in refs[1:]:
            cur.append(nid)
            if use.get(nid, 0) > 1 and nid != refs[-1]:
                pieces.append(cur)
                cur = [nid]
        if len(cur) > 1:
            pieces.append(cur)
        k = 0
        cyc = is_cycleway(tags)
        for piece in pieces:
            coords = [list(nodes[n]) for n in piece]
            subs = segmentise(coords)
            for j, sub in enumerate(subs):
                feats.append({"type": "Feature", "properties": {
                    "segment_id": f"seg_{w['id']}_{k}", "way_id": int(w["id"]),
                    "name": tags.get("name") or ("Unnamed cycleway" if cyc else "Unnamed road"),
                    "highway": tags.get("highway", ""), "is_cycleway": cyc,
                    "length_m": round(polyline_length_m(sub), 1),
                    "start_node": int(piece[0]) if j == 0 else None,
                    "end_node": int(piece[-1]) if j == len(subs) - 1 else None},
                    "geometry": {"type": "LineString",
                                 "coordinates": [[round(a, 7), round(b, 7)] for a, b in sub]}})
                k += 1
    return {"type": "FeatureCollection",
            "attribution": "© OpenStreetMap contributors (ODbL)",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "features": feats}


def tracks_bbox(tracks: str | Path, buffer_m: float = 300.0
                ) -> tuple[tuple[float, float, float, float], int]:
    """(south, west, north, east) of all GPS fixes of the ride files under `tracks`, + buffer."""
    from smartcycle.io.loader import load_ride

    p = Path(tracks)
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*.csv") if f.name not in ("labels.csv", "rides.csv"))
    lats, lons = [], []
    for f in files:
        try:
            df = load_ride(f)
        except ValueError:
            continue
        g = df[df["latitude"].notna() & df["longitude"].notna()]
        lats.append(g["latitude"].to_numpy())
        lons.append(g["longitude"].to_numpy())
    if not lats or sum(len(a) for a in lats) == 0:
        raise ValueError(f"no GPS fix found in ride files under {tracks}")
    lat, lon = np.concatenate(lats), np.concatenate(lons)
    dlat = buffer_m / EARTH_M_PER_DEG
    dlon = buffer_m / (EARTH_M_PER_DEG * np.cos(np.radians(lat.mean())))
    return ((float(lat.min() - dlat), float(lon.min() - dlon),
             float(lat.max() + dlat), float(lon.max() + dlon)), len(files))


def fetch_osm(tracks: str | Path, out: str | Path, buffer_m: float = 300.0,
              force: bool = False) -> tuple[Path, int, bool]:
    """Returns (path, n_segments, downloaded?). Uses the cached file unless `force`."""
    out = Path(out)
    if out.exists() and not force:
        n = len(json.loads(out.read_text()).get("features", []))
        return out, n, False
    bbox, _ = tracks_bbox(tracks, buffer_m)
    gj = overpass_to_segments(fetch_overpass(overpass_query(bbox)))
    gj["bbox"] = list(bbox)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gj, separators=(",", ":")))
    return out, len(gj["features"]), True
