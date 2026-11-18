"""Refresh ONLY the event-level fields of models/model_card.json from reports/metrics.json.

Used once after the P90 review changed the event metric (one-to-one matching, 2 s units):
the model is NOT retrained (models/model.joblib unchanged), window-level metrics are checked
to be identical, and the card records when and why its event fields were refreshed.
Run: python scripts/refresh_card_events.py [--card models/model_card.json]
                                           [--metrics reports/metrics.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

EVENT_FIELDS = ("events_detected", "false_alarms", "false_alarms_per_km", "events_predicted",
                "event_precision", "event_matching")
POINT = ("Pothole", "Irregularity")


def refresh(card: dict, metrics: dict) -> dict:
    for name, cv in card["cv_metrics"].items():
        m = metrics["models"][name]
        if abs(cv["macro_f1"] - m["macro_f1"]) > 1e-12 or abs(cv["accuracy"] - m["accuracy"]) > 1e-12:
            raise SystemExit(f"{name}: window-level metrics differ from the card; retrain instead")
        e = m["events"]
        cv.update({"events_detected": {c: e[c]["detected_of"] for c in POINT},
                   "false_alarms": {c: e[c]["false_alarms"] for c in POINT},
                   "false_alarms_per_km": {c: e[c]["false_alarms_per_km"] for c in POINT},
                   "events_predicted": {c: e[c]["n_predicted"] for c in POINT},
                   "event_precision": {c: e[c]["event_precision"] for c in POINT},
                   "event_matching": e["matching"]})
    card["event_metrics_refreshed"] = {
        "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "reason": "P90 review: event metric changed to one-to-one matching with 2 s units; "
                  "model not retrained (window-level metrics identical)",
        "fields": list(EVENT_FIELDS)}
    return card


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--card", default="models/model_card.json")
    ap.add_argument("--metrics", default="reports/metrics.json")
    a = ap.parse_args()
    cp = Path(a.card)
    card = refresh(json.loads(cp.read_text()), json.loads(Path(a.metrics).read_text()))
    cp.write_text(json.dumps(card, indent=1))
    print(f"refreshed event fields of {cp}")
