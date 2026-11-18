# Dashboard test (P04a)

## Automated (headless Chrome, no npm package)
Run on a COPY of the database: the self-test writes a rider detail and a photo.
```
cp data/smartcycle_real.db /tmp/sc_copy.db
SMARTCYCLE_PHOTOS=/tmp/sc_photos python -m smartcycle serve --port 8765 --db /tmp/sc_copy.db &
node scripts/cdp_check.mjs --url 'http://127.0.0.1:8765/?selftest=1' --wait-selftest
node scripts/cdp_check.mjs --url 'http://127.0.0.1:8765/?selftest=1' --wait-selftest --reduced-motion
node scripts/cdp_check.mjs --url 'http://127.0.0.1:8765/?view=partner&theme=dark' --dark --shot /tmp/partner_dark.png
```
The script prints the request hosts (expected: the server, `tile.openstreetmap.org`,
`fonts.googleapis.com`, `fonts.gstatic.com`), console errors / exceptions, and the 21 self-test
lines. Useful URL parameters: `view=partner`, `select=<event id>`, `dialog=model`,
`theme=light|dark`, `replay=<ride id>`, `replay_ms=<ms>`.

## Manual checklist (browser, `python -m smartcycle serve`, http://127.0.0.1:8000/)
1. Map shows OSM tiles + attribution, the network (green = cycleway, grey = other paths), and the
   reported points; the view is centred on the data.
2. Public view: neutral dots, "Latest reports" (6), the partner-cities note, no severity anywhere
   (lists, panel, ride log).
3. Partner city view: key form (default `demo-partner`) -> segments coloured Good / Needs attention
   / Repair soon, Rough stretches thick orange, dot size = severity, Alerts + Priority repairs.
   A wrong key shows "Wrong partner key."
4. Click a list item -> the map flies to it and the panel shows it. Tab to a dot, press Enter ->
   selected (focus ring visible).
5. Selected point: type ("Pothole / bump"), id, lat/lon + way name, date, reports; severity +
   confidence only in the partner view. Send a detail (reason + text, 200 chars max) -> "Saved".
6. Photo: the privacy reminder is visible before choosing a file; upload a JPEG with GPS EXIF ->
   thumbnail appears; `curl` the photo URL and check with Pillow that EXIF is empty.
7. Upload ride CSV (e.g. a synthetic `data/demo_rides/rides/syn_00.csv`, or a real demo ride on a
   DB copy) -> ride log lists detections with time (and confidence in the partner view).
8. Replay demo ride: the rider dot moves along the track (~10 s), rings pulse, log lines appear;
   with a real ride the "training data, replayed for demonstration" warning is shown.
9. Model badge -> dialog: provenance badge, rides / windows per class, RF vs rule macro-F1, false
   alarms per km with the evaluated distance, 3-class diagnostic, caveats, both report images.
10. Theme button switches light/dark; both readable. OS "reduce motion" -> no ring animation.
11. Edge states: synthetic DB -> red "SYNTHETIC DEMO DATA" banner; `SMARTCYCLE_NETWORK=/none`
    -> "No OSM network loaded" note, points at raw GPS; API stopped / page served elsewhere ->
    "The API is not reachable" + Retry; no model -> banner with the train hint.
