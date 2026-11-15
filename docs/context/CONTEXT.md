# Smart Cycle — Project context (condensed from the 3 source documents)

Source files (read-only, same folder): `Smart_Cycle.pptx` (pitch deck), `Data_Collection_Plan.pdf`, `Smart_Cycle.html` (marketing website with a SIMULATED demo).
This file is the authoritative summary. Do not edit the source files.

## 1. Project
- **Name:** Smart Cycle. **Team:** Paul Emptoz, Gonzalo Adarraga. **Course:** "AI for Engineering". **Deadline of the MVP: Wednesday 18 Nov 2026.**
- **Pitch:** AI-powered smartphone app that detects bumps, cracks and potholes on Dublin cycleways while people ride, geolocates them, and turns them into a map cities can act on. Links to UN SDG 11 (sustainable cities).
- **Problem:** cycleways are rarely inspected; manual inspection is expensive; cycling budgets are being cut; rough surfaces hurt safety, accessibility, comfort. Dublin adults cycling 5+ days/week: 8% (2023) -> 12% (2025).
- **Value for cities:** no inspection vehicles, real-time data, targeted maintenance, lower logistics cost, fewer accidents, more citizen participation.
- **Stakeholders:** Government/partner cities (full analysis, heatmap of road quality, alerts on worst sections), Riders (safer cycleways, can back a report with a photo), Public (all reported data viewable online, without the analysis), Investors.

## 2. The three promised MVP results (slide "Expected Results")
1. **Trained Road Anomaly Classifier**
2. **Functional Smartphone Data Pipeline**
3. **Map-Matching & GPS Alignment**
Plus (slides "Our Solution"): automatic road-quality detection + GPS, and the possibility to share photos after the ride.

## 3. Data collection plan (facts)
**Data needed**
- Motion: 3-axis accelerometer, m/s², WITH gravity (user confirmed), high frequency (sampling rate not yet known).
- Spatial: latitude, longitude, GPS accuracy radius.
- Temporal: millisecond timestamps, to sync fast vibrations with slow GPS updates.
- Ground-truth labels (training only), exactly 4 classes:
  - `Smooth` — Z (vertical, gravity) steady ~9.8 m/s²; X/Y minimal noise.
  - `Rough` — continuous rapid low-amplitude Z oscillations (cobblestones, gravel, degraded tarmac); trajectory straight.
  - `Pothole` — isolated large spike then immediate drop on Z (wheel drops into cavity, hits far edge).
  - `Irregularity` — sudden sharp spikes on X (swerve) or Y (hard braking): rider's reaction to hazards (glass, parked car in lane, a pothole, sunken drain cover). The alert is deliberately general; the user can detail afterwards what happened.
**Sample CSV in the plan** (columns): `Timestamp (ms), Accel_X (m/s²), Accel_Y, Accel_Z, Latitude, Longitude, Ground_Truth_Label`
e.g. `1700468404000, -1.10, 1.30, 21.45, 53.34990, -6.26040, Pothole` (the sample is 1 row/s and only illustrative; real data is high frequency; GPS is Dublin city centre ~53.350, -6.260).
**Where/how**
- Real bicycle lanes, urban + suburban, varying surface conditions (week 8); controlled test routes with known imperfections; optionally external scientific datasets (must be validated: methodology may differ).
- Smartphone accelerometer via **Phyphox** (accelerometer + GPS recorded concurrently), manual verification/labelling, optionally with video; export CSV, synchronise timestamps, map acceleration peaks to GPS locations.
- Model training week 9; data processing week 10 (plan order is inconsistent — we process before training).
- Risk named in the plan: model too insensitive to certain inputs; external data collected with a different methodology.
- Privacy: self-collected data, no identifiable elements in photos, no identifying info in feedback results, respondents informed, data deleted on request. Comply with GDPR/anonymisation.
**Reference** methodology: Lee, Yoo, Lee, Jeong (2025) smartphone-based bicycle pavement assessment (Bicycle Road Roughness Index, Faulting Impact Index) — inspiration for roughness/impact features. NOTE: the plan footnote cites *Sensors* 25, 3520 while the deck/website cite *Sustainability* 17, 7488 (different papers) — cite both carefully in the report, do not merge them.

## 4. Decisions taken for the MVP (with the user)
- Classes: ONLY the 4 plan labels. The website's `Crack / Faulting / Debris` are NOT detectable and are dropped. Website `Pothole / Rough surface` map to `Pothole / Rough`; new `Irregularity` replaces the rest.
- Real data will be collected on **Mon 16 Nov**. Until then, everything is built and tested on **synthetic** data (clearly labelled). Reported results must come from real data only.
- Phone mount: handlebar preferred, pocket as fallback -> features must be orientation-robust; keep a "fixed-axes" feature set for comparison.
- One CSV per ride containing accelerometer (with gravity) AND GPS; sampling rate unknown -> never hardcode; GPS rows may be sparse (NaN between fixes).
- Model runs **server-side**; everything must be **free** (local run is the baseline; optional free deployment).
- Phone for the demo: **Android (Chrome)**. UI/code/docs in **English**; the assistant talks to the user in French.
- User codes everything with Claude Code, alone, in ~3 days: **scope is deliberately small, "it must work", not perfect.**
- Map matching: real OpenStreetMap extract (cached), snap to cycleway segments.
- Severity (1–5) is an **uncalibrated heuristic** (not in the plan): derived from peak magnitude; must be labelled as such in the UI.

## 5. Website (`Smart_Cycle.html`) — what to reuse
It is a static marketing page with a fake demo (seeded random anomalies on a stylised map). Reuse its **look and UX**, not its data.
- Font: Outfit (Google Fonts), system fallback. Light + dark themes via `prefers-color-scheme` and `data-theme`.
- Tokens (light): `--bg #f6faf2; --ink #16241a; --mute #4d6153; --green #2f5d2c; --mint #bfdbab; --panel #fff; --line #cfdcc6; --water #d6e8f2; --lo #3f9a4a; --mid #d99a12; --hi #c93b27; --dot #2b3a2e`.
- Tokens (dark): `--bg #0f1a11; --ink #e6efe3; --mute #a3b7a8; --green #8cc57f; --mint #24402a; --panel #162419; --line #2d4632; --water #1a3140; --lo #5cc06a; --mid #e8b030; --hi #ef6a55; --dot #d5e3d3`.
- Demo UX to port: route selector + "Simulate a ride" (rider dot animates, ring pulse on each detection, ride log lines); **Public view** (raw dots, "Latest reports", no severity, note "analysis is for partner cities") vs **Partner city view** (segments coloured good/needs attention/repair soon, dot size = severity, **Alerts** = severity>=4 in last 10 days top 4, **Priority repairs** = top 5 segments by score); selected-point panel (type, id like `A-001`, GPS, date, #reports, severity only in partner view); "Attach photo" flow; legend; accessibility (aria labels, keyboard activation of markers, focus rings, reduced-motion).
- Heat colour: score >= 6 hi, >= 3 mid, else lo. Severity colour: >= 4 hi, >= 3 mid, else lo.
- Footer disclaimer pattern: "Sample data for illustration" -> in the MVP replace with provenance badge ("Real data" / "Synthetic demo data").

## 6. Original timeline (deck)
Oct–Dec: data collection (10 h), solution design & scope/assumptions (30 h), app development + testing (30 h), report (10 h), external feedback from Government / Investors / Citizens.
