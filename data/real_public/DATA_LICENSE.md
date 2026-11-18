# Data license — Smart Cycle real recordings

The files in this directory (`rides/*.csv` accelerometer + GPS recordings, `labels.csv`,
`rides.csv`) are licensed under the **Creative Commons Attribution 4.0 International License
(CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

Attribution: "Smart Cycle recordings, Paul Emptoz and Gonzalo Adarraga, 2026, CC BY 4.0"

Anyone may share and adapt these recordings for any purpose, including commercially, provided
the attribution above is given and changes are indicated. The recorder gave written consent to
this public release (see `docs/DATA.md`, "Collection, consent and sharing").

Independent licenses in the same repository:
- Source code: MIT (`LICENSE` at the repository root).
- OpenStreetMap extract `data/osm/cycleways.geojson`: ODbL 1.0, "© OpenStreetMap contributors".

Contents: 5 rides recorded on 16 November 2026 on the UCD campus (Dublin). Ride CSVs use the
canonical format (`timestamp_ms` relative to the ride start, accel in m/s² including gravity,
GPS fields empty between fixes); `rides.csv` gives `started_at_utc` (Phyphox START time).
No device metadata is included. Built from the local raw data by `scripts/make_real_public.py`.
