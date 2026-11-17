# Demo rides

Rides loaded into the demo database with `python -m smartcycle ingest --dir data/demo_rides`.

For now they are SYNTHETIC (regenerable, gitignored), with a seed different from the training set
so the demo is not in-sample:

```
python -m smartcycle synth --out data/demo_rides --rides 3 --minutes 2 --seed 7 --plot ''
SMARTCYCLE_NETWORK=data/synthetic/routes.geojson python -m smartcycle ingest --dir data/demo_rides
```

`ingest --dir` reads `rides.csv` (mount per ride) and `MANIFEST.json` (provenance: synthetic).
Small real demo rides may replace them after the real data collection (P05/P06).

## Real demo rides (P05)
`data/demo_rides/real/{bump,irregs,rug,smooth}.csv` are copies of the normalised real rides
(37–62 s each, handlebar). They contain raw GPS traces and are **gitignored** (not committed until
the repo is confirmed private). They are TRAINING rides of the final model: replaying them is an
in-sample demo, not an evaluation.

```
SMARTCYCLE_DB=data/smartcycle_real.db python -m smartcycle serve --port 8000
curl -F file=@data/demo_rides/real/irregs.csv -F provenance=real localhost:8000/api/rides/upload-csv
```
The real-data database is built with
`python -m smartcycle ingest --dir data/raw --db data/smartcycle_real.db`.
