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
