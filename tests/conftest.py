"""Shared fixtures for the geo / API tests: a tiny SYNTHETIC dataset, a fast RF model trained
on it, and the synthetic route network. No network access anywhere."""

import json

import joblib
import pytest

from smartcycle.features.build import build_windows
from smartcycle.features.features import config_columns
from smartcycle.features.preprocess import PipelineConfig
from smartcycle.io import load_dataset
from smartcycle.models.zoo import make_model
from smartcycle.synth import generate
from smartcycle.synth.routes import routes_geojson


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_geo")
    generate(d, n_rides=3, seed=11, minutes=1.5)
    return d


@pytest.fixture(scope="session")
def tiny_model(tiny_data, tmp_path_factory):
    """(model.joblib path, model_card.json path) — RF fitted on all tiny windows."""
    cfg = PipelineConfig()
    wt = build_windows(load_dataset(tiny_data), cfg)
    cols = config_columns(list(wt.X.columns), cfg)
    model = make_model("rf").fit(wt.X[cols], wt.meta["label"])
    out = tmp_path_factory.mktemp("model")
    bundle = {"model": model, "model_name": "rf", "config": cfg.to_dict(),
              "feature_columns": cols, "classes": list(model.classes_),
              "data_provenance": "synthetic"}
    joblib.dump(bundle, out / "model.joblib")
    (out / "model_card.json").write_text(json.dumps({"model": "rf",
                                                     "data_provenance": "synthetic"}))
    return out / "model.joblib", out / "model_card.json"


@pytest.fixture(scope="session")
def routes_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("net") / "routes.geojson"
    p.write_text(json.dumps(routes_geojson()))
    return p
