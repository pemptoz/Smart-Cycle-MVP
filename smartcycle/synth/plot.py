"""Plot one example window per class from a SYNTHETIC dataset (eyeball check of the signals)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from smartcycle.io.dataset import load_dataset  # noqa: E402
from smartcycle.io.labels import LABELS  # noqa: E402

AXIS_COLORS = {"accel_x": "#2a78d6", "accel_y": "#eb6834", "accel_z": "#1baf7a"}
INK, MUTED, GRID, SHADE = "#1f1f1e", "#6b6a63", "#e4e3dd", "#f1efe8"


def _pick_window(df: pd.DataFrame, label: str, half: float) -> tuple[float, float, float, float]:
    """Return (t0, t1, lab_start, lab_end) centred on a labelled run of `label`."""
    lab = df["label"].fillna("")
    run = (lab != lab.shift()).cumsum()
    runs = df[lab == label].groupby(run[lab == label])["t_s"].agg(["min", "max"])
    runs = runs[runs["min"] > 10]  # skip the initial standstill
    if label in ("Smooth", "Rough"):
        best = runs.assign(d=runs["max"] - runs["min"]).sort_values("d").iloc[-1]
    else:
        best = runs.iloc[len(runs) // 2]
    c = 0.5 * (best["min"] + best["max"])
    return c - half, c + half, float(best["min"]), float(best["max"])


def plot_examples(data_dir: str | Path, out_png: str | Path, window_s: float = 3.0) -> Path:
    ds = load_dataset(data_dir)
    hb = ds.meta.index[ds.meta["mount"] == "handlebar"]
    rid = str(hb[0]) if len(hb) else next(iter(ds.rides))
    df = ds.rides[rid]
    fig, axes = plt.subplots(len(LABELS), 1, figsize=(9, 9), sharey=True)
    for ax, label in zip(axes, LABELS):
        t0, t1, a, b = _pick_window(df, label, window_s / 2)
        w = df[(df["t_s"] >= t0) & (df["t_s"] <= t1)]
        if label in ("Pothole", "Irregularity"):
            ax.axvspan(max(a, t0), min(b, t1), color=SHADE, zorder=0, lw=0)
        for col, color in AXIS_COLORS.items():
            ax.plot(w["t_s"], w[col], color=color, lw=1.2, label=col.replace("accel_", ""))
        ax.set_title(f"{label}  ({rid}, t = {t0:.1f}-{t1:.1f} s)", loc="left", color=INK,
                     fontsize=10)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_ylabel("m/s²", color=MUTED, fontsize=8)
    axes[-1].set_xlabel("time from ride start (s)", color=MUTED, fontsize=8)
    axes[0].legend(ncol=3, frameon=False, fontsize=8, loc="upper right",
                   bbox_to_anchor=(1.0, 1.32), title="phone axis", title_fontsize=8)
    fig.suptitle(f"SYNTHETIC data - one example window per class (mount: "
                 f"{ds.meta.loc[rid, 'mount']}, gravity included; shaded = labelled event)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, facecolor="white")
    plt.close(fig)
    return out_png
