"""Physics-inspired SYNTHETIC accelerometer + GPS signals for one ride.

Bike frame: x = lateral, y = longitudinal (forward), z = vertical (up). The accelerometer
reads +9.81 m/s^2 on the up axis at rest (gravity included, Android convention).
Classes follow the data collection plan:
- Smooth: low road-texture noise.
- Rough: continuous 8-25 Hz low-amplitude vertical oscillation (cobbles, gravel).
- Pothole: isolated 0.1-0.3 s damped oscillation with a big vertical spike (+ then -).
- Irregularity: 0.3-0.8 s lateral swerve (x) or braking pulse (y), small vertical response.
Unlabelled distractors (bumps, small swerves) and weak patches make classes overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.spatial.transform import Rotation

from smartcycle.synth.routes import EARTH_M_PER_DEG, Polyline

G = 9.81


@dataclass
class SimParams:
    hard: bool = False
    duration_s: float | None = None  # None -> U(180, 480)
    mount: str = "handlebar"
    event_mean_gap_s: float = 6.0


@dataclass
class RideSim:
    t: np.ndarray  # accel sample times (s from first sample), after gaps
    acc: np.ndarray  # (n, 3) m/s^2 incl. gravity, phone frame
    gps: dict[str, np.ndarray]  # t, latitude, longitude, accuracy, speed
    intervals: list[tuple[float, float, str]]  # label intervals (s)
    info: dict = field(default_factory=dict)


def _noise(rng: np.random.Generator, n: int, fs: float, lo: float | None, hi: float | None
           ) -> np.ndarray:
    """Unit-std filtered white noise (band, low or high pass)."""
    pad = int(fs)
    x = rng.standard_normal(n + pad)
    nyq = 0.5 * fs
    if lo and hi:
        sos = butter(4, [lo, min(hi, 0.9 * nyq)], btype="band", fs=fs, output="sos")
    elif hi:
        sos = butter(4, min(hi, 0.9 * nyq), btype="low", fs=fs, output="sos")
    else:
        sos = butter(4, lo, btype="high", fs=fs, output="sos")
    y = sosfilt(sos, x)[pad:]
    return y / (y.std() + 1e-12)


def _smooth(x: np.ndarray, fs: float, width_s: float) -> np.ndarray:
    w = max(1, int(width_s * fs))
    k = np.hanning(w + 2)[1:-1]
    k /= k.sum()
    return np.convolve(np.pad(x, (w, w), mode="edge"), k, mode="same")[w:-w]


def _speed_profile(rng, t, fs) -> tuple[np.ndarray, list[tuple[float, float]]]:
    dur = t[-1]
    base = rng.uniform(3.0, 7.0)
    v = base + sum(rng.uniform(0.2, 0.6) * np.sin(2 * np.pi * t / rng.uniform(20, 90)
                                                    + rng.uniform(0, 6.3)) for _ in range(3))
    v = np.clip(v, 2.5, 8.0)
    mask = np.ones_like(t)
    stops = [(0.0, rng.uniform(3.0, 6.0))]  # rides start at rest (phone just started)
    for _ in range(rng.integers(1, 3)):
        a = rng.uniform(0.25, 0.85) * dur
        stops.append((a, a + rng.uniform(5.0, 15.0)))
    for a, b in stops:
        mask[(t >= a) & (t < b)] = 0.0
    mask = _smooth(mask, fs, 3.0)
    mask[t < stops[0][1]] = 0.0
    return v * mask, stops


def _pothole(tau: np.ndarray, d: float) -> np.ndarray:
    """Positive then negative spike, then decaying ringing (duration ~2d)."""
    inside = (tau >= 0) & (tau <= 2 * d)
    tc = np.where(inside, tau, 0.0)
    return np.where(inside, np.exp(-1.5 * tc / d) * np.sin(2 * np.pi * tc / d), 0.0)


def simulate_ride(rng: np.random.Generator, route: Polyline, fs: float, p: SimParams) -> RideSim:
    dur = p.duration_s if p.duration_s else rng.uniform(180.0, 480.0)
    n = int(dur * fs)
    t = np.arange(n) / fs
    v, stops = _speed_profile(rng, t, fs)
    vs = (v / 5.0) ** 0.7  # road excitation grows with speed, zero at rest
    hard = 2.0 if p.hard else 1.0
    pocket = p.mount == "pocket"

    # --- surface timeline: Smooth / Rough segments -----------------------------------
    intervals: list[tuple[float, float, str]] = []
    rough_env = np.zeros(n)
    smooth_amp = np.zeros(n)
    s = 0.0
    while s < dur:
        e = min(dur, s + rng.uniform(5.0, 30.0))
        m = (t >= s) & (t < e)
        if rng.random() < 0.38:
            weak = rng.random() < (0.25 * hard)
            amp = rng.uniform(0.4, 0.8) if weak else rng.uniform(0.8, 2.5) / (1 + 0.3 * p.hard)
            rough_env[m] = amp
            intervals.append((s, e, "Rough"))
        else:
            smooth_amp[m] = rng.uniform(0.08, 0.25) * (1.8 if p.hard else 1.0)
            intervals.append((s, e, "Smooth"))
        s = e
    rough_env = _smooth(rough_env, fs, 0.3)
    smooth_amp = _smooth(smooth_amp, fs, 0.3)

    dyn = np.zeros((n, 3))  # road-induced (damped in a pocket)
    body = np.zeros((n, 3))  # rider-induced (pedalling, manoeuvres)

    # texture
    tex = _noise(rng, n, fs, 2.0, 20.0)
    dyn[:, 2] += smooth_amp * vs * tex
    dyn[:, :2] += 0.4 * (smooth_amp * vs)[:, None] * np.column_stack(
        [_noise(rng, n, fs, 2.0, 20.0), _noise(rng, n, fs, 2.0, 20.0)])
    rough = _noise(rng, n, fs, 8.0, 25.0)
    dyn[:, 2] += rough_env * vs * rough
    dyn[:, :2] += 0.35 * (rough_env * vs)[:, None] * np.column_stack(
        [_noise(rng, n, fs, 8.0, 25.0), _noise(rng, n, fs, 8.0, 25.0)])

    # pedalling (with coasting phases); much larger for a phone in a trouser pocket
    cad = rng.uniform(1.0, 1.5)
    pedal_on = _smooth((rng.random(n // int(fs) + 1) < 0.75).repeat(int(fs))[:n].astype(float),
                       fs, 1.0)
    ped_amp = rng.uniform(1.5, 3.5) if pocket else rng.uniform(0.3, 0.8)
    ps = np.clip((v - 0.5) / 2.0, 0, 1) * pedal_on * ped_amp
    ph = 2 * np.pi * cad * t + rng.uniform(0, 6.3)
    body[:, 0] += ps * np.sin(ph)
    body[:, 1] += 0.6 * ps * np.sin(2 * ph + 0.5)
    body[:, 2] += 0.3 * ps * np.sin(2 * ph + 1.0)
    # handling noise (also at rest: the rider holds the bike)
    body += 0.04 * np.column_stack([_noise(rng, n, fs, None, 5.0) for _ in range(3)])

    # --- discrete events ----------------------------------------------------------------
    kinds = ["Pothole", "Irregularity", "bump", "swerve"]
    probs = np.array([0.38, 0.32, 0.18 * hard, 0.12 * hard])
    probs /= probs.sum()
    events = []
    te = rng.uniform(6.0, 10.0)
    last = -10.0
    while te < dur - 3.0:
        i = int(te * fs)
        if v[i] > 2.5 and te - last >= 2.0:
            kind = str(rng.choice(kinds, p=probs))
            k = np.sqrt(max(v[i], 1.0) / 5.0)
            if kind in ("Pothole", "bump"):
                d = rng.uniform(0.1, 0.3)
                A = (rng.uniform(4.0, 18.0) if p.hard else rng.uniform(8.0, 25.0)) * k
                if kind == "bump":  # unlabelled curb/joint bump: overlaps with mild potholes
                    A = (rng.uniform(3.0, 9.0) if p.hard else rng.uniform(2.0, 6.0)) * k
                w = _pothole(t - te, d)
                dyn[:, 2] += A * w
                dyn[:, 0] += 0.12 * A * rng.choice([-1, 1]) * w
                dyn[:, 1] -= 0.15 * A * np.abs(w)
                end = te + 2 * d
            else:
                d = rng.uniform(0.3, 0.8)
                A = rng.uniform(2.0, 6.0) if p.hard else rng.uniform(3.0, 8.0)
                if kind == "swerve":
                    A = rng.uniform(2.0, 4.5) if p.hard else rng.uniform(1.0, 3.0)
                tau = t - te
                inside = (tau >= 0) & (tau <= d)
                half = np.where(inside, np.sin(np.pi * tau / d), 0.0)
                if rng.random() < 0.5:  # swerve: left then right
                    body[:, 0] += A * rng.choice([-1, 1]) * np.where(
                        inside, np.sin(2 * np.pi * tau / d), 0.0)
                    body[:, 2] += 0.1 * A * half
                else:  # hard braking: deceleration pulse + fork dive
                    body[:, 1] -= A * half
                    body[:, 2] += 0.2 * A * half
                end = te + d
            events.append((te, end, kind))
            if kind in ("Pothole", "Irregularity"):
                intervals.append((max(0.0, te - 0.4), min(dur, end + 0.4), kind))
            last = end
        te += max(2.0, rng.exponential(p.event_mean_gap_s))

    # --- mount: damping + orientation ----------------------------------------------------
    if pocket:  # soft tissue + fabric act as a low-pass filter
        sos = butter(2, min(12.0, 0.45 * fs), btype="low", fs=fs, output="sos")
        dyn = 0.8 * sosfilt(sos, dyn, axis=0)
        r0 = Rotation.random(random_state=rng)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = np.radians(rng.uniform(2, 6)) * np.sin(2 * np.pi * t / rng.uniform(60, 200))
        rot = Rotation.from_rotvec(axis[None, :] * ang[:, None]) * r0
    else:
        tilt = rng.normal(size=3)
        tilt = tilt / np.linalg.norm(tilt) * np.radians(rng.uniform(0, 10))
        rot = Rotation.from_rotvec(tilt)
    a_bike = dyn + body + np.array([0.0, 0.0, G])
    acc = rot.inv().apply(a_bike)  # bike frame -> phone frame
    acc = acc +rng.uniform(-0.08, 0.08, size=3) + rng.uniform(0.03, 0.10) * rng.standard_normal(
        (n, 3))

    # --- timing: jitter + gaps -------------------------------------------------------------
    jitter = np.clip(rng.normal(0, 0.05 / fs, n), -0.4 / fs, 0.4 / fs)
    jitter[0] = 0.0
    t_obs = t + jitter
    keep = np.ones(n, dtype=bool)
    for _ in range(rng.integers(0, 3)):
        a = rng.uniform(15.0, dur - 5.0)
        keep[(t >= a) & (t < a + rng.uniform(0.3, 1.0))] = False

    # --- GPS: ~1 Hz fixes along the route with correlated noise ------------------------------
    dist = np.cumsum(v) / fs
    t_fix = np.arange(rng.uniform(0.3, 1.0), dur, 1.0)
    t_fix = t_fix + rng.normal(0, 0.03, t_fix.size)
    t_fix = t_fix[(rng.random(t_fix.size) > 0.05) & (t_fix > 0) & (t_fix < dur)]
    ll = route.at(np.interp(t_fix, t, dist))
    sd = rng.uniform(3.0, 6.0)
    noise = np.zeros((t_fix.size, 2))
    for j in range(1, t_fix.size):
        noise[j] = 0.7 * noise[j - 1] + rng.normal(0, sd * np.sqrt(1 - 0.49), 2)
    lat = ll[:, 0] + noise[:, 1] / EARTH_M_PER_DEG
    lon = ll[:, 1] + noise[:, 0] / (EARTH_M_PER_DEG * np.cos(np.radians(ll[:, 0])))
    accuracy = np.clip(_smooth(rng.uniform(3, 10, t_fix.size), 1.0, 5.0), 3.0, 10.0)
    speed = np.clip(np.interp(t_fix, t, v) + rng.normal(0, 0.2, t_fix.size), 0, None)

    intervals.sort(key=lambda x: (x[0], x[1]))
    return RideSim(
        t=t_obs[keep], acc=acc[keep],
        gps={"t": t_fix, "latitude": lat, "longitude": lon, "accuracy": accuracy, "speed": speed},
        intervals=intervals,
        info={"fs": fs, "duration_s": dur, "mount": p.mount, "stops": stops,
              "n_events": {k: sum(e[2] == k for e in events) for k in kinds}},
    )
