import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import numpy as np

from cr_event_analysis import preclassify_events


def _make_case(shape=(1, 33, 33), noise_sigma=0.35, seed=0):
    rng = np.random.default_rng(seed)
    data_cube = rng.normal(loc=0.0, scale=noise_sigma, size=shape).astype(np.float32)
    medians = np.zeros(shape[0], dtype=np.float32)
    return data_cube, medians


def _run_single_event_case(data_cube, medians, frame, y, x, **kwargs):
    events = np.array([[frame, y, x]], dtype=int)
    rows = preclassify_events(
        f=frame,
        idxs=np.array([0], dtype=int),
        events=events,
        data_cube=data_cube,
        medians=medians,
        **kwargs,
    )
    assert len(rows) == 1
    return rows[0]


def test_preclassify_events_single_peak_with_noise():
    data_cube, medians = _make_case(seed=1)
    frame, y, x = 0, 16, 16
    data_cube[frame, y, x] += 40.0

    row = _run_single_event_case(data_cube, medians, frame, y, x)

    assert row["frame"] == frame
    assert row["y"] == y
    assert row["x"] == x
    assert row["peak_val"] > 30.0
    assert row["n_secondary_5x5"] == 0
    assert row["r3"] >= 0.0
    assert row["r5"] >= row["r3"]
    assert np.isfinite(row["linearity"]) or np.isinf(row["linearity"])


def test_preclassify_events_three_nearby_peaks_with_noise():
    data_cube, medians = _make_case(seed=2)
    frame, y, x = 0, 16, 16

    data_cube[frame, 16, 16] += 35.0
    data_cube[frame, 16, 18] += 28.0
    data_cube[frame, 18, 17] += 24.0

    row = _run_single_event_case(data_cube, medians, frame, y, x)

    assert row["peak_val"] > 25.0
    assert row["n_secondary_5x5"] >= 1
    assert row["r5"] > 0.0


def test_preclassify_events_thin_streak_with_noise():
    data_cube, medians = _make_case(seed=3)
    frame = 0

    pts = [(12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17)]
    amps = [14, 18, 22, 26, 22, 18]
    for (yy, xx), a in zip(pts, amps):
        data_cube[frame, yy, xx] += float(a)

    row = _run_single_event_case(data_cube, medians, frame, 15, 15)

    assert row["peak_val"] > 20.0
    assert row["r5"] > 0.5
    assert row["linearity"] >= 1.0


def test_preclassify_events_thick_streak_with_noise():
    data_cube, medians = _make_case(seed=4)
    frame = 0

    core = [(12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17)]
    flank1 = [(12, 13), (13, 14), (14, 15), (15, 16), (16, 17)]
    flank2 = [(13, 12), (14, 13), (15, 14), (16, 15), (17, 16)]

    for yy, xx in core:
        data_cube[frame, yy, xx] += 18.0
    for yy, xx in flank1 + flank2:
        data_cube[frame, yy, xx] += 9.0

    row = _run_single_event_case(data_cube, medians, frame, 15, 15)

    assert row["peak_val"] > 10.0
    assert row["r5"] > 1.0
    assert row["linearity"] >= 1.0


def test_preclassify_events_relative_behavior():
    # Single peak
    dc1, med1 = _make_case(seed=11)
    dc1[0, 16, 16] += 40.0
    row_single = _run_single_event_case(dc1, med1, 0, 16, 16)

    # Thin streak
    dc2, med2 = _make_case(seed=12)
    for (yy, xx), a in zip(
        [(12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17)],
        [14, 18, 22, 26, 22, 18],
    ):
        dc2[0, yy, xx] += float(a)
    row_thin = _run_single_event_case(dc2, med2, 0, 15, 15)

    # Three nearby peaks
    dc3, med3 = _make_case(seed=13)
    dc3[0, 16, 16] += 35.0
    dc3[0, 16, 18] += 28.0
    dc3[0, 18, 17] += 24.0
    row_three = _run_single_event_case(dc3, med3, 0, 16, 16)

    assert row_three["n_secondary_5x5"] >= row_single["n_secondary_5x5"]
    assert row_thin["r5"] >= row_single["r5"]
