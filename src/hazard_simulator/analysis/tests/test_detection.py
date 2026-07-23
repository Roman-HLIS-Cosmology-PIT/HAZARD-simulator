import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from cr_event_analysis import find_peaks_for_frame


def make_test_cube(shape=(1, 21, 21), noise_sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    data_cube = rng.normal(loc=0.0, scale=noise_sigma, size=shape).astype(np.float32)
    return data_cube


def add_peak(data_cube, frame, y, x, amp):
    data_cube[frame, y, x] += amp


def peak_set(peaks):
    return {(int(f), int(y), int(x)) for f, y, x in peaks}


def test_find_peaks_for_frame_finds_isolated_peak():
    data_cube = make_test_cube(seed=1)
    badpix_mask = np.zeros((21, 21), dtype=bool)

    add_peak(data_cube, frame=0, y=10, x=10, amp=50.0)

    peaks, _ , threshold = find_peaks_for_frame(
        data_cube=data_cube,
        index=0,
        badpix_mask=badpix_mask,
        sigma_thresh=5.0,
    )

    pset = peak_set(peaks)

    assert (0, 10, 10) in pset
    assert threshold > 0


def test_find_peaks_for_frame_rejects_bad_pixel_peak():
    data_cube = make_test_cube(seed=2)
    badpix_mask = np.zeros((21, 21), dtype=bool)

    # put a huge signal exactly on a bad pixel
    badpix_mask[10, 10] = True
    add_peak(data_cube, frame=0, y=10, x=10, amp=500.0)

    peaks, _ , _ = find_peaks_for_frame(
        data_cube=data_cube,
        index=0,
        badpix_mask=badpix_mask,
        sigma_thresh=5.0,
    )

    pset = peak_set(peaks)

    assert (0, 10, 10) not in pset


def test_find_peaks_for_frame_rejects_neighbor_of_bad_pixel_when_enabled():
    data_cube = make_test_cube(seed=3)
    badpix_mask = np.zeros((21, 21), dtype=bool)

    # bad pixel in center, bright nearby pixel adjacent to it
    badpix_mask[10, 10] = True
    add_peak(data_cube, frame=0, y=10, x=11, amp=60.0)

    peaks, _ , _ = find_peaks_for_frame(
        data_cube=data_cube,
        index=0,
        badpix_mask=badpix_mask,
        sigma_thresh=5.0,
        exclude_badpix_neighbors=True,
    )

    pset = peak_set(peaks)

    assert (0, 10, 11) not in pset


def test_find_peaks_for_frame_still_rejects_nearby_bad_pixel_when_disabled():
    """
    The final badpix_veto_radius check is always applied, even when
    exclude_badpix_neighbors=False.
    """
    data_cube = make_test_cube(seed=4)
    badpix_mask = np.zeros((21, 21), dtype=bool)

    badpix_mask[10, 10] = True

    # This peak is one pixel away from the bad pixel and should be vetoed.
    add_peak(
        data_cube,
        frame=0,
        y=10,
        x=11,
        amp=60.0,
    )

    peaks, _, _ = find_peaks_for_frame(
        data_cube=data_cube,
        index=0,
        badpix_mask=badpix_mask,
        sigma_thresh=5.0,
        exclude_badpix_neighbors=False,
    )

    pset = peak_set(peaks)

    assert (0, 10, 11) not in pset


def test_find_peaks_for_frame_keeps_peak_outside_badpix_veto_radius():
    """
    A peak more than badpix_veto_radius pixels away from the bad pixel
    should remain eligible for detection.
    """
    data_cube = make_test_cube(seed=6)
    badpix_mask = np.zeros((21, 21), dtype=bool)

    badpix_mask[10, 10] = True

    # The current veto radius is 3 pixels.
    # This peak is four columns away and should survive.
    add_peak(
        data_cube,
        frame=0,
        y=10,
        x=14,
        amp=60.0,
    )

    peaks, _, _ = find_peaks_for_frame(
        data_cube=data_cube,
        index=0,
        badpix_mask=badpix_mask,
        sigma_thresh=5.0,
        exclude_badpix_neighbors=False,
    )

    pset = peak_set(peaks)

    assert (0, 10, 14) in pset
    

def test_find_peaks_for_frame_finds_multiple_well_separated_peaks():
    data_cube = make_test_cube(seed=5)
    badpix_mask = np.zeros((21, 21), dtype=bool)

    add_peak(data_cube, frame=0, y=5, x=5, amp=45.0)
    add_peak(data_cube, frame=0, y=15, x=15, amp=55.0)
    add_peak(data_cube, frame=0, y=5, x=15, amp=65.0)

    peaks, _ , _ = find_peaks_for_frame(
        data_cube=data_cube,
        index=0,
        badpix_mask=badpix_mask,
        sigma_thresh=5.0,
    )

    pset = peak_set(peaks)

    assert (0, 5, 5) in pset
    assert (0, 15, 15) in pset
    assert (0, 5, 15) in pset

