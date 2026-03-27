import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from cr_event_analysis import blob_pca_metrics


def test_blob_pca_single_pixel():
    coords = np.array([[5, 5]])
    m = blob_pca_metrics(coords)
    assert m["major_extent_pix"] == 0.0
    assert m["minor_extent_pix"] == 0.0
    assert m["aspect_ratio"] == 1.0
    assert m["orientation_deg"] == 0.0


def test_blob_pca_empty_input():
    coords = np.empty((0, 2), dtype=int)
    m = blob_pca_metrics(coords)
    assert m["major_extent_pix"] == 0.0
    assert m["minor_extent_pix"] == 0.0
    assert m["aspect_ratio"] == 1.0
    assert m["orientation_deg"] == 0.0


def test_blob_pca_horizontal_line():
    coords = np.array([[10, 3], [10, 4], [10, 5], [10, 6], [10, 7]])
    m = blob_pca_metrics(coords)
    assert m["major_extent_pix"] > 3.9
    assert m["minor_extent_pix"] < 1e-10
    assert np.isinf(m["aspect_ratio"])
    assert abs(m["orientation_deg"]) < 1e-6


def test_blob_pca_vertical_line():
    coords = np.array([[3, 10], [4, 10], [5, 10], [6, 10], [7, 10]])
    m = blob_pca_metrics(coords)
    assert m["major_extent_pix"] > 3.9
    assert m["minor_extent_pix"] < 1e-10
    assert np.isinf(m["aspect_ratio"])
    assert abs(abs(m["orientation_deg"]) - 90.0) < 1e-6


def test_blob_pca_square_blob():
    coords = np.array([
        [0, 0], [0, 1], [0, 2],
        [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2],
    ])
    m = blob_pca_metrics(coords)
    assert m["major_extent_pix"] >= m["minor_extent_pix"]
    assert abs(m["aspect_ratio"] - 1.0) < 0.25