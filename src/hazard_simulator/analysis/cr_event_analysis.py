# script for analyzing real and simulated cosmic ray events
# Initial creation date: 23-Mar-2026
# Developers: Anthony Harbo Torres
# with technical advice by Christopher Hirata
# Notes: Gaussian smoothing and edge detection idea
# provided by Emily Koivu
# version 0.16

import os
import time
import json
import argparse

import numpy as np 
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    maximum_filter,
    gaussian_filter,
    label,
    find_objects
)
from tqdm.contrib.concurrent import thread_map 
from concurrent.futures import ThreadPoolExecutor
from astropy.stats import sigma_clipped_stats 
from collections import Counter
from functools import partial
from astropy.io import fits 

from cr_analysis_plotting import generate_diagnostic_plots

def load_data(fits_path):
    """
    Loads FITS files

    Parameters
    ----------
    fits_path: str
        File path string

    Returns
    -------
    data: np.array
        Typically, a datacube of 100 frames of x-ray data
    """
    with fits.open(fits_path) as hdulist:
        # hdulist is a list of HDU (Header/Data Unit) objects
        primary_hdu = hdulist[0]
        data = primary_hdu.data      # NumPy array of the data
        header = primary_hdu.header  # FITS header metadata

    print(f"Data shape: {data.shape}")
    print("Header keys:", list(header.keys())[:10])
    return(data)

def compute_hot_pixel_mask(data, sigma_cutoff):
    """
    Finds outlier pixels by comparing against the median of all frames

    Parameters
    ----------
    data: np.array
        Datacube of differences images
    sigma_cutoff: float
        Desired multiple of the variance to use as a hot pixel cutoff

    Returns
    -------
    hot_pix_mask: np.array
        Array of ones and zeroes
    """
    print("Looking for hot pixels…")
    median_img = np.median(data, axis=0)
    mad        = np.median(np.abs(median_img - np.median(median_img)))
    sigma_est  = 1.4826 * mad
    thresh_med = np.median(median_img) + sigma_cutoff * sigma_est
    hot_pix_mask   = median_img > thresh_med
    print(f"Done searching for hot pixels (σ={sigma_est:.3f}, thresh={thresh_med:.1f})")
    return hot_pix_mask

def compute_veryhot_pixel_mask(data, sigma_cutoff):
    """
    Finds outlier pixels by comparing against the median of the first frame

    Parameters
    ----------
    data: np.array
        Datacube of differences images

    Returns
    -------
    data: np.array
        Typically, a datacube of 100 frames of x-ray data
    """
    print("Looking for very hot pixels…")
    first_img  = data[0]
    med_first  = np.median(first_img)
    mad_first  = np.median(np.abs(first_img - med_first))
    sigma_est  = 1.4826 * mad_first
    thresh0    = med_first + sigma_cutoff * sigma_est
    mask0      = first_img > thresh0
    print(f"Done searching for very hot pixels (σ={sigma_est:.3f}, thresh={thresh0:.1f})")
    return mask0

def compute_unresponsive_mask(data, sat_cut):
    print("Looking for non-responsive pixels…")
    # If you wanted a row-wise tqdm you could replace the next line with a loop + tqdm
    frame_diff = np.abs(np.diff(data, axis=0))       # (Nframe-1, 4096,4096)
    med_diff   = np.median(frame_diff, axis=0)
    mask_non_res   = med_diff < sat_cut
    print(f"Done searching for non-responsive pixesls (median(med_diff)={np.median(med_diff):.3e})")
    return mask_non_res

def filter_transient_events(events, transient_verification="full_exposure"):
    """
    Filter out non-transient peaks based on temporal behavior.

    Parameters
    ----------
    events : (N, 3) ndarray
        Array of (frame, y, x) peak detections.
    transient_verification : str
        One of:
            - "previous_frame" : remove peaks that also appear in frame f-1
            - "full_exposure"  : remove peaks that appear in any other frame

    Returns
    -------
    single_epoch_events : (M, 3) ndarray
        Filtered list of transient (single-epoch) events.
    """

    if len(events) == 0:
        return events.copy()

    events = np.asarray(events)

    # METHOD 1: full exposure (global Counter method)
    if transient_verification == "full_exposure":

        coord_counts = Counter(map(tuple, events[:, 1:]))

        keep = np.array(
            [coord_counts[(y, x)] == 1 for _, y, x in events],
            dtype=bool,
        )

        return events[keep]

    # METHOD 2: strict previous-frame-only check
    elif transient_verification == "previous_frame":
        # Build lookup: frame -> set of (y,x)
        frame_to_coords = {}
        for f, y, x in events:
            frame_to_coords.setdefault(f, set()).add((int(y), int(x)))

        keep_mask = np.zeros(len(events), dtype=bool)

        for i, (f, y, x) in enumerate(events):
            f = int(f)
            y = int(y)
            x = int(x)

            # If no previous frame exists, keep it
            if (f - 1) not in frame_to_coords:
                keep_mask[i] = True
                continue

            # Check if this coordinate exists in previous frame
            if (y, x) not in frame_to_coords[f - 1]:
                keep_mask[i] = True

        return events[keep_mask]

    # INVALID OPTION
    else:
        raise ValueError(
            f"Invalid transient_verification='{transient_verification}'. "
            "Choose from {'previous_frame', 'full_exposure'}."
        )


def find_peaks_for_frame(data_cube, index, badpix_mask, sigma_thresh,
    exclude_badpix_neighbors=False, veto_radius=3):
    image   = data_cube[index]
    _, median, _ = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    mad     = np.median(np.abs(image - median))
    sigma_e = mad * 1.4826
    threshold = median + sigma_thresh * sigma_e

    if exclude_badpix_neighbors:
        reject_mask = binary_dilation(badpix_mask, structure=np.ones((3,3)))
    else:
        reject_mask = badpix_mask

    image_for_max = image.astype(np.float32, copy=True)
    image_for_max[reject_mask] = -np.inf

    local_max = maximum_filter(image_for_max, size=3, mode="nearest")

    cand = (
        (image_for_max == local_max)
        & (~reject_mask)
        & np.isfinite(image_for_max)
        & (image > threshold)
    )

    ys, xs = np.where(cand)

    # code to deal with peaks near badpix
    peaks  = []
    ny, nx = image.shape

    for y, x in zip(ys, xs):
        ylo = max(0, y - veto_radius)
        yhi = min(ny, y + veto_radius + 1)
        xlo = max(0, x - veto_radius)
        xhi = min(nx, x + veto_radius + 1)

        if np.any(badpix_mask[ylo:yhi, xlo:xhi]):
            continue

        peaks.append((index, int(y), int(x)))
    
    return peaks, median, threshold

def summed_area_table(image):
    """
    Build a summed-area table (integral image) for fast box sums.
    """
    sat = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    sat[1:, 1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return sat


def box_sum_from_sat(sat, y0, y1, x0, x1):
    """
    Sum image[y0:y1, x0:x1] using a summed-area table.
    """
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]


def extract_box_bounds(y, x, shape, half_size):
    """
    Clip a square box centered on (y, x) to image boundaries.
    """
    ny, nx = shape
    y0 = max(0, y - half_size)
    y1 = min(ny, y + half_size + 1)
    x0 = max(0, x - half_size)
    x1 = min(nx, x + half_size + 1)
    return y0, y1, x0, x1


def count_secondary_local_peaks(
    image,
    y,
    x,
    half_size=2,
    rel_thresh=0.35,
    abs_thresh=None,
    footprint_size=3,
):
    """
    Count non-central local maxima in a small ROI around a peak.
    """
    y0, y1, x0, x1 = extract_box_bounds(y, x, image.shape, half_size)
    roi = image[y0:y1, x0:x1]

    cy = y - y0
    cx = x - x0
    center_val = roi[cy, cx]

    thresh = rel_thresh * center_val
    if abs_thresh is not None:
        thresh = max(thresh, abs_thresh)

    local_max = roi == maximum_filter(roi, size=footprint_size, mode="nearest")
    peak_mask = local_max & (roi >= thresh)

    # exclude the center peak itself
    peak_mask[cy, cx] = False

    return int(np.count_nonzero(peak_mask))

def preclassify_events(
    f,
    idxs,
    events,
    data_cube,
    medians,
    support3_thresh=0.18,
    support5_thresh=2.0,
    secondary_peak_rel_thresh=0.35,
    secondary_peak_abs_thresh=None,
    max_secondary_peaks_for_isolated=0,
):
    """
    Frame-level preclassification on raw, unmerged peaks.

    Uses:
      - 3x3 neighbor-only support ratio
      - 5x5 neighbor-only support ratio
      - secondary local-peak count in 11x11 (not final)
    Important:
      The integral image (summed-area table) is built ONCE per frame,
      not once per event.
    """
    image_raw = data_cube[f].astype(np.float32, copy=True)
    image_bg_subtract = image_raw - np.float32(medians[f])

    sat = summed_area_table(image_bg_subtract)

    rows = []
    for idx in idxs:
        _, y, x = events[idx].astype(int)

        p = float(image_bg_subtract[y, x])

        if p <= 0:
            rows.append({
                "event_index": int(idx),
                "frame": int(f),
                "y": int(y),
                "x": int(x),
                "class": "ambiguous",
                "peak_val": p,
                "r3": np.nan,
                "r5": np.nan,
                "n_secondary_in_5x5": -1,
            })
            continue

        # 3x3
        y0, y1, x0, x1 = extract_box_bounds(y, x, image_bg_subtract.shape, half_size=1)
        s3 = box_sum_from_sat(sat, y0, y1, x0, x1)

        # 5x5
        y0, y1, x0, x1 = extract_box_bounds(y, x, image_bg_subtract.shape, half_size=2)
        s5 = box_sum_from_sat(sat, y0, y1, x0, x1)
        # calculate the relative differences
        r3 = (s3 - p) / p
        r5 = (s5 - p) / p

        nsec = count_secondary_local_peaks(
            image_bg_subtract,
            y,
            x,
            half_size=2,
            rel_thresh=secondary_peak_rel_thresh,
            abs_thresh=secondary_peak_abs_thresh,
            footprint_size=3,
        )

        # 11x11
        y0, y1, x0, x1 = extract_box_bounds(y, x, image_bg_subtract.shape, half_size=5)
        s3 = box_sum_from_sat(sat, y0, y1, x0, x1)
        roi = image_bg_subtract[y0:y1, x0:x1]

        roi_pos = np.clip(roi, 0.0, None)
        sum_pos = float(roi_pos.sum())

        # fraction of positive signal in the center peak
        peak_fraction = p / sum_pos if sum_pos > 0 else 1.0

        # count of support pixels above a modest threshold
        support_floor = max(0.20 * p, 3.0)
        support_mask = roi > support_floor
        n_support = int(np.count_nonzero(support_mask))

        # connected support size containing the center pixel
        cy = y - y0
        cx = x - x0
        lab, _ = label(support_mask, structure=np.ones((3, 3), dtype=bool))
        if support_mask[cy, cx]:
            center_label = lab[cy, cx]
            center_cc_size = int(np.count_nonzero(lab == center_label))
        else:
            center_cc_size = 0


        # threshold to select signal pixels
        #mask = roi > (0.35 * p) THIS WAS CAUSING HIGH ENERGY STREAKS TO BE IGNORED

        #instead we'll use
        shape_floor_abs = 30.0          # same logic as edge_thresh/CDS argument
        shape_floor_rel = 0.01 * p      # much less aggressive than 0.35*p

        mask = roi > max(shape_floor_abs, shape_floor_rel)

        coords = np.argwhere(mask)

        major_axis_extent = 0.0
        minor_axis_extent = 0.0
        aspect_ratio = 1.0

        if len(coords) >= 2:
            vals = roi[mask]
            vals_pos = np.clip(vals, 0.0, None)

            pca_metrics = blob_pca_metrics(coords, weights=vals_pos)

            major_axis_extent = pca_metrics["major_extent_pix"]
            minor_axis_extent = pca_metrics["minor_extent_pix"]
            aspect_ratio = pca_metrics["aspect_ratio"]

        bbox_h = 0
        bbox_w = 0

        bbox_area = 0
        fill_frac = 1.0

        long_axis_bbox = 0
        short_axis_bbox = 0
        bbox_aspect_ratio = 1.0

        n_mask = len(coords)


        if n_mask > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            bbox_h = int(y_max - y_min + 1)
            bbox_w = int(x_max - x_min + 1)

            bbox_area = int(bbox_h * bbox_w)

            long_axis_bbox = int(max(bbox_h, bbox_w))
            short_axis_bbox = int(min(bbox_h, bbox_w))

            bbox_aspect_ratio = (
                long_axis_bbox / max(short_axis_bbox, 1)
            )


            fill_frac = (
                n_mask / bbox_area
                if bbox_area > 0
                else 0.0
            )

        if n_mask >= 2:
            vals = roi[mask]
            vals_pos = np.clip(vals, 0.0, None)

            pca_metrics = blob_pca_metrics(coords, weights=vals_pos)

            linearity = (
                pca_metrics["major_extent_pix"] /
                max(pca_metrics["minor_extent_pix"], 1e-6)
            )

            denom = (
                pca_metrics["major_extent_pix"] +
                pca_metrics["minor_extent_pix"]
            )

            anisotropy = (
                (pca_metrics["major_extent_pix"] - pca_metrics["minor_extent_pix"]) / denom
                if denom > 1e-6 else 0.0
            )
        else:
            linearity = 1.0
            anisotropy = 0.0

        # bbox / mask morphology categories
        is_tiny_bbox = (
            (n_mask <= 1)
            or (bbox_area <= 1)
        )

        is_morph_noise = (
            (n_support <= 2)
            or (center_cc_size <= 2)
            or is_tiny_bbox
            or (peak_fraction >= 0.85 and r5 < 1.5)
        )

        is_low_signal = (p < 100)

        is_noise_like = is_morph_noise and is_low_signal

        is_isolated = (
            (r3 < support3_thresh)
            and (r5 < support5_thresh)
            and (nsec <= max_secondary_peaks_for_isolated)
            and (linearity < 2.5)
            and (anisotropy < 0.55)
        ) and not is_low_signal

        is_streak_like = (
            (
                (aspect_ratio >= 1.4)     # Primary discriminator
                or
                (long_axis_bbox >= 5)         # fallback for very large events
            )
            and (major_axis_extent >= 3.5)     # must be spatially extended
            and (n_support >= 3)
        ) and not is_low_signal

        if is_noise_like:
            cls = "noise"
        elif is_isolated:
            cls = "likely_xray"
        elif is_streak_like:
            cls = "likely_streak"
        else:
            cls = "ambiguous"


        rows.append({
            "event_index": int(idx),
            "frame": int(f),
            "y": int(y),
            "x": int(x),
            "class": cls,
            "peak_val": p,
            "r3": float(r3),
            "r5": float(r5),
            "n_secondary_in_5x5": int(nsec),
            "linearity": float(linearity),
            "anisotropy": float(anisotropy),
            "bbox_h_5x5": int(bbox_h),
            "bbox_w_5x5": int(bbox_w),
            "n_mask_5x5": int(n_mask),
            "bbox_area_5x5": int(bbox_area),
            "long_axis_bbox_5x5": int(long_axis_bbox),
            "short_axis_bbox_5x5": int(short_axis_bbox),
            "bbox_aspect_ratio_5x5": float(bbox_aspect_ratio),
            "fill_frac_5x5": float(fill_frac),
            "major_axis_extent": float(major_axis_extent),
            "minor_axis_extent": float(minor_axis_extent),
            "aspect_ratio": float(aspect_ratio),
        })

    return rows

def assign_peaks_to_labels(coords, lab_img, search_radius=2):
    """
    Assign each peak coordinate to a nearby nonzero label in lab_img.

    If the exact peak pixel has label 0, search a small neighborhood
    for the nearest labeled pixel.

    Returns
    -------
    labels : 1D int ndarray
    """
    h, w = lab_img.shape
    labels = np.zeros(len(coords), dtype=int)

    for i, (y, x) in enumerate(coords):
        label0 = lab_img[y, x]
        if label0 != 0:
            labels[i] = label0
            continue

        y0 = max(0, y - search_radius)
        y1 = min(h, y + search_radius + 1)
        x0 = max(0, x - search_radius)
        x1 = min(w, x + search_radius + 1)

        patch = lab_img[y0:y1, x0:x1]
        nonzero = np.argwhere(patch > 0)

        if nonzero.size == 0:
            labels[i] = 0
            continue

        # nearest labeled pixel in the local patch
        dy = nonzero[:, 0] + y0 - y
        dx = nonzero[:, 1] + x0 - x
        d2 = dy * dy + dx * dx
        j = np.argmin(d2)

        yy = nonzero[j, 0] + y0
        xx = nonzero[j, 1] + x0
        labels[i] = lab_img[yy, xx]

    return labels

def build_event_neighborhood_mask(coords, h, w, radius=12):
    """
    Build a boolean mask that is True only in square neighborhoods around
    merged-event peak coordinates.

    Parameters
    ----------
    coords : (N, 2) ndarray
        Event coordinates as (y, x).
    h, w : int
        Frame shape.
    radius : int
        Half-size of the neighborhood box around each event.

    Returns
    -------
    mask : 2D boolean ndarray
    """
    mask = np.zeros((h, w), dtype=bool)

    for y, x in coords:
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        mask[y0:y1, x0:x1] = True

    return mask


def edge_pixels_from_mask(mask, structure=None):
    """
    Return a 1-pixel-wide edge mask for a filled boolean region.
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask, structure=structure, border_value=0)
    edge = mask & (~eroded)
    return edge


def build_smoothed_seeded_blob_labels(
    image_bg_subtract,
    coords,
    neighborhood_mask=None,
    structure=None,
    gaussian_sigma=0.8,
    edge_thresh=24.0,
    seed_thresh=32.0,
    min_blob_pixels=1,
    fill_holes=True,
    return_debug=False,
):
    """
    Build event labels from a smoothed ROI, but keep the final measurements tied
    to the original unsmoothed background-subtracted ROI.

    Strategy
    --------
    1. Smooth the local ROI to suppress pixel-scale jaggedness/noise.
    2. Threshold the smoothed image at a lower 'edge/support' threshold.
    3. Keep only connected support components that contain at least one event seed.
    4. Fill holes to get a solid footprint.
    5. Return a labeled image for downstream metrics on the ORIGINAL image_bg_subtract.

    Parameters
    ----------
    image_bg_subtract : 2D ndarray
        Background-subtracted ROI image.
    coords : (N, 2) ndarray
        Peak/event coordinates in ROI-local (y, x) coordinates.
    neighborhood_mask : 2D bool ndarray or None
        Optional mask restricting where footprint finding is allowed.
    structure : 2D bool ndarray or None
        Connectivity structure. Defaults to 3x3 full connectivity.
    gaussian_sigma : float
        Gaussian sigma in pixels for ROI smoothing.
    edge_thresh : float
        Lower threshold on the smoothed ROI used to define support/footprint.
    seed_thresh : float
        Higher threshold on the smoothed ROI used only for sanity checks.
        A component will be kept if it contains a seed; this threshold is
        mainly diagnostic protection against weak fuzzy junk.
    min_blob_pixels : int
        Minimum pixel count for a kept blob.
    fill_holes : bool
        If True, fill internal holes in each kept component.
    return_debug : bool
        If True, also return intermediate masks useful for plotting.

    Returns
    -------
    label_img : 2D int ndarray
        Final label image, 0 background, 1..N blobs.
    debug : dict, optional
        Returned only when return_debug=True.
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)

    h, w = image_bg_subtract.shape
    label_img = np.zeros((h, w), dtype=np.int32)

    if len(coords) == 0:
        if return_debug:
            return label_img, {
                "smoothed": np.zeros_like(image_bg_subtract, dtype=np.float32),
                "support_mask": np.zeros_like(label_img, dtype=bool),
                "edge_mask": np.zeros_like(label_img, dtype=bool),
                "seed_mask": np.zeros_like(label_img, dtype=bool),
            }
        return label_img

    # Restrict the smoothing/thresholding domain if requested.
    work = image_bg_subtract.astype(np.float32, copy=True)

    if neighborhood_mask is not None:
        work[~neighborhood_mask] = 0.0

    smoothed = gaussian_filter(work, sigma=gaussian_sigma, mode="nearest")

    # Thresholds on smoothed ROI
    support_mask = smoothed > edge_thresh
    seed_mask = smoothed > seed_thresh

    if neighborhood_mask is not None:
        support_mask &= neighborhood_mask
        seed_mask &= neighborhood_mask

    if not np.any(support_mask):
        if return_debug:
            return label_img, {
                "smoothed": smoothed,
                "support_mask": support_mask,
                "kept_mask": np.zeros_like(support_mask, dtype=bool),
                "edge_mask": np.zeros_like(support_mask, dtype=bool),
                "seed_mask": seed_mask,
            }
        return label_img

    # Connected components on smoothed support
    cc_img, n_cc = label(support_mask, structure=structure)

    if n_cc == 0:
        if return_debug:
            return label_img, {
                "smoothed": smoothed,
                "support_mask": support_mask,
                "kept_mask": np.zeros_like(support_mask, dtype=bool),
                "edge_mask": np.zeros_like(support_mask, dtype=bool),
                "seed_mask": seed_mask,
            }
        return label_img

    # Assign each seed to a support component.
    seed_cc_labels = assign_peaks_to_labels(coords, cc_img, search_radius=2)

    kept_components = []
    for cc_id in np.unique(seed_cc_labels):
        if cc_id <= 0:
            continue

        comp = (cc_img == cc_id)

        # protection against weak fuzzy patches:
        # keep if the component contains either a seed pixel assignment
        # (already true by construction) and is big enough.
        if fill_holes:
            comp = binary_fill_holes(comp)

        if int(np.count_nonzero(comp)) < min_blob_pixels:
            continue

        kept_components.append(comp)

    # Build final contiguous labels
    next_label = 1
    final_mask = np.zeros_like(support_mask, dtype=bool)

    for comp in kept_components:
        label_img[comp] = next_label
        final_mask |= comp
        next_label += 1

    # fallback: if smoothing was too aggressive and no support CC was
    # retained, create a tiny label around each seed above edge_thresh in the
    # ORIGINAL ROI.
    if next_label == 1:
        for (y, x) in coords:
            if not (0 <= y < h and 0 <= x < w):
                continue
            if image_bg_subtract[y, x] <= edge_thresh:
                continue

            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)

            tiny = np.zeros_like(final_mask, dtype=bool)
            tiny[y0:y1, x0:x1] = True

            if neighborhood_mask is not None:
                tiny &= neighborhood_mask

            label_img[tiny] = next_label
            final_mask |= tiny
            next_label += 1

    edge_mask = edge_pixels_from_mask(final_mask, structure=structure)

    if return_debug:
        return label_img, {
            "smoothed": smoothed,
            "support_mask": support_mask,
            "kept_mask": final_mask,
            "edge_mask": edge_mask,
            "seed_mask": seed_mask,
        }

    return label_img

def analyze_blobs_by_frame(
    f,
    idxs,
    events,
    data_cube,
    medians,
    h,
    w,
    small_struct,   
    peak_assign_radius=2,
    seed_thresh=32.0,
    edge_thresh=24.0,
    event_neighborhood_radius=16,
    gaussian_sigma=1.0,
    min_blob_pixels=1,
    fill_holes=True,
):

    # Frame setup + preprocessing
    t_frame_start = time.perf_counter()
    t0 = time.perf_counter()
    
    coords = events[idxs, 1:].astype(int)

    # Background-subtracted frame
    image_bg_subtract = data_cube[f].astype(np.float32, copy=True)
    image_bg_subtract -= np.float32(medians[f])

    # crop to ROI
    y0, y1, x0, x1 = build_event_roi(
        coords, h, w, radius=event_neighborhood_radius, pad=2
    )

    im_roi = image_bg_subtract[y0:y1, x0:x1]
    coords_roi = coords.copy()
    coords_roi[:, 0] -= y0
    coords_roi[:, 1] -= x0
    h_roi, w_roi = im_roi.shape

    t1 = time.perf_counter()

    # Neighborhood + grow mask
    t_mask_start = time.perf_counter()

    # Restrict analysis to ROI neighborhoods near merged peaks
    event_neighborhood_mask = build_event_neighborhood_mask(
        coords_roi, h_roi, w_roi, radius=event_neighborhood_radius
    )


    t_mask_end = time.perf_counter()

    # Build smart seeded labels
    t_label_start = time.perf_counter()

    lab_img_roi = build_smoothed_seeded_blob_labels(
        image_bg_subtract=im_roi,
        coords=coords_roi,
        neighborhood_mask=event_neighborhood_mask,
        structure=small_struct,
        gaussian_sigma=gaussian_sigma,
        edge_thresh=edge_thresh,
        seed_thresh=seed_thresh,
        min_blob_pixels=min_blob_pixels,
        fill_holes=fill_holes,
    )

    t_label_end = time.perf_counter()

    n_blobs = int(lab_img_roi.max())

    # Blob metrics (PCA + gini)
    t_metrics_start = time.perf_counter()

    # Blob-level arrays
    sums = np.zeros(n_blobs, dtype=np.float32)
    counts = np.zeros(n_blobs, dtype=int)
    major_extent_geom = np.zeros(n_blobs, dtype=np.float32)
    minor_extent_geom = np.zeros(n_blobs, dtype=np.float32)
    major_extent_pix = np.zeros(n_blobs, dtype=np.float32)
    minor_extent_pix = np.zeros(n_blobs, dtype=np.float32)
    aspect_ratios = np.zeros(n_blobs, dtype=np.float32)
    orientations = np.zeros(n_blobs, dtype=np.float32)
    
    #ginis = np.zeros(n_blobs, dtype=np.float32)
    #new two Gini code below

    gini_pixels = np.zeros(n_blobs, dtype=np.float32)

    gini_longitudinal = np.full(
        n_blobs,
        np.nan,
        dtype=np.float32,
    )

    longitudinal_peak_fraction = np.full(
        n_blobs,
        np.nan,
        dtype=np.float32,
    )

    longitudinal_cv = np.full(
        n_blobs,
        np.nan,
        dtype=np.float32,
    )

    longitudinal_end_asymmetry = np.full(
        n_blobs,
        np.nan,
        dtype=np.float32,
    )

    longitudinal_peak_offset = np.full(
        n_blobs,
        np.nan,
        dtype=np.float32,
    )

    n_longitudinal_bins = np.zeros(
        n_blobs,
        dtype=np.int32,
    )

    blob_slices = find_objects(lab_img_roi)

    for blob_label, slc in enumerate(blob_slices, start=1):
        if slc is None:
            continue

        # labeled-image subarray
        lab_sub = lab_img_roi[slc]

        # Matching subimage from the background-subtracted frame
        im_sub = im_roi[slc]

        blob_mask = (lab_sub == blob_label)

        blob_coords = np.argwhere(blob_mask)
        blob_coords[:, 0] += slc[0].start
        blob_coords[:, 1] += slc[1].start

        blob_vals = im_sub[blob_mask]
        blob_vals_pos = np.clip(blob_vals, 0.0, None)

        sums[blob_label - 1] = float(np.sum(blob_vals))
        counts[blob_label - 1] = int(blob_mask.sum())

        n_blob_pix = blob_coords.shape[0]

        if n_blob_pix > 10000:
            print(f"Warning: frame {f}, blob {blob_label} has {n_blob_pix} pixels")
            major_extent_geom[blob_label - 1] = np.nan
            minor_extent_geom[blob_label - 1] = np.nan
            major_extent_pix[blob_label - 1] = np.nan
            minor_extent_pix[blob_label - 1] = np.nan
            aspect_ratios[blob_label - 1] = np.nan
            orientations[blob_label - 1] = np.nan
        else:
            metrics = blob_pca_metrics(blob_coords, weights=blob_vals_pos)
            major_extent_geom[blob_label - 1] = metrics["major_extent_geom"]
            minor_extent_geom[blob_label - 1] = metrics["minor_extent_geom"]
            major_extent_pix[blob_label - 1] = metrics["major_extent_pix"]
            minor_extent_pix[blob_label - 1] = metrics["minor_extent_pix"]
            aspect_ratios[blob_label - 1] = metrics["aspect_ratio"]
            orientations[blob_label - 1] = metrics["orientation_deg"]

        #ginis[blob_label - 1] = _gini_coefficient(blob_vals)
        blob_index = blob_label - 1

        # Existing pixel-value Gini.
        gini_pixels[blob_index] = _gini_coefficient(blob_vals)

        # Only interpret the longitudinal direction when the object has
        # a sufficiently well-defined streak geometry.
        has_stable_major_axis = (
            np.isfinite(aspect_ratios[blob_index])
            and np.isfinite(major_extent_pix[blob_index])
            and aspect_ratios[blob_index] >= 1.4
            and major_extent_pix[blob_index] >= 3.5
        )

        if has_stable_major_axis:
            longitudinal = longitudinal_streak_metrics(
                coords=blob_coords,
                values=blob_vals_pos,
                bin_width=1.0,
                charge_weighted_axis=False,
            )

            gini_longitudinal[blob_index] = (
                longitudinal["gini_longitudinal"]
            )

            longitudinal_peak_fraction[blob_index] = (
                longitudinal["longitudinal_peak_fraction"]
            )

            longitudinal_cv[blob_index] = (
                longitudinal["longitudinal_cv"]
            )

            longitudinal_end_asymmetry[blob_index] = (
                longitudinal["longitudinal_end_asymmetry"]
            )

            longitudinal_peak_offset[blob_index] = (
                longitudinal["longitudinal_peak_offset"]
            )

            n_longitudinal_bins[blob_index] = (
                longitudinal["n_longitudinal_bins"]
            )

    t_metrics_end = time.perf_counter()

    # Assign peaks to labels
    t_assign_start = time.perf_counter()

    # Each merged peak gets assigned to the final smart label image
    # using ROI-local coordinates, since lab_img_roi is ROI-sized.
    hit_labels = assign_peaks_to_labels(
        coords_roi,
        lab_img_roi,
        search_radius=peak_assign_radius,
    )

    t_assign_end = time.perf_counter()

    t_frame_end = time.perf_counter()

    # PRINT TIMING SUMMARY
    print(
        f"[Frame {f}] "
        f"setup={t1 - t0:.2f}s | "
        f"mask={t_mask_end - t_mask_start:.2f}s | "
        f"label={t_label_end - t_label_start:.2f}s | "
        f"metrics={t_metrics_end - t_metrics_start:.2f}s | "
        f"assign={t_assign_end - t_assign_start:.2f}s | "
        f"TOTAL={t_frame_end - t_frame_start:.2f}s"
    )

    return {
        "frame": f,
        "idxs": idxs,
        "sums": sums,
        "counts": counts,
        "major_extent_geom": major_extent_geom,
        "minor_extent_geom": minor_extent_geom,
        "major_extent_pix": major_extent_pix,
        "minor_extent_pix": minor_extent_pix,
        "aspect_ratios": aspect_ratios,
        "orientations": orientations,
        #"ginis": ginis,
        "gini_pixels": gini_pixels,
        "gini_longitudinal": gini_longitudinal,
        "longitudinal_peak_fraction": longitudinal_peak_fraction,
        "longitudinal_cv": longitudinal_cv,
        "longitudinal_end_asymmetry": longitudinal_end_asymmetry,
        "longitudinal_peak_offset": longitudinal_peak_offset,
        "n_longitudinal_bins": n_longitudinal_bins,
        "hit_labels": hit_labels,
    }


def process_hit(
    hit,
    data_cube,
    medians,
    gain_array,
    supercell_size,
    blob_sums,
    blob_counts,
    blob_major_extent_geom,
    blob_minor_extent_geom,
    blob_major_extent_pix,
    blob_minor_extent_pix,
    blob_aspect_ratios,
    blob_orientations,
    gini_pixels_blob,
    gini_longitudinal_blob,
    longitudinal_peak_fraction,
    longitudinal_cv,
    longitudinal_end_asymmetry,
    longitudinal_peak_offset,
    n_longitudinal_bins,
):
    """
    Build one final-result row for a detected event with a valid blob label.

    The blob-level metric dictionaries are indexed first by frame and
    then by blob_label - 1.
    """
    frame, y, x, blob_label = hit.astype(int)

    img_raw = data_cube[frame].astype(
        np.float32,
        copy=False,
    )

    med = float(medians[frame])
    img_bgsub = img_raw - np.float32(med)

    sc_row = y // supercell_size
    sc_col = x // supercell_size

    sc_gain = float(
        gain_array[sc_row, sc_col]
    )

    sum3_bgsub_DN = float(
        _clipped_box_sum(
            img_bgsub,
            y,
            x,
            radius=1,
        )
    )

    sum5_bgsub_DN = float(
        _clipped_box_sum(
            img_bgsub,
            y,
            x,
            radius=2,
        )
    )

    # Blob labels are numbered 1...N, while arrays are indexed 0...N-1.
    blob_index = blob_label - 1

    sum_blob = float(blob_sums[frame][blob_index])
    n_pix_blob = int(blob_counts[frame][blob_index])
    major_blob_geom = float(blob_major_extent_geom[frame][blob_index])
    minor_blob_geom = float(blob_minor_extent_geom[frame][blob_index])
    major_blob = float(blob_major_extent_pix[frame][blob_index])
    minor_blob = float(blob_minor_extent_pix[frame][blob_index])
    aspect_blob = float(blob_aspect_ratios[frame][blob_index])
    orient_blob = float(blob_orientations[frame][blob_index])
    gini_pixel = float(gini_pixels_blob[frame][blob_index])
    gini_longitudinal = float(gini_longitudinal_blob[frame][blob_index])
    longitudinal_peak_fraction = float(longitudinal_peak_fraction[frame][blob_index])
    longitudinal_cv = float(longitudinal_cv[frame][blob_index])
    longitudinal_end_asymmetry = float(longitudinal_end_asymmetry[frame][blob_index])
    longitudinal_peak_offset = float(longitudinal_peak_offset[frame][blob_index])
    n_longitudinal_bins = int(n_longitudinal_bins[frame][blob_index])

    return {
        "frame": frame, "y": y, "x": x, "median": med,
        "sum3x3_bgsub_DN": sum3_bgsub_DN, "sum3x3_bgsub_e": sum3_bgsub_DN * sc_gain,
        "sum5x5_bgsub_DN": sum5_bgsub_DN, "sum5x5_bgsub_e": sum5_bgsub_DN * sc_gain,
        "blob_label": blob_label, "blob_DN": sum_blob,
        "blob_e": sum_blob * sc_gain, "n_pix_blob": n_pix_blob,
        "major_extent_geom": major_blob_geom, "minor_extent_geom": minor_blob_geom,
        "major_extent_pix": major_blob, "major_extent_um": major_blob * 10.0,
        "minor_extent_pix": minor_blob, "minor_extent_um": minor_blob * 10.0,
        "aspect_ratio_blob": aspect_blob, "orientation_deg_blob": orient_blob,

        # Backward-compatible alias
        "gini_blob": gini_pixel,

        # Explicit new Gini metrics.
        "gini_pixel_blob": gini_pixel,
        "gini_longitudinal_blob": gini_longitudinal,

        # Other longitudinal-profile diagnostics.
        "longitudinal_peak_fraction": longitudinal_peak_fraction,
        "longitudinal_cv": longitudinal_cv,
        "longitudinal_end_asymmetry": longitudinal_end_asymmetry,
        "longitudinal_peak_offset": longitudinal_peak_offset,
        "n_longitudinal_bins": n_longitudinal_bins,
        "supercell_gain": sc_gain,
    }


def build_event_roi(coords, h, w, radius=12, pad=2):
    y_min = max(0, coords[:, 0].min() - radius - pad)
    y_max = min(h, coords[:, 0].max() + radius + pad + 1)
    x_min = max(0, coords[:, 1].min() - radius - pad)
    x_max = min(w, coords[:, 1].max() + radius + pad + 1)
    return y_min, y_max, x_min, x_max

# HELPER FUNCTIONS

def _clipped_box_sum(img, y, x, radius):
    h, w = img.shape
    y0, y1 = max(y - radius, 0), min(y + radius + 1, h)
    x0, x1 = max(x - radius, 0), min(x + radius + 1, w)
    return img[y0:y1, x0:x1].sum()


def _gini_coefficient(values):
    """
    Gini coefficient of a 1D array of nonnegative values.
    Returns 0 for empty arrays or all-zero arrays.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return 0.0

    # For morphology / charge concentration, negative background-subtracted
    # values are not physically useful here.
    x = np.clip(x, 0.0, None)

    if np.all(x == 0):
        return 0.0

    x = np.sort(x)
    n = x.size
    if n < 2 or np.sum(x) <= 0:
        return 0.0
    
    index = np.arange(1, n + 1)

    return (np.sum((2 * index - n - 1) * x)) / ((n -1)* np.sum(x))


def blob_pca_metrics(coords, weights=None):
    """
    Compute PCA-based morphology metrics for a set of blob pixels.

    Parameters
    ----------
    coords : either
        - (N, 2) array of (y, x) pixel coordinates, or
        - 2D mask array, where nonzero pixels define the blob
    weights : (N,) array or None
        Optional nonnegative weights for coordinate inputs.
        If coords is a mask, weights are ignored unless you later add
        explicit support for weighted masks.

    Returns
    -------
    metrics : dict
    """
    arr = np.asarray(coords)

    # Accept mask input directly
    if arr.ndim == 2 and arr.shape[1] != 2:
        coords = np.argwhere(arr > 0).astype(np.float64)
        weights = None
    else:
        coords = np.asarray(arr, dtype=np.float64)

    # Strict shape check for coordinate input
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("blob_pca_metrics expects either a 2D mask or an (N,2) coordinate array.")

    if coords.shape[0] == 0:
        return {
            "major_extent_geom": 0.0,
            "minor_extent_geom": 0.0,
            "major_extent_pix": 0.0,
            "minor_extent_pix": 0.0,
            "aspect_ratio": 1.0,
            "orientation_deg": 0.0,
        }

    if coords.shape[0] == 1:
        return {
            "major_extent_geom": 1.0,
            "minor_extent_geom": 1.0,
            "major_extent_pix": 1.0,
            "minor_extent_pix": 1.0,
            "aspect_ratio": 1.0,
            "orientation_deg": 0.0,
        }

    YX = coords.astype(np.float64)

    if weights is None:
        center = YX.mean(axis=0)
        Xc = YX - center
        cov = (Xc.T @ Xc) / max(len(Xc), 1)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)

        if np.sum(w) <= 0:
            center = YX.mean(axis=0)
            Xc = YX - center
            cov = (Xc.T @ Xc) / max(len(Xc), 1)
        else:
            wsum = np.sum(w)
            center = np.sum(YX * w[:, None], axis=0) / wsum
            Xc = YX - center
            cov = (Xc.T @ (Xc * w[:, None])) / wsum

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    major_axis = evecs[:, 0]
    minor_axis = evecs[:, 1]

    proj_major = Xc @ major_axis
    proj_minor = Xc @ minor_axis

    # Geometric span in projected coordinates
    major_extent_geom = float(proj_major.max() - proj_major.min() + 1.0)
    minor_extent_geom = float(proj_minor.max() - proj_minor.min() + 1.0)

    # Convert projected geometric span to pixel-count-like span
    major_step_scale = max(abs(major_axis[0]), abs(major_axis[1]))
    minor_step_scale = max(abs(minor_axis[0]), abs(minor_axis[1]))

    major_extent_pix = 1.0 + (major_extent_geom - 1.0) * major_step_scale
    minor_extent_pix = 1.0 + (minor_extent_geom - 1.0) * minor_step_scale

    if major_extent_pix < minor_extent_pix:
        major_extent_pix, minor_extent_pix = minor_extent_pix, major_extent_pix
        major_extent_geom, minor_extent_geom = minor_extent_geom, major_extent_geom
        major_axis, minor_axis = minor_axis, major_axis

    aspect_ratio = major_extent_pix / minor_extent_pix if minor_extent_pix > 0 else np.inf

    dy, dx = major_axis[0], major_axis[1]
    orientation_deg = float(np.degrees(np.arctan2(dy, dx)))

    return {
        "major_extent_geom": major_extent_geom,
        "minor_extent_geom": minor_extent_geom,
        "major_extent_pix": major_extent_pix,
        "minor_extent_pix": minor_extent_pix,
        "aspect_ratio": aspect_ratio,
        "orientation_deg": orientation_deg,
    }


def _principal_axis_basis(coords, weights=None):
    """
    Find the center and PCA major/minor axes for (y, x) coordinates.

    Parameters
    ----------
    coords : (N, 2) array
        Pixel coordinates in (y, x) order.
    weights : (N,) array or None
        Optional nonnegative PCA weights.

    Returns
    -------
    center : (2,) ndarray
        PCA center in (y, x).
    major_axis : (2,) ndarray
        Unit vector along the major axis.
    minor_axis : (2,) ndarray
        Unit vector along the minor axis.
    """
    coords = np.asarray(coords, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")

    if len(coords) < 2:
        return (
            coords.mean(axis=0) if len(coords) else np.zeros(2),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
        )

    if weights is None:
        center = coords.mean(axis=0)
        centered = coords - center
        covariance = centered.T @ centered / len(centered)

    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)

        if weights.shape != (len(coords),):
            raise ValueError(
                "weights must contain one value per coordinate."
            )

        weight_sum = weights.sum()

        if weight_sum <= 0:
            center = coords.mean(axis=0)
            centered = coords - center
            covariance = centered.T @ centered / len(centered)

        else:
            center = np.sum(
                coords * weights[:, None],
                axis=0,
            ) / weight_sum

            centered = coords - center

            covariance = (
                centered.T
                @ (centered * weights[:, None])
                / weight_sum
            )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]

    major_axis = eigenvectors[:, order[0]]
    minor_axis = eigenvectors[:, order[1]]

    return center, major_axis, minor_axis


def longitudinal_streak_metrics(
    coords,
    values,
    bin_width=1.0,
    charge_weighted_axis=False,
):
    """
    Collapse a 2D streak across its width and measure the resulting
    1D charge profile along its PCA major axis.

    Parameters
    ----------
    coords : (N, 2) array
        Blob-pixel coordinates in (y, x) order.
    values : (N,) array
        Background-subtracted blob-pixel values.
    bin_width : float
        Longitudinal bin width in pixels.
    charge_weighted_axis : bool
        If True, use signal-weighted PCA to determine the axis.
        If False, use the blob geometry alone.

    Returns
    -------
    metrics : dict
        Longitudinal profile and summary metrics.
    """
    coords = np.asarray(coords, dtype=np.float64)
    charge = np.asarray(values, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")

    if charge.shape != (len(coords),):
        raise ValueError(
            "values must contain one entry per coordinate."
        )

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    valid = (
        np.all(np.isfinite(coords), axis=1)
        & np.isfinite(charge)
    )

    coords = coords[valid]
    charge = np.clip(charge[valid], 0.0, None)

    if len(coords) < 2 or charge.sum() <= 0:
        return {
            "gini_longitudinal": 0.0,
            "longitudinal_peak_fraction": 1.0,
            "longitudinal_cv": 0.0,
            "longitudinal_end_asymmetry": 0.0,
            "longitudinal_peak_offset": 0.0,
            "n_longitudinal_bins": int(len(coords) > 0),
            "longitudinal_profile": charge.copy(),
        }

    axis_weights = charge if charge_weighted_axis else None

    center, major_axis, _ = _principal_axis_basis(
        coords,
        weights=axis_weights,
    )

    centered = coords - center

    # Position of every pixel along the major axis, in pixel units.
    longitudinal_position = centered @ major_axis

    position_min = longitudinal_position.min()

    bin_index = np.floor(
        (longitudinal_position - position_min) / bin_width
    ).astype(int)

    n_bins = int(bin_index.max()) + 1

    # Sum across the transverse direction within each longitudinal slice.
    profile = np.bincount(
        bin_index,
        weights=charge,
        minlength=n_bins,
    ).astype(np.float64)

    total_charge = profile.sum()

    gini_longitudinal = _gini_coefficient(profile)

    peak_bin = int(np.argmax(profile))

    longitudinal_peak_fraction = (
        profile[peak_bin] / total_charge
    )

    mean_profile = profile.mean()

    longitudinal_cv = (
        profile.std() / mean_profile
        if mean_profile > 0
        else 0.0
    )

    # Orientation-invariant displacement of the brightest bin:
    # 0 means central; 1 means at either endpoint.
    if n_bins > 1:
        peak_fractional_position = peak_bin / (n_bins - 1)
        longitudinal_peak_offset = (
            2.0 * abs(peak_fractional_position - 0.5)
        )
    else:
        longitudinal_peak_offset = 0.0

    # Compare charge in the first and last quarters of the track.
    n_end_bins = max(1, int(np.ceil(0.25 * n_bins)))

    first_end_charge = profile[:n_end_bins].sum()
    last_end_charge = profile[-n_end_bins:].sum()

    longitudinal_end_asymmetry = (
        abs(first_end_charge - last_end_charge)
        / total_charge
    )

    return {
        "gini_longitudinal": float(gini_longitudinal),
        "longitudinal_peak_fraction": float(
            longitudinal_peak_fraction
        ),
        "longitudinal_cv": float(longitudinal_cv),
        "longitudinal_end_asymmetry": float(
            longitudinal_end_asymmetry
        ),
        "longitudinal_peak_offset": float(
            longitudinal_peak_offset
        ),
        "n_longitudinal_bins": int(n_bins),

        # Useful for diagnostic plots; do not put the array into the CSV.
        "longitudinal_profile": profile,
    }


def _timestamped_name(base_name, timestamp, on_hpc):
    name, ext = os.path.splitext(base_name)
    if on_hpc:
        job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        return f"{name}_{timestamp}_job{job_id}{ext}"
    return f"{name}_{timestamp}{ext}"


def load_sim_data(sim_data_path, sim_metadata_path, pixel_size=10.0,):
    """
    Load a simulated 2D DN image and its associated energy-loss metadata.

    The metadata CSV contains one row per propagation step. PIDs use the
    following 32-bit encoding:

        7 bits  : species index
        11 bits : primary-particle index
        14 bits : delta-ray index

    Rows belonging to a primary particle and all of its delta rays are grouped
    together using parent_PID, which is obtained by clearing the lower
    14 delta-ray bits.

    Parameters
    ----------
    sim_data_path : str or path-like
        Path to the 2D NumPy array containing the simulated DN image.

    sim_metadata_path : str or path-like
        Path to the energy-loss CSV containing at least:
        PID, step, x, y, z, dE, and delta_energy.

    pixel_size : float
        Pixel pitch in microns.

    Returns
    -------
    sim_data : np.ndarray
        Simulated 2D DN image as float32.

    sim_metadata_df : pd.DataFrame
        Event-level metadata with one row per parent simulated event.
    """
    # Load the simulated image
    sim_data = np.load(sim_data_path)

    if sim_data.ndim != 2:
        raise ValueError(
            f"Sim data must be 2D, got shape {sim_data.shape}."
        )

    if sim_data.shape != (4096, 4096):
        raise ValueError(
            "Sim data must have shape (4096, 4096), "
            f"got {sim_data.shape}."
        )

    sim_data = sim_data.astype(np.float32, copy=False)

    # Load the step-level simulation metadata
    sim_step_metadata_df = pd.read_csv(sim_metadata_path)

    required_columns = {
        "PID",
        "step",
        "x",
        "y",
        "z",
        "dE",
        "delta_energy",
    }

    missing_columns = required_columns - set(sim_step_metadata_df.columns)

    if missing_columns:
        raise ValueError(
            "Simulation metadata CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(sim_step_metadata_df) == 0:
        raise ValueError("Simulation metadata CSV contains no rows.")

    sim_step_metadata_df = sim_step_metadata_df.copy()

    # Ensure PID is stored as an integer before applying bit operations.
    sim_step_metadata_df["PID"] = pd.to_numeric(
        sim_step_metadata_df["PID"],
        errors="raise",
    ).astype(np.int64)

    pid_values = sim_step_metadata_df["PID"].to_numpy(dtype=np.int64)

    if np.any(pid_values < 0) or np.any(pid_values > (2**32 - 1)):
        raise ValueError(
            "Simulation PIDs must be unsigned 32-bit integer values."
        )

    # Decode PID bit fields
    delta_bits = 14
    primary_bits = 11
    species_bits = 7

    delta_mask = (1 << delta_bits) - 1
    primary_mask = (1 << primary_bits) - 1
    species_mask = (1 << species_bits) - 1

    # Lowest 14 bits
    sim_step_metadata_df["delta_index"] = (
        pid_values & delta_mask
    )

    # Next 11 bits
    sim_step_metadata_df["primary_index"] = (
        (pid_values >> delta_bits) & primary_mask
    )

    # Highest 7 bits
    sim_step_metadata_df["species_index"] = (
        pid_values >> (delta_bits + primary_bits)
    ) & species_mask

    # Clear the lower 14 delta-ray bits.
    sim_step_metadata_df["parent_PID"] = (
        pid_values & ~delta_mask
    )

    sim_step_metadata_df["is_delta_ray"] = (
        sim_step_metadata_df["delta_index"] != 0
    )

    # Used to count distinct delta-ray PIDs during aggregation.
    sim_step_metadata_df["delta_PID"] = (
        sim_step_metadata_df["PID"].where(
            sim_step_metadata_df["is_delta_ray"]
        )
    )

    sim_step_metadata_df["sim_x"] = np.floor(
        pd.to_numeric(
            sim_step_metadata_df["x"],
            errors="raise",
        ) / pixel_size).astype(np.int64)

    sim_step_metadata_df["sim_y"] = np.floor(
        pd.to_numeric(
            sim_step_metadata_df["y"],
            errors="raise",
        ) / pixel_size).astype(np.int64)

    # Collapse the step-level table into one row per parent event
    sim_metadata_df = (
        sim_step_metadata_df
        .groupby("parent_PID", as_index=False, sort=True)
        .agg(
            species_index=("species_index", "first"),
            primary_index=("primary_index", "first"),

            # Number of distinct particle trajectories:
            # one primary plus any associated delta rays.
            n_particle_PIDs=("PID", "nunique"),

            # Number of distinct delta-ray trajectories.
            n_delta_rays=("delta_PID", "nunique"),

            # Metadata row and step counts.
            n_steps=("PID", "size"),
            n_primary_steps=(
                "is_delta_ray",
                lambda values: int((~values).sum()),
            ),
            n_delta_steps=("is_delta_ray", "sum"),

            # Full trajectory bounds in microns.
            x_min_um=("x", "min"),
            x_max_um=("x", "max"),
            y_min_um=("y", "min"),
            y_max_um=("y", "max"),
            z_min_um=("z", "min"),
            z_max_um=("z", "max"),

            # Energy quantities.
            total_dE_MeV=("dE", "sum"),
            total_delta_energy_MeV=("delta_energy", "sum"),
        )
    )

    sim_metadata_df["n_delta_steps"] = (
        sim_metadata_df["n_delta_steps"].astype(int)
    )

    # Find the pixel bounds of energy deposited into the image
    deposit_rows = sim_step_metadata_df.loc[
        pd.to_numeric(
            sim_step_metadata_df["dE"],
            errors="coerce",
        ).fillna(0.0) > 0.0
    ].copy()

    deposit_bounds_df = (
        deposit_rows
        .groupby("parent_PID", as_index=False, sort=True)
        .agg(
            sim_x0=("sim_x", "min"),
            sim_x_last=("sim_x", "max"),
            sim_y0=("sim_y", "min"),
            sim_y_last=("sim_y", "max"),
        )
    )

    # Convert inclusive maximum pixel indices into half-open bounds:
    # [sim_y0:sim_y1, sim_x0:sim_x1]
    deposit_bounds_df["sim_x1"] = (
        deposit_bounds_df["sim_x_last"] + 1
    )

    deposit_bounds_df["sim_y1"] = (
        deposit_bounds_df["sim_y_last"] + 1
    )

    deposit_bounds_df = deposit_bounds_df.drop(
        columns=["sim_x_last", "sim_y_last"]
    )

    # Attach the deposited-energy pixel bounds to each parent event.
    sim_metadata_df = sim_metadata_df.merge(
        deposit_bounds_df,
        on="parent_PID",
        how="left",
        validate="one_to_one",
    )

    # Validate and clip the pixel bounds
    bounds_columns = [
        "sim_x0",
        "sim_x1",
        "sim_y0",
        "sim_y1",
    ]

    missing_bounds = sim_metadata_df[bounds_columns].isna().any(axis=1)

    if missing_bounds.any():
        missing_parent_pids = sim_metadata_df.loc[
            missing_bounds,
            "parent_PID",
        ].tolist()

        raise ValueError(
            "Some parent events have no positive-dE spatial bounds. "
            f"Parent PIDs: {missing_parent_pids}"
        )

    sim_h, sim_w = sim_data.shape

    sim_metadata_df["sim_x0"] = (
        sim_metadata_df["sim_x0"]
        .clip(0, sim_w)
        .astype(np.int64)
    )

    sim_metadata_df["sim_x1"] = (
        sim_metadata_df["sim_x1"]
        .clip(0, sim_w)
        .astype(np.int64)
    )

    sim_metadata_df["sim_y0"] = (
        sim_metadata_df["sim_y0"]
        .clip(0, sim_h)
        .astype(np.int64)
    )

    sim_metadata_df["sim_y1"] = (
        sim_metadata_df["sim_y1"]
        .clip(0, sim_h)
        .astype(np.int64)
    )

    sim_metadata_df["n_delta_steps"] = (
        sim_metadata_df["n_delta_steps"].astype(int)
    )

    # A readable numeric label that does not require importing GCRsim.
    sim_metadata_df["parent_label"] = (
        "S"
        + sim_metadata_df["species_index"].astype(str)
        + "-P"
        + sim_metadata_df["primary_index"].map(
            lambda value: f"{value:04d}"
        )
    )

    n_step_rows = len(sim_step_metadata_df)
    n_particle_pids = sim_step_metadata_df["PID"].nunique()
    n_parent_events = len(sim_metadata_df)
    n_delta_pids = int(
        sim_step_metadata_df.loc[
            sim_step_metadata_df["is_delta_ray"],
            "PID",
        ].nunique()
    )

    print(f"Loaded simulated image with shape {sim_data.shape}.")
    print(f"Loaded {n_step_rows} simulation metadata rows.")
    print(f"Found {n_particle_pids} distinct particle PIDs.")
    print(f"Found {n_parent_events} distinct parent simulated events.")
    print(f"Found {n_delta_pids} distinct delta-ray PIDs.")

    return sim_data, sim_metadata_df


def extract_sim_data(sim_data, threshold=1e-6, min_pixels=1, structure=None, return_metadata=False):
    """
    Find connected simulated events in a noiseless sim image and return them
    as cutouts with local masks.

    Returns
    -------
    cutouts : list of dict
        Each entry contains:
            image   : float32 cutout with original DN values
            mask    : bool cutout mask
            bbox    : (y0, y1, x0, x1) in the sim frame
            n_pix   : number of mask pixels
            peak_dn : peak DN in cutout
            sum_dn  : total DN in cutout
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)

    sim_pos = np.asarray(sim_data, dtype=np.float32)
    event_mask = sim_pos > threshold

    lab, nlab = label(event_mask, structure=structure)

    metadata = {
        "sim_shape": tuple(sim_data.shape),
        "threshold": float(threshold),
        "min_pixels": int(min_pixels),
        "n_connected_components_raw": int(nlab),
        "n_pixels_above_threshold": int(np.count_nonzero(event_mask)),
    }

    if nlab == 0:
        metadata["n_cutouts_kept"] = 0
        if return_metadata:
            return [], metadata
        return []

    objs = find_objects(lab)
    cutouts = []

    for lab_id, slc in enumerate(objs, start=1):
        if slc is None:
            continue

        sub_lab = lab[slc]
        sub_mask = (sub_lab == lab_id)

        n_pix = int(sub_mask.sum())
        if n_pix < min_pixels:
            continue

        sub_img = sim_pos[slc].copy()
        sub_img[~sub_mask] = 0.0

        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop

        cutouts.append({
            "image": sub_img.astype(np.float32, copy=False),
            "mask": sub_mask,
            "bbox": (y0, y1, x0, x1),
            "n_pix": n_pix,
            "peak_dn": float(sub_img[sub_mask].max()),
            "sum_dn": float(sub_img[sub_mask].sum()),
        })

    metadata["n_cutouts_kept"] = int(len(cutouts))
    metadata["n_cutouts_rejected_min_pixels"] = (
        metadata["n_connected_components_raw"] - metadata["n_cutouts_kept"]
    )

    if return_metadata:
        return cutouts, metadata
    
    return cutouts


def choose_injection_location(frame_shape, cutout_shape, rng, border=0):
    """
    Choose a random valid upper-left insertion location for a cutout.
    """
    h, w = frame_shape
    ch, cw = cutout_shape

    if ch + 2 * border > h or cw + 2 * border > w:
        raise ValueError(
            f"Cutout shape {cutout_shape} does not fit into frame shape {frame_shape}"
        )

    y0 = rng.integers(border, h - ch - border + 1)
    x0 = rng.integers(border, w - cw - border + 1)
    return int(y0), int(x0)


def inject_sim_data(
    data_cube,
    sim_cutouts,
    n_injections=10,
    rng=None,
    frame_indices=None,
    allow_reuse=False,
    border=32,
):
    """
    Add selected simulated event cutouts into random frames/locations.

    Parameters
    ----------
    data_cube : (Nframe, h, w) ndarray
        Real data cube to modify.
    sim_cutouts : list of dict
        Output of extract_sim_data.
    n_injections : int
        Number of cutouts to inject in total.
    rng : np.random.Generator or None
    frame_indices : sequence[int] or None
        Restrict injections to this subset of frames.
    allow_reuse : bool
        If True, the same simulated event can be used more than once.
    border : int
        Keep injections this far from the image edge.

    Returns
    -------
    injected_cube : ndarray
        Copy of input cube with simulated events added.
    truth_df : pd.DataFrame
        Ground-truth table for injected events.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(sim_cutouts) == 0:
        raise ValueError("No simulated cutouts found to inject.")

    injected_cube = data_cube.astype(np.float32, copy=True)

    nframe, h, w = injected_cube.shape

    if frame_indices is None:
        frame_indices = np.arange(nframe, dtype=int)
    else:
        frame_indices = np.asarray(frame_indices, dtype=int)

    if len(frame_indices) == 0:
        raise ValueError("frame_indices is empty.")

    if allow_reuse:
        chosen_cutout_ids = rng.integers(0, len(sim_cutouts), size=n_injections)
    else:
        if n_injections > len(sim_cutouts):
            raise ValueError(
                "n_injections exceeds number of available sim cutouts when allow_reuse=False."
            )
        chosen_cutout_ids = rng.choice(
            len(sim_cutouts), size=n_injections, replace=False
        )

    truth_rows = []

    for inj_id, cutout_id in enumerate(chosen_cutout_ids):
        cut = sim_cutouts[int(cutout_id)]
        cut_img = cut["image"]
        ch, cw = cut_img.shape

        frame = int(rng.choice(frame_indices))
        y0, x0 = choose_injection_location((h, w), (ch, cw), rng=rng, border=border)
        y1 = y0 + ch
        x1 = x0 + cw

        injected_cube[frame, y0:y1, x0:x1] += cut_img

        mask = cut["mask"]
        yy, xx = np.where(mask)
        peak_local_idx = np.argmax(cut_img[mask])
        peak_y_local = int(yy[peak_local_idx])
        peak_x_local = int(xx[peak_local_idx])

        src_y0, src_y1, src_x0, src_x1 = cut["bbox"]

        truth_rows.append({
            "injection_id": int(inj_id),
            "source_cutout_id": int(cutout_id),
            "is_sim": True,

            # Parent simulation identity
            "parent_PID": int(cut.get("parent_PID", -1)),
            "species_index": int(cut.get("species_index", -1)),
            "primary_index": int(cut.get("primary_index", -1)),

            # Quality of the source cutout-to-metadata mapping
            "metadata_match_status": cut.get(
                "metadata_match_status",
                "unknown",
            ),

            "metadata_exact_bbox_match": bool(
                cut.get(
                    "metadata_exact_bbox_match",
                    False,
                )
            ),

            "metadata_bbox_overlap_pixels": int(
                cut.get(
                    "metadata_bbox_overlap_pixels",
                    0,
                )
            ),

            "metadata_bbox_iou": float(
                cut.get(
                    "metadata_bbox_iou",
                    0.0,
                )
            ),

            "metadata_cutout_overlap_fraction": float(
                cut.get(
                    "metadata_cutout_overlap_fraction",
                    0.0,
                )
            ),

            "metadata_center_distance_pix": float(
                cut.get(
                    "metadata_center_distance_pix",
                    np.nan,
                )
            ),

            "n_parent_candidates": int(
                cut.get(
                    "n_parent_candidates",
                    0,
                )
            ),

            # Injected position in the real FITS frame
            "frame": frame,
            "y0": int(y0),
            "y1": int(y1),
            "x0": int(x0),
            "x1": int(x1),
            "peak_y": int(y0 + peak_y_local),
            "peak_x": int(x0 + peak_x_local),

            # Original source position in sim_data
            "source_y0": int(src_y0),
            "source_y1": int(src_y1),
            "source_x0": int(src_x0),
            "source_x1": int(src_x1),

            "n_pix_sim": int(cut["n_pix"]),
            "peak_dn_sim": float(cut["peak_dn"]),
            "sum_dn_sim": float(cut["sum_dn"]),
        })


    truth_df = pd.DataFrame(truth_rows)
    print(f"Simulated objects inject into frames:{sorted(truth_df['frame'].unique())}")
    return injected_cube, truth_df


def add_is_sim_flag(detections_df, sim_truth_df, padding=2):
    """
    Flag detected peaks that fall inside an injected simulated-event footprint.

    A detection is considered simulated when:
      1. It occurs in the same frame as an injected event.
      2. Its (y, x) peak coordinate falls inside the injected event's
         bounding box, expanded by `padding` pixels.

    Parameters
    ----------
    detections_df : pd.DataFrame
        Detection table containing frame, y, and x columns.
    sim_truth_df : pd.DataFrame or None
        Injection truth table returned by inject_sim_data().
    padding : int
        Number of pixels by which to expand each injection bounding box.

    Returns
    -------
    flagged_df : pd.DataFrame
        Copy of detections_df with a boolean is_sim column.
    """
    flagged_df = detections_df.copy()
    flagged_df["is_sim"] = False
    flagged_df["sim_PID"] = np.int64(-1)
    flagged_df["sim_injection_id"] = np.int64(-1)

    if (
        sim_truth_df is None
        or len(sim_truth_df) == 0
        or len(flagged_df) == 0
    ):
        print(
            "No sim truth dataframe provided; injected sim events "
            "will not be auto-flagged."
        )
        return flagged_df

    required_truth_columns = {
        "injection_id",
        "parent_PID",
        "frame",
        "y0",
        "y1",
        "x0",
        "x1",
    }

    missing = required_truth_columns - set(sim_truth_df.columns)

    if missing:
        raise ValueError(
            "sim_truth_df is missing required columns: "
            f"{sorted(missing)}"
        )

    recovered_injection_ids = set()

    for sim_row in sim_truth_df.itertuples(index=False):
        inside_sim_event = (
            (flagged_df["frame"] == sim_row.frame)
            & (flagged_df["y"] >= sim_row.y0 - padding)
            & (flagged_df["y"] < sim_row.y1 + padding)
            & (flagged_df["x"] >= sim_row.x0 - padding)
            & (flagged_df["x"] < sim_row.x1 + padding)
        )

        if inside_sim_event.any():
            recovered_injection_ids.add(int(sim_row.injection_id))

            flagged_df.loc[inside_sim_event, "is_sim"] = True

            flagged_df.loc[
                inside_sim_event,
                "sim_PID",
            ] = int(sim_row.parent_PID)

            flagged_df.loc[
                inside_sim_event,
                "sim_injection_id",
            ] = int(sim_row.injection_id)

    flagged_df["sim_PID"] = flagged_df["sim_PID"].astype(np.int64)
    flagged_df["sim_injection_id"] = (
        flagged_df["sim_injection_id"].astype(np.int64)
    )

    print(
        f"Flagged detections associated with "
        f"{len(recovered_injection_ids)}/{len(sim_truth_df)} "
        "injected simulated events."
    )

    return flagged_df


def add_derived_signal_columns(df):
    """
    Add reconstructed 3x3 and/or 5x5 background-subtracted signal columns when
    the required peak and r3/r5 columns are available.

    The preclassifier defines:

        r5 = (sum5x5 - peak) / peak

    so:

        sum5x5 = peak * (1 + r5), same for r3 and sum3x3
    """
    out = df.copy()

    # Standard preclassification dataframe
    if ("sum3x3_bgsub_DN" not in out.columns
        and {"peak_val", "r3"}.issubset(out.columns)):
        out["sum3x3_bgsub_DN"] = (
            pd.to_numeric(out["peak_val"], errors="coerce")
            * (1.0 + pd.to_numeric(out["r3"], errors="coerce"))
        )

    if ("sum5x5_bgsub_DN" not in out.columns
        and {"peak_val", "r5"}.issubset(out.columns)):
        out["sum5x5_bgsub_DN"] = (
            pd.to_numeric(out["peak_val"], errors="coerce")
            * (1.0 + pd.to_numeric(out["r5"], errors="coerce"))
        )

    return out

def count_recovered_injections(events, sim_truth_df, padding=2):
    """
    Count distinct injected events that contain at least one detected peak.
    """
    events = np.asarray(events)

    recovered_ids = []
    match_counts = {}

    for sim_row in sim_truth_df.itertuples(index=False):
        same_frame = events[:, 0] == sim_row.frame

        inside_box = (
            same_frame
            & (events[:, 1] >= sim_row.y0 - padding)
            & (events[:, 1] < sim_row.y1 + padding)
            & (events[:, 2] >= sim_row.x0 - padding)
            & (events[:, 2] < sim_row.x1 + padding)
        )

        n_matches = int(np.count_nonzero(inside_box))
        match_counts[int(sim_row.injection_id)] = n_matches

        if n_matches > 0:
            recovered_ids.append(int(sim_row.injection_id))

    return recovered_ids, match_counts


def map_parent_pids_to_sim_cutouts(
    sim_cutouts,
    sim_metadata_df,
    bbox_padding=1,
):
    """
    Match connected-component cutouts to event-level simulation metadata.

    The metadata must contain one row per parent event with pixel-space
    bounding boxes:

        sim_y0, sim_y1, sim_x0, sim_x1

    Matching priority:
      1. Exact bounding-box match.
      2. Highest bounding-box intersection-over-union.
      3. Shortest center-to-center distance as a tie breaker.

    Parameters
    ----------
    sim_cutouts : list of dict
        Output from extract_sim_data(). Each cutout must contain bbox.

    sim_metadata_df : pd.DataFrame
        One row per parent simulated event.

    bbox_padding : int, default=1
        Number of pixels by which to expand metadata bounds when testing
        overlap. This accommodates small differences caused by thresholding
        or rounding.

    Returns
    -------
    mapped_cutouts : list of dict
        Copies of sim_cutouts with parent PID metadata attached.

    cutout_pid_map_df : pd.DataFrame
        Summary of the cutout-to-parent mapping.
    """
    required_columns = {
        "parent_PID",
        "species_index",
        "primary_index",
        "sim_y0",
        "sim_y1",
        "sim_x0",
        "sim_x1",
    }

    missing = required_columns - set(sim_metadata_df.columns)

    if missing:
        raise ValueError(
            "Event-level sim_metadata_df is missing columns: "
            f"{sorted(missing)}"
        )

    metadata = sim_metadata_df.copy()

    integer_columns = [
        "parent_PID",
        "species_index",
        "primary_index",
        "sim_y0",
        "sim_y1",
        "sim_x0",
        "sim_x1",
    ]

    for column in integer_columns:
        metadata[column] = pd.to_numeric(
            metadata[column],
            errors="raise",
        ).astype(np.int64)

    mapped_cutouts = []
    mapping_rows = []

    for cutout_id, original_cutout in enumerate(sim_cutouts):
        cutout = original_cutout.copy()

        cut_y0, cut_y1, cut_x0, cut_x1 = cutout["bbox"]

        cut_height = cut_y1 - cut_y0
        cut_width = cut_x1 - cut_x0
        cut_area = max(cut_height * cut_width, 1)

        scores = metadata.copy()

        # Check for exact equality before applying padding.
        scores["exact_bbox_match"] = (
            (scores["sim_y0"] == cut_y0)
            & (scores["sim_y1"] == cut_y1)
            & (scores["sim_x0"] == cut_x0)
            & (scores["sim_x1"] == cut_x1)
        )

        # Expand the event-level metadata bounds slightly.
        event_y0 = scores["sim_y0"] - bbox_padding
        event_y1 = scores["sim_y1"] + bbox_padding
        event_x0 = scores["sim_x0"] - bbox_padding
        event_x1 = scores["sim_x1"] + bbox_padding

        intersection_height = np.maximum(
            0,
            np.minimum(cut_y1, event_y1)
            - np.maximum(cut_y0, event_y0),
        )

        intersection_width = np.maximum(
            0,
            np.minimum(cut_x1, event_x1)
            - np.maximum(cut_x0, event_x0),
        )

        scores["intersection_area"] = (
            intersection_height * intersection_width
        )

        event_area = np.maximum(
            (event_y1 - event_y0) * (event_x1 - event_x0),
            1,
        )

        union_area = (
            cut_area
            + event_area
            - scores["intersection_area"]
        )

        scores["bbox_iou"] = (
            scores["intersection_area"]
            / np.maximum(union_area, 1)
        )

        scores["cutout_overlap_fraction"] = (
            scores["intersection_area"] / cut_area
        )

        # Center-distance tie breaker.
        cut_center_y = 0.5 * (cut_y0 + cut_y1)
        cut_center_x = 0.5 * (cut_x0 + cut_x1)

        event_center_y = 0.5 * (
            scores["sim_y0"] + scores["sim_y1"]
        )

        event_center_x = 0.5 * (
            scores["sim_x0"] + scores["sim_x1"]
        )

        scores["center_distance_pix"] = np.sqrt(
            (event_center_y - cut_center_y) ** 2
            + (event_center_x - cut_center_x) ** 2
        )

        candidates = scores.loc[
            scores["intersection_area"] > 0
        ].copy()

        if len(candidates) == 0:
            parent_PID = -1
            species_index = -1
            primary_index = -1

            match_status = "unmatched"
            exact_bbox_match = False
            intersection_area = 0
            bbox_iou = 0.0
            cutout_overlap_fraction = 0.0
            center_distance_pix = np.nan
            n_parent_candidates = 0

        else:
            candidates = candidates.sort_values(
                by=[
                    "exact_bbox_match",
                    "bbox_iou",
                    "cutout_overlap_fraction",
                    "intersection_area",
                    "center_distance_pix",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
            ).reset_index(drop=True)

            best = candidates.iloc[0]

            parent_PID = int(best["parent_PID"])
            species_index = int(best["species_index"])
            primary_index = int(best["primary_index"])

            exact_bbox_match = bool(
                best["exact_bbox_match"]
            )

            intersection_area = int(
                best["intersection_area"]
            )

            bbox_iou = float(best["bbox_iou"])

            cutout_overlap_fraction = float(
                best["cutout_overlap_fraction"]
            )

            center_distance_pix = float(
                best["center_distance_pix"]
            )

            n_parent_candidates = int(len(candidates))

            if exact_bbox_match:
                match_status = "exact_bbox"
            elif n_parent_candidates == 1:
                match_status = "bbox_overlap"
            else:
                match_status = "multiple_bbox_candidates"

        # Attach identity and match diagnostics to the cutout.
        cutout["source_cutout_id"] = int(cutout_id)
        cutout["parent_PID"] = int(parent_PID)
        cutout["species_index"] = int(species_index)
        cutout["primary_index"] = int(primary_index)

        cutout["metadata_match_status"] = match_status
        cutout["metadata_exact_bbox_match"] = exact_bbox_match
        cutout["metadata_bbox_overlap_pixels"] = (
            intersection_area
        )
        cutout["metadata_bbox_iou"] = bbox_iou
        cutout["metadata_cutout_overlap_fraction"] = (
            cutout_overlap_fraction
        )
        cutout["metadata_center_distance_pix"] = (
            center_distance_pix
        )
        cutout["n_parent_candidates"] = (
            n_parent_candidates
        )

        mapped_cutouts.append(cutout)

        mapping_rows.append({
            "source_cutout_id": int(cutout_id),
            "source_y0": int(cut_y0),
            "source_y1": int(cut_y1),
            "source_x0": int(cut_x0),
            "source_x1": int(cut_x1),
            "n_pix_sim": int(cutout["n_pix"]),
            "parent_PID": int(parent_PID),
            "species_index": int(species_index),
            "primary_index": int(primary_index),
            "metadata_match_status": match_status,
            "metadata_exact_bbox_match": exact_bbox_match,
            "metadata_bbox_overlap_pixels": (
                intersection_area
            ),
            "metadata_bbox_iou": bbox_iou,
            "metadata_cutout_overlap_fraction": (
                cutout_overlap_fraction
            ),
            "metadata_center_distance_pix": (
                center_distance_pix
            ),
            "n_parent_candidates": (
                n_parent_candidates
            ),
        })

    cutout_pid_map_df = pd.DataFrame(mapping_rows)

    matched_mask = cutout_pid_map_df["parent_PID"] >= 0
    n_matched = int(matched_mask.sum())

    duplicated_parent_mask = (cutout_pid_map_df.loc[matched_mask, "parent_PID"].duplicated(keep=False) )
    duplicated_parent_ids = (cutout_pid_map_df.loc[matched_mask].loc[duplicated_parent_mask, "parent_PID"].unique() )

    print(f"Mapped {n_matched}/{len(cutout_pid_map_df)} sim cutouts to parent PIDs.")

    if len(duplicated_parent_ids) > 0:
        print(f"Warning: multiple cutouts were assigned to the same parent PID: {duplicated_parent_ids.tolist()}" )

    return mapped_cutouts, cutout_pid_map_df


# MAIN FUNCTION BELOW
def cr_analysis(fits_path, gain_path, params, badpix_mask = None):
    """
    Run the CR-analysis pipeline.
    """

    if params is None:
        params = {}

    default_params = {
        "on_HPC": False,
        "channel_size": 32,
        "supercell_size": 128,
        "sigma_mult": 12,
        "sat_cut": 5.999,
        "sigma_thresh": 5.0,

        # output control
        "save_dataframe": True,
        "output_csv": "cr_event_analysis_results.csv",
        "output_parquet": None,
        "output_xray_csv": "xray_event_analysis_results.csv",
        "output_xray_parquet": None,

        #transient verification mode
        "transient_verification": "previous_frame",

        # optional bypass for event pre-classifier
        "use_preclassification_filter": True,

        # preclassification params
        "keep_ambiguous_events": False,

        # post-processing of candidate events
        "peak_assign_radius": 2,
        "seed_thresh": 32.0,
        "event_neighborhood_radius": 22,

        #gaussian smooth and edge detec
        "gaussian_sigma": 0.7,
        "edge_thresh": 24.0,
        "min_blob_pixels": 2,
        "fill_holes": True,

        # optional simulated event injection
        "add_sim_data": False,
        "sim_data_path": None,
        "sim_metadata_path": None,
        "sim_threshold": 1e-6,
        "sim_min_pixels": 1,
        "n_sim_injections": 10,
        "sim_random_seed": 12345,
        "sim_allow_reuse": False,
        "sim_injection_border": 32,
        "sim_frame_indices": None,
        "save_sim_truth": True,
        "save_sim_diagnostics": True,
        "sim_truth_csv": "cr_event_analysis_sim_truth.csv",
        "sim_processed_md_csv": "cr_event_analysis_processed_sim_md.csv",
        "sim_cutout_pid_map_csv": "cr_event_analysis_sim_cutout_pid_map.csv",
        "sim_recovery_csv": "cr_event_analysis_sim_recovery.csv",
        "sim_diagnostics_json": "cr_event_analysis_sim_diagnostics.json",
        "sim_injection_border": 32,
        "sim_match_padding": 2,
        "sim_frame_indices": None,

        # automatic diagnostic plotting
        "make_plots": True,
        "plot_output_dir": "cr_event_analysis_plots",
        "plot_dpi": 180,
    }

    params = {**default_params, **params}

    on_HPC = params.get("on_HPC", False) or ("SLURM_JOB_ID" in os.environ)
    channel_size = params["channel_size"]
    supercell_size = params["supercell_size"]
    sigma_mult = params["sigma_mult"]
    sat_cut = params["sat_cut"]
    sigma_thresh = params["sigma_thresh"]

    transient_verification = params["transient_verification"]
    use_preclassification_filter = params["use_preclassification_filter"]
    keep_ambiguous_events = params["keep_ambiguous_events"]

    peak_assign_radius = params["peak_assign_radius"]
    seed_thresh = params["seed_thresh"]
    edge_thresh = params["edge_thresh"]
    event_neighborhood_radius = params["event_neighborhood_radius"]
    gaussian_sigma = params["gaussian_sigma"]
    min_blob_pixels = params["min_blob_pixels"]
    fill_holes = params["fill_holes"]

    save_dataframe = params.get("save_dataframe", True)
    output_csv = params.get("output_csv", "cr_event_analysis_results.csv")
    output_parquet = params.get("output_parquet", None)
    output_xray_csv = params.get("output_xray_csv", "xray_event_analysis_results.csv")
    output_xray_parquet = params.get("output_xray_parquet", None)  

    add_sim_data = params.get("add_sim_data", False)
    sim_data_path = params.get("sim_data_path", None)
    sim_metadata_path = params.get("sim_metadata_path", None)
    sim_threshold = params.get("sim_threshold", 1e-6)
    sim_min_pixels = params.get("sim_min_pixels", 1)
    n_sim_injections = params.get("n_sim_injections", 10)
    sim_random_seed = params.get("sim_random_seed", 12345)
    sim_allow_reuse = params.get("sim_allow_reuse", False)
    sim_injection_border = params.get("sim_injection_border", 32)
    sim_frame_indices = params.get("sim_frame_indices", None)
    save_sim_truth = params.get("save_sim_truth", True)
    save_sim_diagnostics = params.get("save_sim_diagnostics", save_sim_truth)
    sim_truth_csv = params.get("sim_truth_csv", "cr_event_analysis_sim_truth.csv")
    sim_processed_md_csv = params.get("sim_processed_md_csv","cr_event_analysis_processed_sim_md.csv")
    sim_cutout_pid_map_csv = params.get("sim_cutout_pid_map_csv","cr_event_analysis_sim_cutout_pid_map.csv")
    sim_recovery_csv = params.get("sim_recovery_csv","cr_event_analysis_sim_recovery.csv")
    sim_diagnostics_json = params.get("sim_diagnostics_json","cr_event_analysis_sim_diagnostics.json")
    sim_injection_border = params.get("sim_injection_border", 32)
    sim_match_padding = params.get("sim_match_padding", 2)
    sim_frame_indices = params.get("sim_frame_indices", None)
    badpix_veto_radius = params.get("badpix_veto_radius", 2)

    make_plots = params.get("make_plots", True)
    plot_output_dir = params.get(
        "plot_output_dir",
        "cr_event_analysis_plots",
    )
    plot_dpi = params.get("plot_dpi", 180)

    #check the time before starting
    start_time = time.perf_counter()

    # load in FITS data cube and gain array, 
    # initialize size of each supercell in the gain array
    data_cube  = load_data(fits_path)
    gain_array = np.loadtxt(gain_path)[:, 5].reshape((channel_size, channel_size))

    #data dimensions
    Nframe, h, w = data_cube.shape
    print("Number of frames in data cube:", Nframe)
    #Load in sim data, if needed
    sim_truth_df = None
    sim_metadata_df = None
    cutout_pid_map_df = None
    sim_recovery_df = None

    matches_before = {}
    matches_after = {}

    recovered_before_transient = []
    recovered_after_transient = []

    if add_sim_data:
        if sim_data_path is None:
            raise ValueError(
                "params['add_sim_data'] is True, but params['sim_data_path'] is None."
            )

        if sim_metadata_path is None:
            raise ValueError(
                "params['add_sim_data'] is True, "
                "but params['sim_metadata_path'] is None."
            )

        print(f"Loading simulated event image from: {sim_data_path} and metadata from: {sim_metadata_path}")

        sim_data, sim_metadata_df = load_sim_data(sim_data_path, sim_metadata_path)

        rng = np.random.default_rng(sim_random_seed)

        sim_cutouts, extraction_info = extract_sim_data(
            sim_data,
            threshold=sim_threshold,
            min_pixels=sim_min_pixels,
            structure=np.ones((3, 3), dtype=bool),
            return_metadata=True,
        )

        print(
            f"Sim array contains {extraction_info['n_connected_components_raw']} "
            f"connected objects above threshold; "
            f"{extraction_info['n_cutouts_kept']} passed min_pixels."
            f"The 'sim_metadata_df' has {len(sim_metadata_df)} total elements."
        )

        if len(sim_cutouts) == 0:
            raise ValueError("No simulated events found above threshold in sim_data.")


        sim_cutouts, cutout_pid_map_df = map_parent_pids_to_sim_cutouts(
            sim_cutouts=sim_cutouts,
            sim_metadata_df=sim_metadata_df,
            bbox_padding=1,
        )

        data_cube, sim_truth_df = inject_sim_data(
            data_cube=data_cube,
            sim_cutouts=sim_cutouts,
            n_injections=n_sim_injections,
            rng=rng,
            frame_indices=sim_frame_indices,
            allow_reuse=sim_allow_reuse,
            border=sim_injection_border,
        )

        if len(sim_truth_df) == extraction_info['n_cutouts_kept']:
            print("Number of found sim events matches supplied metadata.")
        else:
            print("Number of found sim events DOES NOT match supplied metadata. Verify supplied paths are correct.")


        print(f"Injected {len(sim_truth_df)} simulated events into the FITS cube"
              f" from {extraction_info['n_cutouts_kept']} available such events")

        sim_truth_df["n_sim_events_available"] = extraction_info["n_cutouts_kept"]
        sim_truth_df["n_sim_components_raw"] = extraction_info["n_connected_components_raw"]
        sim_truth_df["sim_threshold"] = extraction_info["threshold"]
        sim_truth_df["sim_min_pixels"] = extraction_info["min_pixels"]

    #check the time
    now = time.perf_counter()
    load_time = now - start_time
    if add_sim_data:
        print(f"Time to load the data cube and inject simulated data: {load_time}s")
    else:
        print(f"Time to load the data cube: {load_time}s")
    
    #enter number of available cores
    if on_HPC:
        num_of_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        print(f"Number of cores available for parallelization = {num_of_cores}")
    else:
        num_of_cores = os.cpu_count() + 4
        print(f"Number of cores available for parallelization = {num_of_cores - 4}")

    mask_workers = min(num_of_cores, 16)
    peak_workers = min(num_of_cores, 12)
    merge_workers = min(num_of_cores, 8)
    blob_workers = min(num_of_cores, 10)

    #X-ray energy (in eV), will need this later
    xray_en = 5898.75

    #check the time
    now = time.perf_counter()
    
    #set up a list of tasks we want to run to extract three different types
    # of badpix (hot, very hot, unresponsive) using different params
    if badpix_mask is None:
        tasks = [
        (compute_hot_pixel_mask,   sigma_mult),
        (compute_veryhot_pixel_mask, sigma_mult),
        (compute_unresponsive_mask, sat_cut),
        ]
        
        mask_hot, mask_veryhot, mask_non_res = thread_map(
            lambda fn, param: fn(data_cube, param),
            [fn for fn, _ in tasks],
            [param for _, param in tasks],
            max_workers=mask_workers,
            desc="Computing all masks",
            unit="mask"
        )
        #check the time
        badpix_search_time = time.perf_counter() - now
        total_time = time.perf_counter() - start_time
        print(f"Time to find badpix: {badpix_search_time}s; total time elapsed: {total_time}s")
        now = time.perf_counter()

        # Print the results of each badpix mask
        print("Combining masks into one boolean array…")
        base_mask = mask_hot | mask_veryhot | mask_non_res

        # create a mask for pixels adjacent to a pixel with flagged response: any neighbor of the base_mask
        print("Finding all adjacent pixels…")

        #create small structure (3x3 grid) for binary dilation and future uses
        small_struct =np.ones((3,3), dtype=bool)
        mask_adj  = binary_dilation(base_mask, structure=small_struct) & ~base_mask
        print("Done with adjacent pixel mask")

        print("Combining all masks into final array…")
        badpix_mask = base_mask | mask_adj
        print("badpix_mask ready, shape =", badpix_mask.shape)

        print("Comparing to percentages from Hirata, 2024, Table 2:")
        # fractions in percent
        frac_non_res   = mask_non_res.mean()   * 100  # mask.mean() = mask.sum() / mask.size
        frac_hot   = mask_hot.mean()   * 100  
        frac_veryhot = mask_veryhot.mean()       * 100  
        frac_adj = mask_adj.mean() * 100
        frac_all   = badpix_mask.mean()   * 100  # union 

        print(f"Non-resp pixels: {frac_non_res:.2f}% (vs. 0.53%)")
        print(f"Hot pixels: {frac_hot:.2f}% (vs. 0.20%)")
        print(f"Very hot pixels: {frac_veryhot:.2f}% (vs. 0.11%)")
        print(f"Adjacent pixels: {frac_adj:.2f}% (vs. 2.47%)")
        print(f"Union:       {frac_all:.2f}%  (vs. 3.01%)")
        
        #check the time
        badpix_time = time.perf_counter() - now
        total_time = time.perf_counter() - start_time
        print(f"Time to combine badpix masks and add adjacent pix: {badpix_time}s; total time elapsed: {total_time}s")

        #save badpix_mask for later use
        print("Saving badpix mask")
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        badpix_mask_name =_timestamped_name("cr_event_analysis_badpix_mask.npy",timestamp,on_HPC)
        np.save(badpix_mask_name, badpix_mask)
        print(f"Badpix mask saved as {badpix_mask_name}")
    else:
        small_struct = np.ones((3,3), dtype=bool)
        #check the time
        badpix_time = time.perf_counter() - now
        total_time = time.perf_counter() - start_time
        print("Using provided bad pixel mask (skipping computation)")
        print(f"Time to load badpix mask: {badpix_time}s; total time elapsed: {total_time}s")

    #check to see if simulated events are near/on badpix
    sim_badpix_rows = []

    for sim_row in sim_truth_df.itertuples(index=False):
        frame = int(sim_row.frame)
        y = int(sim_row.peak_y)
        x = int(sim_row.peak_x)

        ylo = max(0, y - badpix_veto_radius)
        yhi = min(h, y + badpix_veto_radius + 1)
        xlo = max(0, x - badpix_veto_radius)
        xhi = min(w, x + badpix_veto_radius + 1)

        peak_on_badpix = bool(badpix_mask[y, x])
        badpix_in_veto_box = bool(
            np.any(badpix_mask[ylo:yhi, xlo:xhi])
        )

        sim_badpix_rows.append({
            "injection_id": int(sim_row.injection_id),
            "frame": frame,
            "peak_y": y,
            "peak_x": x,
            "peak_on_badpix": peak_on_badpix,
            "badpix_in_veto_box": badpix_in_veto_box,
        })

    sim_badpix_df = pd.DataFrame(sim_badpix_rows)

    #check the time
    now = time.perf_counter()

    # Peak finding via median and MAD
    peak_threshold_estimate = sigma_thresh * (6) * 1.4826 #should I adjust the multiplier ?
    print(f"Calculating median of each frame to identify outliers." \
    " \n These outlier peaks will be our event candidates" \
        f"\n Using a threshold sigma of {sigma_thresh} * MAD * 1.4826 ~ {peak_threshold_estimate} DN")

    print(f"Rejecting events if they land on or near a badpix with a veto radius of {badpix_veto_radius}")
    print("Injected peaks directly on bad pixels:",
          int(sim_badpix_df["peak_on_badpix"].sum())
    )
    print(f"Injected peaks rejected by the {badpix_veto_radius}-pixel badpix veto:",
        int(sim_badpix_df["badpix_in_veto_box"].sum())
    )

    find_peaks_worker = partial(
        find_peaks_for_frame,
        data_cube,
        badpix_mask=badpix_mask,
        sigma_thresh=sigma_thresh,
        veto_radius = badpix_veto_radius,
    )

    peak_results = thread_map(
        find_peaks_worker,
        range(Nframe),
        max_workers=peak_workers,
        desc="Finding peaks",
        unit="frame"
    )

    all_events = []
    frame_medians = np.zeros(Nframe, dtype=float)
    frame_thresholds = np.zeros(Nframe, dtype=float)

    for frame_index, result in enumerate(peak_results):
        peaks, median, threshold = result
        frame_thresholds[frame_index] = threshold
        frame_medians[frame_index] = median
        all_events.extend(peaks)
    
    print(f"Frame medians: {frame_medians}")
    print(f"Frame thresholds: {frame_thresholds}")

    if len(all_events) == 0:
        print("No candidate peaks found.")
        empty_cols_streak = [
            "frame", "y", "x", "event_index", "class", "is_sim", "sim_PID", "sim_injection_id",
            "median", "sum3x3_bgsub_DN", "sum3x3_bgsub_e", "sum5x5_bgsub_DN", "sum5x5_bgsub_e",
            "blob_label", "blob_DN", "blob_e", "n_pix_blob",
        ]

        empty_cols_xray = [
            "frame", "y", "x", "event_index", "class", "is_sim", "sim_PID", "sim_injection_id",
            "median", "sum3x3_bgsub_DN", "sum3x3_bgsub_e", "sum5x5_bgsub_DN", "sum5x5_bgsub_e",
            "blob_label", "blob_DN", "blob_e", "n_pix_blob",
        ]
        return pd.DataFrame(columns=empty_cols_streak), pd.DataFrame(columns=empty_cols_xray)

    events = np.array(all_events, dtype=int)
    print(f"Found {len(events)} peaks")
    peak_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to detect candidate peaks: {peak_time:.2f}s; total time elapsed: {total_time:.2f}s")
    now = time.perf_counter()

    if sim_truth_df is not None:
        recovered_before_transient, matches_before = count_recovered_injections(
            events,
            sim_truth_df,
            padding=sim_match_padding,
        )

        print(
            "Injected events represented after peak finding:",
            f"{len(recovered_before_transient)}/{len(sim_truth_df)}",
        )

    # Previous-frame / full-exposure transient verification
    print("Events of interest are transients, so we'll use a transient-only filter")
    print(f"Applying transient verification mode: {transient_verification}")

    single_epoch_events = filter_transient_events(
        events,
        transient_verification=transient_verification,
    )
    print(f"Removed {len(events)-len(single_epoch_events)} events")
    print(f"Single-epoch peaks kept: {len(single_epoch_events)}/{len(events)}, proportion: {(len(single_epoch_events) / len(events)):.2%}")

    #check to see if transient verification removed any sim events
    if sim_truth_df is not None:
        recovered_after_transient, matches_after = count_recovered_injections(
            single_epoch_events,
            sim_truth_df,
            padding=sim_match_padding,
        )

        print("Injected events represented after transient filtering:",
            f"{len(recovered_after_transient)}/{len(sim_truth_df)}")

        lost_by_transient = sorted(set(recovered_before_transient) - set(recovered_after_transient))

        print("Injection IDs lost during transient filtering:", lost_by_transient)

    # generate initial pandas dataframe

    # Build frame -> event indices mapping
    event_idxs = {
        f: np.where(single_epoch_events[:, 0] == f)[0]
        for f in np.unique(single_epoch_events[:, 0])
    }

    # preclassification on raw peaks
    support3_thresh = 0.22 # was 0.18
    support5_thresh = 0.40 # was 0.35
    secondary_peak_rel_thresh = 0.30  #was 0.35
    secondary_peak_abs_thresh = 16.0 # was None

    print("Preclassifying raw peaks...")
    print("Using the following classification parameters:")
    print(f"3x3 support = {support3_thresh}")
    print(f"5x5 support = {support5_thresh}")
    print(f"Secondary peak relative threshold = {secondary_peak_rel_thresh*100}%")
    print(f"Secondary peak absolute threshold = {secondary_peak_abs_thresh} DN")

    pre_rows = []

    if use_preclassification_filter:
        def _preclass_one_frame(f):
            return preclassify_events(
                f=f,
                idxs=event_idxs[f],
                events=single_epoch_events,
                data_cube=data_cube,
                medians=frame_medians,
                support3_thresh=support3_thresh,
                support5_thresh=support5_thresh,
                secondary_peak_rel_thresh=secondary_peak_rel_thresh,
                secondary_peak_abs_thresh=secondary_peak_abs_thresh,
                max_secondary_peaks_for_isolated=0,
            )

        with ThreadPoolExecutor(max_workers=merge_workers) as exe:
            for rows_f in tqdm(
                exe.map(_preclass_one_frame, sorted(event_idxs.keys())),
                total=len(event_idxs),
                desc="Preclassifying frames",
                unit="frame",
            ):
                pre_rows.extend(rows_f)
    else:
        for idx, (f, y, x) in enumerate(single_epoch_events):
            pre_rows.append({
                "event_index": int(idx),
                "frame": int(f),
                "y": int(y),
                "x": int(x),
                "class": "ambiguous",
                "peak_val": np.nan,
                "r3": np.nan,
                "r5": np.nan,
                "n_secondary_in_5x5": np.nan,
                "linearity": np.nan,
                "anisotropy": np.nan,
                "bbox_h_5x5": np.nan,
                "bbox_w_5x5": np.nan,
            })
    
    pre_df = pd.DataFrame(pre_rows)

    if len(pre_df) == 0:
        print("Preclassification produced no rows.")
        return pd.DataFrame(), pd.DataFrame()
    
    #create a "total_signal (background substracted)" column from peak and r5 values
    pre_df = add_derived_signal_columns(pre_df)

    # Assign simulation origin once at the preclassification stage.
    # Downstream dataframes inherit this value from pre_df.
    pre_df = add_is_sim_flag(
        detections_df=pre_df,
        sim_truth_df=sim_truth_df,
        padding=sim_match_padding,
    )

    #check to see what happens to injected sim events
    recovered_in_pre, pre_match_counts = count_recovered_injections(
        pre_df[["frame", "y", "x"]].to_numpy(),
        sim_truth_df,
        padding=sim_match_padding,
    )

    missing_in_pre = sim_truth_df.loc[
        ~sim_truth_df["injection_id"].isin(recovered_in_pre),
        [
            "injection_id",
            "frame",
            "peak_y",
            "peak_x",
            "peak_dn_sim",
            "n_pix_sim",
            "y0",
            "y1",
            "x0",
            "x1",
        ],
    ].copy()

    print(
        f"Distinct injections recovered in pre_df: "
        f"{len(recovered_in_pre)}/{len(sim_truth_df)}"
    )

    print("Missing simulated injections:")
    print(missing_in_pre.to_string(index=False))


    pre_match_df = (
        pd.Series(
            pre_match_counts,
            name="n_pre_matches",
            dtype=np.int64,
        )
        .rename_axis("injection_id")
        .reset_index()
    )

    sim_recovery_df = sim_truth_df.merge(
        pre_match_df,
        on="injection_id",
        how="left",
        validate="one_to_one",
    )

    # Detection counts at each stage
    sim_recovery_df["n_peak_matches_before_transient"] = (
        sim_recovery_df["injection_id"]
        .map(matches_before)
        .fillna(0)
        .astype(int)
    )
    sim_recovery_df["n_peak_matches_after_transient"] = (
        sim_recovery_df["injection_id"]
        .map(matches_after)
        .fillna(0)
        .astype(int)
    )
    sim_recovery_df["n_pre_matches"] = (
        sim_recovery_df["n_pre_matches"]
        .fillna(0)
        .astype(int)
    )

    # Boolean recovery flags
    sim_recovery_df["recovered_after_peak_finding"] = (
        sim_recovery_df[
            "n_peak_matches_before_transient"
        ] > 0
    )
    sim_recovery_df["recovered_after_transient"] = (
        sim_recovery_df[
            "n_peak_matches_after_transient"
        ] > 0
    )
    sim_recovery_df["recovered_in_pre"] = (
        sim_recovery_df["n_pre_matches"] > 0
    )

    # Diagnostics for event loss or duplication
    sim_recovery_df["n_matches_removed_by_transient"] = (
        sim_recovery_df[
            "n_peak_matches_before_transient"
        ]
        - sim_recovery_df[
            "n_peak_matches_after_transient"
        ]
    )
    sim_recovery_df["lost_during_transient_filter"] = (
        sim_recovery_df[
            "recovered_after_peak_finding"
        ]
        & ~sim_recovery_df[
            "recovered_after_transient"
        ]
    )
    sim_recovery_df["multiple_matches_before_transient"] = (
        sim_recovery_df[
            "n_peak_matches_before_transient"
        ] > 1
    )
    sim_recovery_df["multiple_matches_after_transient"] = (
        sim_recovery_df[
            "n_peak_matches_after_transient"
        ] > 1
    )
    sim_recovery_df["multiple_pre_matches"] = (
        sim_recovery_df["n_pre_matches"] > 1
    )

    print("\nPreclassification match-count distribution:")
    print(
        sim_recovery_df["n_pre_matches"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nDistinct injections recovered: "
        f"{sim_recovery_df['recovered_in_pre'].sum()}/"
        f"{len(sim_recovery_df)}"
    )

    print(
        f"Total pre_df rows matched to simulation boxes: "
        f"{sim_recovery_df['n_pre_matches'].sum()}"
    )

    print(
        f"Injections producing multiple detected peaks: "
        f"{sim_recovery_df['multiple_pre_matches'].sum()}"
    )

    print("Preclassification counts:")
    class_counts = pre_df["class"].value_counts(dropna=False)
    print(class_counts.to_dict())
    #save dataframe for debugging and analysis
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    pre_df_name = _timestamped_name("cr_event_analysis_preclassifier.csv",timestamp,on_HPC)
    medians_name =_timestamped_name("cr_event_analysis_preclassifier_medians.npy",timestamp,on_HPC)
    sim_recovery_name = _timestamped_name("cr_event_analysis_sim_recovery.csv", timestamp, on_HPC)
    sim_recovery_df.to_csv(sim_recovery_name, index=False)
    pre_df.to_csv(pre_df_name, index=False)
    np.save(medians_name, frame_medians)
    print(f"Pre-classification dataframe saved as {pre_df_name}, medians numpy array saved as {medians_name}")
    print(f"Saved simulation recovery diagnostics to: {sim_recovery_name}")

    if add_sim_data and sim_truth_df is not None:
        sim_truth_csv_final = _timestamped_name(sim_truth_csv,timestamp,on_HPC)
        sim_processed_md_csv_final = _timestamped_name(sim_processed_md_csv,timestamp,on_HPC)
        sim_cutout_pid_map_csv_final = _timestamped_name(sim_cutout_pid_map_csv,timestamp,on_HPC)
        sim_recovery_csv_final = _timestamped_name(sim_recovery_csv,timestamp,on_HPC)
        sim_diagnostics_json_final = _timestamped_name(sim_diagnostics_json,timestamp,on_HPC)

        if save_sim_truth:
            sim_truth_df.to_csv(sim_truth_csv_final,index=False)
            sim_metadata_df.to_csv(sim_processed_md_csv_final,index=False)

            print("Saved simulated-event truth table to: " f"{sim_truth_csv_final}")
            print("Saved processed simulation metadata to: " f"{sim_processed_md_csv_final}")

        if save_sim_diagnostics:
            if cutout_pid_map_df is not None:
                cutout_pid_map_df.to_csv(sim_cutout_pid_map_csv_final,index=False)
                print("Saved cutout-to-PID mapping diagnostics to: "f"{sim_cutout_pid_map_csv_final}")

            if sim_recovery_df is not None:
                sim_recovery_df.to_csv(sim_recovery_csv_final,index=False)
                print("Saved injection-recovery diagnostics to: "f"{sim_recovery_csv_final}")

            diagnostic_summary = {
            "timestamp_utc": timestamp,
            "on_HPC": bool(on_HPC),
            "slurm_job_id": (
                os.environ.get("SLURM_JOB_ID")
                if on_HPC
                else None
            ),

            "sim_data_path": str(sim_data_path),
            "sim_metadata_path": str(
                sim_metadata_path
            ),

            "sim_random_seed": int(
                sim_random_seed
            ),

            "sim_threshold": float(
                sim_threshold
            ),

            "sim_min_pixels": int(
                sim_min_pixels
            ),

            "sim_match_padding": int(
                sim_match_padding
            ),

            "transient_verification": (
                transient_verification
            ),

            "n_sim_components_raw": int(
                extraction_info[
                    "n_connected_components_raw"
                ]
            ),

            "n_sim_cutouts_kept": int(
                extraction_info[
                    "n_cutouts_kept"
                ]
            ),

            "n_sim_injections": int(
                len(sim_truth_df)
            ),

            "n_cutouts_mapped_to_pid": int(
                (
                    cutout_pid_map_df["parent_PID"] >= 0
                ).sum()
            ),

            "n_recovered_after_peak_finding": int(
                sim_recovery_df[
                    "recovered_after_peak_finding"
                ].sum()
            ),

            "n_recovered_after_transient": int(
                sim_recovery_df[
                    "recovered_after_transient"
                ].sum()
            ),

            "n_recovered_in_pre": int(
                sim_recovery_df[
                    "recovered_in_pre"
                ].sum()
            ),

            "n_lost_during_transient_filter": int(
                sim_recovery_df[
                    "lost_during_transient_filter"
                ].sum()
            ),

            "output_files": {
                "sim_truth": sim_truth_csv_final,
                "sim_metadata": (
                    sim_processed_md_csv_final
                ),
                "cutout_pid_map": (
                    sim_cutout_pid_map_csv_final
                ),
                "sim_recovery": (
                    sim_recovery_csv_final
                ),
            },
        }

        with open(
            sim_diagnostics_json_final,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                diagnostic_summary,
                file,
                indent=2,
            )

        print(
            "Saved simulation diagnostic summary to: "
            f"{sim_diagnostics_json_final}"
        )

    # Early sanity check:
    # if likely_streak is too small, stop early
    # n_total = len(pre_df)
    n_streaks = int(class_counts.get("likely_streak", 0))

    print(f"likely_streak found = {n_streaks}")

    if n_streaks < 5: # change this to be algorithmic later, perhaps based on percent
        raise RuntimeError(
            f"Early exit: only {n_streaks} likely streaks found."
        )


    pre_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time for preclassification: {pre_time:.2f}s; total time elapsed: {total_time:.2f}s")
    now = time.perf_counter()

    # Separate likely x-rays immediately
    df_xrays = pre_df.loc[pre_df["class"] == "likely_xray"].copy()

    if len(df_xrays):
        xray_cols = [
            "frame", "y", "x", "event_index", "class", "is_sim", "sim_PID", "sim_injection_id",
            "peak_val", "r3", "r5", "sum3x3_bgsub_DN", "sum5x5_bgsub_DN", "n_secondary_in_5x5",
        ]
        df_xrays = df_xrays[xray_cols]

    # streak selection

    if keep_ambiguous_events:
        keep_classes = {"likely_streak", "ambiguous"}
    else:
        keep_classes = {"likely_streak"}

    streak_candidate_idx = pre_df.loc[
        pre_df["class"].isin(keep_classes),
        "event_index"
    ].to_numpy(dtype=int)

    streak_candidate_idx = np.unique(streak_candidate_idx)

    print(f"Post-classification streak candidates: {len(streak_candidate_idx)} / {len(single_epoch_events)} raw-peak candidates")


    if len(streak_candidate_idx) == 0:
        print("No post-classification streak candidates found.")
        df_streaks = pd.DataFrame(columns=[
            "frame", "y", "x", "event_index", "class", "is_sim",  "sim_PID", "sim_injection_id",
            "median", "sum3x3_bgsub_DN", "sum3x3_bgsub_e", "sum5x5_bgsub_DN", "sum5x5_bgsub_e",
            "blob_label", "blob_DN", "blob_e", "n_pix_blob",
            "major_extent_geom", "minor_extent_geom",
            "major_extent_pix", "major_extent_um",
            "minor_extent_pix", "minor_extent_um",
            "aspect_ratio_blob", "orientation_deg_blob",
            "gini_blob", "gini_pixel_blob", "gini_longitudinal_blob",
            "longitudinal_peak_fraction", "longitudinal_cv",
            "longitudinal_end_asymmetry", "longitudinal_peak_offset",
            "n_longitudinal_bins","supercell_gain",
            "peak_val", "r3", "r5", "n_secondary_in_5x5",
        ])
        return df_streaks, pre_df

    idxs_by_frame_streak_candidate = {}
    for idx in streak_candidate_idx:
        f = int(single_epoch_events[idx, 0])
        idxs_by_frame_streak_candidate.setdefault(f, []).append(idx)

    idxs_by_frame_streak_candidate = {
        f: np.asarray(v, dtype=int)
        for f, v in idxs_by_frame_streak_candidate.items()
    }


    frame_items = list(idxs_by_frame_streak_candidate.items())
    print("Frames with streak candidates:", len(frame_items))


    # blob analysis only on streak candidates
    print("Analyzing streak candidates blobs...")

    blob_results = thread_map(
        lambda item: analyze_blobs_by_frame(
            f=item[0],
            idxs=item[1],
            events=single_epoch_events,
            data_cube=data_cube,
            medians=frame_medians,
            h=h,
            w=w,
            small_struct=small_struct,
            peak_assign_radius=peak_assign_radius,
            seed_thresh=seed_thresh,
            edge_thresh=edge_thresh,
            event_neighborhood_radius=event_neighborhood_radius,
            gaussian_sigma=gaussian_sigma,
            min_blob_pixels=min_blob_pixels,
            fill_holes=fill_holes,

        ),
        frame_items,
        max_workers=blob_workers,
        desc="Analyzing event blobs",
        unit="frame",
    )

    blob_sums = {}
    blob_counts = {}
    blob_major_extent_pix = {}
    blob_minor_extent_pix = {}
    blob_aspect_ratios = {}
    blob_orientations = {}
    blob_major_extent_geom = {}
    blob_minor_extent_geom = {}
    #blob_ginis = {}


    gini_pixels_blob = {}
    gini_longitudinal_blob = {}
    longitudinal_peak_fraction = {}
    longitudinal_cv = {}
    longitudinal_end_asymmetry = {}
    longitudinal_peak_offset = {}
    n_longitudinal_bins = {}

    hit_blob_label = np.zeros(len(single_epoch_events), dtype=int)

    for out in blob_results:
        f = int(out["frame"])
        idxs = out["idxs"]

        blob_sums[f] = out["sums"]
        blob_counts[f] = out["counts"]
        blob_major_extent_pix[f] = out["major_extent_pix"]
        blob_minor_extent_pix[f] = out["minor_extent_pix"]
        blob_aspect_ratios[f] = out["aspect_ratios"]
        blob_orientations[f] = out["orientations"]
        blob_major_extent_geom[f] = out["major_extent_geom"]
        blob_minor_extent_geom[f] = out["minor_extent_geom"]
        #blob_ginis[f] = out["ginis"]
        gini_pixels_blob[f] = out["gini_pixels"]
        gini_longitudinal_blob[f] = out["gini_longitudinal"]
        longitudinal_peak_fraction[f] = out["longitudinal_peak_fraction"]
        longitudinal_cv[f] = out["longitudinal_cv"]
        longitudinal_end_asymmetry[f] = out["longitudinal_end_asymmetry"]
        longitudinal_peak_offset[f] = out["longitudinal_peak_offset"]
        n_longitudinal_bins[f] = out["n_longitudinal_bins"]

        hit_blob_label[idxs] = out["hit_labels"]

    blob_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to analyze streak candidates blobs: {blob_time:.2f}s; total time elapsed: {total_time:.2f}s")
    now = time.perf_counter()

    # Build final dataframe
    print("Building streak candidates dataframe...")

    final_rows = []

    pre_lookup = pre_df.set_index("event_index")

    for idx in streak_candidate_idx:
        frame, y, x = single_epoch_events[idx].astype(int)
        blob_label = int(hit_blob_label[idx])

        pre_row = pre_lookup.loc[idx]

        row_pre = {
            "event_index": int(idx),
            "class": pre_row["class"],
            # Simulation identity inherited from pre_df
            "is_sim": bool(pre_row["is_sim"]),
            "sim_PID": int(pre_row["sim_PID"]),
            "sim_injection_id": int(pre_row["sim_injection_id"]),

            "peak_val": pre_row["peak_val"],
            "r3": pre_row["r3"],
            "r5": pre_row["r5"],
            "n_secondary_in_5x5": pre_row["n_secondary_in_5x5"],
        }

        if blob_label <= 0:
            row = {
                "frame": int(frame),
                "y": int(y),
                "x": int(x),
                "median": frame_medians[frame],
                "sum3x3_bgsub_DN": np.nan,
                "sum3x3_bgsub_e": np.nan,
                "sum5x5_bgsub_DN": np.nan,
                "sum5x5_bgsub_e": np.nan,
                "blob_label": 0,
                "blob_DN": np.nan,
                "blob_e": np.nan,
                "n_pix_blob": np.nan,
                "major_extent_geom": np.nan,
                "minor_extent_geom": np.nan,
                "major_extent_pix": np.nan,
                "major_extent_um": np.nan,
                "minor_extent_pix": np.nan,
                "minor_extent_um": np.nan,
                "aspect_ratio_blob": np.nan,
                "orientation_deg_blob": np.nan,

                "gini_blob": np.nan,
                "gini_pixel_blob": np.nan,
                "gini_longitudinal_blob": np.nan,
                "longitudinal_peak_fraction": np.nan,
                "longitudinal_cv": np.nan,
                "longitudinal_end_asymmetry": np.nan,
                "longitudinal_peak_offset": np.nan,
                "n_longitudinal_bins": np.nan,

                "supercell_gain": np.nan,
            }
            row.update(row_pre)
            final_rows.append(row)
            continue

        hit = np.array([frame, y, x, blob_label], dtype=int)

        row = process_hit(
            hit=hit,
            data_cube=data_cube,
            medians=frame_medians,
            gain_array=gain_array,
            supercell_size=supercell_size,
            blob_sums=blob_sums,
            blob_counts=blob_counts,
            blob_major_extent_geom=blob_major_extent_geom,
            blob_minor_extent_geom=blob_minor_extent_geom,
            blob_major_extent_pix=blob_major_extent_pix,
            blob_minor_extent_pix=blob_minor_extent_pix,
            blob_aspect_ratios=blob_aspect_ratios,
            blob_orientations=blob_orientations,

            #blob_ginis=blob_ginis,
            gini_pixels_blob=gini_pixels_blob,
            gini_longitudinal_blob=gini_longitudinal_blob,
            longitudinal_peak_fraction=longitudinal_peak_fraction,
            longitudinal_cv=longitudinal_cv,
            longitudinal_end_asymmetry=longitudinal_end_asymmetry,
            longitudinal_peak_offset=longitudinal_peak_offset,
            n_longitudinal_bins=n_longitudinal_bins,
        )

        row.update(row_pre)
        final_rows.append(row)

    df_streaks = pd.DataFrame(final_rows)
    df_streaks = add_derived_signal_columns(df_streaks)

    preferred_cols = [
        "frame", "y", "x", "event_index", "class",
        "is_sim", "sim_PID", "sim_injection_id",
        "median", "sum3x3_bgsub_DN", "sum3x3_bgsub_e",
        "sum5x5_bgsub_DN", "sum5x5_bgsub_e", "blob_label",
        "blob_DN", "blob_e", "n_pix_blob",
        "major_extent_geom", "minor_extent_geom",
        "major_extent_pix", "major_extent_um",
        "minor_extent_pix", "minor_extent_um",
        "aspect_ratio_blob", "orientation_deg_blob",
        "gini_blob", "gini_pixel_blob", "gini_longitudinal_blob",
        "longitudinal_peak_fraction", "longitudinal_cv",
        "longitudinal_end_asymmetry", "longitudinal_peak_offset",
        "n_longitudinal_bins","supercell_gain",
        "peak_val", "r3", "r5", "n_secondary_in_5x5",
    ]

    if len(df_streaks):
        cols = [c for c in preferred_cols if c in df_streaks.columns] + [
            c for c in df_streaks.columns if c not in preferred_cols
        ]
        df_streaks = df_streaks[cols]

    build_df_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to create output dataframes: {build_df_time:.2f}s; total time elapsed: {total_time:.2f}s")
    now = time.perf_counter()

    # Save outputs
    output_csv_final = _timestamped_name(output_csv, timestamp, on_HPC)
    output_xray_csv_final = _timestamped_name(output_xray_csv, timestamp, on_HPC)

    if save_dataframe:
        if len(df_streaks):
            df_streaks.to_csv(output_csv_final, index=False)
            print(f"Saved streak candidates dataframe to: {output_csv_final}")
        else:
            print("No streak candidate rows to save for streak dataframe.")

        if len(df_xrays):
            df_xrays.to_csv(output_xray_csv_final, index=False)
            print(f"Saved x-ray dataframe to: {output_xray_csv_final}")
        else:
            print("No likely_xray rows to save for x-ray dataframe.")

        if output_parquet and len(df_streaks):
            df_streaks.to_parquet(output_parquet, index=False)
            print(f"Saved streak dataframe to Parquet: {output_parquet}")

        if output_xray_parquet and len(df_xrays):
            df_xrays.to_parquet(output_xray_parquet, index=False)
            print(f"Saved x-ray dataframe to Parquet: {output_xray_parquet}")


    if make_plots:
        plot_run_directory = generate_diagnostic_plots(
            pre_df=pre_df,
            df_streaks=df_streaks,
            output_root=plot_output_dir,
            timestamp=timestamp,
            random_seed=sim_random_seed,
            dpi=plot_dpi,
        )

        print(
            f"Automatic diagnostic plots saved under: "
            f"{plot_run_directory}"
        )

    total_time = time.perf_counter() - start_time
    print(f"Total runtime: {total_time:.2f}s")

    if add_sim_data:
        return data_cube, events, single_epoch_events, sim_data, sim_metadata_df, extraction_info, rng, sim_cutouts, sim_truth_df, pre_df, frame_medians, df_xrays, df_streaks
    else:
        return data_cube, events, single_epoch_events, pre_df, frame_medians, df_xrays, df_streaks


#---------end main function------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cr analysis pipeline")
    parser.add_argument("fits_path", help="Path to FITS file")
    parser.add_argument("gain_path", help="Path to gain file")
    parser.add_argument(
        "--params",
        type=str,
        help="Path to JSON file with parameters",
        default=None
    )
    parser.add_argument("--badpix_mask", default=None,
                        help="Path to .npy bad pixel mask")
    
    args = parser.parse_args()

    # Load params dict
    if args.params:
        with open(args.params, "r") as f:
            params = json.load(f)
    else:
        params = {}

    # Welcome msg
    start_time_string = time.strftime("%B %d, %Y at %I:%M:%S %p", time.localtime() )
    print("-... . --. .. -.")
    print(f"Cosmic ray event analysis started on {start_time_string}. Please wait.")

    # Load badpix mask if provided
    badpix_mask = None
    if args.badpix_mask is not None:
        print(f"Loading badpix mask from {args.badpix_mask}")
        badpix_mask = np.load(args.badpix_mask)

    results = cr_analysis(args.fits_path, args.gain_path, params, badpix_mask=badpix_mask)
    # Final msg
    end_time_string = time.strftime("%B %d, %Y at %I:%M:%S %p", time.localtime() )
    print(f"Cosmic ray event analysis completed on {end_time_string}. Thank you for your patience.")
    print(". -. -..")