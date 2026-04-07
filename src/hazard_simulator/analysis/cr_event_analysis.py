# script for analyzing real and simulated cosmic ray events
# Initial creation date: 23-Mar-2026
# Developers: Anthony Harbo Torres
# version 0.13

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import (
    binary_dilation,
    maximum_filter,
    label,
    find_objects
)
from tqdm.contrib.concurrent import thread_map
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import Delaunay, ConvexHull
from astropy.stats import sigma_clipped_stats
from matplotlib.path import Path
from collections import Counter
from functools import partial
from astropy.io import fits


def load_data(fits_path):   
    with fits.open(fits_path) as hdulist:
        # hdulist is a list of HDU (Header/Data Unit) objects
        primary_hdu = hdulist[0]
        data = primary_hdu.data      # NumPy array of your image/spectrum/whatever
        header = primary_hdu.header  # FITS header metadata

    print(f"Data shape: {data.shape}")
    print("Header keys:", list(header.keys())[:10])
    return(data)

def compute_mask_med_frame(data, sigma_mult):
    print("Finding hot pixels…")
    median_img = np.median(data, axis=0)
    mad        = np.median(np.abs(median_img - np.median(median_img)))
    sigma_est  = 1.4826 * mad
    thresh_med = np.median(median_img) + sigma_mult * sigma_est
    mask_med   = median_img > thresh_med
    print(f"Done looking for hot pixels (σ={sigma_est:.3f}, thresh={thresh_med:.1f})")
    return mask_med

def compute_mask_first_frame(data, sigma_mult):
    print("Finding very hot pixels…")
    first_img  = data[0]
    med_first  = np.median(first_img)
    mad_first  = np.median(np.abs(first_img - med_first))
    sigma_est  = 1.4826 * mad_first
    thresh0    = med_first + sigma_mult * sigma_est
    mask0      = first_img > thresh0
    print(f"Done looking for very hot pixels (σ={sigma_est:.3f}, thresh={thresh0:.1f})")
    return mask0

def compute_mask_no_response(data, sat_cut):
    print("⏳ Finding non-responsive pixels…")
    # If you wanted a row-wise tqdm you could replace the next line with a loop + tqdm
    frame_diff = np.abs(np.diff(data, axis=0))       # (Nframe-1, 4096,4096)
    med_diff   = np.median(frame_diff, axis=0)
    mask_non_res   = med_diff < sat_cut
    print(f"Done looking for non-responsive pixesls (median(med_diff)={np.median(med_diff):.3e})")
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

    # ------------------------------------------------------------
    # METHOD 1: full exposure (global Counter method)
    # ------------------------------------------------------------
    if transient_verification == "full_exposure":

        coord_counts = Counter(map(tuple, events[:, 1:]))

        keep = np.array(
            [coord_counts[(y, x)] == 1 for _, y, x in events],
            dtype=bool,
        )

        return events[keep]

    # ------------------------------------------------------------
    # METHOD 2: strict previous-frame-only check
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # INVALID OPTION
    # ------------------------------------------------------------
    else:
        raise ValueError(
            f"Invalid transient_verification='{transient_verification}'. "
            "Choose from {'previous_frame', 'full_exposure'}."
        )


def find_peaks_for_frame(data_cube, index, badpix_mask, sigma_thresh,
    exclude_badpix_neighbors=False):
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
    peaks  = [(index, int(y), int(x)) for y, x in zip(ys, xs)]
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
      - secondary local-peak count in 5x5

    Important:
      The integral image (summed-area table) is built ONCE per frame,
      not once per event.
    """
    im_corr = data_cube[f].astype(np.float32, copy=True)
    im_corr -= np.float32(medians[f])

    sat = summed_area_table(im_corr)

    rows = []
    for idx in idxs:
        _, y, x = events[idx].astype(int)

        p = float(im_corr[y, x])

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
                "n_secondary_5x5": -1,
            })
            continue

        # 3x3
        y0, y1, x0, x1 = extract_box_bounds(y, x, im_corr.shape, half_size=1)
        s3 = box_sum_from_sat(sat, y0, y1, x0, x1)

        # 5x5
        y0, y1, x0, x1 = extract_box_bounds(y, x, im_corr.shape, half_size=2)
        s5 = box_sum_from_sat(sat, y0, y1, x0, x1)
        # calculate the relative differences
        r3 = (s3 - p) / p
        r5 = (s5 - p) / p

        nsec = count_secondary_local_peaks(
            im_corr,
            y,
            x,
            half_size=2,
            rel_thresh=secondary_peak_rel_thresh,
            abs_thresh=secondary_peak_abs_thresh,
            footprint_size=3,
        )

        # local linearity via PCA on small ROI
        y0, y1, x0, x1 = extract_box_bounds(y, x, im_corr.shape, half_size=2)
        roi = im_corr[y0:y1, x0:x1]

        # 5x5 local ROI already exists here
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
        mask = roi > (0.35 * p)

        coords = np.argwhere(mask)

        bbox_h = 0
        bbox_w = 0

        bbox_area = 0
        elongation_bbox = 1.0
        fill_frac = 1.0

        n_mask = len(coords)
        if n_mask >= 2:
            bbox_h = int(coords[:, 0].max() - coords[:, 0].min() + 1)
            bbox_w = int(coords[:, 1].max() - coords[:, 1].min() + 1)
            bbox_area = bbox_h * bbox_w
            elongation_bbox = max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w))
            fill_frac = n_mask / max(1, bbox_area)

            if bbox_h >= 2 and bbox_w >= 2:
                coords = coords.astype(float)
                center = coords.mean(axis=0)
                X = coords - center
                cov = (X.T @ X) / len(X)
                evals, _ = np.linalg.eigh(cov)
                evals = np.sort(evals)[::-1]

                lam1 = float(evals[0])
                lam2 = float(evals[1])
                denom = lam1 + lam2

                if denom > 1e-6 and lam2 > 1e-6:
                    linearity = lam1 / lam2
                    anisotropy = (lam1 - lam2) / denom
                elif denom > 1e-6:
                    linearity = np.inf
                    anisotropy = 1.0
                else:
                    linearity = 1.0
                    anisotropy = 0.0
            else:
                linearity = 1.0
                anisotropy = 0.0
        else:
            linearity = 1.0
            anisotropy = 0.0


        # -----------------------------
        # bbox / mask morphology categories
        # -----------------------------
        is_tiny_bbox = (
            (n_mask <= 1)
            or (bbox_area <= 1)
        )

        bbox_supports_streak = (
            (elongation_bbox >= 1.5)
            or (bbox_h >= 4)
            or (bbox_w >= 4)
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
            (anisotropy >= 0.80)
            and (linearity >= 5.0)
            and (r5 >= 1.5)
            and (nsec >= 2)
            and (n_support >= 4)
            and (center_cc_size >= 3)
            and bbox_supports_streak
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
            "n_secondary_5x5": int(nsec),
            "linearity": float(linearity),
            "anisotropy": float(anisotropy),
            "bbox_h_5x5": int(bbox_h),
            "bbox_w_5x5": int(bbox_w),
            "n_mask_5x5": int(n_mask),
            "bbox_area_5x5": int(bbox_area),
            "elongation_bbox_5x5": float(elongation_bbox),
            "fill_frac_5x5": float(fill_frac),
        })

    return rows

def cluster_nearby_peaks(coords, max_dist=6.0):
    """
    Cluster peaks that are close and approximately collinear.

    Parameters
    ----------
    coords : (N, 2) int ndarray
        Peak coordinates as (y, x).
    max_dist : float
        Maximum Euclidean distance for connecting two peaks.

    Returns
    -------
    groups : list[list[int]]
        Each group is a list of indices into coords.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)

    if n <= 1:
        return [list(range(n))]

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            dy, dx = coords[j] - coords[i]
            dist = float(np.hypot(dy, dx))
            if dist > max_dist:
                continue

            # with only two points, alignment is implicit
            union(i, j)

    groups_dict = {}
    for i in range(n):
        groups_dict.setdefault(find(i), []).append(i)

    return list(groups_dict.values())

def representative_peak_indices(coords, im_corr, groups):
    """
    Choose one representative peak per cluster: the brightest peak.
    """
    rep_idxs = []
    for grp in groups:
        if len(grp) == 1:
            rep_idxs.append(grp[0])
            continue

        vals = [im_corr[int(coords[i, 0]), int(coords[i, 1])] for i in grp]
        rep_local = grp[int(np.argmax(vals))]
        rep_idxs.append(rep_local)

    return rep_idxs

def build_group_axis_models(
    cc_img,
    seed_groups,
    seed_pixels,
    max_endcap_extra=4.0,
):
    """
    Build a lightweight geometric model for each seed group.

    Returns
    -------
    group_models : list[dict]
        One dict per group with:
          label
          seed_ids
          center
          major_axis
          minor_axis
          proj_min
          proj_max
    """
    group_models = []

    for label_id, group in enumerate(seed_groups, start=1):
        # Gather all seed points in this group
        grp_seed_pts = np.array(
            [seed_pixels[sid] for sid in group if sid in seed_pixels],
            dtype=float,
        )

        if len(grp_seed_pts) == 0:
            continue

        # Try to get a richer point cloud from the connected core component(s)
        cc_ids = []
        for p in grp_seed_pts.astype(int):
            y, x = p
            cc_id = cc_img[y, x]
            if cc_id > 0:
                cc_ids.append(cc_id)

        cc_ids = np.unique(cc_ids)

        if len(cc_ids) > 0:
            grp_core_pts = np.argwhere(np.isin(cc_img, cc_ids))
            pts = grp_core_pts.astype(float)
        else:
            pts = grp_seed_pts

        # Degenerate case: one point only
        if len(pts) == 1:
            center = pts[0]
            major_axis = np.array([0.0, 1.0], dtype=float)  # default x-like direction in (y,x) basis
            minor_axis = np.array([1.0, 0.0], dtype=float)
            proj = np.array([0.0])
        else:
            center = pts.mean(axis=0)
            X = pts - center
            cov = (X.T @ X) / max(len(X), 1)

            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evecs = evecs[:, order]

            major_axis = evecs[:, 0]
            minor_axis = evecs[:, 1]

            proj = X @ major_axis

        proj_min = float(np.min(proj) - max_endcap_extra)
        proj_max = float(np.max(proj) + max_endcap_extra)

        group_models.append({
            "label": label_id,
            "seed_ids": group,
            "center": center,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "proj_min": proj_min,
            "proj_max": proj_max,
            "seed_pts": grp_seed_pts,
        })

    return group_models

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


def recover_labeled_blob_footprints(
    lab_img,
    im_corr,
    recover_thresh=3.0,
    structure=None,
    pad=2,
    max_recover_pixels=20000,
):
    """
    Expand each already-labeled blob into connected lower-threshold pixels.

    This is a second hysteresis-style growth stage:
      - The initial smart-seeded labels define the trusted event cores.
      - Each label is then allowed to grow into nearby pixels with
        im_corr > recover_thresh, but only if those pixels are connected
        to that label through iterative dilation.

    Parameters
    ----------
    lab_img : 2D int ndarray
        Initial label image. 0 = background, 1..N = blob labels.
    im_corr : 2D ndarray
        Background-subtracted image in the same ROI as lab_img.
    recover_thresh : float
        Lower threshold for the footprint-recovery step.
    structure : 2D boolean ndarray or None
        Connectivity structure for binary_dilation. Defaults to 3x3.
    pad : int
        Padding added around each blob slice before recovery.
    max_recover_pixels : int
        Bail out if a blob local region gets too large.

    Returns
    -------
    lab_out : 2D int ndarray
        Label image after per-blob footprint recovery.
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)

    lab_out = np.array(lab_img, copy=True)
    n_labels = int(lab_img.max())

    if n_labels <= 0:
        return lab_out

    blob_slices = find_objects(lab_img)

    for blob_label, slc in enumerate(blob_slices, start=1):
        if slc is None:
            continue

        ysl, xsl = slc

        # Add a little padding so recovery can reach beyond the initial slice
        y0 = max(0, ysl.start - pad)
        y1 = min(lab_img.shape[0], ysl.stop + pad)
        x0 = max(0, xsl.start - pad)
        x1 = min(lab_img.shape[1], xsl.stop + pad)

        lab_sub = lab_out[y0:y1, x0:x1]
        im_sub = im_corr[y0:y1, x0:x1]

        current = (lab_sub == blob_label)
        if not np.any(current):
            continue

        if current.size > max_recover_pixels:
            print(
                f"Recovery skipped for blob {blob_label}: "
                f"local region has {current.size} pixels"
            )
            continue

        # Pixels that are allowed to be added during recovery
        allowed = im_sub > recover_thresh

        # Prevent stealing pixels already assigned to another label
        other_labels = (lab_sub != 0) & (lab_sub != blob_label)
        allowed &= ~other_labels

        # Iteratively grow only into allowed pixels
        while True:
            grown = binary_dilation(current, structure=structure) & allowed
            new_current = current | grown

            if np.array_equal(new_current, current):
                break

            current = new_current

        # Rewrite this local patch:
        # remove old instances of this label, then write recovered footprint
        lab_sub[lab_sub == blob_label] = 0
        lab_sub[current] = blob_label

    return lab_out

def analyze_blobs_by_frame(
    f,
    idxs,
    events,
    data_cube,
    medians,
    h,
    w,
    small_struct,
    blob_signal_thresh=5.0,   
    peak_assign_radius=2,
    seed_thresh=20.0,
    grow_thresh=None,
    recover_thresh=3.0,
    alpha=1.5,
    min_points_for_hull=6,
    max_assign_dist=12.0,
    event_neighborhood_radius=16,
    seed_radius=1,
    max_seed_link_dist=16.0,
    bridge_min_frac=0.7,
    elongated_merge_aspect=3.0,
):
    t_frame_start = time.perf_counter()
    # Setup + preprocessing
    t0 = time.perf_counter()
    
    coords = events[idxs, 1:].astype(int)

    # Background-subtracted frame
    im_corr = data_cube[f].astype(np.float32, copy=True)
    im_corr -= np.float32(medians[f])

    if grow_thresh is None:
        grow_thresh = blob_signal_thresh

    #  cluster nearby survivor peaks into grouped seeds
    peak_groups = cluster_nearby_peaks(coords, max_dist=4.0)
    rep_local_idxs = representative_peak_indices(coords, im_corr, peak_groups)
    coords_grouped = coords[rep_local_idxs]


    # crop to ROI
    y0, y1, x0, x1 = build_event_roi(
        coords_grouped, h, w, radius=event_neighborhood_radius, pad=2
    )

    im_roi = im_corr[y0:y1, x0:x1]
    coords_roi = coords.copy()
    coords_roi[:, 0] -= y0
    coords_roi[:, 1] -= x0
    h_roi, w_roi = im_roi.shape

    # grouped coords for seed building
    coords_grouped_roi = coords_grouped.copy()
    coords_grouped_roi[:, 0] -= y0
    coords_grouped_roi[:, 1] -= x0


    t1 = time.perf_counter()

    # Neighborhood + grow mask
    t_mask_start = time.perf_counter()

    # Restrict analysis to ROI neighborhoods near merged peaks
    event_neighborhood_mask = build_event_neighborhood_mask(
        coords_grouped_roi, h_roi, w_roi, radius=event_neighborhood_radius
    )


    t_mask_end = time.perf_counter()

    # Build smart seeded labels
    t_label_start = time.perf_counter()


    lab_img_roi = build_alpha_shape_labels(
        im_corr=im_roi,
        coords=coords_grouped_roi,
        neighborhood_mask=event_neighborhood_mask,
        seed_thresh=seed_thresh,
        grow_thresh=grow_thresh,
        recover_thresh=recover_thresh,
        small_struct=small_struct,
        seed_radius=seed_radius,
        max_seed_link_dist=max_seed_link_dist,
        bridge_min_frac=bridge_min_frac,
        elongated_merge_aspect=elongated_merge_aspect,
        alpha=alpha,
        min_points_for_hull=min_points_for_hull,
        max_assign_dist=max_assign_dist,
    )

    lab_img_roi = recover_labeled_blob_footprints(
        lab_img=lab_img_roi,
        im_corr=im_roi,
        recover_thresh=recover_thresh,
        structure=small_struct,
        pad=2,
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
    ginis = np.zeros(n_blobs, dtype=np.float32)

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

        ginis[blob_label - 1] = _gini_coefficient(blob_vals)

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
        "ginis": ginis,
        "hit_labels": hit_labels,
    }


def build_candidate_support_mask(
    im_corr,
    neighborhood_mask=None,
    grow_thresh=6.5,
    recover_thresh=3.0,
):
    support = im_corr > recover_thresh
    core = im_corr > grow_thresh

    if neighborhood_mask is not None:
        support &= neighborhood_mask
        core &= neighborhood_mask

    return core, support

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
    blob_ginis,
):
    frame, y, x, blob_label = hit.astype(int)

    img = data_cube[frame].astype(np.int32)
    med = medians[frame]

    sc_row = y // supercell_size
    sc_col = x // supercell_size
    sc_gain = gain_array[sc_row, sc_col]

    sum3 = _clipped_box_sum(img, y, x, radius=1)
    sum5 = _clipped_box_sum(img, y, x, radius=2)

    sum_blob = int(blob_sums[frame][blob_label - 1])
    n_pix_blob = int(blob_counts[frame][blob_label - 1])

    major_blob_geom = float(blob_major_extent_geom[frame][blob_label - 1])
    minor_blob_geom = float(blob_minor_extent_geom[frame][blob_label - 1])    
    major_blob = float(blob_major_extent_pix[frame][blob_label - 1])
    minor_blob = float(blob_minor_extent_pix[frame][blob_label - 1])
    aspect_blob = float(blob_aspect_ratios[frame][blob_label - 1])
    orient_blob = float(blob_orientations[frame][blob_label - 1])

    gini_blob = float(blob_ginis[frame][blob_label - 1])

    return {
        "frame": frame,
        "y": y,
        "x": x,
        "median": med,
        "sum3x3_DN": sum3,
        "sum3x3_e": sum3 * sc_gain,
        "sum5x5_DN": sum5,
        "sum5x5_e": sum5 * sc_gain,
        "blob_label": blob_label,
        "blob_DN": sum_blob,
        "blob_e": sum_blob * sc_gain,
        "n_pix_blob": n_pix_blob,
        "major_extent_geom": major_blob_geom,
        "minor_extent_geom": minor_blob_geom,
        "major_extent_pix": major_blob,
        "major_extent_um": major_blob * 10.0,
        "minor_extent_pix": minor_blob,
        "minor_extent_um": minor_blob * 10.0,
        "aspect_ratio_blob": aspect_blob,
        "orientation_deg_blob": orient_blob,
        "gini_blob": gini_blob,
        "supercell_gain": sc_gain,
    }

def _line_bridge_score(p0, p1, mask):
    """
    Fraction of sampled points along the line from p0 to p1 that fall on True pixels in mask.

    Parameters
    ----------
    p0, p1 : iterable of length 2
        (y, x) endpoints
    mask : 2D boolean ndarray

    Returns
    -------
    score : float in [0, 1]
    """
    y0, x0 = p0
    y1, x1 = p1

    dy = y1 - y0
    dx = x1 - x0
    n = int(max(abs(dy), abs(dx))) + 1
    if n <= 1:
        return 1.0

    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)

    h, w = mask.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)

    return float(np.mean(mask[ys, xs]))


def _component_seed_groups(
    cc_mask,
    coords_in_cc,
    local_seed_ids,
    grow_mask,
    max_seed_link_dist=9.0,
    bridge_min_frac=0.7,
    elongated_merge_aspect=3.0,
):
    """
    Decide how seeds inside one low-threshold connected component should be grouped.

    Strategy
    --------
    1. If only one seed is present -> one group.
    2. If the component is elongated, allow more aggressive seed merging.
    3. Merge two seeds if:
       - they are not too far apart, and
       - the line between them is mostly supported by grow_mask
         OR the component is strongly elongated.

    Returns
    -------
    groups : list[list[int]]
        Each inner list contains seed ids that should become one event group.
    """
    if len(local_seed_ids) <= 1:
        return [list(local_seed_ids)]

    cc_coords = np.argwhere(cc_mask)
    cc_metrics = blob_pca_metrics(cc_coords)
    cc_aspect = cc_metrics["aspect_ratio"]

    seed_pts = np.array([coords_in_cc[sid] for sid in local_seed_ids], dtype=float)
    n = len(local_seed_ids)

    # simple union-find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            p0 = seed_pts[i]
            p1 = seed_pts[j]
            dist = float(np.hypot(*(p1 - p0)))

            if dist > max_seed_link_dist:
                continue

            bridge_score = _line_bridge_score(p0, p1, grow_mask)

            # Merge if strongly bridged, or if the whole component is very elongated
            # and the seeds are not too far apart.
            if (bridge_score >= bridge_min_frac) or (
                cc_aspect >= elongated_merge_aspect and dist <= 0.75 * max_seed_link_dist
            ):
                union(i, j)

    groups_dict = {}
    for i, sid in enumerate(local_seed_ids):
        root = find(i)
        groups_dict.setdefault(root, []).append(sid)

    return list(groups_dict.values())

def find_seed_pixels(
    im_corr,
    coords,
    seed_thresh=25.0,
    seed_radius=1,
    fallback_mask=None,
):
    """
    Find one seed pixel near each merged-event coordinate.

    Parameters
    ----------
    im_corr : 2D ndarray
        Background-subtracted ROI image.
    coords : (N, 2) ndarray
        Event coordinates as (y, x).
    seed_thresh : float
        Threshold for selecting a strong local seed.
    seed_radius : int
        Search radius around each event coordinate.
    fallback_mask : 2D boolean ndarray or None
        If no strong seed is found, allow fallback to the original coord
        if that coord lies inside fallback_mask.

    Returns
    -------
    seed_pixels : dict[int, ndarray]
        seed_pixels[seed_id] = np.array([y, x], dtype=int)
    """
    h, w = im_corr.shape
    coords = np.asarray(coords, dtype=int)

    seed_pixels = {}

    for seed_id, (y, x) in enumerate(coords):
        y0 = max(0, y - seed_radius)
        y1 = min(h, y + seed_radius + 1)
        x0 = max(0, x - seed_radius)
        x1 = min(w, x + seed_radius + 1)

        patch = im_corr[y0:y1, x0:x1]
        seed_patch = patch > seed_thresh

        if np.any(seed_patch):
            locs = np.argwhere(seed_patch)
            vals = patch[seed_patch]
            k = int(np.argmax(vals))
            yy, xx = locs[k]
            seed_pixels[seed_id] = np.array([y0 + yy, x0 + xx], dtype=int)
        else:
            if (
                fallback_mask is not None
                and 0 <= y < h
                and 0 <= x < w
                and fallback_mask[y, x]
            ):
                seed_pixels[seed_id] = np.array([y, x], dtype=int)

    return seed_pixels

def assign_support_pixels_to_seed_groups(
    support_mask,
    group_models,
    max_assign_dist=12.0,
    max_perp_dist=2.5,
    longitudinal_weight=0.15,
    perp_weight=1.0,
):
    """
    Assign support pixels to the nearest plausible seed-group, with a
    directional penalty based on perpendicular distance to the group's
    major axis.

    Parameters
    ----------
    support_mask : 2D boolean ndarray
        Candidate support pixels.
    group_models : list[dict]
        Output of build_group_axis_models(...).
    max_assign_dist : float
        Max nearest-seed distance allowed.
    max_perp_dist : float
        Max perpendicular distance from the group's major axis.
    longitudinal_weight : float
        Small penalty on longitudinal offset from the group center.
    perp_weight : float
        Stronger penalty on perpendicular distance.
    """
    support_pts = np.argwhere(support_mask).astype(float)

    points_by_label = {}
    if support_pts.size == 0 or len(group_models) == 0:
        return points_by_label

    max_d2 = float(max_assign_dist ** 2)

    assigned_lists = {gm["label"]: [] for gm in group_models}

    for p in support_pts:
        best_label = None
        best_score = np.inf

        for gm in group_models:
            seed_pts = gm["seed_pts"]
            center = gm["center"]
            major_axis = gm["major_axis"]
            minor_axis = gm["minor_axis"]

            # 1) keep old nearest-seed distance gate
            d2_seed = np.min(np.sum((seed_pts - p) ** 2, axis=1))
            if d2_seed > max_d2:
                continue

            # 2) directional coordinates relative to group center
            dp = p - center
            longi = float(dp @ major_axis)
            perp = float(abs(dp @ minor_axis))

            # 3) reject points too far off-axis
            if perp > max_perp_dist:
                continue

            # 4) reject points far beyond current ends
            if longi < gm["proj_min"] or longi > gm["proj_max"]:
                continue

            # 5) score: nearest seed + strong perp penalty
            score = (
                d2_seed
                + longitudinal_weight * (longi ** 2)
                + perp_weight * (perp ** 2)
            )

            if score < best_score:
                best_score = score
                best_label = gm["label"]

        if best_label is not None:
            assigned_lists[best_label].append(p.astype(int))

    for lab, lst in assigned_lists.items():
        if len(lst) > 0:
            points_by_label[lab] = np.array(lst, dtype=int)

    return points_by_label

def alpha_shape_mask_from_points(points_xy, shape, alpha=1.5):
    """
    Rasterize an alpha shape from a set of 2D points.

    Parameters
    ----------
    points_xy : (N, 2) ndarray
        Point coordinates in (x, y) order.
    shape : tuple[int, int]
        Output mask shape as (h, w).
    alpha : float
        Alpha-shape parameter. Larger alpha keeps more triangles and
        approaches a convex hull; smaller alpha is more concave.

    Returns
    -------
    mask : 2D boolean ndarray
        Filled alpha-shape mask.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    h, w = shape

    mask = np.zeros((h, w), dtype=bool)

    if points_xy.shape[0] == 0:
        return mask

    if points_xy.shape[0] < 4:
        xs = np.clip(np.rint(points_xy[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.rint(points_xy[:, 1]).astype(int), 0, h - 1)
        mask[ys, xs] = True
        return mask

    def rasterize_polygon(poly_xy, out_mask):
        xmin = max(0, int(np.floor(np.min(poly_xy[:, 0]))))
        xmax = min(w - 1, int(np.ceil(np.max(poly_xy[:, 0]))))
        ymin = max(0, int(np.floor(np.min(poly_xy[:, 1]))))
        ymax = min(h - 1, int(np.ceil(np.max(poly_xy[:, 1]))))

        if xmin > xmax or ymin > ymax:
            return

        xx, yy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        pts = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])

        path = Path(poly_xy)
        inside = path.contains_points(pts, radius=1e-9).reshape(yy.shape)
        out_mask[ymin:ymax + 1, xmin:xmax + 1] |= inside

    try:
        tri = Delaunay(points_xy)
    except Exception:
        # Fallback: convex hull
        hull = ConvexHull(points_xy)
        poly_xy = points_xy[hull.vertices]
        rasterize_polygon(poly_xy, mask)
        return mask

    kept_any = False
    thresh = 1.0 / max(alpha, 1e-12)

    for simplex in tri.simplices:
        pa, pb, pc = points_xy[simplex]

        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)

        s = 0.5 * (a + b + c)
        area2 = s * (s - a) * (s - b) * (s - c)

        if area2 <= 0:
            continue

        area = np.sqrt(area2)
        circum_r = (a * b * c) / (4.0 * area)

        if circum_r < thresh:
            kept_any = True
            poly_xy = np.array([pa, pb, pc])
            rasterize_polygon(poly_xy, mask)

    if not kept_any:
        hull = ConvexHull(points_xy)
        poly_xy = points_xy[hull.vertices]
        rasterize_polygon(poly_xy, mask)

    return mask

def build_alpha_shape_labels(
    im_corr,
    coords,
    neighborhood_mask=None,
    seed_thresh=25.0,
    grow_thresh=6.5,
    recover_thresh=3.0,
    small_struct=None,
    seed_radius=1,
    max_seed_link_dist=9.0,
    bridge_min_frac=0.7,
    elongated_merge_aspect=3.0,
    alpha=1.5,
    min_points_for_hull=6,
    max_assign_dist=16.0,
):
    """
    Build ROI-local event labels using:
      1. thresholded support/core masks,
      2. seed finding,
      3. seed grouping inside connected core components,
      4. nearest-group assignment of support pixels,
      5. alpha-shape rasterization per group.

    Returns
    -------
    label_img : 2D int ndarray
        0 = background, 1..N = blob labels
    """
    h, w = im_corr.shape
    coords = np.asarray(coords, dtype=int)

    if small_struct is None:
        small_struct = np.ones((3, 3), dtype=bool)

    label_img = np.zeros((h, w), dtype=np.int32)

    if coords.size == 0:
        return label_img

    core_mask, support_mask = build_candidate_support_mask(
        im_corr=im_corr,
        neighborhood_mask=neighborhood_mask,
        grow_thresh=grow_thresh,
        recover_thresh=recover_thresh,
    )

    seed_pixels = find_seed_pixels(
        im_corr=im_corr,
        coords=coords,
        seed_thresh=seed_thresh,
        seed_radius=seed_radius,
        fallback_mask=core_mask,
    )

    if len(seed_pixels) == 0:
        return label_img

    # Connected components of the confident core
    cc_img, _ = label(core_mask, structure=small_struct)

    # Assign each seed to a core connected component
    seed_cc_ids = np.zeros(len(coords), dtype=np.int32)
    for sid, yx in seed_pixels.items():
        y, x = yx
        seed_cc_ids[sid] = cc_img[y, x]

    seeds_by_cc = {}
    for sid, cc_id in enumerate(seed_cc_ids):
        if cc_id <= 0:
            continue
        seeds_by_cc.setdefault(cc_id, []).append(sid)

    cc_slices = find_objects(cc_img)

    global_seed_groups = []

    for cc_id, slc in enumerate(cc_slices, start=1):
        if slc is None:
            continue

        local_seed_ids = seeds_by_cc.get(cc_id, [])
        if len(local_seed_ids) == 0:
            continue

        ysl, xsl = slc
        cc_sub = (cc_img[ysl, xsl] == cc_id)
        core_sub = core_mask[ysl, xsl]

        coords_in_cc_local = {
            sid: np.array(
                [
                    seed_pixels[sid][0] - ysl.start,
                    seed_pixels[sid][1] - xsl.start,
                ],
                dtype=int,
            )
            for sid in local_seed_ids
        }

        seed_groups = _component_seed_groups(
            cc_mask=cc_sub,
            coords_in_cc=coords_in_cc_local,
            local_seed_ids=local_seed_ids,
            grow_mask=core_sub,
            max_seed_link_dist=max_seed_link_dist,
            bridge_min_frac=bridge_min_frac,
            elongated_merge_aspect=elongated_merge_aspect,
        )

        global_seed_groups.extend(seed_groups)

    if len(global_seed_groups) == 0:
        # fallback: one group per found seed
        global_seed_groups = [[sid] for sid in seed_pixels.keys()]

    group_models = build_group_axis_models(
        cc_img=cc_img,
        seed_groups=global_seed_groups,
        seed_pixels=seed_pixels,
        max_endcap_extra=4.0,
    )

    points_by_label = assign_support_pixels_to_seed_groups(
        support_mask=support_mask,
        group_models=group_models,
        max_assign_dist=max_assign_dist,
        max_perp_dist=2.0,
        longitudinal_weight=0.05,
        perp_weight=2.0,
    )

    for blob_label, pts in points_by_label.items():
        if len(pts) == 0:
            continue

        if len(pts) < min_points_for_hull:
            label_img[pts[:, 0], pts[:, 1]] = blob_label
            continue

        pts_xy = pts[:, [1, 0]]
        mask = alpha_shape_mask_from_points(
            points_xy=pts_xy,
            shape=(h, w),
            alpha=alpha,
        )
        label_img[mask] = blob_label

    return label_img

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
    index = np.arange(1, n + 1)

    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))


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

def _timestamped_name(base_name, timestamp, on_hpc):
    name, ext = os.path.splitext(base_name)
    if on_hpc:
        job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        return f"{name}_{timestamp}_job{job_id}{ext}"
    return f"{name}_{timestamp}{ext}"


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
        "sigma_thresh": 4.51,

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
        "min_xray_fraction" :0.50,
        "keep_ambiguous_events": False,

        # post-processing of candidate events
        "blob_signal_thresh": 5.0,
        "peak_assign_radius": 2,
        "seed_thresh": 20.0,
        "grow_thresh": None,
        "recover_thresh": 3.0,
        "event_neighborhood_radius": 16,
        "seed_radius": 1,
        "max_seed_link_dist": 16.0,
        "bridge_min_frac": 0.7,
        "elongated_merge_aspect": 3.0,

        # alpha-shape / geometric settings
        "alpha": 1.5,
        "min_points_for_hull": 6,
        "max_assign_dist": 12.0,
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

    min_xray_fraction = params.get("min_xray_fraction", 0.50)
    keep_ambiguous_events = params["keep_ambiguous_events"]

    blob_signal_thresh = params["blob_signal_thresh"]
    peak_assign_radius = params["peak_assign_radius"]
    seed_thresh = params["seed_thresh"]
    grow_thresh = params["grow_thresh"]
    recover_thresh = params["recover_thresh"]
    event_neighborhood_radius = params["event_neighborhood_radius"]
    seed_radius = params["seed_radius"]
    max_seed_link_dist = params["max_seed_link_dist"]
    bridge_min_frac = params["bridge_min_frac"]
    elongated_merge_aspect = params["elongated_merge_aspect"]

    alpha = params["alpha"]
    min_points_for_hull = params["min_points_for_hull"]
    max_assign_dist = params["max_assign_dist"]

    save_dataframe = params.get("save_dataframe", True)
    output_csv = params.get("output_csv", "cr_event_analysis_results.csv")
    output_parquet = params.get("output_parquet", None)
    output_xray_csv = params.get("output_xray_csv", "xray_event_analysis_results.csv")
    output_xray_parquet = params.get("output_xray_parquet", None)  

    #check the time before starting
    start_time = time.perf_counter()

    # load in FITS data cube and gain array, 
    # initialize size of each supercell in the gain array
    data_cube  = load_data(fits_path)
    gain_array = np.loadtxt(gain_path)[:, 5].reshape((channel_size, channel_size))

    #data dimensions
    Nframe, h, w = data_cube.shape


    #new hull search params
    blob_alpha = 1.5
    blob_min_points_for_hull = 6
    blob_max_assign_dist = 12.0

    alpha = params.get("alpha", blob_alpha)
    min_points_for_hull = params.get("min_points_for_hull", blob_min_points_for_hull)
    max_assign_dist = params.get("max_assign_dist", blob_max_assign_dist)


    #check the time
    now = time.perf_counter()
    load_time = now - start_time
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
        (compute_mask_med_frame,   sigma_mult),
        (compute_mask_first_frame, sigma_mult),
        (compute_mask_no_response, sat_cut),
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
        
    #check the time
    now = time.perf_counter()

    # Peak finding via median and MAD
    print("Calculating median of each frame to identify outliers." \
    " \n These outlier peaks will be our event candidates")

    find_peaks_worker = partial(
        find_peaks_for_frame,
        data_cube,
        badpix_mask=badpix_mask,
        sigma_thresh=sigma_thresh,
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
        

    if len(all_events) == 0:
        print("No candidate peaks found.")
        empty_cols_streak = [
            "frame", "y", "x", "event_index", "class",
            "median", "sum3x3_DN", "sum3x3_e", "sum5x5_DN", "sum5x5_e",
            "blob_label", "blob_DN", "blob_e", "n_pix_blob",
        ]

        empty_cols_xray = [
            "frame", "y", "x", "event_index", "class",
            "median", "sum3x3_DN", "sum3x3_e", "sum5x5_DN", "sum5x5_e",
            "blob_label", "blob_DN", "blob_e", "n_pix_blob",
        ]
        return pd.DataFrame(columns=empty_cols_streak), pd.DataFrame(columns=empty_cols_xray)

    events = np.array(all_events, dtype=int)
    print(f"Found {len(events)} peaks")
    peak_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to detect candidate peaks: {peak_time:.2f}s; total time elapsed: {total_time:.2f}s")
    now = time.perf_counter()


    # Previous-frame / full-exposure transient verification
    print("Events of interest are transients, so we'll use a transient-only filter")
    print(f"Applying transient verification mode: {transient_verification}")

    single_epoch_events = filter_transient_events(
        events,
        transient_verification=transient_verification,
    )
    print(f"Removed {len(events)-len(single_epoch_events)} events")
    print(f"Single-epoch peaks kept: {len(single_epoch_events)}/{len(events)}, proportion: {(len(single_epoch_events) / len(events)):.2%}")

    # generate initial pandas dataframe

    # Build frame -> event indices mapping
    event_idxs = {
        f: np.where(single_epoch_events[:, 0] == f)[0]
        for f in np.unique(single_epoch_events[:, 0])
    }

    # preclassification on raw peaks

    print("Preclassifying raw peaks...")
    pre_rows = []

    if use_preclassification_filter:
        def _preclass_one_frame(f):
            return preclassify_events(
                f=f,
                idxs=event_idxs[f],
                events=single_epoch_events,
                data_cube=data_cube,
                medians=frame_medians,
                support3_thresh=0.18,
                support5_thresh=0.35,
                secondary_peak_rel_thresh=0.35,
                secondary_peak_abs_thresh=None,
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
                "n_secondary_5x5": np.nan,
                "linearity": np.nan,
                "anisotropy": np.nan,
                "bbox_h_5x5": np.nan,
                "bbox_w_5x5": np.nan,
            })

    pre_df = pd.DataFrame(pre_rows)

    if len(pre_df) == 0:
        print("Preclassification produced no rows.")
        return pd.DataFrame(), pd.DataFrame()

    print("Preclassification counts:")
    class_counts = pre_df["class"].value_counts(dropna=False)
    print(class_counts.to_dict())
    #save dataframe for debugging and analysis
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    pre_df_name = _timestamped_name("cr_event_analysis_preclassifier.csv",timestamp,on_HPC)
    medians_name =_timestamped_name("cr_event_analysis_preclassifier_medians.npy",timestamp,on_HPC)
    np.save(medians_name, frame_medians)
    pre_df.to_csv(pre_df_name, index=False)
    print(f"Pre-classification dataframe saved as {pre_df_name}, medians numpy array saved as {medians_name}")

    # Early sanity check:
    # if likely_streak is too small, stop early
    # n_total = len(pre_df)
    n_streaks = int(class_counts.get("likely_streak", 0))

    print(f"likely_streak found = {n_streaks}")

    if n_streaks < 10: # change this to be algorithmic later, perhaps based on percent
        print(
            f"Early exit: number of likely streaks is too low")

        return 0


    pre_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time for preclassification: {pre_time:.2f}s; total time elapsed: {total_time:.2f}s")
    now = time.perf_counter()

    # Separate likely x-rays immediately
    df_xrays = pre_df.loc[pre_df["class"] == "likely_xray"].copy()

    if len(df_xrays):
        xray_cols = [
            "frame", "y", "x", "event_index", "class",
            "peak_val", "r3", "r5", "n_secondary_5x5",
        ]
        df_xrays = df_xrays[xray_cols]

    # survivor selection

    if keep_ambiguous_events:
        keep_classes = {"likely_streak", "ambiguous"}
    else:
        keep_classes = {"likely_streak"}

    survivor_idx = pre_df.loc[
        pre_df["class"].isin(keep_classes),
        "event_index"
    ].to_numpy(dtype=int)

    survivor_idx = np.unique(survivor_idx)

    print(f"Post-classification survivors: {len(survivor_idx)} / {len(single_epoch_events)} raw-peak candidates")

    if len(survivor_idx) == 0:
        print("No post-classification survivors found.")
        df_streaks = pd.DataFrame(columns=[
            "frame", "y", "x", "event_index", "class",
            "median", "sum3x3_DN", "sum3x3_e", "sum5x5_DN", "sum5x5_e",
            "blob_label", "blob_DN", "blob_e", "n_pix_blob",
            "major_extent_geom", "minor_extent_geom",
            "major_extent_pix", "major_extent_um",
            "minor_extent_pix", "minor_extent_um",
            "aspect_ratio_blob", "orientation_deg_blob",
            "gini_blob", "supercell_gain",
            "peak_val_pre", "r3_pre", "r5_pre", "n_secondary_5x5_pre",
        ])
        return df_streaks, pre_df

    idxs_by_frame_survivor = {}
    for idx in survivor_idx:
        f = int(single_epoch_events[idx, 0])
        idxs_by_frame_survivor.setdefault(f, []).append(idx)

    idxs_by_frame_survivor = {
        f: np.asarray(v, dtype=int)
        for f, v in idxs_by_frame_survivor.items()
    }


    # blob analysis only on survivors
    print("Analyzing survivor blobs...")
    frame_items = list(idxs_by_frame_survivor.items())

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
            blob_signal_thresh=blob_signal_thresh,
            peak_assign_radius=peak_assign_radius,
            seed_thresh=seed_thresh,
            grow_thresh=grow_thresh,
            recover_thresh=recover_thresh,
            alpha=alpha,
            min_points_for_hull=min_points_for_hull,
            max_assign_dist=max_assign_dist,
            event_neighborhood_radius=event_neighborhood_radius,
            seed_radius=seed_radius,
            max_seed_link_dist=max_seed_link_dist,
            bridge_min_frac=bridge_min_frac,
            elongated_merge_aspect=elongated_merge_aspect,
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
    blob_ginis = {}
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
        blob_ginis[f] = out["ginis"]

        hit_blob_label[idxs] = out["hit_labels"]

    blob_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to analyze survivor blobs: {blob_time:.2f}s; total time elapsed: {total_time:.2f}s")
    now = time.perf_counter()

    # Build final dataframe
    print("Building survivor dataframe...")

    final_rows = []

    pre_lookup = pre_df.set_index("event_index")

    for idx in survivor_idx:
        frame, y, x = single_epoch_events[idx].astype(int)
        blob_label = int(hit_blob_label[idx])

        pre_row = pre_lookup.loc[idx]

        row_pre = {
            "event_index": int(idx),
            "class": pre_row["class"],
            "peak_val_pre": pre_row["peak_val"],
            "r3_pre": pre_row["r3"],
            "r5_pre": pre_row["r5"],
            "n_secondary_5x5_pre": pre_row["n_secondary_5x5"],
        }

        if blob_label <= 0:
            row = {
                "frame": int(frame),
                "y": int(y),
                "x": int(x),
                "median": frame_medians[frame],
                "sum3x3_DN": np.nan,
                "sum3x3_e": np.nan,
                "sum5x5_DN": np.nan,
                "sum5x5_e": np.nan,
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
            blob_ginis=blob_ginis,
        )

        row.update(row_pre)
        final_rows.append(row)

    df_streaks = pd.DataFrame(final_rows)

    preferred_cols = [
        "frame", "y", "x", "event_index", "class",
        "median",
        "sum3x3_DN", "sum3x3_e",
        "sum5x5_DN", "sum5x5_e",
        "blob_label",
        "blob_DN", "blob_e", "n_pix_blob",
        "major_extent_geom", "minor_extent_geom",
        "major_extent_pix", "major_extent_um",
        "minor_extent_pix", "minor_extent_um",
        "aspect_ratio_blob", "orientation_deg_blob",
        "gini_blob", "supercell_gain",
        "peak_val_pre", "r3_pre", "r5_pre", "n_secondary_5x5_pre",
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

    print(f"Streak/survivor output filename: {output_csv_final}")
    print(f"X-ray output filename: {output_xray_csv_final}")

    if save_dataframe:
        if len(df_streaks):
            df_streaks.to_csv(output_csv_final, index=False)
            print(f"Saved streak/survivor dataframe to: {output_csv_final}")
        else:
            print("No survivor rows to save for streak dataframe.")

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

    total_time = time.perf_counter() - start_time
    print(f"Total runtime: {total_time:.2f}s")

    return df_streaks, df_xrays


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

    # Load badpix mask if provided
    badpix_mask = None
    if args.badpix_mask is not None:
        print(f"Loading badpix mask from {args.badpix_mask}")
        badpix_mask = np.load(args.badpix_mask)

    results = cr_analysis(args.fits_path, args.gain_path, params, badpix_mask=badpix_mask)

    print("Cosmic ray analysis complete.")