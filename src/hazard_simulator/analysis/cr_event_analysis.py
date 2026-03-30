# script for analyzing real and simulated cosmic ray events
# Initial creation date: 23-Mar-2026
# Developers: Anthony Harbo Torres
# version 0.11

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from scipy.ndimage import (
    binary_dilation,
    binary_closing,
    maximum_filter,
    label,
    find_objects,
    sum as ndi_sum,
)
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from pathlib import Path



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
    print("⏳ Finding hot pixels…")
    median_img = np.median(data, axis=0)
    mad        = np.median(np.abs(median_img - np.median(median_img)))
    sigma_est  = 1.4826 * mad
    thresh_med = np.median(median_img) + sigma_mult * sigma_est
    mask_med   = median_img > thresh_med
    print(f"✅ Done looking for hot pixels (σ={sigma_est:.3f}, thresh={thresh_med:.1f})")
    return mask_med

def compute_mask_first_frame(data, sigma_mult):
    print("⏳ Finding very hot pixels…")
    first_img  = data[0]
    med_first  = np.median(first_img)
    mad_first  = np.median(np.abs(first_img - med_first))
    sigma_est  = 1.4826 * mad_first
    thresh0    = med_first + sigma_mult * sigma_est
    mask0      = first_img > thresh0
    print(f"✅ Done looking for very hot pixels (σ={sigma_est:.3f}, thresh={thresh0:.1f})")
    return mask0

def compute_mask_no_response(data, sat_cut):
    print("⏳ Finding non-responsive pixels…")
    # If you wanted a row-wise tqdm you could replace the next line with a loop + tqdm
    frame_diff = np.abs(np.diff(data, axis=0))       # (Nframe-1, 4096,4096)
    med_diff   = np.median(frame_diff, axis=0)
    mask_non_res   = med_diff < sat_cut
    print(f"✅ Done looking for non-responsive pixesls (median(med_diff)={np.median(med_diff):.3e})")
    return mask_non_res

def find_peaks_for_frame(data_cube, index, badpix_mask, sigma_thresh):
    image   = data_cube[index]
    _, med, _ = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    mad     = np.median(np.abs(image - med))
    sigma_e = mad * 1.4826
    threshold = med + sigma_thresh * sigma_e

    local_max = maximum_filter(image, size=3)
    cand      = (image == local_max) & (~badpix_mask) & (image > threshold)

    ys, xs = np.where(cand)
    peaks  = [(index, int(y), int(x)) for y, x in zip(ys, xs)]
    return peaks, threshold

def merge_peaks(events, data, proximity_radius=2, max_workers=2):
    """
    Merge spatially adjacent peaks within each frame.

    Parameters
    ----------
    events : (M, 3)-ndarray of int
        List of (frame, y, x) peaks already found.
    data : ndarray, shape (Nframe, h, w)
        Full FITS data cube, needed for intensity weighting.
    proximity_radius : int
        Merge any two peaks whose pixel-centers are within
        `proximity_radius` in Chebyshev distance.

    Returns
    -------
    merged : (K,3)-ndarray of int
        New list of (frame, y, x), one per merged object.
    """

    # Pre-bucket original peaks by frame for O(1) lookup
    events_by_frame = {
        f: events[events[:,0] == f, 1:]
        for f in np.unique(events[:,0])
    }

    # A 3×3 struct for the actual label() call
    small_struct = np.ones((3,3), dtype=bool)
    # The large footprint we use for dilation
    big_struct   = np.ones((2*proximity_radius+1,
                            2*proximity_radius+1), dtype=bool)

    def process_frame(f):
        coords = events_by_frame.get(f)
        if coords is None or len(coords)==0:
            return []

        # build a 1-pixel mask of your raw hits
        mask = np.zeros((data.shape[1], data.shape[2]), bool)
        mask[coords[:,0], coords[:,1]] = True

        # dilate by the big_struct so any hits within r pixels merge
        mask_dil = binary_dilation(mask, structure=big_struct)

        # now label the dilated mask with the 3×3 struct
        labeled, ncomp = label(mask_dil, structure=small_struct)

        # figure out which original coords belong to which label
        labels_at_peaks = labeled[coords[:,0], coords[:,1]]

        merged = []
        for lab in range(1, ncomp+1):
            inds = np.where(labels_at_peaks == lab)[0]
            cluster = coords[inds]   # all (y,x) in this cluster

            if len(cluster)==1:
                y0, x0 = cluster[0]
            else:
                # intensity‐weighted centroid over the ORIGINAL points
                ys = cluster[:,0].astype(float)
                xs = cluster[:,1].astype(float)
                ws = data[f, ys.astype(int), xs.astype(int)].astype(float)
                y0 = int(round(np.average(ys, weights=ws)))
                x0 = int(round(np.average(xs, weights=ws)))

            merged.append((f, y0, x0))

        return merged

    # dispatch in parallel
    merged = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for sub in tqdm(
            exe.map(process_frame, range(data.shape[0])),
            total=data.shape[0],
            desc="Processing frames",
            unit="frame"
        ):
            merged.extend(sub)

    return np.array(merged, dtype=int)

def build_signal_mask(im_corr, blob_signal_thresh=0.0):
    """
    Build a binary mask of signal-bearing pixels from a background-subtracted image.

    Parameters
    ----------
    im_corr : 2D ndarray
        Background-subtracted frame.
    blob_signal_thresh : float
        Threshold for including pixels in the signal mask.

    Returns
    -------
    signal_mask : 2D boolean ndarray
    """
    signal_mask = im_corr > blob_signal_thresh

    return signal_mask

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

def analyze_blobs_by_frame(
    f,
    idxs,
    merged_events,
    data_cube,
    medians,
    h,
    w,
    small_struct,
    blob_signal_thresh=6.5,   # used as grow_thresh now
    peak_assign_radius=2,
    seed_thresh=25.0,
    grow_thresh=None,
    event_neighborhood_radius=9,
    seed_radius=1,
    max_seed_link_dist=9.0,
    bridge_min_frac=0.7,
    elongated_merge_aspect=3.0,
):
    t_frame_start = time.perf_counter()
    # Setup + preprocessing
    t0 = time.perf_counter()
    
    coords = merged_events[idxs, 1:].astype(int)

    # Background-subtracted frame
    im_corr = data_cube[f].astype(np.float32, copy=True)
    im_corr -= np.float32(medians[f])

    if grow_thresh is None:
        grow_thresh = blob_signal_thresh

    # crop to ROI
    y0, y1, x0, x1 = build_event_roi(
        coords, h, w, radius=event_neighborhood_radius, pad=2
    )

    im_roi = im_corr[y0:y1, x0:x1]
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


    lab_img_roi = build_smart_seeded_labels(
        im_corr=im_roi,
        coords=coords_roi,
        neighborhood_mask=event_neighborhood_mask,
        seed_thresh=seed_thresh,
        grow_thresh=grow_thresh,
        small_struct=small_struct,
        seed_radius=seed_radius,
        max_seed_link_dist=max_seed_link_dist,
        bridge_min_frac=bridge_min_frac,
        elongated_merge_aspect=elongated_merge_aspect,
    )

    t_label_end = time.perf_counter()

    n_blobs = int(lab_img_roi.max())

    # Blob metrics (PCA + gini)
    t_metrics_start = time.perf_counter()

    # Blob-level arrays
    sums = np.zeros(n_blobs, dtype=np.float32)
    counts = np.zeros(n_blobs, dtype=int)
    major_extents = np.zeros(n_blobs, dtype=np.float32)
    minor_extents = np.zeros(n_blobs, dtype=np.float32)
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
            major_extents[blob_label - 1] = np.nan
            minor_extents[blob_label - 1] = np.nan
            aspect_ratios[blob_label - 1] = np.nan
            orientations[blob_label - 1] = np.nan
        else:
            metrics = blob_pca_metrics(blob_coords, weights=blob_vals_pos)
            major_extents[blob_label - 1] = metrics["major_extent_pix"]
            minor_extents[blob_label - 1] = metrics["minor_extent_pix"]
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
        "major_extents": major_extents,
        "minor_extents": minor_extents,
        "aspect_ratios": aspect_ratios,
        "orientations": orientations,
        "ginis": ginis,
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
    blob_major_extents,
    blob_minor_extents,
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

    major_blob = float(blob_major_extents[frame][blob_label - 1])
    minor_blob = float(blob_minor_extents[frame][blob_label - 1])
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


def build_smart_seeded_labels(
    im_corr,
    coords,
    neighborhood_mask=None,
    seed_thresh=25.0,
    grow_thresh=6.5,
    small_struct=None,
    seed_radius=1,
    max_seed_link_dist=9.0,
    bridge_min_frac=0.7,
    elongated_merge_aspect=3.0,
):
    """
    Build event labels using a smart seeded-growth strategy.

    Pipeline
    --------
    1. Build a low-threshold grow mask.
    2. Build initial seeds near each merged-event coordinate.
    3. For each connected component of the grow mask:
       - if one seed is present, keep it as one label
       - if multiple seeds are present, decide whether to merge them into
         groups using bridge support / elongation
       - if more than one final seed-group remains, split the component by
         nearest-group assignment

    Returns
    -------
    label_img : 2D int ndarray
    """
    h, w = im_corr.shape
    coords = np.asarray(coords, dtype=int)

    if small_struct is None:
        small_struct = np.ones((3, 3), dtype=bool)

    grow_mask = im_corr > grow_thresh
    if neighborhood_mask is not None:
        grow_mask &= neighborhood_mask

    # Find one local seed pixel per merged-event coord
    # seed_id here is local to this frame's coords array: 0..len(coords)-1
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
            # fallback: use original coord if it is inside the grow mask
            if 0 <= y < h and 0 <= x < w and grow_mask[y, x]:
                seed_pixels[seed_id] = np.array([y, x], dtype=int)

    # Connected components of the grow mask
    cc_img, n_cc = label(grow_mask, structure=small_struct)
    label_img = np.zeros((h, w), dtype=np.int32)

    next_label = 1

    # Assign each seed to its connected component exactly once
    # seed_cc_ids[sid] = cc_id containing that seed, or 0 if none
   
    seed_cc_ids = np.zeros(len(coords), dtype=np.int32)
    for sid, yx in seed_pixels.items():
        y, x = yx
        seed_cc_ids[sid] = cc_img[y, x]

    # Build reverse lookup: cc_id -> list of seed ids in that component
    seeds_by_cc = {}
    for sid, cc_id in enumerate(seed_cc_ids):
        if cc_id <= 0:
            continue
        seeds_by_cc.setdefault(cc_id, []).append(sid)

    # Component bounding boxes
    cc_slices = find_objects(cc_img)

    # Process each component using its slice only
    for cc_id, slc in enumerate(cc_slices, start=1):
        if slc is None:
            continue

        local_seed_ids = seeds_by_cc.get(cc_id, [])
        if len(local_seed_ids) == 0:
            continue

        ysl, xsl = slc
        cc_sub = (cc_img[ysl, xsl] == cc_id)
        grow_sub = grow_mask[ysl, xsl]
        label_sub = label_img[ysl, xsl]

        cc_pts = np.argwhere(cc_sub)
        n_cc_pix = cc_pts.shape[0]

        max_component_pixels = 10000
        max_component_seeds = 18

        # Cheap bailout before expensive grouping
        if n_cc_pix > max_component_pixels:
            label_sub[cc_sub] = next_label
            print(
                f"Max component pixels exceeded on cc_id {cc_id} "
                f"[# pix = {n_cc_pix}], assigning label {next_label}"
            )
            next_label += 1
            continue

        if len(local_seed_ids) > max_component_seeds:
            label_sub[cc_sub] = next_label
            print(
                f"Max component seeds exceeded on cc_id {cc_id} "
                f"[# seeds = {len(local_seed_ids)}], assigning label {next_label}"
            )
            next_label += 1
            continue

        # Convert seed coordinates into the LOCAL coordinate system
        # of this component slice, so they match cc_sub and grow_sub.
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

        # Smart grouping of seeds inside this connected component
        seed_groups = _component_seed_groups(
            cc_mask=cc_sub,
            coords_in_cc=coords_in_cc_local,
            local_seed_ids=local_seed_ids,
            grow_mask=grow_sub,
            max_seed_link_dist=max_seed_link_dist,
            bridge_min_frac=bridge_min_frac,
            elongated_merge_aspect=elongated_merge_aspect,
        )

        if len(seed_groups) == 1:
            label_sub[cc_sub] = next_label
            next_label += 1
            continue

        # Otherwise split the component by nearest seed-group
        group_seed_pts = []
        for group in seed_groups:
            pts = np.array([coords_in_cc_local[sid] for sid in group], dtype=float)
            group_seed_pts.append(pts)

        for yy, xx in cc_pts:
            best_group = None
            best_d2 = np.inf

            p = np.array([yy, xx], dtype=float)

            for gi, pts in enumerate(group_seed_pts):
                d2 = np.min(np.sum((pts - p) ** 2, axis=1))
                if d2 < best_d2:
                    best_d2 = d2
                    best_group = gi

            label_sub[yy, xx] = next_label + best_group

        next_label += len(seed_groups)

    return label_img

def build_event_roi(coords, h, w, radius=12, pad=2):
    y_min = max(0, coords[:, 0].min() - radius - pad)
    y_max = min(h, coords[:, 0].max() + radius + pad + 1)
    x_min = max(0, coords[:, 1].min() - radius - pad)
    x_max = min(w, coords[:, 1].max() + radius + pad + 1)
    return y_min, y_max, x_min, x_max

# HELPER FUNCTIONS
def _prep_frame(data_cube, frame_index):
    img   = data_cube[frame_index].astype(np.float32)
    _, med, _ = sigma_clipped_stats(img, sigma=3.0, maxiters=15)
    return frame_index, med

def _clipped_box_sum(img, y, x, radius):
    h, w = img.shape
    y0, y1 = max(y - radius, 0), min(y + radius + 1, h)
    x0, x1 = max(x - radius, 0), min(x + radius + 1, w)
    return img[y0:y1, x0:x1].sum()

def _compute_frame_median(frame_index, data_cube):
    img = data_cube[frame_index].astype(np.float32)
    _, med, _ = sigma_clipped_stats(img, sigma=3.0, maxiters=15)
    return frame_index, med

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
            "major_extent_pix": 0.0,
            "minor_extent_pix": 0.0,
            "aspect_ratio": 1.0,
            "orientation_deg": 0.0,
        }

    if coords.shape[0] == 1:
        return {
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

    # center-to-center span; +1 makes single-row/column blobs come out in pixel units
    #major_extent = float(proj_major.max() - proj_major.min() + 1.0)
    #minor_extent = float(proj_minor.max() - proj_minor.min() + 1.0)
    #major_sigma_pix = float(np.sqrt(max(evals[0], 0.0)))
    #minor_sigma_pix = float(np.sqrt(max(evals[1], 0.0)))
    #aspect_ratio = major_extent / minor_extent if minor_extent > 0 else 1.0

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


# MAIN FUNCTION BELOW
def cr_analysis(fits_path, gain_path, params):
    if params is None:
        params = {}

    default_params = {
        "on_HPC": False,
        "channel_size": 32,
        "supercell_size": 128,
        "sigma_mult": 12,
        "sat_cut": 5.999,
        "sigma_thresh": 4.51,
        "save_dataframe": True,
        "output_csv": "cr_event_analysis_results.csv",
        "output_parquet": None
    }

    params = {**default_params, **params}

    on_HPC = params.get("on_HPC", False) or ("SLURM_JOB_ID" in os.environ)
    channel_size = params["channel_size"]
    supercell_size = params["supercell_size"]
    sigma_mult = params["sigma_mult"]
    sat_cut = params["sat_cut"]
    sigma_thresh = params["sigma_thresh"]
    

    #check the time before starting
    start_time = time.perf_counter()

    # load in FITS data cube and gain array, 
    # initialize size of each supercell in the gain array
    data_cube  = load_data(fits_path)
    gain_array = np.loadtxt(gain_path)[:, 5].reshape((channel_size, channel_size))

    #data dimensions
    Nframe, h, w = data_cube.shape

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
    median_workers = min(num_of_cores, 8)
    blob_workers = min(num_of_cores, 10)
    hit_workers = min(num_of_cores, 12)

    #X-ray energy (in eV), will need this later
    xray_en = 5898.75

    #check the time
    now = time.perf_counter()
    
    #set up a list of tasks we want to run to extract three different types
    # of badpix (hot, very hot, unresponsive) using different params
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
    print("🔗 Combining masks into one boolean array…")
    base_mask = mask_hot | mask_veryhot | mask_non_res

    # create a mask for pixels adjacent to a pixel with flagged response: any neighbor of the base_mask
    print("⏳ Finding all adjacent pixels…")
    mask_adj  = binary_dilation(base_mask, structure=np.ones((3,3)), border_value=0) & ~base_mask
    print("✅ Done with adjacent pixel mask")

    print("🔗 Combining all masks into final array…")
    maskArray = base_mask | mask_adj
    print("🎉 maskArray ready, shape =", maskArray.shape)

    print("Comparing to percentages from Hirata, 2024, Table 2:")
    # fractions in percent
    frac_non_res   = mask_non_res.mean()   * 100  # mask.mean() = mask.sum() / mask.size
    frac_hot   = mask_hot.mean()   * 100  
    frac_veryhot = mask_veryhot.mean()       * 100  
    frac_adj = mask_adj.mean() * 100
    frac_all   = maskArray.mean()   * 100  # union 

    print(f"Non-resp pixels: {frac_non_res:.2f}% (vs. 0.53%)")
    print(f"Hot pixels: {frac_hot:.2f}% (vs. 0.20%)")
    print(f"Very hot pixels: {frac_veryhot:.2f}% (vs. 0.11%)")
    print(f"Adjacent pixels: {frac_adj:.2f}% (vs. 2.47%)")
    print(f"Union:       {frac_all:.2f}%  (vs. 3.01%)")

    #check the time
    comb_and_comp_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to combine badpix masks and add adjacent pix: {comb_and_comp_time}s; total time elapsed: {total_time}s")
    now = time.perf_counter()

    #find candidate events in the data
    mask_expanded = maximum_filter(maskArray.astype(int), size=5) > 0

    # run in parallel with a tqdm bar
    results = thread_map(
        lambda i: find_peaks_for_frame(data_cube, i, mask_expanded, sigma_thresh), # worker fn
        range(Nframe),       # first iterable
        max_workers=peak_workers, # second iterable
        desc="Finding peaks",     # bar label
        unit="frame"              # units on bar
    )

    #check the time
    candidate_search_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to find candidate events: {candidate_search_time}s; total time elapsed: {total_time}s")
    now = time.perf_counter()

    # unzip them into two lists
    all_frame_peaks, thresholds = zip(*results)

    # previous-frame filtering
    filtered_events = []
    for f, peaks in tqdm(enumerate(all_frame_peaks),
                         total=Nframe,
                         desc='Verifying single-epoch occurrence',
                         unit='frame'):
        prev_f   = (f - 1) % Nframe
        prev_pos = {(y, x) for (_, y, x) in all_frame_peaks[prev_f]}
        for (_, y, x) in peaks:
            if (y, x) not in prev_pos:
                filtered_events.append((f, y, x))

    events = np.array(filtered_events, dtype=int)
    print(f"Found {len(events)} x-ray & cosmic-ray-like peaks "
          f"with ≥{sigma_thresh:.1f} σ cut")
    
    # merge nearby events
    print("Merging candidate events that are spatially related")
    merged_events = merge_peaks(events, data_cube)
    events_difference = len(events) -len(merged_events)

    print(f"{len(events)} → {len(merged_events)} merged events, a difference of {events_difference}")

    #check the time
    verify_and_merge_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to verify and merge events: {verify_and_merge_time}s; total time elapsed: {total_time}s")
    now = time.perf_counter()

    # generate initial pandas dataframe
    print("Computing frame medians to acquire signal background")
    # ---- frame medians ----
    medians = np.zeros(Nframe, dtype=float)

    compute_frame_median_worker = partial(_compute_frame_median, data_cube=data_cube)

    median_results = thread_map(
        compute_frame_median_worker,
        range(Nframe),
        max_workers=median_workers,
        desc="Computing frame medians",
        unit="frame"
    )

    for frame_index, median in median_results:
        medians[frame_index] = median

    print("Summing up event pixels")

    event_idxs = {
        f: np.where(merged_events[:, 0] == f)[0]
        for f in np.unique(merged_events[:, 0])
    }


    small_struct = np.ones((3, 3), dtype=bool)

    blob_sums = {}
    blob_counts = {}
    blob_major_extent_pix = {}
    blob_minor_extent_pix = {}
    blob_aspect_ratios = {}
    blob_orientations = {}
    blob_major_extent_geom = {}
    blob_minor_extent_geom = {}
    blob_ginis = {}
    hit_blob_label = np.zeros(len(merged_events), dtype=int)

    frame_items = list(event_idxs.items())
    blob_signal_thresh =  5.0
    peak_assign_radius =  2


    seed_thresh = params.get("seed_thresh", 20.0)
    grow_thresh = params.get("grow_thresh", blob_signal_thresh)
    event_neighborhood_radius = params.get("event_neighborhood_radius", 12)
    seed_radius = params.get("seed_radius", 1)
    max_seed_link_dist = params.get("max_seed_link_dist", 12.0)
    bridge_min_frac = params.get("bridge_min_frac", 0.7)
    elongated_merge_aspect = params.get("elongated_merge_aspect", 3.0)

    blob_results = thread_map(
        lambda item: analyze_blobs_by_frame(
            f=item[0],
            idxs=item[1],
            merged_events=merged_events,
            data_cube=data_cube,
            medians=medians,
            h=h,
            w=w,
            small_struct=small_struct,
            blob_signal_thresh=blob_signal_thresh,
            peak_assign_radius=peak_assign_radius,
            seed_thresh=seed_thresh,
            grow_thresh=grow_thresh,
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

    for result in blob_results:
        f = result["frame"]
        idxs = result["idxs"]

        blob_sums[f] = result["sums"]
        blob_counts[f] = result["counts"]
        blob_major_extent_geom[f] = result["major_extent_geom"]
        blob_minor_extent_geom[f] = result["minor_extent_geom"]
        blob_major_extent_pix[f] = result["major_extent_pix"]
        blob_minor_extent_pix[f] = result["minor_extent_pix"]
        blob_aspect_ratios[f] = result["aspect_ratios"]
        blob_orientations[f] = result["orientations"]
        blob_ginis[f] = result["ginis"]

        hit_blob_label[idxs] = result["hit_labels"]

    events_aug = np.column_stack([merged_events, hit_blob_label])

    process_hit_worker = partial(
        process_hit,
        data_cube=data_cube,
        medians=medians,
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

    rows = thread_map(
        process_hit_worker,
        events_aug,
        max_workers=hit_workers,
        desc="Processing hits",
        unit="hit"
    )

    print("Preparing results dataframe")
    df = pd.DataFrame(rows)
    print(df.head())

    save_dataframe = params.get("save_dataframe", True)
    output_csv = params.get("output_csv", "cr_event_analysis_results.csv")
    output_parquet = params.get("output_parquet", None)

    #check the time
    pd_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to create initial dataframe: {pd_time}s; total time elapsed: {total_time}s")
    now = time.perf_counter()
    
    # save the dataframe as an output
    # --- Build timestamp ---
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    
    # --- Get base filename from params ---
    base_name = params.get("output_csv", "cr_event_analysis_results.csv")
    name, ext = os.path.splitext(base_name)
    
    # --- Check HPC mode ---
    on_HPC = params.get("on_HPC", False)
    
    if on_HPC:
        job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        output_csv = f"{name}_{timestamp}_job{job_id}{ext}"
    else:
        output_csv = f"{name}_{timestamp}{ext}"
    
    print(f"Output file will be: {output_csv}")
    
    # --- Save dataframe ---
    save_dataframe = params.get("save_dataframe", True)
    
    if save_dataframe:
        df.to_csv(output_csv, index=False)
        print(f"Saved dataframe to: {output_csv}")

        if output_parquet:
            df.to_parquet(output_parquet, index=False)
            print(f"Saved dataframe to Parquet: {output_parquet}")
    
    return df


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

    args = parser.parse_args()

    # Load params dict
    if args.params:
        with open(args.params, "r") as f:
            params = json.load(f)
    else:
        params = {}

    results = cr_analysis(args.fits_path, args.gain_path, params)

    print("Cosmic ray analysis complete.")