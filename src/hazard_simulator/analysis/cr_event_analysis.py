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
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import binary_dilation, maximum_filter, label, find_objects, sum as ndi_sum
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
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
        for sub in exe.map(process_frame, range(data.shape[0])):
            merged.extend(sub)

    return np.array(merged, dtype=int)

def analyze_blobs_by_frame(
    f,
    idxs,
    merged_events,
    data_cube,
    medians,
    h,
    w,
    big_struct,
    small_struct,
):
    """
    Analyze all merged-event blobs in one frame.
    Returns everything needed to build blob dictionaries and hit_blob_label.
    """
    coords = merged_events[idxs, 1:].astype(int)

    # Mark peak pixels
    mask_peaks = np.zeros((h, w), dtype=bool)
    mask_peaks[coords[:, 0], coords[:, 1]] = True

    # Merge nearby peaks into blobs
    dil = binary_dilation(mask_peaks, structure=big_struct)
    lab_img, n_blobs = label(dil, structure=small_struct)

    # Background-subtracted working image
    im_corr = data_cube[f].astype(np.float32, copy=True)
    im_corr -= np.float32(medians[f])

    labels_idx = np.arange(1, n_blobs + 1)

    # Blob sums and counts
    sums = ndi_sum(im_corr, lab_img, labels_idx)
    counts = np.bincount(lab_img.ravel(), minlength=n_blobs + 1)[1:]

    # Per-blob morphology
    track_lengths = np.zeros(n_blobs, dtype=np.float32)
    ginis = np.zeros(n_blobs, dtype=np.float32)

    # Bounding boxes for each blob label
    blob_slices = find_objects(lab_img)

    for blob_label, slc in enumerate(blob_slices, start=1):
        if slc is None:
            continue

        lab_sub = lab_img[slc]
        im_sub = im_corr[slc]

        local_mask = (lab_sub == blob_label)

        # local coords inside the box
        local_coords = np.argwhere(local_mask)

        # shift to full-image coords
        local_coords[:, 0] += slc[0].start
        local_coords[:, 1] += slc[1].start

        blob_vals = im_sub[local_mask]

        track_lengths[blob_label - 1] = _blob_track_length(local_coords)
        ginis[blob_label - 1] = _gini_coefficient(blob_vals)

    # Which blob each merged event belongs to
    hit_labels = lab_img[coords[:, 0], coords[:, 1]]

    return {
        "frame": f,
        "idxs": idxs,
        "sums": sums,
        "counts": counts,
        "track_lengths": track_lengths,
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
    blob_track_lengths,
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
    track_length_blob = float(blob_track_lengths[frame][blob_label - 1])
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
        "track_length_pix": track_length_blob,
        "track_length_um": track_length_blob * 10.0,
        "gini_blob": gini_blob,
        "supercell_gain": sc_gain,
    }

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


def _blob_track_length(coords):
    """
    Maximum Euclidean separation between any two blob pixels.
    coords should be an (N, 2) array of (y, x) pixel coordinates.
    Returns length in pixels.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)

    if n <= 1:
        return 0.0

    # Cheap exact method for typical small blobs
    diffs = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    return float(np.sqrt(np.max(d2)))


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

    proximity_radius = 2
    big_struct = np.ones((2 * proximity_radius + 1, 2 * proximity_radius + 1), dtype=bool)
    small_struct = np.ones((3, 3), dtype=bool)

    blob_sums = {}
    blob_counts = {}
    blob_track_lengths = {}
    blob_ginis = {}
    hit_blob_label = np.zeros(len(merged_events), dtype=int)

    frame_items = list(event_idxs.items())

    blob_results = thread_map(
        lambda item: analyze_blobs_by_frame(
            f=item[0],
            idxs=item[1],
            merged_events=merged_events,
            data_cube=data_cube,
            medians=medians,
            h=h,
            w=w,
            big_struct=big_struct,
            small_struct=small_struct,
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
        blob_track_lengths[f] = result["track_lengths"]
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
        blob_track_lengths=blob_track_lengths,
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
    
    if save_dataframe:
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Saved dataframe to CSV: {output_csv}")
    
        if output_parquet:
            df.to_parquet(output_parquet, index=False)
            print(f"Saved dataframe to Parquet: {output_parquet}")

    #check the time
    pd_time = time.perf_counter() - now
    total_time = time.perf_counter() - start_time
    print(f"Time to create initial dataframe: {pd_time}s; total time elapsed: {total_time}s")
    now = time.perf_counter()
    
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