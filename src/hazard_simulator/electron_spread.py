# electron_spread.py

import argparse
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import nbinom
from tqdm import tqdm

from .ffrng import FastForwardRNG


def _rng_from_state(rng_state: dict) -> np.random.Generator:
    bitgen = np.random.PCG64()
    bitgen.state = rng_state
    return np.random.Generator(bitgen)


def _tile_worker_histblur(args):
    (
        ty,
        tx,
        tile_pixels,
        tile_h,
        tile_w,
        n_pixels,
        pixel_size_micron,
        hi_res_grid_spacing_micron,
        sigma_micron,
        N_sigma,
        x_um,
        y_um,
        dE_MeV,
        tile_event_indices,
        rng_state,
    ) = args

    rng = _rng_from_state(rng_state)

    r = int(round(pixel_size_micron / hi_res_grid_spacing_micron))
    sigma_hi = sigma_micron / hi_res_grid_spacing_micron
    pad_hi = int(np.ceil(N_sigma * sigma_hi))
    pad_um = pad_hi * hi_res_grid_spacing_micron

    # Detector tile bounds in microns
    tile_um = tile_pixels * pixel_size_micron
    x0_um = tx * tile_um
    y0_um = ty * tile_um
    x1_um = x0_um + tile_w * pixel_size_micron
    y1_um = y0_um + tile_h * pixel_size_micron

    # Expanded bounds (for blur support)
    ex0_um = x0_um - pad_um
    ey0_um = y0_um - pad_um
    ex1_um = x1_um + pad_um
    ey1_um = y1_um + pad_um

    # Clip expanded bounds to detector physical extent
    det_um = n_pixels * pixel_size_micron
    ex0_um_c = max(0.0, ex0_um)
    ey0_um_c = max(0.0, ey0_um)
    ex1_um_c = min(det_um, ex1_um)
    ey1_um_c = min(det_um, ey1_um)

    if ex1_um_c <= ex0_um_c or ey1_um_c <= ey0_um_c:
        return None

    idx = tile_event_indices
    if idx.size == 0:
        return None

    # Pull candidate events then exact-filter
    xu = x_um[idx]
    yu = y_um[idx]
    dEsub = dE_MeV[idx]

    m = (
        (xu >= ex0_um_c)
        & (xu < ex1_um_c)
        & (yu >= ey0_um_c)
        & (yu < ey1_um_c)
        & np.isfinite(dEsub)
        & (dEsub > 0)
    )
    if not np.any(m):
        return None

    xu = xu[m]
    yu = yu[m]
    dEsub = dEsub[m]

    # Hi-res impulse image size (expanded region)
    w_hi = int(np.ceil((ex1_um_c - ex0_um_c) / hi_res_grid_spacing_micron))
    h_hi = int(np.ceil((ey1_um_c - ey0_um_c) / hi_res_grid_spacing_micron))
    w_hi = max(w_hi, 1)
    h_hi = max(h_hi, 1)

    impulse = np.zeros((h_hi, w_hi), dtype=np.float32)

    # Convert event positions to hi-res indices relative to expanded origin
    x_idx = np.floor((xu - ex0_um_c) / hi_res_grid_spacing_micron).astype(np.int64)
    y_idx = np.floor((yu - ey0_um_c) / hi_res_grid_spacing_micron).astype(np.int64)

    x_idx = np.clip(x_idx, 0, w_hi - 1)
    y_idx = np.clip(y_idx, 0, h_hi - 1)

    # Sample electrons for this tile using this tile's RNG
    n_e_int = electron_conversion_nb_array_gamma_poisson(dEsub, rng=rng, fano_factor=2.71, w_eV=2.509)
    if np.all(n_e_int <= 0):
        return None

    # Histogram electrons into impulse image
    np.add.at(impulse, (y_idx, x_idx), n_e_int.astype(np.float32, copy=False))

    # Gaussian blur
    blurred = gaussian_filter(impulse, sigma=sigma_hi, mode="constant", truncate=float(N_sigma))

    # Crop blurred image down to tile region
    cx0 = int(np.floor((x0_um - ex0_um_c) / hi_res_grid_spacing_micron))
    cy0 = int(np.floor((y0_um - ey0_um_c) / hi_res_grid_spacing_micron))
    cx1 = cx0 + tile_w * r
    cy1 = cy0 + tile_h * r

    cx0c = max(0, cx0)
    cy0c = max(0, cy0)
    cx1c = min(w_hi, cx1)
    cy1c = min(h_hi, cy1)
    if cx1c <= cx0c or cy1c <= cy0c:
        return None

    tile_hi = blurred[cy0c:cy1c, cx0c:cx1c]

    # Pad to multiple of r for reshape
    ph, pw = tile_hi.shape
    pad_h = (-ph) % r
    pad_w = (-pw) % r
    if pad_h or pad_w:
        tile_hi = np.pad(tile_hi, ((0, pad_h), (0, pad_w)), mode="constant")
        ph, pw = tile_hi.shape

    block = tile_hi.reshape(ph // r, r, pw // r, r).sum(axis=(1, 3)).astype(np.float32)

    y0_det = ty * tile_pixels + max(0, (cy0c - cy0) // r)
    x0_det = tx * tile_pixels + max(0, (cx0c - cx0) // r)

    return (y0_det, x0_det, block)


def electron_conversion(dE_MeV, fano_factor=2.71, w_eV=2.509):
    """
    Convert deposited energy [MeV] to the number of electrons via the negative binomial distribution.
    """
    if dE_MeV <= 0:
        return 0
    dE_eV = dE_MeV * 1e6  # MeV -> eV
    mu_nb = dE_eV / w_eV
    p = 1.0 / fano_factor
    if not (0 < p < 1):
        return 0
    r = mu_nb * (p / (1.0 - p))
    if r <= 0:
        return 0
    return nbinom(r, p).rvs()


def electron_conversion_nb_array_gamma_poisson(
    dE_MeV: np.ndarray,
    rng: np.random.Generator,
    fano_factor: float = 2.71,
    w_eV: float = 2.509,
) -> np.ndarray:
    """
    Vectorized equivalent of:
        p = 1/fano_factor
        mu_nb = dE_eV/w_eV
        r = mu_nb * (p/(1-p))
        return nbinom(r, p).rvs()

    Uses Gamma–Poisson mixture (supports non-integer r efficiently):
        lam ~ Gamma(shape=r, scale=(1-p)/p)
        k   ~ Poisson(lam)

    Returns int32 electrons (>=0).
    """
    dE = np.asarray(dE_MeV, dtype=np.float32)
    out = np.zeros(dE.shape, dtype=np.int32)

    p = np.float32(1.0 / fano_factor)
    if not (0.0 < p < 1.0):
        return out

    m = (dE > 0) & np.isfinite(dE)
    if not np.any(m):
        return out

    dE_eV = dE[m] * np.float32(1e6)
    mu_nb = dE_eV / np.float32(w_eV)
    r = mu_nb * (p / (1.0 - p))

    good = r > 0
    if not np.any(good):
        return out

    scale = np.float32((1.0 - p) / p)
    lam = rng.gamma(shape=r[good], scale=scale)
    k = rng.poisson(lam=lam).astype(np.int32, copy=False)

    tmp = np.zeros_like(r, dtype=np.int32)
    tmp[good] = k
    out[m] = tmp
    return out


def kernel_size_from_sigma(sigma_um, grid_spacing_um, N_sigma=6):
    """Odd integer kernel size to cover ±N_sigma*sigma."""
    size = int(np.ceil(2 * N_sigma * sigma_um / grid_spacing_um)) + 1
    if size % 2 == 0:
        size += 1
    return size


def min_region_size_um_for_kernel(sigma_um, grid_spacing_um, min_region_um=50, N_sigma=6):
    """Ensure region is large enough (microns) for a given sigma/grid."""
    kernel_size = kernel_size_from_sigma(sigma_um, grid_spacing_um, N_sigma)
    region_um = kernel_size * grid_spacing_um
    return max(min_region_um, region_um)


def gaussian_sum_kernel(size, sigma_um, grid_spacing_um=1.0, w_list=None, c_list=None):
    """
    Generate a normalized 2D charge-diffusion kernel as a weighted sum of Gaussians.

    The kernel is evaluated on a square micron-scale grid and normalized such that
    its elements sum to 1. It is intended for modeling non-Gaussian charge diffusion
    in detector simulations.

    Parameters
    ----------
    size : int
        Kernel dimension (number of grid points per side).
        Units: grid cells.
    sigma_um : float
        Base diffusion scale used to set the Gaussian widths.
        Units: microns.
    grid_spacing_um : float, default=1.0
        Physical spacing between grid points.
        Units: microns per grid cell.
    w_list : sequence of float, optional
        Relative weights of the Gaussian components.
        Units: dimensionless.
    c_list : sequence of float, optional
        Scale factors applied to ``sigma_um`` for each Gaussian component.
        Units: dimensionless.

    Returns
    -------
    kernel : numpy.ndarray
        Normalized 2D diffusion kernel with shape ``(size, size)``.
        Units: dimensionless.
    """
    if w_list is None:
        w_list = [0.17519, 0.53146, 0.29335]
    if c_list is None:
        c_list = [0.4522, 0.8050, 1.4329]

    ax = (np.arange(size) - size // 2) * grid_spacing_um
    xx, yy = np.meshgrid(ax, ax)
    rr2 = xx**2 + yy**2
    kernel = np.zeros_like(xx, dtype=float)
    for w, c in zip(w_list, c_list, strict=False):
        s = sigma_um * c
        norm = 2 * np.pi * (s**2)
        kernel += w * np.exp(-rr2 / (2 * s**2)) / norm
    kernel = np.maximum(kernel, 0)
    kernel /= kernel.sum() if kernel.sum() > 0 else 1
    return kernel


def _flatten_streaks(streaks):
    """
    Accept either:
      - flat list of streak tuples, or
      - nested [species][bin][streak] structure (as saved/loaded by GCRsim)
    and yield streak tuples.
    """
    if streaks is None:
        return
    # Heuristic: nested if first element is list-like and not a streak tuple
    if isinstance(streaks, list) and streaks and isinstance(streaks[0], list):
        for species in streaks:
            for bin_streaks in species:
                for st in bin_streaks:
                    yield st
    else:
        for st in streaks:
            yield st


def _events_df_from_streaks(streaks, dtype_xy=np.float32, dtype_dE=np.float32):
    """
    Convert streak tuples into an events DataFrame with columns: x, y, dE, PID

    Notes:
      - positions are stored in microns in GCRsim (x0/y0 built in um and appended)
      - energy_changes for primaries is stored as (dE, T_delta) so we take the first element.
    """
    xs, ys, dEs, pids = [], [], [], []

    for st in _flatten_streaks(streaks):
        positions = st[0]
        pid = st[1]
        energy_changes = st[10]

        n = min(len(positions), len(energy_changes))
        if n <= 0:
            continue

        for i in range(n):
            x_um, y_um, _z_um = positions[i]

            ec = energy_changes[i]
            # ec may be a scalar or tuple/list; primaries store (dE, T_delta)
            if isinstance(ec, (tuple, list, np.ndarray)):  # noqa: UP038
                dE = float(ec[0]) if len(ec) > 0 else 0.0
            else:
                dE = float(ec)

            xs.append(x_um)
            ys.append(y_um)
            dEs.append(dE)
            pids.append(pid)

    if not xs:
        raise ValueError("No events found in streaks (empty or mismatched positions/energy_changes).")

    return pd.DataFrame(
        {
            "x": np.asarray(xs, dtype=dtype_xy),
            "y": np.asarray(ys, dtype=dtype_xy),
            "dE": np.asarray(dEs, dtype=dtype_dE),
            "PID": np.asarray(pids, dtype=np.int64),
        }
    )


def process_electrons_to_DN(
    rng_ff=None,
    csvfile=None,
    streaks=None,
    gain_txt=None,
    n_pixels=4096,
    pixel_size_micron=10.0,
    hi_res_grid_spacing_micron=2.0,
    sigma_micron=3.14,
    N_sigma=6,
    tile_pixels=256,
    n_workers=None,
    chunk_tiles=16,  
    apply_gain=True,
    output_array_path=None,
    one_explicit=False,
):
    """
    Convert deposited electron events into a detector DN (Digital Number) map
    using a tiled, parallel Gaussian charge-diffusion model with deterministic RNG.
    Only tiles containing events (plus a configurable halo region to support the
    Gaussian kernel extent) are processed, significantly reducing computational cost
    for sparse events.

    Parameters
    ----------
    rng_ff : FastForwardRNG or None, optional
        A fast-forwardable random number generator used to create deterministic,
        independent RNG streams for each tile. Required for reproducible stochastic
        sampling inside workers. Must implement ``spawn_generators_by_jump``.
    csvfile : str or None, optional
        Path to a CSV file containing energy-deposition events. Must include the columns
        ``["x", "y", "dE", "PID"]`` with coordinates in microns. If ``None``, ``streaks``
        must be provided.
    streaks : list or None, optional
        In-memory list of streak objects (as produced by the GCR simulation pipeline).
        Used as an alternative to ``csvfile``.
    gain_txt : str or None, optional
        Path to a gain-map text file (e.g., 32×32 supercell gain values). Required if
        ``apply_gain=True``. Ignored if ``apply_gain=False``.
    n_pixels : int, default=4096
        Number of detector pixels per side (assumes a square detector).
        Units: pixels.
    pixel_size_micron : float, default=10.0
        Physical size of a detector pixel.
        Units: microns.
    hi_res_grid_spacing_micron : float, default=2.0
        Spacing of the high-resolution grid used for charge diffusion before
        downsampling to detector pixels.
        Units: microns.
    sigma_micron : float, default=3.14
        Standard deviation of the Gaussian charge-diffusion kernel.
        Units: microns.
    N_sigma : int, default=6
        Half-width of the Gaussian kernel in units of ``sigma`` (i.e., kernel extends
        to ±N_sigma·sigma).
    tile_pixels : int, default=256
        Tile size (in detector pixels) used to partition the detector. Each tile is
        processed independently with its own RNG stream.
    n_workers : int or None, optional
        Number of worker processes to use for parallel tile processing.
        If ``None``, uses ``(os.cpu_count() - 1)`` (minimum 1).
    chunk_tiles : int, default=16
        Number of tiles submitted per batch to the process pool. Controls scheduling
        overhead for large numbers of tiles.
    apply_gain : bool, default=True
        If True, applies the gain map to convert electrons to DN.
        If False, returns electrons per pixel without gain conversion.
    output_array_path : str or None, optional
        If provided, saves the output array (electrons or DN) to this path as a ``.npy`` file.
    one_explicit : bool, optional
        If True, extracts one process from the loop and does it separately for coverage tracking.
        Leave off for normal usage.

    Returns
    -------
    H_detector : numpy.ndarray
        If ``apply_gain=False``, returns the 2D array of electrons per pixel.
        Shape: ``(n_pixels, n_pixels)``.
    H_detector_DN : numpy.ndarray
        If ``apply_gain=True``, returns the 2D array of digital numbers (DN) after
        applying the gain map.
        Shape: ``(n_pixels, n_pixels)``.

    Raises
    ------
    ValueError
        If neither ``csvfile`` nor ``streaks`` is provided.
    ValueError
        If ``rng_ff`` is ``None`` (this function requires a fast-forward RNG).
    ValueError
        If ``apply_gain=True`` and ``gain_txt`` is not specified.
    """

    # Load events 
    if csvfile is not None:
        df = pd.read_csv(
            csvfile,
            usecols=["x", "y", "dE", "PID"],
            dtype={"x": np.float32, "y": np.float32, "dE": np.float32, "PID": np.int64},
        )
    else:
        if streaks is None:
            raise ValueError("csvfile is None, so you must pass streaks=<streaks_list>.")
        df = _events_df_from_streaks(streaks)

    dE = df["dE"].to_numpy(np.float32)
    x_um = df["x"].to_numpy(np.float32)
    y_um = df["y"].to_numpy(np.float32)

    # rng should be a numpy.random. Generator passed into your pipeline for determinism
    # only keep physically valid deposits; stochastic sampling happens per-tile in the worker
    keep = np.isfinite(dE) & (dE > 0)
    dE = dE[keep]
    x_um = x_um[keep]
    y_um = y_um[keep]

    # --- Tiling setup ---
    # Allow partial edge tiles (e.g. 4088 with tile 256)
    n_tiles = int(np.ceil(n_pixels / tile_pixels))
    tile_um = tile_pixels * pixel_size_micron

    # Pre-bin events into tile buckets (fast lookup per tile)
    tx_evt = np.floor(x_um / tile_um).astype(np.int32)
    ty_evt = np.floor(y_um / tile_um).astype(np.int32)
    tx_evt = np.clip(tx_evt, 0, n_tiles - 1)
    ty_evt = np.clip(ty_evt, 0, n_tiles - 1)

    buckets = {}
    for i in range(x_um.size):
        key = (int(ty_evt[i]), int(tx_evt[i]))
        buckets.setdefault(key, []).append(i)

    # Determine neighbor radius in tiles for padding reach
    # r = int(round(pixel_size_micron / hi_res_grid_spacing_micron))
    sigma_hi = sigma_micron / hi_res_grid_spacing_micron
    pad_hi = int(np.ceil(N_sigma * sigma_hi))
    pad_um = pad_hi * hi_res_grid_spacing_micron
    neigh = int(np.ceil(pad_um / tile_um))

    # Tiles that contain at least one event
    occupied = set(zip(ty_evt.tolist(), tx_evt.tolist(), strict=False))

    # Tiles we actually process: occupied tiles + neighbor halo for blur support
    tiles_to_process = set()
    for ty, tx in occupied:
        for nny in range(max(0, ty - neigh), min(n_tiles, ty + neigh + 1)):
            for nnx in range(max(0, tx - neigh), min(n_tiles, tx + neigh + 1)):
                tiles_to_process.add((nny, nnx))

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    H_detector = np.zeros((n_pixels, n_pixels), dtype=np.float32)

    jobs = []

    # Iterate only the relevant tiles
    for ty, tx in sorted(tiles_to_process):
        # actual tile size (handles edge tiles)
        y0 = ty * tile_pixels
        x0 = tx * tile_pixels
        tile_h = min(tile_pixels, n_pixels - y0)
        tile_w = min(tile_pixels, n_pixels - x0)
        if tile_h <= 0 or tile_w <= 0:
            continue

        # union of bucket indices from neighboring tiles (coarse candidate set)
        idx_list = []
        for nny in range(max(0, ty - neigh), min(n_tiles, ty + neigh + 1)):
            for nnx in range(max(0, tx - neigh), min(n_tiles, tx + neigh + 1)):
                idx_list.extend(buckets.get((nny, nnx), []))

        if not idx_list:
            # This tile is in halo set but has no nearby candidate events -> skip job
            continue

        idx_arr = np.asarray(idx_list, dtype=np.int64)

        jobs.append(
            (
                ty,
                tx,
                tile_pixels,
                tile_h,
                tile_w,
                n_pixels,
                pixel_size_micron,
                hi_res_grid_spacing_micron,
                sigma_micron,
                N_sigma,
                x_um,
                y_um,
                dE,
                idx_arr,
            )
        )

    if rng_ff is None:
        raise ValueError("Option-2 RNG requires rng_ff=FastForwardRNG(...) to be passed in.")

    # IMPORTANT: stable order so each tile gets a deterministic stream assignment
    # If jobs is a list of tuples that include (ty, tx) early, sort by those:
    jobs.sort(key=lambda j: (j[0], j[1]))  # assumes (ty, tx, ...) are first

    tile_rngs = rng_ff.spawn_generators_by_jump(len(jobs))  
    tile_rng_states = [g.bit_generator.state for g in tile_rngs]  

    # Attach one RNG state per job
    jobs = [(*job, tile_rng_states[i]) for i, job in enumerate(jobs)]

    # Submit in batches so we don’t create thousands of futures at once
    def batched(it, n):
        for i in range(0, len(it), n):
            yield it[i : i + n]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for batch in tqdm(list(batched(jobs, chunk_tiles)), desc=f"Submitting {len(jobs)} tile jobs"):
            _batch = batch[:-1] if one_explicit else batch
            futs = [ex.submit(_tile_worker_histblur, j) for j in _batch]
            for fut in as_completed(futs):
                out = fut.result()
                if out is None:
                    continue
                y0, x0, block = out
                h, w = block.shape
                # Clip just in case edge pads produce slightly off sizes
                H_detector[y0 : y0 + h, x0 : x0 + w] += block
            if one_explicit:
                out = _tile_worker_histblur(batch[-1])
                if out is not None:
                    y0, x0, block = out
                    h, w = block.shape
                    # Clip just in case edge pads produce slightly off sizes
                    H_detector[y0 : y0 + h, x0 : x0 + w] += block

    # Gain + save 
    if not apply_gain:
        if output_array_path:
            np.save(output_array_path, H_detector)
        return H_detector

    if gain_txt is None:
        raise ValueError("gain_txt must be provided when apply_gain=True.")
    gain_array = np.loadtxt(gain_txt)[:, 5]
    nbin = math.isqrt(np.size(gain_array))
    gain_array = gain_array.reshape((nbin, nbin))
    supercell_size = (n_pixels + nbin - 1) // nbin  # supercell size, rounded up
    gain_map = np.kron(gain_array, np.ones((supercell_size, supercell_size)))
    gain_map_safe = np.where(gain_map > 0, gain_map, np.nan)
    if np.shape(gain_map_safe)[-1] > np.shape(H_detector)[-1]:
        # trim reference pixels
        rpix = (np.shape(gain_map_safe)[-1] - np.shape(H_detector)[-1]) // 2
        gain_map_safe = gain_map_safe[rpix:-rpix, rpix:-rpix]

    H_detector_DN = H_detector / gain_map_safe
    if output_array_path:
        np.save(output_array_path, H_detector_DN)

    return H_detector_DN


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spread electrons and optionally convert to DN from cosmic ray sim CSV."
    )
    parser.add_argument(
        "--csvfile", type=str, required=True, help="CSV file with energy loss events (microns, MeV)"
    )
    parser.add_argument(
        "--gain_txt",
        type=str,
        default=None,
        help="Gain map .txt file (column 5 = gain e-/DN). Required if applying gain.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output .npy path (DN if applying gain; electrons if not).",
    )
    # Processing variant selection (keep your blob method as default)
    parser.add_argument(
        "--mode",
        choices=["blob", "stream"],
        default="blob",
        help="'blob' groups by PID; 'stream' reads CSV in chunks.",
    )
    # Gain toggle (default True for backward-compat)
    gain_group = parser.add_mutually_exclusive_group()
    gain_group.add_argument(
        "--apply-gain", dest="apply_gain", action="store_true", help="Apply gain (default)."
    )
    gain_group.add_argument(
        "--no-apply-gain",
        dest="apply_gain",
        action="store_false",
        help="Do not apply gain; output is electrons-per-pixel.",
    )
    parser.set_defaults(apply_gain=True)

    args = parser.parse_args()

    if args.apply_gain and args.gain_txt is None:
        raise SystemExit(
            "ERROR: --gain_txt is required when --apply-gain is set (default). "
            "Use --no-apply-gain to skip gain."
        )

    ffrng = FastForwardRNG()

    if args.csvfile is not None:
        process_electrons_to_DN(
            csvfile=args.csvfile,
            gain_txt=args.gain_txt,
            output_array_path=args.output,
            apply_gain=args.apply_gain,
            rng=ffrng,
        )
    if args.streaks is not None:
        process_electrons_to_DN(
            streaks=args.streaks,
            gain_txt=args.gain_txt,
            output_array_path=args.output,
            apply_gain=args.apply_gain,
            rng=ffrng,
        )        
    if args.streaks is None and args.csvfile is None:
        raise SystemExit(
            "Must pass LET data as either CSV file or streaks data from gcrsim."
        )
