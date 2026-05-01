# electron_spread.py

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import nbinom
from tqdm import tqdm


# new functions to handle 'histogram electrons -> Gaussian blur -> downsample' routine:
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


def _rng_from_state(rng_state: dict) -> np.random.Generator:
    bitgen = np.random.PCG64()
    bitgen.state = rng_state
    return np.random.Generator(bitgen)


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


def spread_electrons_to_patch(array, x_idx, y_idx, n_electrons, kernel):
    """Spread electrons into an array patch using a multinomial distribution."""
    size = kernel.shape[0]
    offset = size // 2

    x0, x1 = x_idx - offset, x_idx + offset + 1
    y0, y1 = y_idx - offset, y_idx + offset + 1

    patch_x0, patch_y0 = max(0, x0), max(0, y0)
    patch_x1, patch_y1 = min(array.shape[1], x1), min(array.shape[0], y1)

    kx0 = patch_x0 - x0
    ky0 = patch_y0 - y0
    kx1 = kx0 + (patch_x1 - patch_x0)
    ky1 = ky0 + (patch_y1 - patch_y0)

    patch_kernel = kernel[ky0:ky1, kx0:kx1]
    patch_kernel = np.maximum(patch_kernel, 0)
    patch_kernel /= patch_kernel.sum() if patch_kernel.sum() > 0 else 1

    draws = np.random.default_rng().multinomial(n_electrons, patch_kernel.ravel())
    # Iterate over patch grid positions (row-major)
    h, w = patch_kernel.shape
    coords = ((i, j) for i in range(h) for j in range(w))
    for count, (dy, dx) in zip(draws, coords, strict=False):
        if count > 0:
            i = patch_y0 + dy
            j = patch_x0 + dx
            array[i, j] += count


def process_electrons_to_DN(
    csvfile,
    gain_txt=None,
    n_pixels=4096,
    pixel_size_micron=10.0,
    hi_res_grid_spacing_micron=2.0,  # changed from 1.0 to 2.0 to see effect on timing
    chunksize=200_000,
    sigma_micron=3.14,
    N_sigma=6,
    output_array_path=None,
    apply_gain=True,
):
    """
    Convert simulated charge deposition events (CSV) to a detector-scale image.

    If apply_gain=True (default):
        - Applies gain map (e-/DN) and returns/saves DN.
    If apply_gain=False:
        - Skips gain, returns/saves electrons-per-pixel.
    """
    # Determine size of hi-res grid
    det_size_micron = n_pixels * pixel_size_micron
    n_hi = int(det_size_micron / hi_res_grid_spacing_micron)
    assert np.isclose(
        n_hi * hi_res_grid_spacing_micron, det_size_micron, atol=1e-5
    ), "Detector size must be divisible by hi-res grid spacing."

    # Kernel (hi-res)
    kernel_size_hi = kernel_size_from_sigma(sigma_micron, hi_res_grid_spacing_micron, N_sigma)
    kernel = gaussian_sum_kernel(kernel_size_hi, sigma_micron, hi_res_grid_spacing_micron)
    H_hi = np.zeros((n_hi, n_hi), dtype=float)

    # Process events
    for chunk in pd.read_csv(csvfile, sep=",", chunksize=chunksize):
        xs_um = chunk["x"].to_numpy()
        ys_um = chunk["y"].to_numpy()
        dEs_MeV = chunk["dE"].to_numpy()
        for x_um, y_um, dE in tqdm(
            zip(xs_um, ys_um, dEs_MeV, strict=False), total=len(xs_um), desc="Processing events"
        ):
            n_electrons = electron_conversion(dE)
            if n_electrons > 0:
                x_hi_idx = int(np.floor(x_um / hi_res_grid_spacing_micron))
                y_hi_idx = int(np.floor(y_um / hi_res_grid_spacing_micron))
                half_patch = kernel_size_hi // 2

                patch = np.zeros((kernel_size_hi, kernel_size_hi), dtype=float)
                spread_electrons_to_patch(patch, half_patch, half_patch, n_electrons, kernel)

                y0 = y_hi_idx - half_patch
                y1 = y0 + kernel_size_hi
                x0 = x_hi_idx - half_patch
                x1 = x0 + kernel_size_hi

                py0 = max(0, -y0)
                py1 = kernel_size_hi - max(0, y1 - n_hi)
                px0 = max(0, -x0)
                px1 = kernel_size_hi - max(0, x1 - n_hi)
                hy0 = max(0, y0)
                hy1 = min(n_hi, y1)
                hx0 = max(0, x0)
                hx1 = min(n_hi, x1)
                H_hi[hy0:hy1, hx0:hx1] += patch[py0:py1, px0:px1]

    # Downsample to detector pixels (sum of electrons per pixel)
    downsample_factor = int(pixel_size_micron / hi_res_grid_spacing_micron)
    assert np.isclose(
        downsample_factor * hi_res_grid_spacing_micron, pixel_size_micron, atol=1e-5
    ), "Detector pixel size must be divisible by hi-res grid spacing."
    H_detector = H_hi.reshape(n_pixels, downsample_factor, n_pixels, downsample_factor).sum(axis=(1, 3))

    if not apply_gain:
        H_out = H_detector  # electrons per pixel
        if output_array_path:
            np.save(output_array_path, H_out)
            print(f"Saved electrons-per-pixel array to {output_array_path}")
        return H_out

    # Apply gain correction (electrons -> DN)
    if gain_txt is None:
        raise ValueError("gain_txt must be provided when apply_gain=True.")
    gain_array = np.loadtxt(gain_txt)[:, 5].reshape((32, 32))
    supercell_size = n_pixels // 32
    gain_map = np.kron(gain_array, np.ones((supercell_size, supercell_size)))
    assert gain_map.shape == H_detector.shape, "Gain map shape does not match detector image."
    gain_map_safe = np.where(gain_map > 0, gain_map, np.nan)
    H_detector_DN = H_detector / gain_map_safe

    if output_array_path:
        np.save(output_array_path, H_detector_DN)
        print(f"Saved DN array to {output_array_path}")

    return H_detector_DN


def process_pid_electrons_zoom(
    csvfile, pid, delta_pids, sigma_micron=3.14, hi_res_grid_spacing_micron=2.0, N_sigma=6
):
    """
    Build a high-res patch (electrons) for a given PID and its delta PIDs.
    """
    wanted_pids = set([pid] + list(delta_pids))
    df = pd.read_csv(csvfile)
    mask = df["PID"].isin(wanted_pids)
    if not mask.any():
        raise ValueError("No events found for this PID + deltas in the CSV.")

    xs = df.loc[mask, "x"]
    ys = df.loc[mask, "y"]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    expand = N_sigma * sigma_micron
    x0_um = x_min - expand
    x1_um = x_max + expand
    y0_um = y_min - expand
    y1_um = y_max + expand

    n_pix_x = int(np.ceil((x1_um - x0_um) / hi_res_grid_spacing_micron))
    n_pix_y = int(np.ceil((y1_um - y0_um) / hi_res_grid_spacing_micron))

    patch = np.zeros((n_pix_y, n_pix_x), dtype=float)

    kernel_size_hi = kernel_size_from_sigma(sigma_micron, hi_res_grid_spacing_micron, N_sigma)
    kernel = gaussian_sum_kernel(kernel_size_hi, sigma_micron, hi_res_grid_spacing_micron)

    dEs = df.loc[mask, "dE"]
    for x_um, y_um, dE in zip(xs, ys, dEs, strict=False):
        n_electrons = electron_conversion(dE)
        if n_electrons > 0:
            x_idx = int(np.floor((x_um - x0_um) / hi_res_grid_spacing_micron))
            y_idx = int(np.floor((y_um - y0_um) / hi_res_grid_spacing_micron))
            spread_electrons_to_patch(patch, x_idx, y_idx, n_electrons, kernel)

    x_coords_um = x0_um + np.arange(n_pix_x) * hi_res_grid_spacing_micron
    y_coords_um = y0_um + np.arange(n_pix_y) * hi_res_grid_spacing_micron
    return patch, x_coords_um, y_coords_um


def _downsample_and_add_patch(H_detector, patch, hi_x0, hi_y0, r):
    """
    Downsample a hi-res patch by block-summing (factor r) and add it into the
    detector image H_detector at the correct location. Handles alignment, padding,
    and clipping to detector bounds.
    """
    h, w = patch.shape

    # pad so the patch starts on an r-aligned boundary
    pad_left = hi_x0 % r
    pad_top = hi_y0 % r
    pad_right = (-(pad_left + w)) % r
    pad_bottom = (-(pad_top + h)) % r

    if pad_top or pad_bottom or pad_left or pad_right:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

    ph, pw = patch.shape
    # block-sum downsample
    patch_ds = patch.reshape(ph // r, r, pw // r, r).sum(axis=(1, 3))

    # destination detector indices
    det_x0 = (hi_x0 - pad_left) // r
    det_y0 = (hi_y0 - pad_top) // r

    Hh, Hw = H_detector.shape

    # clip to detector bounds
    src_x0 = max(0, -det_x0)
    dst_x0 = max(0, det_x0)
    src_y0 = max(0, -det_y0)
    dst_y0 = max(0, det_y0)
    width = min(patch_ds.shape[1] - src_x0, Hw - dst_x0)
    height = min(patch_ds.shape[0] - src_y0, Hh - dst_y0)
    if width <= 0 or height <= 0:
        return

    H_detector[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] += patch_ds[
        src_y0 : src_y0 + height, src_x0 : src_x0 + width
    ]


# new parallel version of above function
def _downsample_patch_to_block(patch, hi_x0, hi_y0, r, det_shape):
    """
    Downsample hi-res patch by factor r and return a clipped (y0, x0, block) suitable
    for adding into H_detector. Returns None if it lands entirely off-detector.
    """
    h, w = patch.shape

    # pad so the patch starts on an r-aligned boundary (same logic as before)
    pad_left = hi_x0 % r
    pad_top = hi_y0 % r
    pad_right = (-(pad_left + w)) % r
    pad_bottom = (-(pad_top + h)) % r

    if pad_top or pad_bottom or pad_left or pad_right:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

    ph, pw = patch.shape
    patch_ds = patch.reshape(ph // r, r, pw // r, r).sum(axis=(1, 3))

    det_x0 = (hi_x0 - pad_left) // r
    det_y0 = (hi_y0 - pad_top) // r

    Hh, Hw = det_shape

    # clip to detector bounds
    src_x0 = max(0, -det_x0)
    dst_x0 = max(0, det_x0)
    src_y0 = max(0, -det_y0)
    dst_y0 = max(0, det_y0)
    width = min(patch_ds.shape[1] - src_x0, Hw - dst_x0)
    height = min(patch_ds.shape[0] - src_y0, Hh - dst_y0)
    if width <= 0 or height <= 0:
        return None

    block = patch_ds[src_y0 : src_y0 + height, src_x0 : src_x0 + width]
    return dst_y0, dst_x0, block


# new worker function that process a chunk of PIDs
def _process_pid_chunk(
    pid_chunk_items,
    kernel,
    kernel_size_hi,
    r,
    hi_res_grid_spacing_micron,
    sigma_micron,
    N_sigma,
    n_pixels,
    pixel_size_micron,
):
    """
    pid_chunk_items: list of (pid, xs_um, ys_um, dEs_MeV) arrays
    Returns: list of (dst_y0, dst_x0, block_array)
    """
    det_shape = (n_pixels, n_pixels)
    det_size_um = n_pixels * pixel_size_micron
    expand = N_sigma * sigma_micron

    out_blocks = []

    for _, xs_um, ys_um, dEs_MeV in pid_chunk_items:
        if len(xs_um) == 0:
            continue

        # bounding box in microns + expand
        x_min = float(np.min(xs_um))
        x_max = float(np.max(xs_um))
        y_min = float(np.min(ys_um))
        y_max = float(np.max(ys_um))

        patch_x0_um = x_min - expand
        patch_x1_um = x_max + expand
        patch_y0_um = y_min - expand
        patch_y1_um = y_max + expand

        # optional: quick reject if fully off detector
        if (
            (patch_x1_um < 0)
            or (patch_y1_um < 0)
            or (patch_x0_um > det_size_um)
            or (patch_y0_um > det_size_um)
        ):
            continue

        patch_w = int(np.ceil((patch_x1_um - patch_x0_um) / hi_res_grid_spacing_micron))
        patch_h = int(np.ceil((patch_y1_um - patch_y0_um) / hi_res_grid_spacing_micron))
        patch_w = max(patch_w, kernel_size_hi)
        patch_h = max(patch_h, kernel_size_hi)

        patch = np.zeros((patch_h, patch_w), dtype=np.float32)

        # stamp each event into the patch
        for x_um, y_um, dE in zip(xs_um, ys_um, dEs_MeV, strict=False):
            n_electrons = electron_conversion(float(dE))
            if n_electrons <= 0:
                continue
            x_idx = int(np.floor((float(x_um) - patch_x0_um) / hi_res_grid_spacing_micron))
            y_idx = int(np.floor((float(y_um) - patch_y0_um) / hi_res_grid_spacing_micron))
            spread_electrons_to_patch(patch, x_idx, y_idx, n_electrons, kernel)

        hi_x0 = int(np.floor(patch_x0_um / hi_res_grid_spacing_micron))
        hi_y0 = int(np.floor(patch_y0_um / hi_res_grid_spacing_micron))

        blk = _downsample_patch_to_block(patch, hi_x0, hi_y0, r, det_shape)
        if blk is not None:
            out_blocks.append(blk)

    return out_blocks


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
        # streak layout in GCRsim_v02ht save/load:
        # (positions, pid, ..., energy_changes, global_steps, ...)
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


def process_electrons_to_DN_by_blob(
    csvfile=None,
    streaks=None,
    gain_txt=None,
    n_pixels=4096,
    pixel_size_micron=10.0,
    hi_res_grid_spacing_micron=2.0,
    sigma_micron=3.14,
    N_sigma=6,
    output_array_path=None,
    apply_gain=True,
    detector_dtype=np.float32,
    n_workers=None,
    chunk_size=64,
):
    """
    Convert deposited electron events into a detector DN (Digital Number) map
    using a high-resolution Gaussian charge-diffusion model.

    This routine:
      1. Loads electron deposition events from a CSV file or in-memory streaks.
      2. Groups events by particle ID (PID).
      3. For each PID, deposits charge onto a high-resolution grid using a
         Gaussian diffusion kernel.
      4. Downsamples the high-resolution grid onto the detector pixel grid.
      5. Optionally applies a gain map to convert electrons to DN.
      6. Optionally saves the resulting array to disk.

    Parallel processing is used to distribute PID groups across multiple workers.

    Parameters
    ----------
    csvfile : str or None, optional
        Path to a CSV file containing energy-deposition events. Must include the columns
        ``["x", "y", "dE", "PID"]`` in microns and energy units consistent with the upstream
        simulation. If ``None``, ``streaks`` must be provided.
    streaks : list or None, optional
        In-memory list of streak objects (as produced by the GCR simulation pipeline).
        Used as an alternative to ``csvfile`` for building the event table.
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
    output_array_path : str or None, optional
        If provided, saves the output array (electrons or DN) to this path as a ``.npy`` file.
    apply_gain : bool, default=True
        If True, applies the gain map to convert electrons to DN.
        If False, returns electrons per pixel without gain conversion.
    detector_dtype : numpy dtype, default=np.float32
        Data type of the output detector array.
    n_workers : int or None, optional
        Number of worker processes to use for parallel PID processing.
        If ``None``, uses ``(os.cpu_count() - 1)`` (minimum 1).
    chunk_size : int, default=64
        Number of PIDs grouped into each processing chunk to reduce multiprocessing overhead.

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
        If ``apply_gain=True`` and ``gain_txt`` is not specified.
    ValueError
        If ``pixel_size_micron`` is not an integer multiple of ``hi_res_grid_spacing_micron``.

    Notes
    -----
    - The high-resolution grid is downsampled by an integer factor
      ``r = pixel_size_micron / hi_res_grid_spacing_micron``.
    - Each PID is processed independently, and partial detector blocks are accumulated
      in the parent process to avoid race conditions.
    - Gain values ≤ 0 are treated as invalid and masked during DN conversion.
    """
    # downsample factor (must be integer)
    r = int(round(pixel_size_micron / hi_res_grid_spacing_micron))
    if not np.isclose(r * hi_res_grid_spacing_micron, pixel_size_micron, atol=1e-6):
        raise ValueError("Detector pixel size must be divisible by hi-res grid spacing.")

    H_detector = np.zeros((n_pixels, n_pixels), dtype=detector_dtype)

    kernel_size_hi = kernel_size_from_sigma(sigma_micron, hi_res_grid_spacing_micron, N_sigma)
    kernel = gaussian_sum_kernel(kernel_size_hi, sigma_micron, hi_res_grid_spacing_micron)

    # --- Get df from CSV or streaks (your earlier change) ---
    if csvfile is not None:
        df = pd.read_csv(
            csvfile,
            usecols=["x", "y", "dE", "PID"],
            dtype={"x": np.float32, "y": np.float32, "dE": np.float32, "PID": np.int64},
        )
        if "PID" not in df.columns:
            raise ValueError("CSV must have a 'PID' column for grouping.")
    else:
        if streaks is None:
            raise ValueError("csvfile is None, so you must pass streaks=<streaks_list>.")
        df = _events_df_from_streaks(streaks)

    # --- build PID items for workers ---
    pid_items = []
    for pid, group in df.groupby(
        "PID"
    ):  # your current serial loop is here :contentReference[oaicite:3]{index=3}
        pid_items.append(
            (
                int(pid),
                group["x"].to_numpy(),
                group["y"].to_numpy(),
                group["dE"].to_numpy(),
            )
        )

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    # chunk PIDs to reduce overhead
    chunks = [pid_items[i : i + chunk_size] for i in range(0, len(pid_items), chunk_size)]

    # --- parallel map: workers return blocks; parent accumulates => no collisions ---
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _process_pid_chunk,
                chunk,
                kernel,
                kernel_size_hi,
                r,
                hi_res_grid_spacing_micron,
                sigma_micron,
                N_sigma,
                n_pixels,
                pixel_size_micron,
            )
            for chunk in chunks
        ]

        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Processing primary GCRs in parallel"
        ):
            blocks = fut.result()
            for y0, x0, block in blocks:
                h, w = block.shape
                H_detector[y0 : y0 + h, x0 : x0 + w] += block

    # --- rest of your function unchanged (gain + save) ---
    if not apply_gain:
        if output_array_path:
            np.save(output_array_path, H_detector)
            print(f"Saved electrons-per-pixel array to {output_array_path}")
        return H_detector

    if gain_txt is None:
        raise ValueError("gain_txt must be provided when apply_gain=True.")
    gain_array = np.loadtxt(gain_txt)[:, 5].reshape((32, 32))
    supercell_size = n_pixels // 32
    gain_map = np.kron(gain_array, np.ones((supercell_size, supercell_size)))
    gain_map_safe = np.where(gain_map > 0, gain_map, np.nan).astype(np.float32)

    H_detector_DN = H_detector / gain_map_safe
    if output_array_path:
        np.save(output_array_path, H_detector_DN)
        print(f"Saved DN array to {output_array_path}")
    return H_detector_DN


# new processing function
def process_electrons_to_DN_by_blob2(
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
    chunk_tiles=16,  # submit tiles in batches to reduce overhead
    apply_gain=True,
    output_array_path=None,
):
    """
    Convert deposited electron events into a detector DN (Digital Number) map
    using a tiled, parallel Gaussian charge-diffusion model with deterministic RNG.

    This routine is a performance-optimized variant of
    ``process_electrons_to_DN_by_blob`` that:

      1. Loads electron deposition events from a CSV file or in-memory streaks.
      2. Filters and vectorizes the event arrays.
      3. Partitions the detector into spatial tiles.
      4. Assigns each tile a deterministic random number stream using a
         fast-forwardable RNG.
      5. Applies Gaussian charge diffusion on a high-resolution grid within each tile.
      6. Downsamples each tile onto the detector pixel grid.
      7. Accumulates all tile contributions into the final detector image.
      8. Optionally applies a gain map to convert electrons to DN.
      9. Optionally saves the output array to disk.

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

    Notes
    -----
    - Events are first binned into coarse spatial buckets for fast lookup by tile.
    - Only tiles containing events, plus a halo of neighboring tiles determined by
      ``N_sigma`` and ``sigma_micron``, are processed.
    - Each tile is assigned an independent, deterministic RNG stream using
      ``rng_ff.spawn_generators_by_jump`` to ensure reproducibility across runs and
      across different process counts.
    - Partial edge tiles are handled automatically when ``n_pixels`` is not an integer
      multiple of ``tile_pixels``.
    - This implementation is optimized for sparse charge-deposition patterns and large
      detectors, significantly reducing runtime compared to global convolution methods.
    """

    # --- Load events (same logic you already added) ---
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

    # Vectorized electron conversion (fast): compute n_e once
    # Replace this with your existing electron_conversion formula if needed.
    dE = df["dE"].to_numpy(np.float32)
    x_um = df["x"].to_numpy(np.float32)
    y_um = df["y"].to_numpy(np.float32)

    # rng should be a numpy.random.Generator passed into your pipeline for determinism
    # Cheap filter: only keep physically valid deposits; stochastic sampling happens per-tile in the worker
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

    # Tiles we will actually process: occupied tiles + neighbor halo for blur support
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
        # --- actual tile size (handles edge tiles if you implemented that earlier) ---
        y0 = ty * tile_pixels
        x0 = tx * tile_pixels
        tile_h = min(tile_pixels, n_pixels - y0)
        tile_w = min(tile_pixels, n_pixels - x0)
        if tile_h <= 0 or tile_w <= 0:
            continue

        # --- union of bucket indices from neighboring tiles (coarse candidate set) ---
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

    tile_rngs = rng_ff.spawn_generators_by_jump(len(jobs))  # you already added this method
    tile_rng_states = [g.bit_generator.state for g in tile_rngs]  # pickle-friendly

    # Attach one RNG state per job
    jobs = [(*job, tile_rng_states[i]) for i, job in enumerate(jobs)]

    # Submit in batches so we don’t create thousands of futures at once
    def batched(it, n):
        for i in range(0, len(it), n):
            yield it[i : i + n]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for batch in tqdm(list(batched(jobs, chunk_tiles)), desc=f"Submitting {len(jobs)} tile jobs"):
            futs = [ex.submit(_tile_worker_histblur, j) for j in batch]
            for fut in as_completed(futs):
                out = fut.result()
                if out is None:
                    continue
                y0, x0, block = out
                h, w = block.shape
                # Clip just in case edge pads produce slightly off sizes
                H_detector[y0 : y0 + h, x0 : x0 + w] += block

    # --- Gain + save (same as your existing path) ---
    if not apply_gain:
        if output_array_path:
            np.save(output_array_path, H_detector)
        return H_detector

    if gain_txt is None:
        raise ValueError("gain_txt must be provided when apply_gain=True.")

    gain_array = np.loadtxt(gain_txt)[:, 5].reshape((32, 32))
    supercell_size = n_pixels // 32
    gain_map = np.kron(gain_array, np.ones((supercell_size, supercell_size), dtype=np.float32))
    gain_map_safe = np.where(gain_map > 0, gain_map, np.nan).astype(np.float32)

    H_detector_DN = H_detector / gain_map_safe
    if output_array_path:
        np.save(output_array_path, H_detector_DN)

    # debug print
    print(
        f"""Occupied tiles: {len(occupied)} | Tiles processed
        (with halo): {len(tiles_to_process)} | Jobs submitted: {len(jobs)}"""
    )
    return H_detector_DN


# -------- CLI --------
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

    if args.mode == "blob":
        process_electrons_to_DN_by_blob(
            args.csvfile, gain_txt=args.gain_txt, output_array_path=args.output, apply_gain=args.apply_gain
        )
    else:
        process_electrons_to_DN(
            args.csvfile, gain_txt=args.gain_txt, output_array_path=args.output, apply_gain=args.apply_gain
        )
