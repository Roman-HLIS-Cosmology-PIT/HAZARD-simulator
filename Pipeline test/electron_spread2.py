# electron_spread.py

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from tqdm import tqdm

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

def gaussian_sum_kernel(
    size,
    sigma_um,
    grid_spacing_um=1.0,
    w_list=[0.17519, 0.53146, 0.29335],
    c_list=[0.4522, 0.8050, 1.4329]
):
    """Sum-of-3-Gaussians, normalized, on a micron grid."""
    ax = (np.arange(size) - size // 2) * grid_spacing_um
    xx, yy = np.meshgrid(ax, ax)
    rr2 = xx**2 + yy**2
    kernel = np.zeros_like(xx, dtype=float)
    for w, c in zip(w_list, c_list):
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
    for count, (dy, dx) in zip(draws, coords):
        if count > 0:
            i = patch_y0 + dy
            j = patch_x0 + dx
            array[i, j] += count

def process_electrons_to_DN(
    csvfile,
    gain_txt=None,
    n_pixels=4096,
    pixel_size_micron=10.0,
    hi_res_grid_spacing_micron=1.0,
    chunksize=200_000,
    sigma_micron=3.14,
    N_sigma=6,
    output_array_path=None,
    apply_gain=True
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
    assert np.isclose(n_hi * hi_res_grid_spacing_micron, det_size_micron, atol=1e-5), \
        "Detector size must be divisible by hi-res grid spacing."

    # Kernel (hi-res)
    kernel_size_hi = kernel_size_from_sigma(sigma_micron, hi_res_grid_spacing_micron, N_sigma)
    kernel = gaussian_sum_kernel(kernel_size_hi, sigma_micron, hi_res_grid_spacing_micron)
    H_hi = np.zeros((n_hi, n_hi), dtype=float)

    # Process events
    for chunk in pd.read_csv(csvfile, sep=',', chunksize=chunksize):
        xs_um = chunk['x'].to_numpy()
        ys_um = chunk['y'].to_numpy()
        dEs_MeV = chunk['dE'].to_numpy()
        for x_um, y_um, dE in tqdm(zip(xs_um, ys_um, dEs_MeV),
                                   total=len(xs_um), desc="Processing events"):
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
    assert np.isclose(downsample_factor * hi_res_grid_spacing_micron, pixel_size_micron, atol=1e-5), \
        "Detector pixel size must be divisible by hi-res grid spacing."
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
    csvfile,
    pid,
    delta_pids,
    sigma_micron=3.14,
    hi_res_grid_spacing_micron=1.0,
    N_sigma=6
):
    """
    Build a high-res patch (electrons) for a given PID and its delta PIDs.
    """
    wanted_pids = set([pid] + list(delta_pids))
    df = pd.read_csv(csvfile)
    mask = df['PID'].isin(wanted_pids)
    if not mask.any():
        raise ValueError("No events found for this PID + deltas in the CSV.")

    xs = df.loc[mask, 'x']
    ys = df.loc[mask, 'y']
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

    dEs = df.loc[mask, 'dE']
    for x_um, y_um, dE in zip(xs, ys, dEs):
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
    import numpy as np
    h, w = patch.shape

    # pad so the patch starts on an r-aligned boundary
    pad_left   = hi_x0 % r
    pad_top    = hi_y0 % r
    pad_right  = (- (pad_left + w)) % r
    pad_bottom = (- (pad_top  + h)) % r

    if pad_top or pad_bottom or pad_left or pad_right:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

    ph, pw = patch.shape
    # block-sum downsample
    patch_ds = patch.reshape(ph // r, r, pw // r, r).sum(axis=(1, 3))

    # destination detector indices
    det_x0 = (hi_x0 - pad_left) // r
    det_y0 = (hi_y0 - pad_top)  // r

    Hh, Hw = H_detector.shape

    # clip to detector bounds
    src_x0 = max(0, -det_x0); dst_x0 = max(0, det_x0)
    src_y0 = max(0, -det_y0); dst_y0 = max(0, det_y0)
    width  = min(patch_ds.shape[1] - src_x0, Hw - dst_x0)
    height = min(patch_ds.shape[0] - src_y0, Hh - dst_y0)
    if width <= 0 or height <= 0:
        return

    H_detector[dst_y0:dst_y0+height, dst_x0:dst_x0+width] += patch_ds[src_y0:src_y0+height, src_x0:src_x0+width]


def process_electrons_to_DN_by_blob(
    csvfile,
    gain_txt=None,
    n_pixels=4096,
    pixel_size_micron=10.0,
    hi_res_grid_spacing_micron=1.0,
    sigma_micron=3.14,
    N_sigma=6,
    output_array_path=None,
    apply_gain=True,
    # optional memory saver: use float32 for the detector grid
    detector_dtype=np.float32,
):
    """
    Memory-safe 'blob' processing:
      * Build a hi-res patch per PID
      * Immediately downsample that patch to detector pixels
      * Accumulate into a single n_pixels×n_pixels image (no global hi-res array)
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # downsample factor (must be integer)
    r = int(round(pixel_size_micron / hi_res_grid_spacing_micron))
    if not np.isclose(r * hi_res_grid_spacing_micron, pixel_size_micron, atol=1e-6):
        raise ValueError("Detector pixel size must be divisible by hi-res grid spacing.")

    # Allocate detector image only (~64MB as float32 for 4088x4088)
    H_detector = np.zeros((n_pixels, n_pixels), dtype=detector_dtype)

    # Kernel on hi-res grid
    kernel_size_hi = kernel_size_from_sigma(sigma_micron, hi_res_grid_spacing_micron, N_sigma)
    kernel = gaussian_sum_kernel(kernel_size_hi, sigma_micron, hi_res_grid_spacing_micron)

    # Read only needed columns, lighter dtypes
    df = pd.read_csv(
        csvfile,
        usecols=["x", "y", "dE", "PID"],
        dtype={"x": np.float32, "y": np.float32, "dE": np.float32, "PID": np.int64},
    )
    if 'PID' not in df.columns:
        raise ValueError("CSV must have a 'PID' column for grouping.")

    det_size_um = n_pixels * pixel_size_micron  # for clipping sanity

    for pid, group in tqdm(df.groupby('PID'), desc="Processing primary GCRs"):
        xs_um = group['x'].to_numpy()
        ys_um = group['y'].to_numpy()
        dEs_MeV = group['dE'].to_numpy()

        # physical patch bounds (+Nσ) on hi-res grid
        x_min, x_max = float(xs_um.min()), float(xs_um.max())
        y_min, y_max = float(ys_um.min()), float(ys_um.max())
        expand = N_sigma * sigma_micron
        patch_x0_um = x_min - expand
        patch_x1_um = x_max + expand
        patch_y0_um = y_min - expand
        patch_y1_um = y_max + expand

        # clip physical patch to detector extent to avoid giant pads when near borders
        patch_x0_um = max(patch_x0_um, 0.0)
        patch_y0_um = max(patch_y0_um, 0.0)
        patch_x1_um = min(patch_x1_um, det_size_um)
        patch_y1_um = min(patch_y1_um, det_size_um)
        if patch_x1_um <= patch_x0_um or patch_y1_um <= patch_y0_um:
            continue

        # hi-res index-space for the patch
        patch_w = int(np.ceil((patch_x1_um - patch_x0_um) / hi_res_grid_spacing_micron))
        patch_h = int(np.ceil((patch_y1_um - patch_y0_um) / hi_res_grid_spacing_micron))
        patch_w = max(patch_w, kernel_size_hi)
        patch_h = max(patch_h, kernel_size_hi)

        patch = np.zeros((patch_h, patch_w), dtype=np.float32)

        # stamp each event into the patch
        for x_um, y_um, dE in zip(xs_um, ys_um, dEs_MeV):
            n_electrons = electron_conversion(float(dE))
            if n_electrons <= 0:
                continue
            x_idx = int(np.floor((float(x_um) - patch_x0_um) / hi_res_grid_spacing_micron))
            y_idx = int(np.floor((float(y_um) - patch_y0_um) / hi_res_grid_spacing_micron))
            spread_electrons_to_patch(patch, x_idx, y_idx, n_electrons, kernel)

        # top-left hi-res indices where this patch lands in the full hi-res frame
        hi_x0 = int(np.floor(patch_x0_um / hi_res_grid_spacing_micron))
        hi_y0 = int(np.floor(patch_y0_um / hi_res_grid_spacing_micron))

        # downsample and add into detector
        _downsample_and_add_patch(H_detector, patch, hi_x0, hi_y0, r)

    if not apply_gain:
        if output_array_path:
            np.save(output_array_path, H_detector)
            print(f"Saved electrons-per-pixel array to {output_array_path}")
        return H_detector

    # Apply gain (e-/DN)
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

# -------- CLI --------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Spread electrons and optionally convert to DN from cosmic ray sim CSV."
    )
    parser.add_argument('--csvfile', type=str, required=True,
                        help='CSV file with energy loss events (microns, MeV)')
    parser.add_argument('--gain_txt', type=str, default=None,
                        help='Gain map .txt file (column 5 = gain e-/DN). Required if applying gain.')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional output .npy path (DN if applying gain; electrons if not).')
    # Processing variant selection (keep your blob method as default)
    parser.add_argument('--mode', choices=['blob', 'stream'], default='blob',
                        help="'blob' groups by PID; 'stream' reads CSV in chunks.")
    # Gain toggle (default True for backward-compat)
    gain_group = parser.add_mutually_exclusive_group()
    gain_group.add_argument('--apply-gain', dest='apply_gain', action='store_true',
                            help='Apply gain (default).')
    gain_group.add_argument('--no-apply-gain', dest='apply_gain', action='store_false',
                            help='Do not apply gain; output is electrons-per-pixel.')
    parser.set_defaults(apply_gain=True)

    args = parser.parse_args()

    if args.apply_gain and args.gain_txt is None:
        raise SystemExit("ERROR: --gain_txt is required when --apply-gain is set (default). "
                         "Use --no-apply-gain to skip gain.")

    if args.mode == 'blob':
        process_electrons_to_DN_by_blob(
            args.csvfile, gain_txt=args.gain_txt, output_array_path=args.output, apply_gain=args.apply_gain
        )
    else:
        process_electrons_to_DN(
            args.csvfile, gain_txt=args.gain_txt, output_array_path=args.output, apply_gain=args.apply_gain
        )
