# electron_spread.py

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from tqdm import tqdm

def electron_conversion(dE, f_eff=2.71, w=2.509):
    """Convert energy loss dE [MeV] to number of electrons."""
    if dE <= 0:
        return 0
    dE = dE*1e6 # MeV to eV
    mu_nb = dE / w
    p = 1.0 / f_eff
    if not (0 < p < 1):
        return 0
    r = mu_nb * (p / (1.0 - p))
    if r <= 0:
        return 0
    return nbinom(r, p).rvs()

def gaussian_sum_kernel(size=50, sigma=0.314, 
                       w_list=[0.17519, 0.53146, 0.29335], 
                       c_list=[0.4522, 0.8050, 1.4329]):
    """Return a (size x size) kernel with sum-of-3-Gaussians probabilities."""
    ax = np.arange(size) - size//2
    xx, yy = np.meshgrid(ax, ax)
    rr2 = xx**2 + yy**2
    kernel = np.zeros_like(xx, dtype=float)
    for w, c in zip(w_list, c_list):
        s = sigma * c
        norm = (4 * np.pi**2 * np.sqrt(c**2 * sigma**2))
        kernel += w * np.exp(-rr2 / (8 * c**2 * np.pi**2 * sigma**2)) / norm
    kernel = np.maximum(kernel, 0)
    kernel /= kernel.sum()
    return kernel

def spread_electrons_to_patch(H, x_pix, y_pix, n_electrons, kernel):
    """Spread n_electrons in H at (x_pix, y_pix) using multinomial over patch, handling edges."""
    size = kernel.shape[0]
    offset = size // 2

    x0 = x_pix - offset
    x1 = x_pix + offset + 1
    y0 = y_pix - offset
    y1 = y_pix + offset + 1

    patch_x0 = max(0, x0)
    patch_y0 = max(0, y0)
    patch_x1 = min(H.shape[1], x1)
    patch_y1 = min(H.shape[0], y1)

    kx0 = patch_x0 - x0
    ky0 = patch_y0 - y0
    kx1 = kx0 + (patch_x1 - patch_x0)
    ky1 = ky0 + (patch_y1 - patch_y0)

    patch_kernel = kernel[ky0:ky1, kx0:kx1]
    patch_kernel = np.maximum(patch_kernel, 0)
    patch_kernel /= patch_kernel.sum() if patch_kernel.sum() > 0 else 1

    draws = np.random.default_rng().multinomial(n_electrons, patch_kernel.ravel())
    for count, (dy, dx) in zip(draws, np.argwhere(np.ones_like(patch_kernel))):
        if count > 0:
            i = patch_y0 + dy
            j = patch_x0 + dx
            H[i, j] += count

def process_electrons_to_DN(
        csvfile,
        gain_txt,
        det_pixels_lo=4096,
        pixel_size_hi=0.1,
        pixel_size_lo=10.0,
        kernel_size_hi=50,
        chunksize=100_000,
        sigma=0.314,
        output_DN_path=None
    ):
    """Process CSV energy loss events and output a DN map array (optionally saves as .npy)"""
    kernel = gaussian_sum_kernel(size=kernel_size_hi, sigma=sigma)
    H_detector = np.zeros((det_pixels_lo, det_pixels_lo), dtype=float)

    for chunk in pd.read_csv(csvfile, sep=',', chunksize=chunksize):
        xs = chunk['x'].to_numpy()
        ys = chunk['y'].to_numpy()
        dEs = chunk['dE'].to_numpy()
        for x, y, dE in tqdm(zip(xs, ys, dEs), desc="Processing events"):
            n_electrons = electron_conversion(dE)
            if n_electrons > 0:
                # Center of high-res patch (event location)
                x_hi = int(np.floor(x / pixel_size_hi))
                y_hi = int(np.floor(y / pixel_size_hi))
                half_patch = kernel_size_hi // 2

                # Build patch
                patch = np.zeros((kernel_size_hi, kernel_size_hi), dtype=float)
                spread_electrons_to_patch(patch, half_patch, half_patch, n_electrons, kernel)

                # Now, for each high-res pixel with electrons, assign to correct DN pixel
                for dy in range(kernel_size_hi):
                    for dx in range(kernel_size_hi):
                        n = patch[dy, dx]
                        if n == 0:
                            continue
                        # Absolute high-res position in microns
                        x_abs = (x_hi + dx - half_patch) * pixel_size_hi
                        y_abs = (y_hi + dy - half_patch) * pixel_size_hi
                        # Corresponding DN pixel (low-res)
                        x_lo = int(np.floor(x_abs / pixel_size_lo))
                        y_lo = int(np.floor(y_abs / pixel_size_lo))
                        # Add to DN map (check bounds!)
                        if 0 <= x_lo < det_pixels_lo and 0 <= y_lo < det_pixels_lo:
                            H_detector[y_lo, x_lo] += n

    # Gain map
    gain_array = np.loadtxt(gain_txt)[:, 5].reshape((32, 32))
    supercell_size = det_pixels_lo // 32
    gain_map = np.kron(gain_array, np.ones((supercell_size, supercell_size)))
    assert gain_map.shape == H_detector.shape

    gain_map_safe = np.where(gain_map > 0, gain_map, np.nan)
    H_detector_DN = H_detector / gain_map_safe

    if output_DN_path:
        np.save(output_DN_path, H_detector_DN)
        print(f"Saved DN array to {output_DN_path}")

    return H_detector_DN

def process_pid_electrons_zoom(
        csvfile,
        pid,
        delta_pids,
        x_center, y_center,         # Center in physical units, e.g., microns
        region_size_um = 20,         # Region width/height in microns (e.g., 20 µm)
        pixel_size_hi=0.1,         # Hi-res pixel size
        kernel_size_hi=50,
        sigma=0.314,
        gain=None,                 # Optional: can skip gain for zoom-in, or load local patch if needed
        chunksize=100_000
    ):
    """
    Returns a hi-res array (mini-image) centered on (x_center, y_center)
    for the selected PID and its deltas, for interactive popup display.
    """
    kernel = gaussian_sum_kernel(size=kernel_size_hi, sigma=sigma)
    n_pix = int(region_size_um / pixel_size_hi)
    H_zoom = np.zeros((n_pix, n_pix), dtype=float)
    
    x0_um = x_center - region_size_um / 2
    y0_um = y_center - region_size_um / 2
    
    wanted_pids = set([pid] + list(delta_pids))

    for chunk in pd.read_csv(csvfile, sep=',', chunksize=chunksize):
        if 'PID' not in chunk.columns:
            raise ValueError("CSV must have PID column.")
        mask = chunk['PID'].isin(wanted_pids)
        filtered = chunk[mask]
        xs = filtered['x'].to_numpy()
        ys = filtered['y'].to_numpy()
        dEs = filtered['dE'].to_numpy()
        for x, y, dE in zip(xs, ys, dEs):
            # Check if the event falls inside the zoom region
            if (x0_um <= x < x0_um + region_size_um) and (y0_um <= y < y0_um + region_size_um):
                n_electrons = electron_conversion(dE)
                if n_electrons > 0:
                    x_pix = int((x - x0_um) / pixel_size_hi)
                    y_pix = int((y - y0_um) / pixel_size_hi)
                    spread_electrons_to_patch(H_zoom, x_pix, y_pix, n_electrons, kernel)

    # (Optional) Gain correction for this patch won't matter if we don't use a legend for the pop up
    # Could be added if we need it

    return H_zoom

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spread electrons and convert to DNs from cosmic ray sim CSV.")
    parser.add_argument('--csvfile', type=str, required=True, help='CSV file with energy loss data')
    parser.add_argument('--gain_txt', type=str, required=True, help='Gain map .txt file')
    parser.add_argument('--output', type=str, default=None, help='Optional output .npy path for DN array')
    args = parser.parse_args()
    process_electrons_to_DN(args.csvfile, args.gain_txt, output_DN_path=args.output)
