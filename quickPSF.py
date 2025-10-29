# point_source_ramp.py
from __future__ import annotations
from matplotlib.patches import Circle
import numpy as np
from scipy.ndimage import shift as nd_shift  # optional subpixel shift
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Dict, Optional, Literal
from math import isfinite

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class PointSource:
    """
    Simple point source model for DN/frame injection.

    Parameters
    ----------
    x : float
        Sub-pixel x position (0 at left edge, increases to the right).
    y : float
        Sub-pixel y position (0 at top edge, increases downward).
    rate_dn_per_frame : float
        Mean source photo-signal rate [DN/frame] integrated over the *entire* PSF.
        The up-the-ramp signal per pixel grows ~ linearly with t * rate.
    fwhm_px : float
        PSF FWHM in pixels for a circularly symmetric Gaussian.
    jitter_frac : float, optional
        Per-frame Gaussian fractional jitter on the source rate (e.g., 0.02 = 2% rms).
        This simulates scintillation / throughput wiggles.
    start_frame : int, optional
        Frame index at which the source starts ramping (0-based). Before this, contributes 0.
    """
    x: float
    y: float
    rate_dn_per_frame: float
    fwhm_px: float
    jitter_frac: float = 0.0
    start_frame: int = 0


# -----------------------------
# PSF & rendering utilities
# -----------------------------
def gaussian_psf_kernel(
    shape: Tuple[int,int],
    fwhm_px: float,
    center: Tuple[float, float],
) -> np.ndarray:
    """
    Build a normalized Gaussian PSF kernel on a size×size grid, centered at a sub-pixel position.

    Notes
    -----
    - center is (x, y) in pixel coordinates within the grid [0..size-1].
    - Kernel sums to 1. You can multiply by total DN for that frame to distribute across pixels.
    """
    ny, nx = shape
    sigma = fwhm_px / (2.0*np.sqrt(2.0*np.log(2.0)) + 1e-12)
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = xx - center[0]
    dy = yy - center[1]
    ker = np.exp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma + 1e-20))
    s = ker.sum()
    if s > 0:
        ker /= s
    return ker


def circular_aperture_sum(img: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Sum pixels whose centers fall within distance r of (cx, cy)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy)
    return img[dist <= r].sum()


# -----------------------------
# Simulation
# -----------------------------
def simulate_point_source_ramp(
    sources: Iterable[PointSource],
    grid_size: Tuple[int, int] = (4, 4),
    n_frames: int = 20,
    read_noise_sigma_dn: float = 1.5,
    background_dn_per_frame: float = 0.0,
    background_jitter_frac: float = 0.0,
    saturation_dn: Optional[float] = None,
    seed: Optional[int] = 1234,
) -> Dict[str, np.ndarray]:
    """
    Simulate an up-the-ramp sequence with injected Gaussian-PSF point sources.

    Parameters
    ----------
    sources : iterable of PointSource
        One or more source definitions.
    grid_size : (int, int)
        (ny, nx) grid size. Default (4,4) = 16 pixels.
    n_frames : int
        Number of frames (time samples).
    read_noise_sigma_dn : float
        Per-frame Gaussian read noise sigma [DN] per pixel, added *after* ramping.
    background_dn_per_frame : float
        Mean uniform background slope [DN/frame] across the whole array (e.g., sky/thermal).
    background_jitter_frac : float
        Fractional frame-to-frame Gaussian jitter on the background slope.
    saturation_dn : float or None
        If set, values are clipped to this DN (simulating well/saturation).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    out : dict
        {
          "cube": (n_frames, ny, nx) float32 array, cumulative DN per frame,
          "per_frame_flux_map": (n_frames, ny, nx) incremental DN added each frame (before read noise),
          "source_truth": list of dicts describing sources,
          "times": (n_frames,) array of frame indices,
        }
    """
    rng = np.random.default_rng(seed)
    ny, nx = grid_size
    times = np.arange(n_frames, dtype=float)  # 0,1,2,...

    # Prepare static PSFs for each source (on full grid, centered at given sub-pixel)
    src_psfs = []
    src_meta = []
    for s in sources:
        psf = gaussian_psf_kernel(shape=(ny, nx), fwhm_px=s.fwhm_px, center=(s.x, s.y))
        src_psfs.append(psf)
        src_meta.append(asdict(s))

    # Container for cumulative signal per frame and incremental frame signal
    cube = np.zeros((n_frames, ny, nx), dtype=np.float32)
    per_frame_flux_map = np.zeros_like(cube)

    # Loop frames
    running = np.zeros((ny, nx), dtype=np.float64)  # cumulative DN (no read noise yet)
    for t in range(n_frames):
        # Background slope for this frame increment
        bg_rate = background_dn_per_frame * (1.0 + background_jitter_frac * rng.normal())
        frame_increment = np.full((ny, nx), bg_rate, dtype=np.float64)

        # Add sources' frame increments
        for psf, s in zip(src_psfs, sources):
            if t >= s.start_frame:
                # Per-frame rate with jitter
                rate_t = s.rate_dn_per_frame * (1.0 + s.jitter_frac * rng.normal())
                frame_increment += rate_t * psf

        per_frame_flux_map[t] = frame_increment.astype(np.float32)

        # Accumulate (up-the-ramp)
        running += frame_increment

        # Apply saturation (optional)
        if saturation_dn is not None:
            running = np.minimum(running, saturation_dn)

        # Add read noise (frame by frame, after charge integration)
        noisy_frame = running + rng.normal(scale=read_noise_sigma_dn, size=running.shape)
        cube[t] = noisy_frame.astype(np.float32)

    return {
        "cube": cube,
        "per_frame_flux_map": per_frame_flux_map,
        "source_truth": src_meta,
        "times": times,
    }


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_pixel_timeseries(
    cube: np.ndarray,
    coords: List[Tuple[int, int]],
    title: str = "Per-pixel Up-the-Ramp Signals",
) -> None:
    """
    Plot DN vs frame for specific (y,x) pixel coordinates.
    """
    n_frames = cube.shape[0]
    t = np.arange(n_frames)
    plt.figure(figsize=(7,5))
    for (y, x) in coords:
        plt.plot(t, cube[:, y, x], marker='o', lw=1, ms=3, label=f"px ({y},{x})")
    plt.xlabel("Frame")
    plt.ylabel("DN")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_aperture_timeseries(
    cube: np.ndarray,
    centers: List[Tuple[float, float]],
    radius_px: float = 1.25,
    title: str = "Aperture-summed Up-the-Ramp Signals",
) -> None:
    """
    Plot DN vs frame for simple circular apertures (sum of pixels whose centers fall within radius).
    """
    n_frames, ny, nx = cube.shape
    t = np.arange(n_frames)
    plt.figure(figsize=(7,5))
    for (cx, cy) in centers:
        vals = np.array([circular_aperture_sum(cube[i], cx, cy, radius_px) for i in range(n_frames)])
        plt.plot(t, vals, marker='o', lw=1, ms=3, label=f"ap (cx={cx:.2f}, cy={cy:.2f}, r={radius_px})")
    plt.xlabel("Frame")
    plt.ylabel("DN (aperture sum)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_final_frame_with_apertures(
    cube: np.ndarray,
    centers: List[Tuple[float, float]],
    radius_px: float = 1.25,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "magma",
    annotate: bool = True,
) -> None:
    """
    Show the final cumulative frame (integrated DN) with red aperture overlays.

    Parameters
    ----------
    cube : array, shape (n_frames, ny, nx)
        Up-the-ramp data (cumulative DN per frame).
    centers : list[(x, y)]
        Sub-pixel aperture centers (x rightward, y downward).
    radius_px : float
        Aperture radius in pixels.
    vmin, vmax : float or None
        Color scale limits. If None, autoscale from data.
    cmap : str
        Matplotlib colormap.
    annotate : bool
        If True, add small coordinate labels near each circle.
    """
    img = cube[-1].astype(float)  # final cumulative signal
    ny, nx = img.shape

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(img, origin="upper", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="DN")

    for (cx, cy) in centers:
        ax.add_patch(Circle((cx, cy), radius_px, fill=False, lw=2.0, ec="red"))
        ax.plot(cx, cy, marker="x", ms=6, mec="red", mfc="none", mew=1.5)
        if annotate:
            ax.text(cx + 0.1, cy - 0.15, f"({cx:.2f},{cy:.2f})", color="red", fontsize=8)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Final cumulative frame with aperture overlays")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# --- BRIDGE: from your widget selection to injection into the PSF sim ----
from typing import Dict

def inject_patch(
    sim: Dict[str, np.ndarray],
    blob_patch: np.ndarray,
    target_frame: int,
    insert_center_yx: Tuple[float, float],
    op: Literal["add", "replace"] = "add",
    scale: float = 1.0,
    saturation_dn: Optional[float] = None,
    allow_crop: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Inject a 2D patch into the simulated datacube *at a specific frame* so that its DN
    persists into all subsequent frames (as a real instantaneous hit would).

    This updates both:
      - sim["per_frame_flux_map"][target_frame]  (incremental),
      - sim["cube"][target_frame:]               (cumulative).

    Parameters
    ----------
    sim : dict
        Output of simulate_point_source_ramp(...).
        Must contain "cube" [T,H,W] and "per_frame_flux_map" [T,H,W].
    blob_patch : array, shape (ph, pw)
        Patch values in DN to insert.
    target_frame : int
        Frame index where the hit occurs.
    insert_center_yx : (float, float)
        Desired center position in the sim grid (y, x). Rounded to nearest pixel for alignment.
    op : {"add","replace"}
        "add" (default) adds the patch to existing values; "replace" overwrites in that ROI for the
        target frame increment and subsequent cumulative frames.
    scale : float
        Multiplicative scale for the patch (e.g., to convert units or tune strength).
    saturation_dn : float or None
        Optional clipping after updating cumulative frames.
    allow_crop : bool
        If False, raises if patch would go out-of-bounds; if True, crops patch at edges.

    Returns
    -------
    sim : dict
        The same dict with arrays updated in-place and returned for convenience.
    """
    cube = sim["cube"]
    inc = sim["per_frame_flux_map"]

    T, H, W = cube.shape
    if not (0 <= target_frame < T):
        raise IndexError("target_frame out of range")

    ph, pw = blob_patch.shape
    cy = int(round(insert_center_yx[0]))
    cx = int(round(insert_center_yx[1]))
    hy = ph // 2
    hx = pw // 2

    y0, y1 = cy - hy, cy - hy + ph
    x0, x1 = cx - hx, cx - hx + pw

    # Compute safe slices
    ys0, ys1 = max(y0, 0), min(y1, H)
    xs0, xs1 = max(x0, 0), min(x1, W)

    if not allow_crop and (ys0 > y0 or ys1 < y1 or xs0 > x0 or xs1 < x1):
        raise ValueError("Patch would exceed bounds and allow_crop=False.")

    # Crop patch if needed
    py0, py1 = ys0 - y0, ys1 - y0
    px0, px1 = xs0 - x0, xs1 - x0
    patch = (scale * blob_patch[py0:py1, px0:px1]).astype(np.float32)

    if op == "replace":
        old = inc[target_frame, ys0:ys1, xs0:xs1].copy()
        delta = patch - old
        inc[target_frame, ys0:ys1, xs0:xs1] = patch
        cube[target_frame:, ys0:ys1, xs0:xs1] += delta[None, :, :]
    else:  # "add"
        inc[target_frame, ys0:ys1, xs0:xs1] += patch
        cube[target_frame:, ys0:ys1, xs0:xs1] += patch[None, :, :]


    # Add to incremental map at target frame
    inc[target_frame, ys0:ys1, xs0:xs1] += patch

    # Add to cumulative frames target_frame..end
    cube[target_frame:, ys0:ys1, xs0:xs1] += patch[None, :, :]

    # Optional saturation clip
    if saturation_dn is not None:
        cube[target_frame:, ys0:ys1, xs0:xs1] = np.minimum(
            cube[target_frame:, ys0:ys1, xs0:xs1], saturation_dn
        )

    return sim



def extract_patch(
    data: np.ndarray,
    frame_idx: int,
    x_center: int,
    y_center: int,
    half_size: int,
    baseline: Literal["none","min","median","mean"] = "median",
) -> np.ndarray:
    """
    Compatible with your widget's outputs.
    Works for 2D (single frame) or 3D (T,H,W) arrays.

    Returns a (2*half_size) x (2*half_size) patch (same slicing as your widget).
    """
    if data.ndim == 3:
        T, H, W = data.shape
        if not (0 <= frame_idx < T):
            raise IndexError(f"frame_idx {frame_idx} out of 0..{T-1}")
        a2d = data[frame_idx]
    elif data.ndim == 2:
        H, W = data.shape
        a2d = data
        frame_idx = 0
    else:
        raise ValueError("data must be 2D or 3D")

    # your widget uses y0:y1 and x0:x1 with no +1 on the end bound
    y0, y1 = y_center - half_size, y_center + half_size
    x0, x1 = x_center - half_size, x_center + half_size

    # bounds-safe crop (pad with zeros if near edges)
    ys0, ys1 = max(y0, 0), min(y1, H)
    xs0, xs1 = max(x0, 0), min(x1, W)

    patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    patch[(ys0 - y0):(ys1 - y0), (xs0 - x0):(xs1 - x0)] = a2d[ys0:ys1, xs0:xs1]

    if baseline != "none":
        if baseline == "median":
            patch -= np.median(patch)
        elif baseline == "mean":
            patch -= np.mean(patch)
        elif baseline == "min":
            patch -= patch.min()
        else:
            raise ValueError("baseline must be one of {'none','min','median','mean'} or 'none'")
    return patch


def inject_from_widget_selection(
    sim: dict,
    data: np.ndarray,
    frame_idx: int,
    x_center: int,
    y_center: int,
    half_size: int,
    target_frame_in_sim: int,
    insert_center_yx_in_sim: Tuple[float, float],
    scale_dn: float = 1.0,
    baseline: Literal["none","min","median","mean"] = "median",
    op: Literal["add","replace"] = "add",
    saturation_dn: Optional[float] = None,
    allow_crop: bool = True,
    subpixel_shift: Optional[Tuple[float, float]] = None,  # (dy, dx) applied to patch before injection
) -> dict:
    """
    1) Extract CR patch from the big numpy array using the widget's ROI.
    2) (Optional) apply a sub-pixel shift to patch.
    3) Inject patch into the PSF simulation at a given sim frame and sim location.

    Notes
    -----
    - This updates BOTH sim['per_frame_flux_map'][target_frame_in_sim] and
      sim['cube'][target_frame_in_sim:].
    - subpixel_shift is applied in patch coordinates: positive dy shifts *down*, dx shifts *right*.
    """

    blob = extract_patch(
        data=data,
        frame_idx=frame_idx,
        x_center=x_center,
        y_center=y_center,
        half_size=half_size,
        baseline=baseline,
    )

    # Optional subpixel shift (keeps shape; order=1 is bilinear)
    if subpixel_shift is not None:
        dy, dx = subpixel_shift
        if all(isfinite(v) for v in (dy, dx)):
            blob = nd_shift(blob, shift=(dy, dx), order=1, mode="nearest")

    # Scale (e.g., energy→DN) if needed
    blob = (scale_dn * blob).astype(np.float32)

    # Inject into the small-grid sim
    return inject_patch(
        sim=sim,
        blob_patch=blob,
        target_frame=target_frame_in_sim,
        insert_center_yx=insert_center_yx_in_sim,
        op=op,
        scale=1.0,
        saturation_dn=saturation_dn,
        allow_crop=allow_crop,
    )




# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Define a 4x4 grid with ~20 frames and two example sources
    sources = [
        PointSource(x=5.5, y=6.5, rate_dn_per_frame=50.0, fwhm_px=1.7, jitter_frac=0.03, start_frame=0),
        PointSource(x=4.2, y=11.8, rate_dn_per_frame=20.0, fwhm_px=0.9, jitter_frac=0.02, start_frame=6),
        PointSource(x=10.4, y=14.3, rate_dn_per_frame=100.0, fwhm_px=3.9, jitter_frac=0.06, start_frame=15)
    ]

    sim = simulate_point_source_ramp(
        sources=sources,
        grid_size=(20, 20),
        n_frames=100,
        read_noise_sigma_dn=2.0,
        background_dn_per_frame=0.5,
        background_jitter_frac=0.05,
        saturation_dn=6e4,
        seed=42,
    )

    cube = sim["cube"]
    times = sim["times"]

    # Plot example per-pixel time series (pick the two brightest-ish pixels by inspection)
    plot_pixel_timeseries(cube, coords=[(5,6), (2,2), (3,12), (10,13)])

    # Plot aperture-summed curves around the nominal source centers
    centers = [(s["x"], s["y"]) for s in sim["source_truth"]]
    plot_aperture_timeseries(cube, centers=centers, radius_px=1.25)
    plot_final_frame_with_apertures(cube, centers=centers, radius_px=1.25)
    
    # Suppose your widget picked this event in the big array:
    # Option A: your actual large array on disk
    sim_data = np.load("Outputs/Sample Outputs/60sec_sim_new_20250801_dn_array_patchByBlob.npy")  # 2D or 3D
    sim_data /= sim_data.max()          # normalize to [0,1]
    sim_data *= 2e3                     # rescale so brightest CR ≈ 2,000 DN

    frame_idx  = 0          # or 61, etc., from your UI
    x_center   = 1420
    y_center   = 990
    half_size  = 24         # same as your widget
    baseline   = "median"   # strip local background around the CR
    scale_dn   = 1.0        # set if you need energy->DN conversion

    # In the PSF sim, choose when & where to inject
    target_frame_in_sim      = 25
    insert_center_yx_in_sim  = (12.0, 12.0)   # (y, x) in the sim grid; can be fractional
    # optional tiny alignment tweak:
    subpixel_shift = (0.0, 0.0)

    sim = inject_from_widget_selection(
        sim=sim,
        data=sim_data,                    # or exp_data_cube
        frame_idx=frame_idx,
        x_center=x_center,
        y_center=y_center,
        half_size=half_size,
        target_frame_in_sim=target_frame_in_sim,
        insert_center_yx_in_sim=insert_center_yx_in_sim,
        scale_dn=scale_dn,
        baseline=baseline,
        op="add",
        saturation_dn=None,
        allow_crop=True,
        subpixel_shift=subpixel_shift,
    )

    # Reuse your existing plots to confirm:
    centers = [(s["x"], s["y"]) for s in sim["source_truth"]]
    plot_final_frame_with_apertures(sim["cube"], centers=centers, radius_px=1.25)
    plot_aperture_timeseries(sim["cube"], centers=[insert_center_yx_in_sim], radius_px=2.0,
                            title="Aperture light curve at injected CR location")

