#zincblende tight binding model

"""
Minimal tight-binding band structure for a zincblende crystal (e.g. GaAs)
using a 2-site / 1-orbital-per-site model on an fcc lattice.

This script:
  * Defines a 2x2 tight-binding Hamiltonian H(k) for zincblende:
        H(k) = [[eps_A, f(k)],
                [f*(k), eps_B]]

  * Evaluates the dispersion E_\pm(k) along a standard fcc high-symmetry path:
        Γ – X – W – K – Γ – L – U – W – L – K

  * Produces:
        (1) Band structure plot E(k) vs path coordinate
        (2) 2D energy map for a kx–ky slice (kz fixed)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


# -----------------------------
# Tight-binding model
# -----------------------------

@dataclass
class ZincblendeTBParams:
    """Parameters for the minimal zincblende tight-binding model."""
    a: float = 1.0          # lattice constant [arbitrary units]
    t: float = -1.0         # nearest-neighbor hopping [energy units]
    eps_A: float = 0.5      # onsite energy for sublattice A
    eps_B: float = -0.5     # onsite energy for sublattice B


def nearest_neighbor_vectors(a: float) -> np.ndarray:
    """
    Return the 4 nearest-neighbor vectors from A to B for a zincblende lattice.

    These are the vectors (a/4)*[±1, ±1, ±1] with an odd number of minus signs.
    (Same as diamond structure; zincblende is two different atoms on those sites.)
    """
    d = (a / 4.0) * np.array(
        [
            [1,  1,  1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=float,
    )
    return d  # shape (4, 3)


def f_k(kvec: np.ndarray, params: ZincblendeTBParams) -> complex:
    """
    Off-diagonal hopping function f(k) = t * sum_j exp(i k·d_j).

    Parameters
    ----------
    kvec : array-like, shape (3,)
        The k-vector in reciprocal space.
    params : ZincblendeTBParams
        Tight-binding parameters.

    Returns
    -------
    complex
        Complex hopping amplitude f(k).
    """
    d_vecs = nearest_neighbor_vectors(params.a)  # (4, 3)
    phases = np.exp(1j * d_vecs @ kvec)         # shape (4,)
    return params.t * np.sum(phases)


def band_energies(kvec: np.ndarray, params: ZincblendeTBParams) -> np.ndarray:
    """
    Compute the two band energies E_±(k) for the 2x2 zincblende TB Hamiltonian.

    E_± = (eps_A + eps_B)/2 ± sqrt( (Δ/2)^2 + |f(k)|^2 ), where Δ = eps_A - eps_B.
    """
    eps_A, eps_B = params.eps_A, params.eps_B
    delta = eps_A - eps_B
    f = f_k(kvec, params)
    center = 0.5 * (eps_A + eps_B)
    split = np.sqrt((delta * 0.5) ** 2 + np.abs(f) ** 2)
    return np.array([center - split, center + split])  # valence, conduction


# -----------------------------
# High-symmetry path in fcc BZ
# -----------------------------

@dataclass
class KPoint:
    label: str
    coord: np.ndarray  # in units of 2π/a * (kx, ky, kz)


def make_fcc_high_symmetry_points() -> List[KPoint]:
    """
    Define standard high-symmetry points in the fcc Brillouin zone
    in units of (2π/a)*(kx, ky, kz) using the conventional cubic basis.

    Common choices:
      Γ = (0, 0, 0)
      X = (0, 1, 0)
      L = (0.5, 0.5, 0.5)
      W = (1, 0.5, 0)
      K = (0.75, 0.75, 0)
      U = (1, 0.25, 0.25)
    """
    return [
        KPoint("Γ", np.array([0.0, 0.0, 0.0])),
        KPoint("X", np.array([0.0, 1.0, 0.0])),
        KPoint("W", np.array([1.0, 0.5, 0.0])),
        KPoint("K", np.array([0.75, 0.75, 0.0])),
        KPoint("Γ", np.array([0.0, 0.0, 0.0])),
        KPoint("L", np.array([0.5, 0.5, 0.5])),
        KPoint("U", np.array([1.0, 0.25, 0.25])),
        KPoint("W", np.array([1.0, 0.5, 0.0])),
        KPoint("L", np.array([0.5, 0.5, 0.5])),
        KPoint("K", np.array([0.75, 0.75, 0.0])),
    ]



def interpolate_k_path(
    kpoints: List[KPoint],
    n_points_per_segment: int = 60,
    a: float = 1.0,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build a k-path through the high-symmetry points.

    Parameters
    ----------
    kpoints : list of KPoint
        Ordered list of high-symmetry points defining the path.
    n_points_per_segment : int
        Number of interpolation points between each pair of k-points.
    a : float
        Lattice constant; used to scale coordinates as k = (2π/a)*coord.

    Returns
    -------
    k_list : (N, 3) array
        List of k-vectors along the path.
    labels : list of str
        Labels for the high-symmetry points in order.
    k_dist : (N,) array
        Cumulative distance along the path.
    """
    coords = np.array([kp.coord for kp in kpoints])
    labels = [kp.label for kp in kpoints]

    # Scale from fractional coords to absolute k (2π/a)
    factor = 2.0 * np.pi / a
    coords_abs = factor * coords

    segments = []
    for i in range(len(coords_abs) - 1):
        k_start = coords_abs[i]
        k_end = coords_abs[i + 1]
        # Include the start of the first segment, then avoid duplicating
        # the first point of subsequent segments.
        if i == 0:
            segment = np.linspace(k_start, k_end, n_points_per_segment, endpoint=False)
        else:
            segment = np.linspace(k_start, k_end, n_points_per_segment, endpoint=False)[1:]
        segments.append(segment)

    # Stack into a single k_list array
    k_list = np.vstack(segments)  # shape (N, 3)

    # Now compute cumulative distance along the path
    N = k_list.shape[0]
    k_dist = np.zeros(N)
    for i in range(1, N):
        dk = k_list[i] - k_list[i - 1]
        k_dist[i] = k_dist[i - 1] + np.linalg.norm(dk)

    return k_list, labels, k_dist

# -----------------------------
# Plotting helpers
# -----------------------------

def plot_band_structure(
    params: ZincblendeTBParams,
    n_points_per_segment: int = 60,
) -> None:
    """
    Compute and plot the band structure E(k) along fcc high-symmetry path.
    """
    kpoints = make_fcc_high_symmetry_points()
    k_list, labels, k_dist = interpolate_k_path(
        kpoints, n_points_per_segment=n_points_per_segment, a=params.a
    )

    # Compute the two bands along the path
    E_vals = np.array([band_energies(kvec, params) for kvec in k_list])
    E_valence = E_vals[:, 0]
    E_conduction = E_vals[:, 1]

    # Figure out tick positions: at each high-symmetry point
    tick_positions = []
    tick_labels = []
    # recompute k_dist at the exact endpoints for ticks
    # (a quick way is to walk through path segments again)
    # simpler hack: sample indices at boundaries:
    pts_per_seg = n_points_per_segment
    idx = 0
    for i, kp in enumerate(kpoints):
        if i == 0:
            idx = 0
        else:
            idx = i * (pts_per_seg - 1)
        tick_positions.append(k_dist[idx])
        tick_labels.append(kp.label)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_dist, E_valence, lw=1.5)
    ax.plot(k_dist, E_conduction, lw=1.5)

    # vertical lines at high-symmetry points
    for x in tick_positions:
        ax.axvline(x=x, color="k", linewidth=0.5, linestyle="--")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Energy (arb. units)")
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_title("Zincblende TB band structure (minimal 2-band model)")

    plt.tight_layout()
    plt.show()


def plot_2d_energy_map(
    params: ZincblendeTBParams,
    nk: int = 201,
    kz: float = 0.0,
    band_index: int = 0,
) -> None:
    """
    Plot a 2D kx–ky slice of one band as a density map.

    Parameters
    ----------
    params : ZincblendeTBParams
        Tight-binding parameters.
    nk : int
        Number of k-points in each dimension.
    kz : float
        Fixed kz value for the slice.
    band_index : int
        Which band to plot: 0 for valence, 1 for conduction.
    """
    # We'll take a simple square region in kx, ky for visualization
    # (not the exact fcc BZ shape, but useful to see dispersion).
    kmax = np.pi / params.a  # roughly half the conventional reciprocal vector
    kx_vals = np.linspace(-kmax, kmax, nk)
    ky_vals = np.linspace(-kmax, kmax, nk)

    E_map = np.zeros((nk, nk))

    for ix, kx in enumerate(kx_vals):
        for iy, ky in enumerate(ky_vals):
            kvec = np.array([kx, ky, kz])
            E = band_energies(kvec, params)[band_index]
            E_map[iy, ix] = E  # note (iy, ix) so ky is vertical axis

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        E_map,
        origin="lower",
        extent=[kx_vals[0], kx_vals[-1], ky_vals[0], ky_vals[-1]],
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Energy (arb. units)")

    band_str = "valence" if band_index == 0 else "conduction"
    ax.set_title(f"2D kx–ky slice (kz = {kz:.2f}) of {band_str} band")
    ax.set_xlabel(r"$k_x$ (1/a)")
    ax.set_ylabel(r"$k_y$ (1/a)")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    params = ZincblendeTBParams(
        a=1.0,       # lattice constant
        t=-1.0,      # hopping
        eps_A=0.5,   # onsite energy A
        eps_B=-0.5,  # onsite energy B
    )

    # 1) Band structure along high-symmetry cuts
    plot_band_structure(params, n_points_per_segment=80)

    # 2) 2D kx–ky energy map (valence band, kz=0)
    plot_2d_energy_map(params, nk=201, kz=0.0, band_index=0)
