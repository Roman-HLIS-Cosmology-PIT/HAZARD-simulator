"""
hgcdte_hptb_model.py

Effective hyperbolic tight-binding / k·p model for HgCdTe,
following Krishnamurthy et al. (1995) for the conduction band
dispersion:

    E_c(k; T) = E_g(T) + [ sqrt(γ(T) * k^2 + c(T)^2) - c(T) ]

with (E_g, γ, c) taken from their Table II for a sample HgCdTe
alloy. See:
  S. Krishnamurthy et al., J. Electron. Mater. 24, 1121 (1995).

This script:
  * Interpolates E_g(T), γ(T), c(T) from the tabulated values.
  * Builds a high-symmetry k-path in the fcc Brillouin zone.
  * Plots valence + conduction band structure along that path.
  * Plots a 2D kx–ky slice of the conduction band as a density map.

You can later replace the (T, E_g, γ, c) arrays with those
appropriate for x = 0.44 once you have them.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# ------------------------------------------------------------
# Constants / units
# ------------------------------------------------------------

# We'll treat k in units of [1/Å], and energies in [eV].
# Handy constant for a parabolic valence band:
# ħ^2 / (2 m_e) ≈ 3.81 eV·Å^2
HBAR2_OVER_2ME_EV_A2 = 3.81


# ------------------------------------------------------------
# Parameters from Krishnamurthy et al. (Table II)
# (example alloy in the paper; swap once you have x=0.44 data)
# ------------------------------------------------------------

# Temperatures in K
T_TABLE = np.array([
      1.0,  10.0,  20.0,  30.0,  40.0,  50.0,  60.0,  70.0,
     80.0,  90.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0,
    400.0, 450.0, 500.0, 550.0, 600.0
])

# Band gap E_g(T) in meV from Table II
EG_ME_V_TABLE = np.array([
    113.60, 112.67, 112.56, 114.44, 117.15, 120.42, 123.96,
    127.65, 131.44, 135.28, 139.17, 158.85, 178.73, 198.68,
    218.66, 238.65, 258.66, 278.67, 298.69, 318.71, 338.74
])

# γ(T) in eV·Å^2 (dimension inferred so that γ k^2 has units of eV)
GAMMA_EVA2_TABLE = np.array([
    47.7656, 47.7553, 47.7169, 47.6421, 47.5582, 47.4461,
    47.3310, 47.2091, 47.0821, 46.9418, 46.7964, 46.1544,
    45.5930, 45.2441, 45.1167, 45.3460, 46.0193, 47.3751,
    49.8338, 54.0581, 61.7125
])

# c(T) in eV
C_EV_TABLE = np.array([
    0.0588, 0.0588, 0.0592, 0.0598, 0.0607, 0.0615, 0.0624,
    0.0634, 0.0643, 0.0653, 0.0662, 0.0712, 0.0767, 0.0832,
    0.0908, 0.1000, 0.1115, 0.1263, 0.1468, 0.1763, 0.2238
])


@dataclass
class HgCdTeHyperbolicParams:
    """
    Effective parameters for the hyperbolic conduction-band model
    at a given temperature T.

    Attributes
    ----------
    T : float
        Temperature [K]
    Eg_eV : float
        Band gap [eV]
    gamma_eVA2 : float
        Hyperbolic γ parameter [eV·Å^2]
    c_eV : float
        Hyperbolic c parameter [eV]
    m_h_over_me : float
        Hole effective mass ratio (for a simple parabolic valence band)
    """
    T: float
    Eg_eV: float
    gamma_eVA2: float
    c_eV: float
    m_h_over_me: float = 0.30  # crude guess; adjust when you have data


def interpolate_hgcdte_params(T: float) -> HgCdTeHyperbolicParams:
    """
    Interpolate Eg(T), gamma(T), c(T) from the Krishnamurthy table.

    Parameters
    ----------
    T : float
        Temperature [K]

    Returns
    -------
    HgCdTeHyperbolicParams
        Interpolated parameters at temperature T.
    """
    # Clamp T to the table range to avoid extrapolation surprises.
    T_clamped = np.clip(T, T_TABLE.min(), T_TABLE.max())

    Eg_meV = np.interp(T_clamped, T_TABLE, EG_ME_V_TABLE)
    gamma_eVA2 = np.interp(T_clamped, T_TABLE, GAMMA_EVA2_TABLE)
    c_eV = np.interp(T_clamped, T_TABLE, C_EV_TABLE)

    Eg_eV = Eg_meV / 1000.0  # convert meV -> eV

    return HgCdTeHyperbolicParams(
        T=T_clamped,
        Eg_eV=Eg_eV,
        gamma_eVA2=gamma_eVA2,
        c_eV=c_eV,
        m_h_over_me=0.30,  # you can refine this later
    )


# ------------------------------------------------------------
# Band-structure formulas
# ------------------------------------------------------------

def conduction_energy(kvec: np.ndarray, p: HgCdTeHyperbolicParams) -> float:
    """
    Conduction band energy E_c(k; T) [eV] following
    Krishnamurthy's hyperbolic form:

        E_c(k; T) = E_g(T) + [ sqrt(γ(T) * k^2 + c(T)^2) - c(T) ],

    where k = |kvec| in units of [1/Å].

    Parameters
    ----------
    kvec : array-like, shape (3,)
        k-vector in reciprocal space [1/Å].
    p : HgCdTeHyperbolicParams
        Effective parameters at temperature T.

    Returns
    -------
    float
        Conduction band energy [eV], measured from the valence band maximum.
    """
    k = np.linalg.norm(kvec)
    return p.Eg_eV + (np.sqrt(p.gamma_eVA2 * k * k + p.c_eV**2) - p.c_eV)


def valence_energy(kvec: np.ndarray, p: HgCdTeHyperbolicParams) -> float:
    """
    Simple parabolic valence band:

        E_v(k; T) = - (ħ^2 / (2 m_h)) k^2,

    with m_h = m_h_over_me * m_e. This is a placeholder; the
    conduction band is the main physically-calibrated piece here.

    Parameters
    ----------
    kvec : array-like, shape (3,)
        k-vector in reciprocal space [1/Å].
    p : HgCdTeHyperbolicParams
        Effective parameters at temperature T.

    Returns
    -------
    float
        Valence band energy [eV], with E_v(0) = 0.
    """
    k = np.linalg.norm(kvec)
    alpha = HBAR2_OVER_2ME_EV_A2 / p.m_h_over_me  # eV·Å^2
    return -alpha * k * k


def band_energies(kvec: np.ndarray, p: HgCdTeHyperbolicParams) -> np.ndarray:
    """
    Return [E_v(k), E_c(k)] as a 2-element array [eV].

    Parameters
    ----------
    kvec : array-like, shape (3,)
    p : HgCdTeHyperbolicParams

    Returns
    -------
    np.ndarray, shape (2,)
        [valence_energy, conduction_energy].
    """
    Ev = valence_energy(kvec, p)
    Ec = conduction_energy(kvec, p)
    return np.array([Ev, Ec])


# ------------------------------------------------------------
# High-symmetry path (fcc)
# ------------------------------------------------------------

@dataclass
class KPoint:
    label: str
    coord: np.ndarray  # fractional coordinates in units of (2π/a)


def make_fcc_high_symmetry_points() -> List[KPoint]:
    """
    Define standard high-symmetry points in the fcc Brillouin zone
    in conventional cubic coordinates (fractional of 2π/a).

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
    a: float = 6.5,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build a k-path through the high-symmetry points.

    Parameters
    ----------
    kpoints : list of KPoint
    n_points_per_segment : int
        Number of interpolation points per segment.
    a : float
        Lattice constant [Å]; k is scaled as (2π/a) * coord.

    Returns
    -------
    k_list : (N, 3) array
        k-vectors along the path [1/Å].
    labels : list of str
        High-symmetry point labels (for x-axis tick labels).
    k_dist : (N,) array
        Cumulative distance along the path [1/Å].
    """
    coords = np.array([kp.coord for kp in kpoints])
    labels = [kp.label for kp in kpoints]

    factor = 2.0 * np.pi / a
    coords_abs = factor * coords  # [1/Å]

    segments = []
    for i in range(len(coords_abs) - 1):
        k_start = coords_abs[i]
        k_end = coords_abs[i + 1]
        if i == 0:
            segment = np.linspace(k_start, k_end, n_points_per_segment, endpoint=False)
        else:
            segment = np.linspace(k_start, k_end, n_points_per_segment, endpoint=False)[1:]
        segments.append(segment)

    k_list = np.vstack(segments)  # (N, 3)

    # cumulative distance along path
    N = k_list.shape[0]
    k_dist = np.zeros(N)
    for i in range(1, N):
        dk = k_list[i] - k_list[i - 1]
        k_dist[i] = k_dist[i - 1] + np.linalg.norm(dk)

    return k_list, labels, k_dist


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_band_structure(
    params: HgCdTeHyperbolicParams,
    n_points_per_segment: int = 80,
    a: float = 6.5,
) -> None:
    """
    Compute and plot band structure along an fcc high-symmetry path.

    Parameters
    ----------
    params : HgCdTeHyperbolicParams
        Effective parameters at the chosen temperature.
    n_points_per_segment : int
        Sampling density per segment.
    a : float
        Lattice constant [Å].
    """
    kpoints = make_fcc_high_symmetry_points()
    k_list, labels, k_dist = interpolate_k_path(
        kpoints, n_points_per_segment=n_points_per_segment, a=a
    )

    E_vals = np.array([band_energies(k, params) for k in k_list])
    E_valence = E_vals[:, 0]
    E_conduction = E_vals[:, 1]

    # tick positions at high-symmetry points
    pts_per_seg = n_points_per_segment
    tick_positions = []
    tick_labels = []
    for i, kp in enumerate(kpoints):
        if i == 0:
            idx = 0
        else:
            idx = i * (pts_per_seg - 1)
        idx = min(idx, len(k_dist) - 1)
        tick_positions.append(k_dist[idx])
        tick_labels.append(kp.label)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_dist, E_valence, lw=1.5, label="Valence (parabolic)")
    ax.plot(k_dist, E_conduction, lw=1.5, label="Conduction (hyperbolic)")

    for x in tick_positions:
        ax.axvline(x=x, color="k", linewidth=0.5, linestyle="--")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Energy [eV]")
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_title(
        f"HgCdTe hyperbolic band model at T = {params.T:.1f} K\n"
        r"$E_c(k) = E_g + (\sqrt{\gamma k^2 + c^2} - c)$"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_2d_conduction_map(
    params: HgCdTeHyperbolicParams,
    nk: int = 201,
    kz: float = 0.0,
    kmax: float = 0.1,
) -> None:
    """
    Plot a 2D kx–ky slice of the conduction band as a density map.

    Parameters
    ----------
    params : HgCdTeHyperbolicParams
    nk : int
        Number of grid points in each direction.
    kz : float
        Fixed kz [1/Å].
    kmax : float
        Maximum |kx| and |ky| sampled [1/Å].
    """
    kx_vals = np.linspace(-kmax, kmax, nk)
    ky_vals = np.linspace(-kmax, kmax, nk)

    E_map = np.zeros((nk, nk))

    for ix, kx in enumerate(kx_vals):
        for iy, ky in enumerate(ky_vals):
            kvec = np.array([kx, ky, kz])
            Ec = conduction_energy(kvec, params)
            E_map[iy, ix] = Ec  # iy = row (y), ix = col (x)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        E_map,
        origin="lower",
        extent=[kx_vals[0], kx_vals[-1], ky_vals[0], ky_vals[-1]],
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Conduction band energy [eV]")

    ax.set_title(
        f"Conduction band E_c(kx, ky; kz={kz:.3f}) at T={params.T:.1f} K\n"
        r"$E_c(k) = E_g + (\sqrt{\gamma k^2 + c^2} - c)$"
    )
    ax.set_xlabel(r"$k_x$ [1/Å]")
    ax.set_ylabel(r"$k_y$ [1/Å]")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    # Choose an operating temperature close to Roman darks
    T_oper = 89.0  # K (example; adjust as appropriate)
    params = interpolate_hgcdte_params(T_oper)

    print(f"Interpolated parameters at T = {params.T:.1f} K:")
    print(f"  Eg(T)   = {params.Eg_eV:.4f} eV")
    print(f"  gamma(T)= {params.gamma_eVA2:.4f} eV·Å^2")
    print(f"  c(T)    = {params.c_eV:.4f} eV")

    # 1) Band structure along high-symmetry path
    plot_band_structure(params, n_points_per_segment=80, a=6.5)

    # 2) 2D kx–ky conduction-band map
    plot_2d_conduction_map(params, nk=201, kz=0.0, kmax=0.08)
