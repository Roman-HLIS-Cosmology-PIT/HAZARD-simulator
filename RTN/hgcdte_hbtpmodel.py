"""
hgcdte_hptb_model.py

Effective hyperbolic band model for Hg1-xCdxTe, combining:

- Hyperbolic conduction-band dispersion from Krishnamurthy et al. (1995):
    Ec(k; T) = Eg(x, T) + [ sqrt(γ(T) * k^2 + c(T)^2) - c(T) ]

- Empirical band gap Eg(x, T) from Chu et al. (1994):
    Eg(x, T) = -0.295 + 1.87 x - 0.28 x^2
               + (6 - 14 x + 3 x^2) * 1e-4 * T
               + 0.35 x^4   [eV]

This script:
  * Computes Eg(x, T) using Chu's formula.
  * Interpolates γ(T), c(T) from Krishnamurthy tables.
  * Builds an fcc high-symmetry k-path and plots Ev, Ec.
  * Plots a 2D kx–ky map of the conduction band.

You can change x_alloy and T_oper to match your detector.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# ------------------------------------------------------------
# Constants / units
# ------------------------------------------------------------

# k in [1/Å], energies in [eV]
HBAR2_OVER_2ME_EV_A2 = 3.81   # ħ^2 / (2 m_e) in eV·Å^2


# ------------------------------------------------------------
# 1. Bandgap Eg(x, T) from Chu et al. (1994), Eq. (6)
# ------------------------------------------------------------

def energy_gap_chu(x: float, T: float) -> float:
    """
    Bandgap Eg(x, T) [eV] for Hg1-xCdxTe from Chu et al. (1994):

        Eg(x, T) = -0.295 + 1.87 x - 0.28 x^2
                   + (6 - 14 x + 3 x^2) * 1e-4 * T
                   + 0.35 x^4

    Valid for 0.170 <= x <= 0.443 and 77 K <= T <= 300 K
    (you are using x ≈ 0.445, T ~ 89 K, which is very close).

    Parameters
    ----------
    x : float
        Cd molar fraction.
    T : float
        Temperature [K].

    Returns
    -------
    float
        Band gap Eg(x, T) in eV.
    """
    return (
        -0.295
        + 1.87 * x
        - 0.28 * x**2
        + (6.0 - 14.0 * x + 3.0 * x**2) * 1.0e-4 * T
        + 0.35 * x**4
    )


# ------------------------------------------------------------
# 2. γ(T), c(T) from Krishnamurthy et al. (1995), Table II
#    (effective hyperbolic parameters; composition not explicit)
# ------------------------------------------------------------

# Temperatures in K (table grid)
T_TABLE = np.array([
      1.0,  10.0,  20.0,  30.0,  40.0,  50.0,  60.0,  70.0,
     80.0,  90.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0,
    400.0, 450.0, 500.0, 550.0, 600.0
])

# γ(T) in eV·Å^2
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
    Effective parameters for the hyperbolic band model.

    Attributes
    ----------
    x : float
        Cd molar fraction.
    T : float
        Temperature [K].
    Eg_eV : float
        Band gap Eg(x, T) [eV] from Chu et al.
    gamma_eVA2 : float
        Hyperbolic γ(T) [eV·Å^2] from Krishnamurthy.
    c_eV : float
        Hyperbolic c(T) [eV] from Krishnamurthy.
    m_h_over_me : float
        Hole mass / free electron mass for simple parabolic Ev(k).
    """
    x: float
    T: float
    Eg_eV: float
    gamma_eVA2: float
    c_eV: float
    m_h_over_me: float = 0.30  # crude hole mass guess


def interpolate_hgcdte_params(T: float, x: float) -> HgCdTeHyperbolicParams:
    """
    Build HgCdTeHyperbolicParams at given T, x.

    Eg(x, T) is computed from Chu's formula.
    γ(T) and c(T) are interpolated from Krishnamurthy's table.

    Parameters
    ----------
    T : float
        Temperature [K].
    x : float
        Cd molar fraction.

    Returns
    -------
    HgCdTeHyperbolicParams
    """
    T_clamped = np.clip(T, T_TABLE.min(), T_TABLE.max())

    gamma_eVA2 = np.interp(T_clamped, T_TABLE, GAMMA_EVA2_TABLE)
    c_eV = np.interp(T_clamped, T_TABLE, C_EV_TABLE)

    Eg_eV = energy_gap_chu(x, T_clamped)

    return HgCdTeHyperbolicParams(
        x=x,
        T=T_clamped,
        Eg_eV=Eg_eV,
        gamma_eVA2=gamma_eVA2,
        c_eV=c_eV,
        m_h_over_me=0.30,
    )


# ------------------------------------------------------------
# 3. Band-structure formulas
# ------------------------------------------------------------

def conduction_energy(kvec: np.ndarray, p: HgCdTeHyperbolicParams) -> float:
    """
    Conduction band energy Ec(k; x, T) [eV]:

        Ec(k; x, T) = Eg(x, T) + [ sqrt(γ(T) k^2 + c(T)^2) - c(T) ],

    with k = |kvec| in [1/Å].
    """
    k = np.linalg.norm(kvec)
    return p.Eg_eV + (np.sqrt(p.gamma_eVA2 * k * k + p.c_eV**2) - p.c_eV)


def valence_energy(kvec: np.ndarray, p: HgCdTeHyperbolicParams) -> float:
    """
    Simple parabolic valence band:

        Ev(k; T) = - (ħ^2 / (2 m_h)) k^2,

    where m_h = m_h_over_me * m_e.
    """
    k = np.linalg.norm(kvec)
    alpha = HBAR2_OVER_2ME_EV_A2 / p.m_h_over_me  # eV·Å^2
    return -alpha * k * k


def band_energies(kvec: np.ndarray, p: HgCdTeHyperbolicParams) -> np.ndarray:
    """Return [Ev(k), Ec(k)] in eV."""
    Ev = valence_energy(kvec, p)
    Ec = conduction_energy(kvec, p)
    return np.array([Ev, Ec])


# ------------------------------------------------------------
# 4. High-symmetry path (fcc)
# ------------------------------------------------------------

@dataclass
class KPoint:
    label: str
    coord: np.ndarray  # fractional (2π/a) coords


def make_fcc_high_symmetry_points() -> List[KPoint]:
    """
    Standard fcc path:

      Γ – X – W – K – Γ – L – U – W – L – K
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
    Build k-path in [1/Å] along the high-symmetry points.
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

    # cumulative distance
    N = k_list.shape[0]
    k_dist = np.zeros(N)
    for i in range(1, N):
        dk = k_list[i] - k_list[i - 1]
        k_dist[i] = k_dist[i - 1] + np.linalg.norm(dk)

    return k_list, labels, k_dist


# ------------------------------------------------------------
# 5. Plotting
# ------------------------------------------------------------

def plot_band_structure(
    params: HgCdTeHyperbolicParams,
    n_points_per_segment: int = 80,
    a: float = 6.5,
) -> None:
    """Plot Ev, Ec along the fcc high-symmetry path."""
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

    for xline in tick_positions:
        ax.axvline(x=xline, color="k", linewidth=0.5, linestyle="--")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Energy [eV]")
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_title(
        f"Hg1-xCdxTe hyperbolic band model\n"
        f"x={params.x:.3f}, T={params.T:.1f} K, Eg={params.Eg_eV:.3f} eV"
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
    """2D kx–ky slice of Ec(k) as a density map."""
    kx_vals = np.linspace(-kmax, kmax, nk)
    ky_vals = np.linspace(-kmax, kmax, nk)

    E_map = np.zeros((nk, nk))
    for ix, kx in enumerate(kx_vals):
        for iy, ky in enumerate(ky_vals):
            kvec = np.array([kx, ky, kz])
            Ec = conduction_energy(kvec, params)
            E_map[iy, ix] = Ec

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        E_map,
        origin="lower",
        extent=[kx_vals[0], kx_vals[-1], ky_vals[0], ky_vals[-1]],
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Ec [eV]")

    ax.set_title(
        f"Conduction band Ec(kx, ky; kz={kz:.3f})\n"
        f"x={params.x:.3f}, T={params.T:.1f} K, Eg={params.Eg_eV:.3f} eV"
    )
    ax.set_xlabel(r"$k_x$ [1/Å]")
    ax.set_ylabel(r"$k_y$ [1/Å]")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------

if __name__ == "__main__":
    # Your Roman-like alloy and operating temperature:
    x_alloy = 0.445
    T_oper = 89.0  # K

    params = interpolate_hgcdte_params(T_oper, x_alloy)

    print(f"Interpolated hyperbolic parameters at x={params.x:.3f}, T={params.T:.1f} K:")
    print(f"  Eg(x,T)   = {params.Eg_eV:.6f} eV (Chu fit)")
    print(f"  gamma(T)  = {params.gamma_eVA2:.4f} eV·Å^2")
    print(f"  c(T)      = {params.c_eV:.4f} eV")

    # 1) Band structure along high-symmetry path
    plot_band_structure(params, n_points_per_segment=80, a=6.5)

    # 2) 2D conduction-band map
    plot_2d_conduction_map(params, nk=201, kz=0.0, kmax=0.08)
