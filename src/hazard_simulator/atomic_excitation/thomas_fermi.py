import numpy as np


def SetupTF(log10rmin, log10rmax, N):
    """
    Setup up self-similar solution for Thomas-Fermi.

    Parameters
    ----------
    log10rmin, log10rmax : float
        Range of radii (log10, in units of the Thomas-Fermi scale length).
    N : int
        Number of points to sample.

    Returns
    -------
    r, phi, phiprime: np.ndarray of float
        The radius, scaled potential, and scaled potential derivatives.

    """

    r = np.logspace(log10rmin, log10rmax, N)
    r_int = np.sqrt(r[1:] * r[:-1])
    phi = np.zeros_like(r)
    phiprime = np.zeros_like(r)
    phi[-1] = 144.0 / r[-1] ** 3  # Sommerfeld's asymptotic solution
    sl = 1.0
    for j in range(53):
        phiprime[-1] = phi[-1] * (-3 * sl / r[-1])
        for i in range(N - 1)[::-1]:
            p_int = phi[i + 1] + phiprime[i + 1] * (r_int[i] - r[i + 1])
            phiprime[i] = phiprime[i + 1] + p_int**1.5 / np.sqrt(r_int[i]) * (r[i] - r[i + 1])
            phi[i] = p_int + phiprime[i] * (r[i] - r_int[i])
            if phi[i] > 1:
                phiprime[i] = 0.0
            if phi[i] <= 0:
                phi[i] = phiprime[i] = 0.0
        if phi[0] < 1.0:
            sl *= 2 ** (0.5**j)
        else:
            sl /= 2 ** (0.5**j)
    return r, phi, phiprime


TFGrid_r, TFGrid_phi, _ = SetupTF(-9, 6, 7501)


def Potential(Z, r):
    """
    Function to get the potential in an atom using the Thomas-Fermi model.

    Parameters
    ----------
    Z : int
        Atomic number.
    r : float
        Radius in meters.

    Returns
    -------
    float
        The potential in Volts.

    """

    return (
        1.4399645432764397e-09
        * Z
        / r
        * np.interp(r / 4.685024802601039e-11 * Z ** (1 / 3), TFGrid_r, TFGrid_phi)
    )
