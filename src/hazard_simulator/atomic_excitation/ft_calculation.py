# Calculating script for F(T)
import numpy as np
import scipy as sp


def ft_calculation(chi_t, chi_l, c, beta, t):
    """Calculate the rate of energy deposition events as a charged particle moves through the detector.
    Parameters:
    ---------
    chi_t : complex
        The transverse susceptibility
    chi_l : complex
        The longitudinal susceptibility.
    c : float
        The speed of light.
    beta : float
        The velocity of the charged particle relative to the speed of light.
    t : float
        Energy transfer.

    Returns:
    --------
    float
        The rate of energy deposition events in terms of t without the prefactor.
    """
    # Successfully convert imaginary numbers to real numbers
    lower_bound = t / (c * beta)
    upper_bound = np.inf

    def integrand(q):
        return (
            ((chi_l.imag) / (abs(1 + chi_l) ** 2))
            + (
                (chi_t.imag)
                / ((abs(1 + chi_t - ((c * q) / t) ** 2)) ** 2)
                * ((((c**2) * (beta**2) * (q**2)) / (t**2)) - 1)
            )
        ) * (1 / q)

    result, error = sp.integrate.quad(integrand, lower_bound, upper_bound)
    return result
