# Calculating script for F(T)
import numpy as np
import scipy as sp

#arbitrary values
ne = 1e28          # electron density, m^-3
quality = 10       # Q-factor
omega_naught = 1e6

def simple_material(ne, quality, omega_naught):
    """Calculate the electrical susceptibility of detector material
    Parameters:
    ----------
    ne: float
        Electron density of the material.
    quality: float
        omega naught/ gamma, scaling "Q-factor" of the material.
    omega_naught: float
        The initial angular frequency of the excitation.
     
    Returns:
    --------
    dict
        The electric susceptibility of the material.
    """
    me = 9.10938356e-31 # kg   
    qe = 1.602176634e-19 # C
    epsilon_0 = 8.854187817e-12 # F/m
    gamma = omega_naught/ quality
    E_min = 0.5 #eV
    E_max = 3 #eV, ballpark, will replace max energy with 2.48 eV (half a micron converted into eV), go up to a keV
    h_bar = 6.582119569e-16 # eV*s
    omega_min = E_min / h_bar
    omega_max = E_max / h_bar
    
    t_array = np.logspace(omega_min, omega_max)
    q_array = np.logspace(1,10)

    q_grid, t_grid = np.meshgrid(q_array, t_array)

    omega_grid = t_grid / h_bar

    x = (1/(omega_naught**2 - (1j*gamma*omega_grid) - omega_grid**2))*(qe/me) ## move this later because x is dependent on other values
    chi_e = (ne*qe*x)/ epsilon_0
    # q_array = np.logspace(omega_min, omega_max) #unit conversions, conversting lattice separations into momenta, 1-1000 for bounds

    chi_dict = {
        "CHI_E": chi_e,
        "Q_ARRAY": q_array,
        "T_ARRAY": t_array
    }
    return chi_dict

sample_material = simple_material(ne, quality, omega_naught)
print(len(sample_material["T_ARRAY"]))


material = simple_material (ne, quality, omega_naught)

def ft_calculation(c, beta, material):  
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
