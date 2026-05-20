#Calculating script for F(T)
import numpy as np
import scipy as sp


def ft_calculation(chi_t, chi_l, c, beta, t, q):
    #Successfully convert imaginary numbers to real numbers
    lower_bound = (t/(c*beta))
    upper_bound = np.inf
    def integrand(q):
        return ((chi_l.imag)/ (abs(1 + chi_l.real)**2)) + ((chi_t.imag)/(abs(1+ chi_t.real - ((c*q)/t)**2))**2)*((((c**2)*(beta**2)*(q**2))/t)-1)
    result, error = sp.integrate.quad(integrand, lower_bound, upper_bound)
    return np.real(result)
