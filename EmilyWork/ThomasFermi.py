import numpy as np   
import sys

def SetupTF(log10rmin, log10rmax, N):

    r = np.logspace(log10rmin, log10rmax, N)
    r_int = np.sqrt(r[1:]*r[:-1])
    phi = np.zeros_like(r)
    phiprime = np.zeros_like(r)
    phi[-1] = 144./r[-1]**3 # Sommerfeld's asymptotic solution
    sl = 1.
    for j in range(53):
        phiprime[-1] = phi[-1]*(-3*sl/r[-1])
        for i in range(N-1)[::-1]:
           p_int = phi[i+1] + phiprime[i+1]*(r_int[i]-r[i+1])
           phiprime[i] = phiprime[i+1] + p_int**1.5/np.sqrt(r_int[i])*(r[i]-r[i+1])
           phi[i] = p_int + phiprime[i]*(r[i]-r_int[i])
           if phi[i]>1: phiprime[i]=0.
           if phi[i]<=0:
              phi[i] = phiprime[i] = 0. 
        if phi[0] < 1.:
           sl *= 2**(.5**j)
        else:
           sl /= 2**(.5**j)
    return r,phi,phiprime

TFGrid_r, TFGrid_phi, _ = SetupTF(-9,6,7501)

def Potential(Z,r):
    """Function to get the potential (in V) at radius r (in m) for atomic number Z"""
    return 1.4399645432764397e-09*Z/r*np.interp(r/4.685024802601039e-11*Z**(1/3), TFGrid_r, TFGrid_phi)

if __name__ == "__main__":
    Z = int(sys.argv[1])
    #print(Z)
    for r in np.logspace(-13,-7,121).tolist():
        print('{:13.7E} {:13.7E}'.format(r, Potential(Z,r)))
