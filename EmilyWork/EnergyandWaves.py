import numpy as np 
#import matplotlib.pyplot as plt
from .ThomasFermi import Potential
import sys
import asdf



r_bohr_H = 5.2946541e-11 # bohr radius in meters
r_bohr_Mg = r_bohr_H/80 
r_min = r_bohr_Mg/3

a_from_paper = 6.468e-10 
r_max = 2.00e-10 #a_from_paper 
overflow_fix_constant = 2.**60

tol=1e-10



class ElectronWaveFunctionGRC:
    
    """
    
    Electron Wave Function class implements the parameters and RK-4 method for the electron's radial part of the 
    wave function.
    
    
    Parameters
    ----------
    
    n: int
    
    k: int
       quantum number 
    
    h: float
       Energy eigenvalue given in J
    
    mu: float
        mass of the electron in kg 
        
    c: float 
        speed of light in m/s
    
    hbar: float 
        reduced Planck's constant in Js 
        
    charge: float
        charge of electron in C
    
    alpha: float 
        fine structure constant 
        
    Z: int
        atomic number 
    
    tol: float 
        tolerance (used for what?)
    
    """

    def __init__(self, n, k, h, tol,Z):
        
        self.n = n
        self.k = k
        self.h = h
        self.mu = 9.1093837e-31 #kilograms #define in right units here
        self.c = 299792458. #m/s  #define in right units here
        self.hbar = 1.054571817e-34 #Js #define in right units here 
        self.charge = -1.60217663e-19 #V
        self.alpha = 0.0072973525643 #alpha 
        self.Z =  Z #define this 
        self.tol = tol

        
        self.nu = np.sqrt(self.k**2 - (self.alpha*self.Z)**2) 
        #print(self.nu,(self.alpha*self.Z)**2)
        
    

    def potential(self, r):
        #return ThomasFermi.Potential(self.Z,r)
        return Potential(self.Z,r)
    
    def diff_eq(self, r, F, G):
        
        dG = (self.h -self.mu*self.c**2 - self.charge*self.potential(r))* F/self.hbar/self.c  - (self.k/r)*G
            
        dF = (self.k/r) * F + (-self.h -self.mu*self.c**2 + self.charge*self.potential(r)) * G /self.hbar/self.c

        return np.asarray([dF, dG])
    
    
    def RK_4(self, r_min, r_inf, N):
        
        #this is in terms of r_star
        
        step_size = (r_inf - r_min) / (N-1)
        #print(step_size)
        
        r_points = np.linspace(r_min, r_inf, N)        
        
        #print(r_points[0])
        
        
        #Define initial values for F and G
        #inner boundary condition
        if r_min<r_inf:
            F = np.sqrt(self.k**2 - self.nu**2)
            G = self.k - self.nu
        else: 
            F = 1.
            G = 1.
        #print(F,G)
        #outer boundary condition is F = G
        
        #Arrays to keep track of r, F and G points:

        F_points = []
        G_points = []
        K_counts = []
        counter_K = 0
        K_check = overflow_fix_constant
        
        for r in r_points:  
            #Update F_points and G_points arrays:
            F_points.append(F)
            G_points.append(G)
            K_counts.append(counter_K)

            #Calculate all the slopes (each variable has a separate k-slope):
            (k1F, k1G) = tuple(step_size * self.diff_eq(r, F, G))

            (k2F, k2G) = tuple(step_size * self.diff_eq(r + 0.5*step_size, F + 0.5*k1F, G + 0.5*k1G))

            (k3F, k3G) = tuple(step_size * self.diff_eq(r + 0.5*step_size, F + 0.5*k2F, G + 0.5*k2G))

            (k4F, k4G) = tuple(step_size * self.diff_eq(r + step_size, F + k3F, G + k3G))


            #Calculate next F_point and G_point
            F = F + (k1F + 2*k2F + 2*k3F + k4F) / 6
            G = G + (k1G + 2*k2G + 2*k3G + k4G) / 6
            if F**2 + G**2 > K_check:
                #print('true') 
                F = F/np.sqrt(K_check)
                G = G/np.sqrt(K_check)
                counter_K += 1 
            #K_counts.append(counter_K)
        #want to normalize the last points of things 
        
        #print('coeff is ' + str(coeff))
        #F_points= F_points/coeff
        #G_points= G_points/coeff                                                                                                            
        return np.array(r_points), np.array(F_points), np.array(G_points), K_counts

        
def iterate_for_zeros(n,k,Z,Nr,hsarray,sarray,nzeros,hmin=8.0109e-20, hmax =1.60218e-12): 

    # number of iterations is limited by how big s can be
    # and the number of bits stored in float64
    #hmin band gap energy eV to J
    #hmax takes me up to a MeV for now to J 
    h = np.sqrt(hmin*hmax)
    s = (hmax/hmin)**.25
    
    for _ in range(53+max(int(np.ceil(np.log2(s))),0)):
        testing = ElectronWaveFunctionGRC(n, k, h, tol,Z)
        temp_data_rs, temp_data_Fs, temp_data_Gs,K_count = testing.RK_4(r_min,r_max, Nr)
        #print(temp_data_rs[0], temp_data_Fs[0], temp_data_Gs[0])
        check_array = (temp_data_Fs- temp_data_Gs)
        #print(check_array)
        crossing_finder = check_array[:-1]*check_array[1:]
        num_zeros = (crossing_finder<0).sum()
        nzeros.append(num_zeros)
        #print(h,s,num_zeros)
        if num_zeros >= n: 
            h = h/s
            hsarray.append(h)
        else: 
            h = h*s
            hsarray.append(h)
        s = np.sqrt(s)
        sarray.append(s)
    return {'rs':temp_data_rs,'Fs_int_out': temp_data_Fs,'Gs_int_out': temp_data_Gs,'K_count': K_count,'h':h }
    
     

#reformat this to take in the eigenvalue h and not have to iterate over finding h. 
def wavefunc_renorm(Z,n,k,bound,h_found,Nr):
    testing = ElectronWaveFunctionGRC(n, k, h_found, tol,Z)
    data_rs, data_Fs, data_Gs,Kcount = testing.RK_4(r_min,r_max, Nr)
    data_Fs = np.array(data_Fs)
    data_Gs = np.array(data_Gs)
    data_rs = np.array(data_rs)
    Kcount = np.array(Kcount)
    rescale_factor = 1
      
    final_fs = data_Fs 
    final_gs = data_Gs
    
    delta_K = np.array(Kcount[:-1])-np.array(Kcount[1:])
    
    
    if bound ==True:
        fout_prime = (data_Fs[:-1]*(overflow_fix_constant)**(delta_K/2)-data_Fs[1:])/(data_rs[:-1]-data_rs[1:])
        crossings = np.array(np.where(fout_prime[:-1]*fout_prime[1:] <0))
        #print(crossings[0])
        # revisit rmatch_index for only 1 crossing
        # rmatch_index = int(np.floor((crossings[0,0]+crossings[0,1])/2)+1)
        rmatch_index = int(np.floor(crossings[0,0])+1)
        #print(rmatch_index)

        instance2 = ElectronWaveFunctionGRC(n, k, h_found, tol,Z)
        temp_data_rs, temp_data_Fs, temp_data_Gs,K_count = instance2.RK_4(r_max,data_rs[rmatch_index], Nr-rmatch_index)
        dK_count = np.array(K_count) - K_count[-1]
        
        
        temp_data_Fs = temp_data_Fs*((overflow_fix_constant)**(np.array(dK_count)/2))
        temp_data_Gs = temp_data_Gs*((overflow_fix_constant)**(np.array(dK_count)/2))
        rescale_factor =  data_Fs[rmatch_index]/temp_data_Fs[-1]
        
        #print(rmatch_index, data['rs'][rmatch_index], rescale_factor)
        #print(data['Fs_int_out'][rmatch_index-1:rmatch_index+2])
        #print(data['Fs_int_out'][rmatch_index]/data['Fs_int_out'][rmatch_index+1]*temp_data_Fs[-2]/temp_data_Fs[-1])
        #print(temp_data_Fs)
        #print(K_count)
        final_fs[rmatch_index:] = temp_data_Fs[::-1]*rescale_factor
        final_gs[rmatch_index:] = temp_data_Gs[::-1]*rescale_factor
        
        
        fwd_dK_count = np.array(Kcount)[:rmatch_index] - Kcount[rmatch_index]
        #print(fwd_dK_count)
        
        final_fs[:rmatch_index] *= (overflow_fix_constant)**(fwd_dK_count/2)
        final_gs[:rmatch_index] *= (overflow_fix_constant)**(fwd_dK_count/2)
        
    
        
    coeff = (np.sum(np.abs(final_fs[1:-1]))**2 + np.sum(np.abs(final_gs[1:-1])**2) + .5*(np.abs(final_fs[0])**2+np.abs(final_fs[-1])**2+ np.abs(final_gs[0])**2+np.abs(final_gs[-1]**2 )))*(data_rs[1] - data_rs[0]) 

        #temp_data_rs, Fs_int_in, Gs_int_in,coeff,K_count_in =  instance2.RK_4(r_max,r_min, Nr)
    return data_rs,final_fs, final_gs
    
if __name__ == "__main__":
    Z = int(sys.argv[1])
    n= int(sys.argv[2])
    k= int(sys.argv[3])
    path_out = sys.argv[4]
    Nr = int(sys.argv[5])
    find_h = sys.argv[6]
    find_wfs = sys.argv[7]
    h_try = float(sys.argv[8])

    tested_hs = []
    tested_s = []
    n_zeros = []
    
    
    #### will be used to compute the energies

    if find_h =="True":
        data = iterate_for_zeros(n,k,Z,Nr,tested_hs,tested_s,n_zeros)
        
        with open(path_out, "a") as f:
            f.write(str(n) +'\t'+str(k) + '\t' +str(data['h']) + '\t' + str(data['h']/1.60218e-19) +'\n')
            
    if find_wfs== "True": 
    
        r,f,g = wavefunc_renorm(Z,n,k,True,h_try,Nr)
        wf_info = {'k':k,'r':r, 'f': f, 'g':g}

        af = asdf.AsdfFile(wf_info)

        # Write the data to a new file
        af.write_to(str(Z)+'_'+str(k)+"WFs.asdf")


