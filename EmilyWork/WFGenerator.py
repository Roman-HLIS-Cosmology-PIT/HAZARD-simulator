import numpy as np 
from . import ThomasFermi
from asdf import AsdfFile
from . import EnergyandWaves 
import sys

Z = int(sys.argv[1])
#n= int(sys.argv[2])
k= int(sys.argv[2])
path_out = sys.argv[3]
Nr = int(sys.argv[4])
#find_h = sys.argv[6]
#find_wfs = sys.argv[7]
h_try = float(sys.argv[5])

Nr = 5000 # gives delta r < pi /(2p_max + k_max ) #basically deta r = 6e-13 
# if p_max = 9.9532518e+12 kgm/s 

n_max = 1200
h_max = 1.1215236e-13 #about 700keV in J 

tested_hs = []
tested_s = []
n_zeros = []

final_h = []
final_fs = []
final_gs = []
    
n = 1 
while h_try < h_max and n<n_max: 

    print(n)
    info1 = EnergyandWaves.iterate_for_zeros(n,k,Z, Nr,tested_hs,tested_s,n_zeros)
    #print('here')
    #print(info1['h'])
    final_h.append(info1['h'])
    print(info1['h'])
    r,f,g = EnergyandWaves.wavefunc_renorm(Z,n,k,True,info1['h'],Nr)
    #print(info2['f'])
    final_fs.append(np.array(f))
    final_gs.append(np.array(g))
    n+=1 
    h_try = info1['h']
    
    
wfdata = {'k': k,'h' : final_h, 'r': r, 'F': final_fs, 'G': final_gs}

af = AsdfFile(wfdata)
af.write_to(path_out+".asdf")
