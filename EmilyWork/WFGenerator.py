import numpy as np 
import os
Z=80
Nr = 5000
with open('Energies80.txt','r') as f: 
    lines = f.readlines()
    for i in range(1,len(lines),1):
        line = lines[i].split('\t')
        #print(line)
        n = line[0]
        k= line[1]
        hs_J = line[2]
        hs_eV = line[3].strip()
        
        os.system("python EnergyandWaves.py "+str(Z) + ' ' + str(n) + ' ' + str(k) + ' Energies80.txt ' +str(Nr)+' False True ' +str(hs_J))



        #data_rs,final_fs, final_gs,coeff = EW.wavefunc_renorm(Z,int(n),int(k),True,float(hs_J))
        
        #np.save("Rs_80_"+str(n)+"_"+str(k)+".npy",final_rs)
        #np.save("Fs_80_"+str(n)+"_"+str(k)+".npy",final_fs)
        #np.save("Gs_80_"+str(n)+"_"+str(k)+".npy",final_gs)