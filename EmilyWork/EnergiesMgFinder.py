import os
import numpy as np

Z = 80 
n = np.arange(1,7,1, dtype =int)
#print(n)
k = np.arange(1,2,1, dtype = int)
Nr = 5000
path = "Energies" + str(Z)+ ".txt"

with open(path, "w") as f:
    f.write('n \t k \t h[J] \t h[eV] \n')

for i in n: 
    for j in k: 
       os.system("python EnergyandWaves.py "+str(Z) + ' ' + str(i) + ' ' + str(j) + ' ' + path +' ' +str(Nr)+' True False 0 ')

