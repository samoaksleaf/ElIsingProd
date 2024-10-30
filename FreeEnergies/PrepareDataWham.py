import numpy as np
import os
import sys
import matplotlib.pyplot as plt
#import pymbar
# Define size of lattice
NS = 15
NSQ = NS * NS
NZ = 15
N = NSQ * NZ
TEMP = 300.00

DIR = "/global/cfs/cdirs/m1864/soak/umbrella_trajectories_zerocorrected/"
filebase = sys.argv[1]

meta_file = filebase + "_WHAMMeta_200_5_800_"
kindex = 0
cindex = -1
num_configs = 51000 # Can be upper bound
eq_sweeps = 200
chunks = 5
def bias_potential(x, k, m):
    return 0.5 * k * (x - m) ** 2

fig, ax = plt.subplots()
last_val = 0
for file in os.scandir(DIR):
    if filebase in file.name and "_processed" not in file.name:
        conc_arr = np.zeros(num_configs)
        step_arr = np.zeros(num_configs, dtype=int)
        count = 0
        params = file.name.replace(filebase, "")
        params = params.replace(".txt", "")
        params = params.split("_")
        fparams = [float(params[i]) for i in range(1,len(params))]
        #print(fparams)
        ks = fparams[kindex]/TEMP
        if ks > N*50:
            continue
        cs = fparams[cindex]

        #print(file.name, ks, cs)
        with open(DIR + file.name, 'r') as f:
            i = 0
            for line in f:
                if i >= (3 + eq_sweeps) and (i - 3 - eq_sweeps) % 5 == 0 and i: 
                    line = line.split()
                    step_arr[count] = int(line[3])
                    conc_arr[count] = float(line[1])
                    count += 1
                i += 1
        
        conc_arr = conc_arr[:count]
        chunks = 5
        chunk_length = len(conc_arr) // chunks
        
        if len(conc_arr) < 800:
            continue
        
        print(file.name, ks, cs, chunk_length)
        for chunk in range(chunks):
            with open(meta_file + "%d.txt" % chunk,'a') as g:
                g.write(DIR + file.name.replace(".txt", "_processed_200_5_800_%d.txt" % chunk) + " %.6f %.6f %d \n" % (cs, ks, 1) )

            with open(DIR + file.name.replace(".txt", "_processed_200_5_800_%d.txt" % chunk), 'w') as g:
                for i in range(chunk_length):
                    g.write("%.6f %.6f \n" % (step_arr[chunk * chunk_length + i], conc_arr[chunk * chunk_length + i]))


