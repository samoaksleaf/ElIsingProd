import numpy as np
import sys
import os
from scipy.stats import iqr

# Script to construct a continuously varying density field from a discrete one (a la Willard-Chandler interface)
import WCHeightField as wc

DIR = "/pscratch/sd/s/soakslea/RodTrajectoriesProd/"
filebase_mu = sys.argv[1]
print("Gathering height field statistics ", wc.NS, wc.NZ, filebase_mu)

max_points = 5
smear_arr = np.array([ [ [wc.gauss(dx, dy, dz) for dx in range(-max_points + 1, max_points)] for dy in range(-max_points + 1, max_points) ] for dz in range(-max_points + 1, max_points) ] )
smear_arr /= np.sum(smear_arr)   
h_stats = open(filebase_mu.replace(".txt", "wch_stats_linear_one.txt"), 'w')

for file in os.scandir(DIR):
    # Read in a discrete density field 
    filename = file.name
    if filename.startswith('t') and filename.endswith(filebase_mu):
        with open(DIR + filename, 'r') as f:
            i = 0
            for line in f:
                if i >= 3:
                    line = line.split()
                    if line[0] == "k_ads": # If there is a trajectory file to which there has been inadvertently added a second trajectory, stop at the first 
                        break
                    time = float(line[0])
                    conc = float(line[1])
                    energy = float(line[2])

                    rho_cont = wc.smear(np.array([int(x) for x in line[4:]]), smear_arr, max_points)
                    hmid = wc.grab_interface(rho_cont, s=0.5)
                    hhigh1 = wc.grab_interface(rho_cont, s=0.4)
                    hhigh2 = wc.grab_interface(rho_cont, s=0.8)
                    hlow1 = wc.grab_interface(rho_cont, s=0.6)
                    hlow2 = wc.grab_interface(rho_cont, s=0.2)

                    midbar = np.mean(hmid)
                    width_1 = np.std(hmid)
                    midhigh = np.mean(hhigh1)
                    midlow = np.mean(hlow1)
                    width_2 = midhigh - midlow
                    width_3 = iqr(hmid)
                    width_4 = np.mean(hhigh2) - np.mean(hlow2)
                    h_stats.write("%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %s \n" % (time, conc, energy, midbar, width_1, midhigh, midlow, width_2, width_3, width_4, np.mean(hhigh2), np.mean(hlow2), filename))

                i += 1
