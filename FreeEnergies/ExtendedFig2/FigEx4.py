import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys 
import math
from matplotlib import ticker

matplotlib.style.use('classic')
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24
matplotlib.rcParams['axes.labelsize'] = 35



MU_ARR = [0.1, 0.2, 0.3, 0.4]
mark = ["o", "s", "^"]
color = ['k', 'y', 'c', 'm']
NS_arr = [15]
title_arr = ["4 nm, N = 1000", "6 nm, N = 3375"]
labels = ["4 nm", "6 nm"]
mu_labels = [r'$\beta\Delta\mu$' + " = %.1f" % mu for mu in MU_ARR]
sfig, sax = plt.subplots(layout = "constrained")
nfig, nax = plt.subplots(layout = "constrained")
num_bins = 492
num_chunks = 5
for NS_i in range(len(NS_arr)):
    c=np.zeros(num_bins)
    fc_total = {}
    fcerr = {}
    NS = NS_arr[NS_i]
    N = NS * NS * NS
    
    for i in range(num_chunks):    
        
        fc=np.zeros(num_bins)
        
        file = "dumbrella_rand_prod_" + str(NS) + "_fes_WHAMMeta_200_5_800_" + str(i) + ".txt"
        with open("../" + file,'r') as f:
            l = 0
            for line in f:
                
                if not line.startswith("#"):
                    line = line.split()
                    c[l] = float(line[0])
                    fc[l] = float(line[1])
                    
                    l += 1
        fc_total[i] = fc  
    
    fc = np.array([np.mean(np.array([fc_total[i][j] for i in range(num_chunks)])) for j in range(num_bins)])
    fcerr  = np.array([3 * np.std(np.array([fc_total[i][j] for i in range(num_chunks)]), ddof = 1) / (num_chunks ** 0.5) for j in range(num_bins)])
    fchigh = fc + fcerr
    fclow = fc - fcerr
   
    fchigh -= np.min(fc)
    fclow -= np.min(fc)
    fc -= np.min(fc)

    sax.fill_between(c, fclow, fchigh, alpha=0.5, color=color[NS_i])
    sax.plot(c, fc, label=labels[NS_i], color=color[NS_i], linewidth=3.0)
    with open("SubFigA/" + file.replace(".txt", "_wham_symmetric.txt"), 'a') as g:
        for i in range(len(c)):
            g.write("%.16e %.16e %.16e \n" % (c[i], fc[i], (fchigh[i] - fc[i])))
   
    for mu_i in range(len(MU_ARR)):
        
        MU = MU_ARR[mu_i]
        nfchigh = np.array([fchigh[i] - MU * N * c[i] for i in range(len(c)) ]) 
        nfclow = np.array([fclow[i] - MU * N * c[i] for i in range(len(c)) ]) 
        nfc = (nfchigh + nfclow) / 2.0
        nc = c
        
        reactant_well = np.nonzero(np.array(np.gradient(nfc) >= 0))[0][0]
        nfchigh -= nfc[reactant_well]
        nfclow -= nfc[reactant_well]
        nfc -= nfc[reactant_well]
        
        nax.fill_between(nc, nfclow, nfchigh, alpha=0.5, color=color[mu_i])
        nax.plot(nc, nfc, label=mu_labels[mu_i], color=color[mu_i], linewidth=3.0)
        with open("SubFigB/" + file.replace(".txt", "_wham_nuc_%.1f.txt" % MU_ARR[mu_i]), 'a') as g:
            for i in range(len(nc)):
                g.write("%.16e %.16e %.16e \n" % (nc[i], nfc[i], (nfchigh[i] - nfc[i])))

nax.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
nax.yaxis.set_major_locator(ticker.FixedLocator([-1000, -500, 0, 300]))
nax.set_xlabel(r"$c_\beta$")
nax.set_ylabel(r"$\beta F(c_\beta)$")
nax.set_xlim(0.0, 1.0)
nax.set_ylim(-1200, 350)
nax.legend(loc='lower left', prop={'size': 24})
nfig.savefig("SubFigB/" + file.replace(".txt", "_wham_nuc.png"))

sax.set_xlabel(r"$c_\beta$")
sax.set_ylabel(r"$\beta F(c_\beta)$")
sax.set_ylim(-5.0, 350.0)
sax.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
sax.yaxis.set_major_locator(ticker.FixedLocator([0, 100, 200, 300]))
sax.set_xlim(0.0, 1.0)
sfig.savefig("SubFigA/" + file.replace(".txt", "_wham_symmetric.png"))

