import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from matplotlib import ticker
import matplotlib
matplotlib.style.use('classic')
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24
matplotlib.rcParams['axes.labelsize'] = 35
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

NZ = 50
meta_dir = "../RodPropData/"
mean_t = 28.0
width_t = 3.0
mu_arr = np.array([30.0, 60.0, 90.0, 120.0, 150.0, 168.8, 187.5, 206.2, 225.0, 262.5, 300.0])


TEMP = 300.0
colors = 'ymk'
num_bins = 200
num_bins_hbar = 20

vfig, vax = plt.subplots( layout="constrained")

NS_ARR = [12, 14, 16]

for NS_i in range(len(NS_ARR)):
    varbymu = {}
    color = colors[NS_i]
    NS = NS_ARR[NS_i]
    NSQ = NS * NS
    label = r"$L_s/a$" + " = %d" % NS
    valid_mus = []
    valid_inds = []
    for mu_i in range(len(mu_arr)):
        mu = mu_arr[mu_i]
        meta_filename = meta_dir + "fullprop_mu_n_T_prod_%d_%d_1_0.0_%.1f_300.0wch_stats_linear_one.txt" % (NS, NZ, mu)
        if not os.path.exists(meta_filename):
             continue
        
        valid_mus.append(mu)
        valid_inds.append(mu_i)
        h = []
        hbar = []

        f = open(meta_filename.replace(".txt", "_allh_byh_%.1f_%.1f.txt" % (mean_t, width_t)), 'r')
        data = f.read().split()[:-1]
        f.close()
        prev_hbar = 50.0
        count_hdata = 0
        chunks = []
        for i in range(len(data)):
            if i % (NSQ + 1) == 0:
                cur_hbar = float(data[i])
                hbar.append(cur_hbar)
                if (cur_hbar - prev_hbar) < -4.0:
                    #print(cur_hbar, prev_hbar)
                    chunks.append(count_hdata)
                prev_hbar = cur_hbar
            else:
                h.append(float(data[i]))
                count_hdata += 1

        total_hhisto = np.zeros((len(chunks)-1, num_bins))
        total_hbarhisto = np.zeros((len(chunks)-1,num_bins_hbar))
        varbymu[mu_i] = []
        for chunk in range(len(chunks) - 1):
            cur_mean = np.mean(h[chunks[chunk]:chunks[chunk + 1]])
            varbymu[mu_i].append(np.mean(np.array(h[chunks[chunk]:chunks[chunk + 1]]) ** 2))
            #print(cur_mean, varbymu[mu_i][-1])                 

    mean_varbymu = [np.sqrt(np.mean(varbymu[i])) for i in valid_inds]
    err_varbymu = np.array([ (1.0 / (2.0 * np.sqrt(np.mean(varbymu[i])))) * np.std(varbymu[i], ddof=1)  / (len(varbymu[i]) ** 0.5) for i in valid_inds])

    #print(mu_arr, mean_varbymu)
    
    new_err_varbymu = np.array([ 2 * (mean_varbymu[i] / mean_varbymu[0]) * np.sqrt( (err_varbymu[i]/mean_varbymu[i]) ** 2 + (err_varbymu[0]/ mean_varbymu[0]) ** 2) for i in range(len(mean_varbymu))])
    mean_varbymu = np.array(mean_varbymu) / mean_varbymu[0]
    
    valid_mus = np.array(valid_mus)
    vax.errorbar(valid_mus/TEMP, mean_varbymu, yerr=new_err_varbymu, marker = 'o', linewidth=3.0, markersize=12, color = color, label=label)
    with open("SubFigB/Ls_%d.txt" % NS, 'w') as f:
        for i in range(len(valid_mus)):
            f.write("%.16e %.16e %.16e \n" % (valid_mus[i]/TEMP,  mean_varbymu[i], new_err_varbymu[i]))
       

vax.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))
vax.yaxis.set_major_locator(ticker.FixedLocator([1.0, 1.2, 1.4, 1.6, 1.8]))
vax.set_xlabel(r"$\beta\Delta\mu$")
vax.set_ylabel(r"$\langle (\delta h)^2 \rangle^{1/2} / \langle (\delta h)^2 \rangle_0^{1/2}$")
vax.set_xlim(0.0,1.05)
vax.set_ylim(0.95,1.85)
vfig.savefig("SubFigB/FigEx4b.png")
plt.close(vfig)


            

            




