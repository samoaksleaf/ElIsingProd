import numpy as np
import matplotlib.pyplot as plt
import sys 
import os

from matplotlib import ticker
import matplotlib
matplotlib.style.use('classic')
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24
matplotlib.rcParams['axes.labelsize'] = 35


def collect_trajectory(t, hbar, w1, w2, w3, file):
    mean_h = 28
    width_h = 3.0
    prev_file = ""
    cur_t = []
    cur_hbar = []
    cur_w1 = []
    cur_w2 = []
    cur_w3 = []
    started=False
    t0 = 0.0
    with open(file, 'r') as f:
        for line in f:
            line = line.split()
            filename = line[-1]
            if filename != prev_file:
                prev_file = filename
                started = False
                if len(cur_t) > 1:
                    t.append(cur_t)
                    hbar.append(cur_hbar)
                    w1.append(cur_w1)
                    w2.append(cur_w2)
                    w3.append(cur_w3)
                
                cur_t = []
                cur_hbar = []
                cur_w1 = []
                cur_w2 = []
                cur_w3 = []
            
            if np.abs(float(line[3]) - mean_h) < width_h:
                if not started:
                    t0 = (float(line[0]))
                    started = True
                
                cur_t.append(float(line[0]) - t0)
                cur_hbar.append(float(line[3]))
                cur_w1.append(float(line[4]))
                cur_w2.append(float(line[7]))
                cur_w3.append(float(line[8]))
        
        t.append(cur_t)
        hbar.append(cur_hbar)
        w1.append(cur_w1)
        w2.append(cur_w2)
        w3.append(cur_w3)

TEMP = 300.0
NS_ARR = [12, 14, 16]
colors = 'ymk'
mu_arr = np.array([30.0, 60.0, 90.0, 120.0, 150.0, 168.8, 187.5, 206.2, 225.0, 262.5, 300.0])
DIR = "../RodPropData/"

mean_h = 28.0
width_h = 3.0


velvmufig, velvmuax = plt.subplots( layout="constrained")
for ns_i in range(len(NS_ARR)):

    vel_v_mu = []
    vel_v_err = []
    valid_mus = []
    ns = NS_ARR[ns_i]
    color = colors[ns_i]
    label = r"$L_s/a$" + " = %d" % ns

    for q in range(len(mu_arr)):
        filename = DIR +  "fullprop_mu_n_T_prod" + "_%d_50_1_0.0_%.1f_300.0wch_stats_linear_one.txt" % (ns, mu_arr[q])
        if not os.path.exists(filename):
             continue

        t = []
        hbar = []
        w1 = []
        w2 = []
        w3 = []

        collect_trajectory(t, hbar, w1, w2, w3, filename)

        num_trajs = len(t)
        num_bins = 10
        if (len(t[0]) == 0):
            continue

        valid_mus.append(mu_arr[q])
        vel_total = np.zeros(num_trajs)
        for traj_i in range(num_trajs):
            vel_total[traj_i] = (hbar[traj_i][-1] - hbar[traj_i][0]) / (t[traj_i][-1] - t[traj_i][0])

        vel_v_mu.append(np.mean(vel_total))
        vel_v_err.append(np.std(vel_total, ddof=1) / (num_trajs ** 0.5))    
    
    new_vel_v_err = np.array([ 2 * (vel_v_mu[i]/vel_v_mu[0]) * np.sqrt( (vel_v_err[i] / vel_v_mu[i]) ** 2 + (vel_v_err[0] / vel_v_mu[0]) ** 2) for i in range(len(vel_v_mu))])
    vel_v_mu = np.array(vel_v_mu)/vel_v_mu[0]
    
    velvmuax.errorbar(np.array(valid_mus) / TEMP, vel_v_mu, yerr=new_vel_v_err, color=color, label=label, marker='o',  linewidth=3.0, markersize=12 ) 
    with open("SubFigC/Ls_%d.txt" % ns ,'w') as f:
        for i in range(len(valid_mus)):
            f.write("%.16e %.16e %.16e \n" % (valid_mus[i]/TEMP, vel_v_mu[i], new_vel_v_err[i]))


velvmuax.legend(loc='upper left', prop={'size': 24})
velvmuax.set_xlabel(r'$\beta\Delta\mu$')
velvmuax.set_ylabel(r'$\langle \bar{v} \rangle / \langle \bar{v} \rangle_0 $')
velvmuax.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))
velvmuax.yaxis.set_major_locator(ticker.FixedLocator([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]))
velvmuax.set_xlim(0.0, 1.05)
velvmuax.set_ylim(0.75, 11.5)
velvmufig.savefig("SubFigC/FigEx4c.png")







