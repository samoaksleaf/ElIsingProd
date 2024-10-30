import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
import math
from matplotlib import ticker
import matplotlib
matplotlib.style.use('classic')
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24
matplotlib.rcParams['axes.labelsize'] = 35
class myhisto():
    def __init__(self, limits, binwidth, logspaced=False, tol=1, base=10):
        self.binwidth = binwidth
        self.lspace = logspaced
        self.tol = tol
        self.num_bins = int(np.ceil((limits[1] - limits[0]) / binwidth))
        
        if logspaced:
            self.edges = np.logspace(limits[0],limits[1], self.num_bins +1, base=base)
            self.vals = np.array([(self.edges[i] + self.edges[i+1])/2.0 for i in range(self.num_bins)])
            self.limits = base**np.array(limits)
        else:
            self.vals = np.arange(limits[0], limits[1], binwidth) + binwidth / 2
            self.limits = limits
        self.histo = np.zeros(self.num_bins)
        self.flots = [[] for i in range(self.num_bins)]
        self.num_samples = 0
        self.highest_counted_bin = 0
        self.lowest_counted_bin = self.num_bins - 1
    
    def add_data(self, data):
        if data >= self.limits[0] and data < self.limits[1]:
            self.num_samples += 1
            if self.lspace:
                for i in range(self.num_bins + 1):
                    if data > self.edges[i] and data < self.edges[i+1]:
                        bin = i
            else:
                bin = int((data - self.limits[0]) / self.binwidth)

            if (bin > self.highest_counted_bin) and self.histo[bin] > self.tol:
                self.highest_counted_bin = bin
            if (bin < self.lowest_counted_bin) and self.histo[bin] > self.tol:
                self.lowest_counted_bin = bin
            self.histo[bin] += 1
    
    def add_flot(self, data, flot):
        if data >= self.limits[0] and data < self.limits[1]:
            self.num_samples += 1
            if self.lspace:
                for i in range(self.num_bins + 1):
                    if data > self.edges[i] and data < self.edges[i+1]:
                        bin = i     
            else:
                bin = int((data - self.limits[0]) / self.binwidth)
            self.flots[bin].append(flot)
            self.histo[bin] += 1
            if (bin > self.highest_counted_bin) and len(self.flots[bin]) > self.tol:
                self.highest_counted_bin = bin
            if (bin < self.lowest_counted_bin) and len(self.flots[bin]) > self.tol:
                self.lowest_counted_bin = bin
    
    def clean(self, tol):
        self.cleaned_histo = []
        self.cleaned_vals = []
        for i in range(self.num_bins):
            if self.histo[i] > tol:
                self.cleaned_histo.append(self.histo[i])
                self.cleaned_vals.append(self.vals[i])
    def normalize(self):
        if self.lspace:
            self.prob = self.histo / np.array([ (self.edges[i+1] - self.edges[i]) for i in range(self.num_bins) ])
        self.prob = self.histo / (self.num_samples * self.binwidth)
    def average(self):
        self.avg = np.array([np.mean(self.flots[i]) for i in range(self.num_bins)])
    def stddev(self):
        self.std = np.array([np.std(self.flots[i], ddof=1) for i in range(self.num_bins)])
    def stderr(self):
        self.err = np.array([np.std(self.flots[i], ddof=1) / (len(self.flots[i]) ** 0.5) for i in range(self.num_bins)]) 
    def cut_out_unused_bins(self):
        self.flots = self.flots[self.lowest_counted_bin:self.highest_counted_bin + 1]
        self.histo = self.histo[self.lowest_counted_bin:self.highest_counted_bin + 1]
        self.vals = self.vals[self.lowest_counted_bin:self.highest_counted_bin + 1]
        self.num_bins = self.highest_counted_bin + 1 - self.lowest_counted_bin
    def lineplot(self):
        plt.plot(self.vals, self.prob, 'ko-')
        # ylim([0, 1.1 * max(self.prob)])
        plt.ylabel('probability distribution', fontsize=14)
        plt.xlabel('sampled variable', fontsize=14)
    def barplot(self):
        plt.bar(self.vals, self.prob, width=0.9 * self.binwidth, edgecolor='k', 
color='Orange')
        plt.ylabel('probability distribution', fontsize=14)
        plt.xlabel('sampled variable', fontsize=14)

def collect_trajectory(t, tw, file, lim, bw, max_ts):
    mean_h = 25
    width_h = 15
    prev_file = ""
    cur_t = myhisto(lim, bw)
    cur_tw = myhisto(lim, bw)
    prev_t = 0
   
    with open(file, 'r') as f:
        for line in f:
            line = line.split()
            filename = line[-1]
            if filename != prev_file:
                prev_file = filename
                started = False
                if cur_t.num_samples > 0:
                    
                    max_ts.append(prev_t)
                    cur_t.cut_out_unused_bins()
                    cur_t.average()
                    cur_tw.cut_out_unused_bins()
                    cur_tw.average()
                    for i in range(len(cur_t.vals)):
                        if not math.isnan(cur_t.avg[i]):
                            t.add_flot(cur_t.vals[i], cur_t.avg[i])
                            tw.add_flot(cur_tw.vals[i], cur_tw.avg[i])
                        #print(cur_t.vals[i], cur_t.avg[i])
                cur_t = myhisto(lim, bw)
                cur_tw = myhisto(lim, bw)
            if np.abs(float(line[3]) - mean_h) < width_h:
                if not started:
                    t0 = float(line[0])
                    h0 = float(line[3])
                    started = True
                cur_t.add_flot(float(line[0])-t0, float(line[3]))
                cur_tw.add_flot(float(line[0])-t0, float(line[4])**2)
                prev_t = float(line[0])-t0
                
        cur_t.cut_out_unused_bins()
        cur_t.average()
        cur_tw.cut_out_unused_bins()
        cur_tw.average()
        for i in range(len(cur_t.vals)):
            if not math.isnan(cur_t.avg[i]):
                t.add_flot(cur_t.vals[i], cur_t.avg[i])
                tw.add_flot(cur_tw.vals[i], cur_tw.avg[i])
bw = 3.0
mu_arr = np.array([30.0, 225.0, 300.0])
limits = [[-bw/2,300 + bw/2]] * len(mu_arr)
binwidths = [bw] * len(mu_arr)
ns_arr = [16, 14, 12]
TEMP = 300.0
colors='kmy'
DIR = "../RodPropData/"

zero_avgs = np.zeros(len(ns_arr))
zero_err = np.zeros(len(ns_arr))

for q in range(len(mu_arr)):
    wfig, wax = plt.subplots(layout="constrained")
    
    for ns_i in range(len(ns_arr)):
        ns = ns_arr[ns_i]
        label = r"$L_s/a$" + " = %d" % ns
        color = colors[ns_i]
        
        filename = DIR +  "fullprop_mu_n_T_prod_%d_50_1_0.0_%.1f_300.0wch_stats_linear_one.txt" % (ns, mu_arr[q])

        max_ts = []
        t = myhisto(limits[q], binwidths[q], tol=100)
        tw = myhisto(limits[q], binwidths[q], tol=100)
        collect_trajectory(t, tw, filename, limits[q], binwidths[q], max_ts)
        tmax = np.min(max_ts)
        print(tw.histo, max_ts)
        tw.cut_out_unused_bins()
        tw.average()
        tw.stderr()
        w_mean = np.sqrt(tw.avg)
        w_err = np.array(tw.err) / (2.0 * w_mean)
        
        if q == 0:
            zero_avgs[ns_i] = np.mean(w_mean)
            zero_err[ns_i] = np.std(w_mean, ddof=1) / (len(w_mean) ** 0.5)
       
        
        new_w_err = np.array([ 2 * (w_mean[i] / zero_avgs[ns_i])  * np.sqrt((w_err[i]/w_mean[i]) ** 2 + (zero_err[ns_i]/zero_avgs[ns_i]) ** 2) for i in range(len(w_mean))])
        w_mean /= zero_avgs[ns_i]               
        print(zero_avgs[ns_i])
        wax.fill_between(tw.vals, (w_mean + new_w_err),  (w_mean - new_w_err), alpha=0.5, color=color)
        wax.plot(tw.vals, w_mean, color=color, linewidth=3.0, label=label)
        with open("SubFigD/DeltaMu_%.2f_Ls_%d.txt" % (mu_arr[q]/TEMP, ns), 'w') as f:
            for i in range(len(tw.vals)):
                if tw.vals[i] < tmax:
                    f.write("%.16e %.16e %.16e \n" % (tw.vals[i], w_mean[i], new_w_err[i]))
    if q != 0:
        wax.yaxis.set_major_locator(ticker.FixedLocator([]))
    else:
        wax.set_ylabel(r'$\langle (\delta h)^2 (t) \rangle^{1/2}/\langle (\delta h)^2 \rangle_0^{1/2} $')
        wax.legend(loc="upper center",prop={'size': 35})
        wax.yaxis.set_major_locator(ticker.FixedLocator([1.0, 1.2, 1.4, 1.6, 1.8, 2.0]))
    
    wax.set_ylim(0.85, 2.05)
    wax.set_xlim(0.0, tmax)
    wax.set_xlabel(r'$\Delta t/\tau_0$')
    wfig.savefig("SubFigD/DeltaMu_%.2f.png" % (mu_arr[q]/TEMP))
   
    plt.close(wfig)

   
   
    
    
    

   



    



       









