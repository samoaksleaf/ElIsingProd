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
bw = 1.0
mu_arr = np.array([30.0, 300.0])
limits = [[-bw/2,300 + bw/2]] * len(mu_arr)
binwidths = [bw] * len(mu_arr)
ns = 16
TEMP = 300.0
colors='rb'
DIR = "../RodPropData/"
hfig, hax = plt.subplots(layout="constrained")

for q in range(len(mu_arr)):
    label = r"$\beta\Delta\mu$" + " = %.1f" % (mu_arr[q]/TEMP)
    color = colors[q]

    filename = DIR +  "fullprop_mu_n_T_prod" + "_%d_50_1_0.0_%.1f_300.0wch_stats_linear_one.txt" % (ns, mu_arr[q])
    
    max_ts = []
    t = myhisto(limits[q], binwidths[q], tol=100)
    tw = myhisto(limits[q], binwidths[q], tol=100)
    collect_trajectory(t, tw, filename, limits[q], binwidths[q], max_ts)
    tmax = np.min(max_ts)
    t.cut_out_unused_bins()
    t.average()
    t.stddev()
    h_mean = np.array(t.avg)
    h_err = np.array(t.std)

    hax.fill_between(t.vals, (h_mean + h_err),  (h_mean - h_err), alpha=0.5, color=color)
    hax.plot(t.vals, h_mean, color=color, linewidth=3.0, label=label)
    with open("SubFigC/DeltaMu_%.1f.txt" % (mu_arr[q]/TEMP), 'w') as f:
        for i in range(len(t.vals)):
            if t.vals[i] < tmax:
                f.write("%.16e %.16e %.16e \n" % (t.vals[i], h_mean[i], h_err[i]))

hax.set_xlabel(r'$\Delta t/\tau_0$')
hax.set_xlim(0.0, tmax)
hax.set_ylim(10.0, 40.0)
hax.legend(loc='upper left', prop={'size': 24})
hax.xaxis.set_major_locator(ticker.FixedLocator([0, 5, 10, 15, 20]))
hax.yaxis.set_major_locator(ticker.FixedLocator([0, 10, 20, 30, 40]))
hax.set_ylabel(r'$\langle \bar{h}(t) \rangle/a $')
hfig.savefig("SubFigC/Fig4c.png")
plt.close(hfig)


    



       









