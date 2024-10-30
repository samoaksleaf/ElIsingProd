#!/bin/bash

# wham program is downloaded from Grossfield, A, “WHAM: an implementation of the weighted histogram analysis method”, http://membrane.urmc.rochester.edu/content/wham/ Version 2.0.10

hist_min="0.008"
hist_max="0.992"
num_bins=492
tol_WHAM="1E-3"
T="1.0"
numpad=0
metabase="./dumbrella_rand_prod_15_WHAMMeta_200_5_800_"
fesbase="./dumbrella_rand_prod_15_fes_WHAMMeta_200_5_800_"
num_mc_trial=0
num_chunks=5
for ((i=0;i<num_chunks;i++)) do
    metafilename="${metabase}${i}.txt"
    fesfilename="${fesbase}${i}.txt"
    ./wham/wham/wham $hist_min $hist_max $num_bins $tol_WHAM $T $numpad $metafilename $fesfilename $num_mc_trial $RANDOM
done