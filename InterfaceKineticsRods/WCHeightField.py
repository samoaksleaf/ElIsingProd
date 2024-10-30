import numpy as np
from numba import njit
import sys
# Script to construct the Willard-Chandler interface for a discrete spin field on a cubic lattice
# Uses periodic boundary conditions in x and y, hard wall boundaries in z

# Describe lattice parameters
NS = int(sys.argv[2]) # Length of axes parallel to interface
NZ = int(sys.argv[3]) # Length of axis perpendicular to interface
NSQ = NS * NS
N = NSQ * NZ

@njit
def gauss(dx, dy, dz):
    """An unnormalized Gaussian for 'smearing' of density field. Implicitly uses lattice constant as discretization length."""
    return np.exp( -(dx ** 2 + dy ** 2 + dz ** 2) / 2 )

@njit
def smear(rho_disc, smear_arr, max_points):
    """Convolutes the discrete density field with Gaussian above, returning convolved field."""
    
    # Note that this procedure produces a field that sums to the current number of particles in the discrete field by construction
    rho_cont = np.zeros((NZ, NS, NS))
    index = 0
    for z in range(NZ):
        for y in range(NS):
            for x in range(NS):
                if rho_disc[index] == 1:
                    rho_cont[z][y][x] += 1.0
                    for i in range(2 * max_points - 1):
                        dz = i - max_points + 1
                        nz = z + dz
                        for j in range(2 * max_points - 1):
                            dy = j - max_points + 1
                            ny = (y + dy) % NS
                            for k in range(2 * max_points - 1):
                                dx = k - max_points + 1
                                nx = (x + dx) % NS
                                if nz >= 0 and nz < NZ:
                                    rho_cont[nz][ny][nx] += smear_arr[i][j][k]
                                    rho_cont[z][y][x] -= smear_arr[i][j][k]
                index +=1
    #print(np.sum(rho_cont))
    return rho_cont

def grab_interface(rho_cont, s=0.5):
    """Performs interpolation on each column of the density field to identify the height at which the density field is equal to 1/2"""
    # Note that using the value of 1/2 assumes that all density fields will have a particle hole symmetry such that rho in the 
    #   liquid phase is 1-rho of the vapor. 
    interface = np.zeros((NS, NS))
    for hy in range(NS):
        for hx in range(NS):
            # pad the density of the column in case the interface is very close the edges of the system
            rho_h = np.array([rho_cont[z][hy][hx] for z in range(NZ)])
            #closest_index = np.argmin(np.abs(rho_h - s))
            closest_list = np.nonzero(rho_h < s)[0]
            
            if len(closest_list) == 0:
                closest_index = NZ - 1
            else:
                closest_index = closest_list[0]
            
            if closest_index == 0:
                interface[hy][hx] = 0.0
            else:
                slope = rho_h[closest_index] - rho_h[closest_index - 1]
                intercept = rho_h[closest_index] - slope * closest_index
                interface[hy][hx] = (s - intercept) / slope
            

    return interface

def get_h_stats(rho_disc):
    interface = grab_interface(smear(rho_disc))
    return np.mean(interface), np.std(interface)









