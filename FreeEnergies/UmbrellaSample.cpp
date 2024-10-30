#include "UmbrellaSample.h"

int main(int argc, char** argv)
{
    std::string data_dir = "UmbrellaTrajectories/";
    std::string meta_dir = "/pscratch/sd/s/soakslea/";
    
    int seed = std::atoi(argv[1]);
    double mu = 0.0; //in kelvin (i.e. per k_B)	
    double temp = 300.0; //in kelvin
    double k_ads = 1.0; // relative to k_diff
    double p = 0.0; //atm

    int ns = std::atoi(argv[2]);
    int nz = ns;
    int num_sites = ns * ns * nz;
    
    // Collect ~10000 samples per umbrella per number of sites in cube
    int total_steps = int(std::atof(argv[3]) * pow(10,4)) * num_sites;  
    int crec = total_steps / (num_sites);
    int drec = total_steps / (num_sites); 

    std::mt19937_64 gen;
    std::uniform_int_distribution<int> dis(0, INT_MAX);
    gen.seed(seed);

    double stiffness = std::atof(argv[4]) * temp * num_sites; // Input spring constant scaled by size and temperature
    double center = std::atof(argv[5]);
        
    std::string filebase = "umbrella_rand_prod_" + std::to_string(ns)  + "_" + std::to_string(stiffness) + "_" + std::to_string(temp) + "_" + std::to_string(mu);
    
    std::string ufilebase = filebase + "_" +  std::to_string(center);
    
    umbrella_nanocube sys(k_ads, mu, temp, p, stiffness, center, ns, nz, ufilebase, meta_dir, data_dir, seed);
    sys.configure_rand(center, dis(gen));
    sys.kmc_run_steps(total_steps, crec, drec, dis(gen), false, true); 
    
    return 0;
}

