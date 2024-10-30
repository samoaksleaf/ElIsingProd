#include "RodPropagation.h"

int main(int argc, char** argv)
{
    std::string meta_dir = "/pscratch/sd/s/soakslea/";
    std::string data_dir = "RodTrajectoriesProd/";
    
    std::string basename = "el_ising_nzr";
    long long int seed = std::atoi(argv[5]);
    
    double p = 0.0; //atm //treated to be effectively zero on the nanoscale
    double temp = 300.0; //in kelvin
    double k_ads = 1.0; // relative to k_diff
    
    double mu = std::atof(argv[3]) * temp; //in kelvin (i.e. per k_B)	
    
    int ns = std::atoi(argv[1]);
    int nz = std::atoi(argv[2]);
    int num_sites = ns * ns * nz;
    int wall_width = 1; 
    
    int num_trials = std::atoi(argv[4]);

    std::stringstream fb_str;
    fb_str << std::setprecision(3) << std::fixed << basename << "_" << mu / temp  << "_" << ns << "_" << nz;
    std::cout << "Beginning Trajectories...\n" << std::flush;
    
    #pragma omp parallel for
    for (long long int i = 0; i < num_trials; i++)
    {
        nanorod sys(k_ads, mu, temp, p, ns, nz, wall_width, fb_str.str(), meta_dir, data_dir, i);
        sys.configure_wall();
        sys.kmc_run_c_up(0.85, 0.1, 0.1, seed * (i + 1), false, true, INT64_MAX); 
    }
    
    return 0;
}
