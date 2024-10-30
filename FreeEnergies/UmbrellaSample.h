//
//  el_ising_umbrella.h
//  el_ising/umbrella/

//  Class for umbrella sampling of the elastic ising model nanocube system (PdH_x)
//  Outputs data from umbrellas in format for ease of analysis with Pymbar

//  Created by Sam Oaks-Leaf on 7/13/23.
//

#ifndef umbrella_nanocube_h
#define umbrella_nanocube_h


#include "helper.h"
#include <armadillo>
#include <iostream>
#include <array>
#include <iomanip>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <limits.h>

class umbrella_nanocube
{
    private:
        const std::vector<std::vector<int> > order = { {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},{0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        const int zlat = 6;
        const double max_time = INFINITY;   
        const std::string vrr_file = "vrr2springs_0K_arma_bin_nokmat_ZeroRemoved_";
        const std::string hr_file = "hr2springs_0K_arma_bin_nokmat_ZeroRemoved_";
    public:
        // Parameter that defines kinetic of adsorption (relative to diffusion)
        double k_ads; // Adsorption rate prefactor, phenoomenological, in units of k_d, diffusion rate presfactor;
        double temp; // temperature units of kelvin
        double del_mu;
        double p;
        double conc;
        arma::vec h; // effective Chemical potential difference accounting for pressure
        
        // We track particle number, total rate, and time elapsed throughout any simulation
        long double k_tot, time;
	double energy;
        long long int count_steps;
        int num_parts;
        int n_s, n_z;
        int n_sq;
        int np, nm;
        
        bool umbrella_on;

        // Umbrella sampling specific parameters
        double spring_stiffness, spring_center; // Stiffness expressed in units of K_B
        arma::Mat<double> v_rr; 

        // Simulation Tools
        std::string filebase;
        std::string meta_direc, direc;
        int trial_number;

        std::mt19937 contgen;
        std::uniform_real_distribution<double> contdis;

        std::ofstream config_stream;
        std::ofstream data_stream;
        std::ofstream full_stream;
        
        std::vector<int> active_sites;
        std::vector<int> open_slots_active_sites;
        int num_open_slots_active_sites;
        int max_size_active_sites;
        int size_active_sites;

        std::vector<int> neighb_m;
        std::vector<int> neighb_p;
        arma::Col<int> spins;
        std::vector<std::array<int, 6> > neighb_inds;
        
        std::vector<std::array<double, 7> > e_diffs_el;
        std::vector<std::array<bool, 7> > active_rates;
        std::vector<int> active_index;

        std::vector<double> partial_sums;
        std::vector<int> partial_sums_indices;
        std::vector<int> partial_sums_labels;
        int current_index_partial_sums;
        
        inline double umbrella_pot(int);
        
        umbrella_nanocube(double k_ads_f, double del_mu_f, double temp_f, double p_f, double k_spring, double c_spring, int ns, int nz, std::string name, std::string meta_dir, std::string dir, int trial_n);

        umbrella_nanocube(std::string infilename, int line_num, double k_ads_f, double del_mu_f, double temp_f, double p_f, int ns, int nz, std::string name, std::string dir, std::string meta_dir, int trial_n, bool preamble);
        
        umbrella_nanocube(std::string infilename, int line_num, double k_ads_f, double del_mu_f, double temp_f, double p_f, double k_spring, double c_spring, int ns, int nz, std::string name, std::string dir, std::string meta_dir, int trial_n, bool preamble);
        
        void configure_band(int width);

        void configure_bare(int);

        void configure_chunk(double);

        void configure_rand(double c, int seed);

        void configure_surface();

        void connect_lat();

        void set_active_sites();

        void compute_veff();

        void load_veff();

        void change_mu(double);

        int set_neighbors(int);

        void check_neighbors();

        double delta_flip_el(int, int);

        double delta_diff_el(int, int);

        void add_active_site(int, int);
        
        void remove_active_site(int);

        void update_neighbors(int, int);

        void execute_flip(int, int);

        void execute_diff(int, int);
        
        void kmc_step();
        
        void kmc_run_time(double total_time, int config_rec, int data_rec, int seed, bool, bool, double, double, int);

        void kmc_run_c_down(double c_end, int config_rec, int data_rec, int seed, bool, bool, int);

        void kmc_run_c_up(double c_end, int config_rec, int data_rec, int seed, bool, bool, int);

        void kmc_run_steps(int num_steps, int config_rec, int data_rec, int seed, bool, bool);

        void add_all_rates(int, int);

        void add_all_rates_plus_k(int);

        void remove_all_rates(int);

        void print_config();
        
        void print_full_config();

        void prepare_run(bool, int);

        void pre_comp_rates();
    
        ~umbrella_nanocube(){}; 
};

inline double umbrella_nanocube::umbrella_pot(int site_index)
{
    return ( (spring_stiffness / num_parts) * ( ( conc - spring_center ) * spins(site_index) - (1.0 / num_parts) ) ) * umbrella_on;
}

umbrella_nanocube::umbrella_nanocube(double k_ads_f, double del_mu_f, double temp_f, double p_f, double k_spring, double c_spring, int ns, int nz, std::string name, std::string meta_dir, std::string dir, int trial_n)
{
    // k_ads should be raltive to k_d
    // temp in kelvin
    // pressure in atm

    trial_number = trial_n;
    filebase = name;
    meta_direc = meta_dir;
    direc = dir;

    k_ads = k_ads_f;
    temp = temp_f;
    del_mu = del_mu_f;
    p = p_f;
    
    umbrella_on = true;
    spring_center = c_spring;
    spring_stiffness = k_spring / temp;

    n_s = ns;
    n_sq = ns * ns;
    n_z = nz;
    num_parts = ns * ns * n_z;
    
    time = 0.0;
    k_tot = 0.0;
    
    spins.resize(num_parts);
    v_rr.resize(num_parts,num_parts);
    neighb_inds.resize(num_parts);
    neighb_m.resize(num_parts);
    neighb_p.resize(num_parts);
   
    active_index.resize(num_parts);

    max_size_active_sites = n_sq * 16 + 4 * n_s * n_z + 2 * n_sq ;
    e_diffs_el.resize(max_size_active_sites);
    active_rates.resize(max_size_active_sites);
    num_open_slots_active_sites = 0;

    active_sites.resize(max_size_active_sites);
    open_slots_active_sites.resize(max_size_active_sites);
    partial_sums.resize(max_size_active_sites * 7);
    partial_sums_indices.resize(max_size_active_sites * 7);
    partial_sums_labels.resize(max_size_active_sites * 7);
}

umbrella_nanocube::umbrella_nanocube(std::string infilename, int line_num, double k_ads_f, double del_mu_f, double temp_f, double p_f, int ns, int nz, std::string name, std::string dir, std::string meta_dir, int trial_n, bool preamble)
{
    umbrella_on = false;
    spring_center = 0.0;
    spring_stiffness = 0.0;
    
    trial_number = trial_n;
    filebase = name;
    meta_direc = meta_dir;
    direc = dir;
    
    k_ads = k_ads_f;
    temp = temp_f;
    del_mu = del_mu_f;
    p = p_f;
    
    n_s = ns;
    n_sq = ns * ns;
    n_z = nz;
    num_parts = ns * ns * n_z;

    k_tot = 0.0;
    
    spins.resize(num_parts);
    v_rr.resize(num_parts,num_parts);
    neighb_inds.resize(num_parts);
    neighb_m.resize(num_parts);
    neighb_p.resize(num_parts);
   
    active_index.resize(num_parts);

    max_size_active_sites = n_sq * 16 + 4 * n_s * n_z + 2 * n_sq ;
    e_diffs_el.resize(max_size_active_sites);
    active_rates.resize(max_size_active_sites);
    num_open_slots_active_sites = 0;

    active_sites.resize(max_size_active_sites);
    open_slots_active_sites.resize(max_size_active_sites);
    partial_sums.resize(max_size_active_sites * 7);
    partial_sums_indices.resize(max_size_active_sites * 7);
    partial_sums_labels.resize(max_size_active_sites * 7);
    
    std::ifstream infile(infilename, std::ios::in);
    std::string t_s;
    
    if (!infile.is_open())
    {
        throw std::invalid_argument("Can't load config");
    }
    
    if (preamble)
    {
        for (int i = 0; i < 8; i++)
        {
            infile >> t_s;
        }

        //infile >> k_ads >> del_mu >> temp >> p >> n_s >> n_z >> t_s >> t_s >> t_s >> t_s >> t_s >> t_s;
        infile >> k_ads >> t_s >> temp >> p >> n_s >> n_z >> t_s >> t_s >> t_s >> t_s >> t_s >> t_s;

        n_sq = n_s * n_s;
        num_parts = n_s * n_s * n_z;
    }
    
    
    for (int count_lines = 0; count_lines < line_num; count_lines++)
    {
        for (int count_pos = 0; count_pos < (num_parts + 3); count_pos++)
        {
            infile >> t_s;
        }
    }
    
    infile >> time >> conc >> t_s;
    //std::cout << time << " " << conc << " " << t_s << "\n";
    np = 0;
    nm = 0;
    spins.resize(num_parts);
    
    int t_i;
    for (int i = 0; i < num_parts; i++)
    {
        infile >> t_i;
        if (t_i == -1)
        {
            nm += 1;
        }
        else
        {
            np +=1;
        }
        spins(i) = t_i;
    }
    
    configure_surface();
}

umbrella_nanocube::umbrella_nanocube(std::string infilename, int line_num, double k_ads_f, double del_mu_f, double temp_f, double p_f, double k_spring, double c_spring, int ns, int nz, std::string name, std::string dir, std::string meta_dir, int trial_n, bool preamble)
{
    
    trial_number = trial_n;
    filebase = name;
    meta_direc = meta_dir;
    direc = dir;
    
    k_ads = k_ads_f;
    temp = temp_f;
    del_mu = del_mu_f;
    p = p_f;
    
    umbrella_on = true;
    spring_center = c_spring;
    spring_stiffness = k_spring / temp;
    
    n_s = ns;
    n_sq = ns * ns;
    n_z = nz;
    num_parts = ns * ns * n_z;

    k_tot = 0.0;
    
    spins.resize(num_parts);
    v_rr.resize(num_parts,num_parts);
    neighb_inds.resize(num_parts);
    neighb_m.resize(num_parts);
    neighb_p.resize(num_parts);
   
    active_index.resize(num_parts);

    max_size_active_sites = n_sq * 16 + 4 * n_s * n_z + 2 * n_sq ;
    e_diffs_el.resize(max_size_active_sites);
    active_rates.resize(max_size_active_sites);
    num_open_slots_active_sites = 0;

    active_sites.resize(max_size_active_sites);
    open_slots_active_sites.resize(max_size_active_sites);
    partial_sums.resize(max_size_active_sites * 7);
    partial_sums_indices.resize(max_size_active_sites * 7);
    partial_sums_labels.resize(max_size_active_sites * 7);
    
    std::ifstream infile(infilename, std::ios::in);
    std::string t_s;
    
    
    if (!infile.is_open())
    {
        throw std::invalid_argument("Can't load config");
    }
    
    if (preamble)
    {
        for (int i = 0; i < 8; i++)
        {
            infile >> t_s;
        }

        infile >> k_ads >> del_mu >> temp >> p >> n_s >> n_z >> t_s >> t_s >> t_s >> t_s >> t_s >> t_s;

        n_sq = n_s * n_s;
        num_parts = n_s * n_s * n_z;
    }
    
    for (int count_lines = 0; count_lines < line_num; count_lines++)
    {
        for (int count_pos = 0; count_pos < (num_parts + 3); count_pos++)
        {
            infile >> t_s;
        }
    }
    
    infile >> time >> conc >> t_s;
    np = 0;
    nm = 0;
    spins.resize(num_parts);
    
    int t_i;
    for (int i = 0; i < num_parts; i++)
    {
        infile >> t_i;
        if (t_i == -1)
        {
            nm += 1;
        }
        else
        {
            np +=1;
        }
        spins(i) = t_i;
    }
    
    infile.close();
    configure_surface();
}  

void umbrella_nanocube::configure_surface()
{
    connect_lat();
    
    load_veff();

    for (int i = 0; i < num_parts; i++)
    {
        h(i) = (p * h(i) - del_mu)/temp;
        for (int j = i; j < num_parts; j++)
        {
            v_rr(i,j) = v_rr(i, j) / temp;
            v_rr(j,i) = v_rr(i, j);
        }
    }
    
    conc = (1.0 * np) / num_parts;
    size_active_sites = 0;
    k_tot = 0.0;
    current_index_partial_sums = 0;
    set_active_sites();
    energy = arma::dot(spins, v_rr * spins) + 0.5 * arma::dot(h, spins);
}

void umbrella_nanocube::change_mu(double new_mu)
{
    double ht;
    
    for (int i = 0; i < num_parts; i++)
    {
        if (p != 0)
        {
            ht = ( (h(i) * temp + del_mu)/p);
        }
        else
        {
            ht = 0;
        }

        h(i) = (p * ht - new_mu)/temp;
    }

    del_mu = new_mu;
    pre_comp_rates();
}

void umbrella_nanocube::configure_bare(int pm)
{
    spins.fill(pm);
    int c = (pm + 1)/2; 
    nm = num_parts * (1-c);
    np = num_parts * c;
    configure_surface();
}

void umbrella_nanocube::configure_rand(double c, int seed)
{
    contgen.seed(seed);
    contdis.param(std::uniform_real_distribution<double>::param_type(0.0, 1.0));
    np = 0;
    nm = 0;
    for (int i = 0; i < num_parts; i++)
    {
        if (contdis(contgen) < c)
        {
            spins(i) = 1;
            np+=1;
        }
        else
        {
            spins(i) = -1;
            nm+=1;
        }
    }
    configure_surface();
}

void umbrella_nanocube::configure_chunk(double c)
{
    np = int(c * num_parts);
    nm = num_parts - np;
    for (int i = 0; i < np; i++)
    {
        spins(i) = 1;
    }
    for (int i = np; i < num_parts; i++)
    {
        spins(i) = -1;
    }
    configure_surface();
}

void umbrella_nanocube::configure_band(int width)
{
    int index;
    spins.fill(-1);
    nm = num_parts;
    np = 0;
    for (int i = (n_z - width)/2; i <= (n_z + width)/2; i++)
    {
        for (int j = 0; j < n_s; j++)
        {
            for (int k = 0; k < n_s; k++)
            {
                index = i * n_sq + j * n_s + k;
                spins(index) = 1;
                np += 1;
                nm -= 1;
            }
        }
    }
    configure_surface();
}

void umbrella_nanocube::connect_lat()
{
    int count = 0;
    int num;
    for (int i = 0; i < num_parts; i++)
    {
        num = set_neighbors(i);
        neighb_m[i] = num / (zlat + 1);
        neighb_p[i] = num % (zlat + 1);
    }
}

void umbrella_nanocube::check_neighbors()
{
    int num;
    for (int i = 0; i < num_parts; i++)
    {
        num = neighb_m[i] * (zlat+1) + neighb_p[i];
        if (num != set_neighbors(i))
        {
            std::cout << "Incorrect neighbors\n";
        }
    }
}

int umbrella_nanocube::set_neighbors(int site_index)
{
    int a1, a2, a3, qtemp;
    a3 = site_index / n_sq;
    a2 = (site_index / n_s) % n_s;
    a1 = site_index % n_s;
    
    int m1, m2, m3, m_index;
    int count_m = 0;
    int count_p = 0;

    for (int l = 0; l < zlat; l++)
    {
        m1 = a1 + order[l][0];
        m2 = a2 + order[l][1];
        m3 = a3 + order[l][2];

        if (m1 >= 0 && m2 >= 0 && m3 >= 0 && m1 < n_s && m2 < n_s && m3 < n_z)
        {
            m_index = m3 * n_sq + m2 * n_s + m1;
            
            if (spins(m_index) == -1)
            {
                count_m += 1;
            }
            else
            {
                count_p += 1;
            }
        }

        else
        {
            m_index = num_parts;
        }  

        neighb_inds[site_index][l] = m_index;
    }

    return (count_m * (zlat + 1) + count_p);
}

void umbrella_nanocube::load_veff()
{
    v_rr.resize(num_parts, num_parts);
    h.resize(num_parts);
    try
    {
        bool found = v_rr.load(meta_direc + "veffstore_prod/" + vrr_file + std::to_string(n_s) + "_" + std::to_string(n_z) + ".txt", arma::arma_binary);
        bool found2 = h.load(meta_direc + "veffstore_prod/" + hr_file + std::to_string(n_s) + "_" + std::to_string(n_z)+ ".txt", arma::arma_binary);
        if (!(found && found2))
        {
            throw std::invalid_argument("Can't find veff or h file " + meta_direc + "veffstore_prod/" + vrr_file + std::to_string(n_s) + "_" + std::to_string(n_z) + ".txt in trial ");
        }
    }
    catch(const std::exception& e)
    {
        std::cout << e.what() << trial_number << "\n";
        time = max_time;
    }

}

void umbrella_nanocube::set_active_sites()
{
    int num;
    for (int i = 0; i < num_parts; i++)
    {
        if ((neighb_m[i] + neighb_p[i]) < zlat || (spins(i) == 1 && neighb_m[i] > 0))
        {
            add_active_site(i, num_parts);
        }
    }
}

void umbrella_nanocube::add_active_site(int site_index, int sc)
{
    int new_active_index;
   
    if (num_open_slots_active_sites > 0)
    {
        new_active_index = open_slots_active_sites[(num_open_slots_active_sites - 1)];
        open_slots_active_sites[(num_open_slots_active_sites - 1)] = 0;
        num_open_slots_active_sites--;
    }

    else
    {
        size_active_sites++;

        if (size_active_sites >= max_size_active_sites)
        {
            throw std::invalid_argument("Reached maximum active sites in trial " + std::to_string(trial_number));
            return;
        }

        new_active_index = size_active_sites;
    }
    
    active_sites[new_active_index] = site_index;
    active_index[site_index] = new_active_index;

    add_all_rates(new_active_index, sc);
}

void umbrella_nanocube::add_all_rates(int act_index, int sc)
{   
    int site_index = active_sites[act_index];
    int val = spins(site_index);
    int num_m = neighb_m[site_index];
    int num_p = neighb_p[site_index];
    
    if ( (num_m + num_p) < zlat )
    {
        active_rates[act_index][zlat] = 1;
        e_diffs_el[act_index][zlat] = delta_flip_el(site_index, val);
    }

    int n_ind, nval;

    if ( val == 1 )
    {
        for (int l = 0; l < zlat; l++)
        {
            n_ind = neighb_inds[site_index][l];

            if (n_ind < num_parts)
            {
                nval = spins(n_ind);

                if (nval == -1)
                {
                    active_rates[act_index][l] = 1;
                    e_diffs_el[act_index][l] = delta_diff_el(site_index, n_ind);
                }
            }    
        }
    }
}

void umbrella_nanocube::add_all_rates_plus_k(int site_index)
{   
    int act_index = active_index[site_index];
    int val = spins(site_index);
    int num_m = neighb_m[site_index];
    int num_p = neighb_p[site_index];
    
    double delta_el;
    double umbrella_contribution;
    if ( (num_m + num_p) < zlat )
    {
        active_rates[act_index][zlat] = 1;
        delta_el = delta_flip_el(site_index, val);
        e_diffs_el[act_index][zlat] = delta_el;
        umbrella_contribution = umbrella_pot(site_index);
        k_tot += k_ads * exp(-(delta_el - umbrella_contribution) / (2.0));
        partial_sums[current_index_partial_sums] = k_tot;
        partial_sums_indices[current_index_partial_sums] = site_index;
        partial_sums_labels[current_index_partial_sums] = zlat;
        current_index_partial_sums++;
    }

    int n_ind;

    if ( val == 1 )
    {
        for (int l = 0; l < zlat; l++)
        {
            n_ind = neighb_inds[site_index][l];

            if (n_ind < num_parts)
            {
                if (spins(n_ind) == -1)
                {
                    active_rates[act_index][l] = 1;
                    delta_el = delta_diff_el(site_index, n_ind);
                    e_diffs_el[act_index][l] = delta_el;
                    k_tot += exp(-(delta_el) / (2.0));
                    partial_sums[current_index_partial_sums] = k_tot;
                    partial_sums_indices[current_index_partial_sums] = site_index;
                    partial_sums_labels[current_index_partial_sums] = l;
                    current_index_partial_sums++;
                }
            }    
        }
    }
}

double umbrella_nanocube::delta_flip_el(int site_index, int val)
{
   return -val * ( 4 * (arma::dot(v_rr.row(site_index), spins) - v_rr(site_index, site_index) * val) + h(site_index) );
}

double umbrella_nanocube::delta_diff_el(int index_p, int index_m)
{
    return 4 * ( arma::dot( (v_rr.row(index_m) - v_rr.row(index_p)), spins ) + v_rr(index_m, index_m) + v_rr(index_p, index_p) - 2 * v_rr(index_m, index_p) ) + h(index_m) - h(index_p);
}

void umbrella_nanocube::remove_all_rates(int active_index)
{
    for (int l = 0; l < zlat; l++)
    {
        active_rates[active_index][l] = 0;
        e_diffs_el[active_index][l] = 0;
    }
    active_rates[active_index][zlat] = 0;
    e_diffs_el[active_index][zlat] = 0; 
}

void umbrella_nanocube::remove_active_site(int vector_index)
{
    int lat_index = active_sites[vector_index];
    num_open_slots_active_sites++;
    open_slots_active_sites[(num_open_slots_active_sites - 1)] = vector_index;
    active_index[lat_index] = 0;
    active_sites[vector_index] = num_parts;
    remove_all_rates(vector_index);
}

void umbrella_nanocube::prepare_run(bool no_config, int seed)
{
    std::stringstream trial_prefix;
    trial_prefix << meta_direc << direc << "t" << trial_number;
    
    if (!no_config)
    {
        std::string config_filename = trial_prefix.str() + "c" + filebase + ".txt";
        config_stream.open( config_filename, std::ofstream::out | std::ofstream::app );
    }

    std::string data_filename = trial_prefix.str() + "d" + filebase + ".txt";
    data_stream.open( data_filename, std::ofstream::out | std::ofstream::app );
    data_stream << std::scientific <<std::setprecision(16);

    std::string full_filename = trial_prefix.str() + "full" + filebase + ".txt";
    full_stream.open(full_filename, std::ofstream::out | std::ofstream::app );
    full_stream << std::fixed <<std::setprecision(16);
    
    if (!no_config)
    {
        config_stream << std::fixed <<std::setprecision(16);
        config_stream <<  "k_ads del_mu temp pres n_s n_z vrr_file hr_file\n";
        config_stream << k_ads << " " << del_mu << " " << temp << " " << p << " " << n_s << " " << n_z << " " << vrr_file << " " << hr_file << "\n";
        config_stream << "val index neighb_1 neighb_2\n";
    }
    data_stream <<  "k_ads del_mu temp pres n_s n_z vrr_file hr_file\n";
    data_stream << k_ads << " " << del_mu << " " << temp << " " << p << " " << n_s << " " << n_z << " " << vrr_file << " " << hr_file << "\n";
    data_stream <<  "np nm time " << seed << "\n";

    full_stream <<  "k_ads del_mu temp pres n_s n_z vrr_file hr_file\n";
    full_stream << k_ads << " " << del_mu << " " << temp << " " << p << " " << n_s << " " << n_z << " " << vrr_file << " " << hr_file << "\n";
    full_stream <<  "np nm time " << seed << "\n";
}

void umbrella_nanocube::update_neighbors(int site_index, int val)
{
    int n_site_index, n_active_index, numm, nump, nval;
    for (int l = 0; l < zlat; l++)
    {
        n_site_index = neighb_inds[site_index][l];
        n_active_index = active_index[n_site_index];
        
        if (n_site_index < num_parts)
        {
            if (val == 1)
            {
                neighb_p[n_site_index] -= 1;
                neighb_m[n_site_index] += 1;
            }
            else
            {
                neighb_p[n_site_index] += 1;
                neighb_m[n_site_index] -= 1;  
            }
            
            numm = neighb_m[n_site_index];
            nump = neighb_p[n_site_index];
            
            nval = spins(n_site_index); 
            
            if (numm < 0 || nump < 0)
            {
                std::cout << "Wrong Neighbors\n";
            } 
            
            if ( (nval == 1 && numm > 0) || ( (numm + nump) < zlat) )
            {
                if (n_active_index == 0)
                {
                    try
                    {
                        add_active_site(n_site_index, site_index);
                    }
                    catch(const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                        print_config();
                        time = max_time;
                        return;
                    }
                }

                else
                {
                    if (val == -1)
                    {
                        active_rates[n_active_index][opp_pos(l)] = 0;
                    }
                    
                    else if (nval == 1)
                    {
                        active_rates[n_active_index][opp_pos(l)] = 1;
                    }
                }
            }
            else if( n_active_index > 0)
            {
                remove_active_site(n_active_index);
            }
        }
    }
}

void umbrella_nanocube::execute_flip(int site_index, int val)
{
    //First update neighbors with no elastic contribution
    np -= val;
    nm += val;
    conc -= (1.0 * val) / num_parts;

    if (np < 0 || np > num_parts)
    {
        std::cout << "Invalid Conc.\n";
    }

    if ( (neighb_m[site_index] + neighb_p[site_index]) >= zlat || neighb_m[site_index] < 0 || neighb_p[site_index] < 0 )
    {
        std::cout << "Wrong Neighbors.\n";
    }

    k_tot = 0.0;
    current_index_partial_sums = 0;
    
    spins(site_index) = -val;
    update_neighbors(site_index, val);

    remove_all_rates(active_index[site_index]);
    add_all_rates_plus_k(site_index);
    
    //Now account for elastic interactions and sum rates of other sites
    
    int curr_site_index;
    for (int i = 1; i <= size_active_sites; i++)
    {
        if (active_sites[i] < num_parts)
        {
            curr_site_index = active_sites[i];
            
            if (curr_site_index != site_index)
            {
                if (spins(curr_site_index) == 1)
                {
                    for (int l = 0; l < zlat; l++)
                    {
                        if (active_rates[i][l])
                        {
                            if (neighb_inds[curr_site_index][l] != site_index)
                            {
                               e_diffs_el[i][l] -= 8 * (v_rr(neighb_inds[curr_site_index][l], site_index) - v_rr(curr_site_index, site_index)) * val;
                            }
                            else
                            {
                                e_diffs_el[i][l] = delta_diff_el(curr_site_index, site_index);
                            }
                            
                            k_tot += exp(-(e_diffs_el[i][l])/ (2.0));
                            partial_sums[current_index_partial_sums] = k_tot;
                            partial_sums_indices[current_index_partial_sums] = curr_site_index;
                            partial_sums_labels[current_index_partial_sums] = l;
                            current_index_partial_sums++;
                        }
                    }
                }
                
                if (active_rates[i][zlat])
                {
                    e_diffs_el[i][zlat] += 8 * val * spins(curr_site_index) * v_rr(curr_site_index, site_index);
                    k_tot += k_ads * exp(-(e_diffs_el[i][zlat] - umbrella_pot(curr_site_index) ) / (2.0));
                    partial_sums[current_index_partial_sums] = k_tot;
                    partial_sums_indices[current_index_partial_sums] = curr_site_index;
                    partial_sums_labels[current_index_partial_sums] = zlat;
                    current_index_partial_sums++;
                }
            }
        }
    }
}

void umbrella_nanocube::execute_diff(int index_p, int index_m)
{
    //First update neighbors with no elastic contribution
    k_tot = 0.0;
    current_index_partial_sums = 0;
    
    if (spins(index_m) != -1 || spins(index_p) != 1)
    {
        std::cout << "Invalid diffusion.\n";
        std::cout << spins(index_m) << " " << spins(index_p) << "\n";
    }

    if ((neighb_m[index_m] + neighb_p[index_m]) > zlat || neighb_m[index_m] < 0 || neighb_p[index_m] < 0 )
    {
        std::cout << "Invalid neighbors.\n";
    }

    if ((neighb_m[index_p] + neighb_p[index_p]) > zlat || neighb_m[index_p] < 0 || neighb_p[index_p] < 0 )
    {
        std::cout << "Invlaid Neighbors.\n";
    }
    spins(index_m) = 1;
    spins(index_p) = -1;
    
    update_neighbors(index_m, -1);
    update_neighbors(index_p, 1);

    remove_all_rates(active_index[index_m]);
    add_all_rates_plus_k(index_m);
    remove_all_rates(active_index[index_p]);
    add_all_rates_plus_k(index_p); 

    //Now account for elastic interactions and sum rates of other sites
    
    int curr_site_index, n_ind;
    for (int i = 1; i <= size_active_sites; i++)
    {
        if (active_sites[i] < num_parts)
        {
            curr_site_index = active_sites[i];
            
            if (curr_site_index != index_m && curr_site_index != index_p)
            {
                if (spins(curr_site_index) == 1)
                {
                    for (int l = 0; l < zlat; l++)
                    {
                        if (active_rates[i][l])
                        {
                            n_ind = neighb_inds[curr_site_index][l];
                            
                            if ( n_ind != index_p)
                            {
                                e_diffs_el[i][l] -= 8 * ( v_rr(n_ind, index_p) - v_rr(curr_site_index, index_p) - v_rr(n_ind, index_m) + v_rr(curr_site_index, index_m));
                            }

                            else
                            {
                                e_diffs_el[i][l] = delta_diff_el(curr_site_index, index_p);
                            }
                            
                            k_tot += exp( - ( e_diffs_el[i][l]) / (2.0));
                            partial_sums[current_index_partial_sums] = k_tot;
                            partial_sums_indices[current_index_partial_sums] = curr_site_index;
                            partial_sums_labels[current_index_partial_sums] = l;
                            current_index_partial_sums++;
                        }
                    }
                }
                if (active_rates[i][zlat])
                {
                    e_diffs_el[i][zlat] += 8 * spins(curr_site_index) * ( v_rr(curr_site_index, index_p) - v_rr(curr_site_index, index_m) );
                    k_tot += k_ads * exp( -(e_diffs_el[i][zlat] - umbrella_pot(curr_site_index)) / (2.0));
                    partial_sums[current_index_partial_sums] = k_tot;
                    partial_sums_indices[current_index_partial_sums] = curr_site_index;
                    partial_sums_labels[current_index_partial_sums] = zlat;
                    current_index_partial_sums++;
                }
            }
        }
    }
}

void umbrella_nanocube::kmc_step()
{
   /* 
    double k_tot_copy = 0.0;
    int site_index;
    int act_index;
    for (int i = 0; i < num_parts; i++)
    {
        site_index = i;
        act_index = active_index[i];
        int val = spins(site_index);
        int num_m = neighb_m[site_index];
        int num_p = neighb_p[site_index];
        
        double delta_el, delta_chem;
        
        if ( (num_m + num_p) < zlat )
        {
            if (act_index == 0)
            {
                std::cout << "Should Flip.\n";
            }
            if (!active_rates[act_index][zlat])
            {
                std::cout << "Should Flip.\n";
            }
            delta_el = delta_flip_el(site_index, val);
            k_tot_copy += k_ads * exp(-(delta_el - umbrella_pot(site_index))/ 2.0);
        }

        int n_ind;

        if ( val == 1 )
        {
            for (int l = 0; l < zlat; l++)
            {
                n_ind = neighb_inds[site_index][l];

                if (n_ind < num_parts)
                {
                    if (spins(n_ind) == -1)
                    {
                        if (!active_rates[act_index][l])
                        {
                            std::cout << "Should Diffuse.\n";
                        }
                        delta_el = delta_diff_el(site_index, n_ind);
                        k_tot_copy += exp(-delta_el / 2.0);
                    }
                }    
            }
        }
    }
    if (abs((k_tot - k_tot_copy)/k_tot) > 0.0000001)
    {
        std::cout << "wrong total k " << abs((k_tot - k_tot_copy)/k_tot) << "\n";
    }
   */

    //check_neighbors();
    
    double cap = contdis(contgen) * k_tot;
    int next_site;
    int next_label;
    bool stop = false;
    int low_range = 0;
    int high_range = current_index_partial_sums - 1;
    int mid_point;

    if (cap < partial_sums[low_range])
    {
        next_site = partial_sums_indices[low_range];
        next_label = partial_sums_labels[low_range];
    }
    
    else
    {
        while (!stop)
        {
            mid_point = low_range + (high_range - low_range) / 2;

            if (cap < partial_sums[mid_point])
            {
                high_range = mid_point;
            }
            else
            {
                low_range = mid_point;
            }

            if (high_range - low_range == 1)
            {
                stop = true;
                next_site = partial_sums_indices[high_range];
                next_label = partial_sums_labels[high_range];
            }
        }
    }
    
    time -= log( contdis(contgen) ) / k_tot;
    energy += e_diffs_el[active_index[next_site]][next_label];

    if (next_label < zlat)
    {
        execute_diff(next_site, neighb_inds[next_site][next_label]);
    }
    
    else
    {
        execute_flip(next_site, spins(next_site));
    }
    count_steps += 1;
    return;
}

void umbrella_nanocube::pre_comp_rates()
{
    k_tot = 0.0 ;
    current_index_partial_sums = 0;
    int lat_ind;
    for (int i = 1; i <= size_active_sites; i++)
    {
        lat_ind = active_sites[i];
        if (lat_ind < num_parts)
        {
            for (int l = 0; l < zlat; l++)
            {
                if (active_rates[i][l])
                {
                    k_tot += exp(-(e_diffs_el[i][l])/(2.0));
                    partial_sums[current_index_partial_sums] = k_tot;
                    partial_sums_indices[current_index_partial_sums] = lat_ind;
                    partial_sums_labels[current_index_partial_sums] = l;
                    current_index_partial_sums++;
                }
            }
            if (active_rates[i][zlat])
                {
                    k_tot += k_ads * exp(-(e_diffs_el[i][zlat] - umbrella_pot(lat_ind))/(2.0));
                    partial_sums[current_index_partial_sums] = k_tot;
                    partial_sums_indices[current_index_partial_sums] = lat_ind;
                    partial_sums_labels[current_index_partial_sums] = zlat;
                    current_index_partial_sums++;
                }
        }
    }
}

void umbrella_nanocube::kmc_run_time(double total_time, int config_rec, int data_rec, int seed, bool no_config, bool high_res, double high_conc, double low_conc, int high_steps)
{
    prepare_run(no_config, seed);
    contgen.seed(seed);
    contdis.param(std::uniform_real_distribution<double>::param_type(0.0, 1.0));
    //double conc;
    double config_time = total_time / config_rec;
    double data_time = total_time  / data_rec;
    
    time = 0.0;
    
    double start_time = time;
    double config_start_time = time;
    double data_start_time = time;

    double data_time_elapsed = 0.0;
    double config_time_elapsed = 0.0;

    pre_comp_rates();
    if (!no_config)
    {
        if (!high_res)
        {
            print_config();
        }
        else
        {
            print_full_config();
        }
    }
    count_steps = 0;
    
    while ( (time - start_time) < total_time && time < max_time )
    {
        kmc_step();
        //conc = (1.0 * np) / num_parts;
        data_time_elapsed = time - data_start_time;
        config_time_elapsed = time - config_start_time;

        if ( data_time_elapsed >= data_time && (time - start_time) < total_time)
        {
            data_stream << conc << " " << time << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";
            //std::cout << conc << " " << time << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";
            data_start_time = time;
        }
    
        if ( config_time_elapsed >= config_time && (time - start_time) < total_time && !no_config)
        {
            if (!high_res)
            {
                print_config();
            }
            else
            {
                print_full_config();
            }
            config_start_time = time;
        }
        if (count_steps > high_steps || conc > high_conc || conc < low_conc)
        {
            break;
        }
    }

    data_stream << conc << " " << time << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";

    if (!no_config)
    {
        if (!high_res)
        {
            print_config();
        }
        else
        {
            print_full_config();
        }
        config_stream.close();
    }
    print_full_config();
    full_stream.close();
    data_stream.close();
}

void umbrella_nanocube::kmc_run_c_down(double c_end, int config_rec, int data_rec, int seed, bool no_config, bool high_res, int maxsteps)
{
    prepare_run(no_config, seed);
    contgen.seed(seed);
    contdis.param(std::uniform_real_distribution<double>::param_type(0.0, 1.0));
    
    //double conc = 1.0 * np / num_parts;
    double config_conc = (conc - c_end) / config_rec;
    double data_conc = (conc - c_end) / data_rec;

    double config_start_conc = conc;
    double data_start_conc = conc;

    double data_conc_elapsed = 0;
    double config_conc_elapsed = 0;

    pre_comp_rates();
    if (!no_config)
    {
        if (!high_res)
        {
            print_config();
        }
        else
        {
            print_full_config();
        }
    }
    count_steps = 0;
    while ( conc > c_end && time < max_time && count_steps < maxsteps)
    {
        kmc_step();

        data_conc_elapsed = data_start_conc - conc;
        config_conc_elapsed = config_start_conc - conc;
    
        if ( data_conc_elapsed >= data_conc)
        {
            data_stream << conc << " " << time << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";
            //std::cout << conc << " " << time << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";
            data_start_conc = conc;
        }
    
        if ( config_conc_elapsed >= config_conc && !no_config)
        {
            if (!high_res)
            {
                 print_config();
            }
            else
            {
                print_full_config();
            }
            config_start_conc = conc;
        }
    }

    data_stream << conc << " " << time << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";

    if (!no_config)
    {
        print_config();
        config_stream.close();
    }

    print_full_config();
    full_stream.close();
    data_stream.close();
}

void umbrella_nanocube::kmc_run_c_up(double c_end, int config_rec, int data_rec, int seed, bool no_config, bool high_res, int maxsteps)
{
    prepare_run(no_config, seed);
    contgen.seed(seed);
    contdis.param(std::uniform_real_distribution<double>::param_type(0.0, 1.0));
    double config_conc = (c_end - conc) / config_rec;
    double data_conc = (c_end - conc) / data_rec;

    double config_start_conc = conc;
    double data_start_conc = conc;

    double data_conc_elapsed = 0;
    double config_conc_elapsed = 0;

    pre_comp_rates();
    if (!no_config)
    {
        if (!high_res)
        {
            print_config();
        }
        else
        {
            print_full_config();
        }
    }
    
    count_steps = 0;
    time = 0.0;
    
    while ( conc < c_end && time < max_time && count_steps < maxsteps)
    {
        kmc_step();
        data_conc_elapsed = conc - data_start_conc;
        config_conc_elapsed = conc - config_start_conc;
        
        if ( (data_conc_elapsed >= data_conc))
        {

            data_stream << time << " " << conc << " " << energy << " " << count_steps << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";
            //std::cout << conc << " " << time << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";
            data_start_conc = conc;
        }
    
    
        if ( config_conc_elapsed >= config_conc && !no_config)
        {
            if (!high_res)
            {
                print_config();
            }
            else
            {
                print_full_config();
            }
            config_start_conc = conc;
        }
    }
    
    data_stream << time << " " << conc << " " << energy << " " << count_steps  << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";
    
    if (!no_config)
    {
        if (!high_res)
        {
            print_config();
        }
        config_stream.close();
    }
    print_full_config();
    
    full_stream.close();
    data_stream.close();
}

void umbrella_nanocube::kmc_run_steps(int num_steps, int config_rec, int data_rec, int seed, bool no_config, bool high_res)
{
    prepare_run(no_config, seed);
    contgen.seed(seed);
    contdis.param(std::uniform_real_distribution<double>::param_type(0.0, 1.0));
    
    if (config_rec > num_steps)
    {
        config_rec = num_steps;
    }

    int config_steps = num_steps / config_rec;
    
    if (data_rec > num_steps)
    {
        data_rec = num_steps;
    }
    
    int data_steps = num_steps / data_rec;
    count_steps = 0;
    
    pre_comp_rates();
    if (!no_config)
    {
        if (!high_res)
        {
            print_config();
        }
        else
        {
            print_full_config();
        }
    }
    
    while (count_steps < num_steps)
    {
        if (time >= max_time)
        {
            break;
        }
        
        kmc_step();

        if ( count_steps % data_steps == 0)
        {
            data_stream << time << " " << conc << " " << energy << " " <<  count_steps  << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n" ;
            data_stream.flush();
            //std::cout << time << " " << conc << " " << energy << " " <<  count_steps  << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";
        }
    
        if ( count_steps % config_steps == 0 && !no_config)
        {
            if (!high_res)
        {
            print_config();
        }
        else
        {
            print_full_config();
        }
        }
    }
    
    data_stream << time << " " << conc << " " << energy << " " << count_steps  << " " << k_tot << " " << size_active_sites << " " << current_index_partial_sums << "\n";

    if (!no_config)
    {
        print_config();
        config_stream.close();
    }
    print_full_config();
    full_stream.close();
    data_stream.close();
}

void umbrella_nanocube::print_config()
{
    config_stream << time << " " << (1.0 * np) / (1.0 * num_parts) << " ";
    int site_index, lat_index;
    for (int r = 1; r <= size_active_sites; r++)
    {
        site_index = active_sites[r];
        if (site_index < num_parts)
        {
            config_stream << spins(site_index) << " " << site_index << " " << neighb_m[site_index] << " " << neighb_p[site_index] << " ";
        }
    }
    config_stream << "\n";
}

void umbrella_nanocube::print_full_config()
{
    full_stream << time << " " << conc << " " << energy << " " << count_steps << " ";
    for (int r = 0; r < num_parts; r++)
    {
        full_stream << spins(r) << " ";     
    }
    full_stream << "\n";
    full_stream.flush();
}

#endif 
