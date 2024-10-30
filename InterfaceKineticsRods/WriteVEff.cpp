// Potential for 3 spring constants on cubic lattice (rectangular nanocrystal). 300 K temp. Con_stants fit to elastic moduli from Hsu, Leisure PRB 20 (4) 1979, mid point used between alpha and beta phase data
#include<armadillo>
#include<fstream>
#include<vector>
#include<cmath>

int main(int argc, char** argv)
{
    // Spring constants in units of K / (Angstrom)^2
    double k_1 = 2.5882 * pow(10,4);//2.5909 * pow(10,4);//
    double k_2 = 2.0084 * pow(10,4);//2.0070 * pow(10,4);//

    // Lattice constant expansion in Angstrom
    double a0 = 3.8900;
    double delta = 0.1387; //0.1362;
    double joule_per_Latm = 0.0098692326671601; // Latm per joule
    double LperA3 = pow(10,-21); //Liter to Angstrom^3
    double kB = 1.380649 * pow(10,-23); //Boltzmann constant in J K^-1

    double p_tilde = (3.0/2.0) * delta * pow( (a0 + delta * 0.5), 2) * LperA3 * joule_per_Latm / kB; //Prefactor to pressure, units of kelvin per atmosphere
    const int zlat_alpha = 6;
    const int zlat_gamma = 12; 

    const int n_s = std::atoi(argv[1]);
    const int n_z = std::atoi(argv[2]);
    
    std::cout << "Beggining construction of interaction potential for dimensions " << n_s << " " << n_z << "\n";

    const int num_parts = n_s * n_s * n_z;
    const int n_sq = n_s * n_s;
    
    std::vector<std::vector<int> > order = { {1, 0, 0},{-1, 0, 0}, {0, 1, 0},{0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}, 
         {1, 0, 1}, {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1},  {0, 1, 1}, {0, -1, -1}, {0, -1, 1}, {0, 1, -1} };
    
    double per_sqrt_2 = 1.0 / sqrt(2);

    std::vector<double> xtl_1_xyz = {1, 0, 0}; 
    std::vector<double> xtl_2_xyz = {0, 1, 0};
    std::vector<double> xtl_3_xyz = {0, 0, 1};
    
    std::vector<std::vector<double> > bonds_alpha;
    
    bonds_alpha.resize(zlat_alpha);
    
    for (int i = 0; i < zlat_alpha; i++)
    {
        bonds_alpha[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            bonds_alpha[i][j] = xtl_1_xyz[j] * order[i][0] + xtl_2_xyz[j] * order[i][1] + xtl_3_xyz[j] * order[i][2];
        }
    }
    
    std::vector<std::vector<std::vector<double> > >alpha_mat;
    alpha_mat.resize(zlat_alpha);
    
    for (int i = 0; i < zlat_alpha; i++)
    {
        alpha_mat[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            alpha_mat[i][j].resize(3);
            for (int k = 0; k < 3; k++)
            {
                alpha_mat[i][j][k] = bonds_alpha[i][j] * bonds_alpha[i][k];
            }
        }
    }

    std::vector<std::vector<double> > bonds_gamma;
    bonds_gamma.resize(zlat_gamma);
    
    for (int i = 0; i < zlat_gamma; i++)
    {
        bonds_gamma[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            bonds_gamma[i][j] = per_sqrt_2 * (xtl_1_xyz[j] * order[i + zlat_alpha][0] + xtl_2_xyz[j] * order[i + zlat_alpha][1] + xtl_3_xyz[j] * order[i + zlat_alpha][2]);
        }
    }
    
    std::vector<std::vector<std::vector<double> > >gamma_mat;
    gamma_mat.resize(zlat_gamma);
    for (int i = 0; i < zlat_gamma; i++)
    {
        gamma_mat[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            gamma_mat[i][j].resize(3);
            for (int k = 0; k < 3; k++)
            {
                gamma_mat[i][j][k] = bonds_gamma[i][j] * bonds_gamma[i][k];
            }
        }
    }

    arma::mat dyn(3 * num_parts, 3 * num_parts, arma::fill::zeros);
    arma::mat coup(3 * num_parts, num_parts, arma::fill::zeros);
    arma::mat spin(num_parts, num_parts, arma::fill::zeros);
    arma::vec z_vec(num_parts);
    
    int a1, a2, a3, b1, b2, b3;
    // index = a3 * n_sq + a2 * ns + a1 convention for ordering
    
    int neighb_ind;
    double z_r;
    for (int i = 0; i < num_parts; i++)
    {
        a3 = i / n_sq;
        a2 = (i / n_s) % n_s;
        a1 = i % n_s; 
        for (int j = 0; j < num_parts; j++)
        {
            z_r = 0.0;
            for (int l = 0; l < zlat_alpha; l++)
            {
                b1 = a1 + order[l][0];
                b2 = a2 + order[l][1];
                b3 = a3 + order[l][2];
                
                if (b1 >= 0 && b2 >= 0 && b3 >= 0 && b1 < n_s && b2 < n_s && b3 < n_z)
                {
                    neighb_ind = b3 * n_sq + b2 * n_s + b1;
                }
                else
                {
                    neighb_ind = num_parts;
                }
                
                if (neighb_ind < num_parts)
                {
                    z_r += k_1 / 2.0;
                    if (i == j)
                    {
                        spin(i, j) += k_1 * delta * delta / 32.0;
                        for (int x = 0; x < 3; x++)
                        {
                            coup(i * 3 + x, j) -= bonds_alpha[l][x] * k_1 * delta / 4.0;
                            
                            for (int y = 0; y < 3; y++)
                            {
                                dyn(i * 3 + x, j * 3 + y) += alpha_mat[l][x][y] * k_1 / 2.0;
                            }
                        }
                    }
                    else if (neighb_ind == j)
                    {
                        spin(i, j) += k_1 * delta * delta / 32.0;
                        for (int x = 0; x < 3; x++)
                        {
                            coup(i * 3 + x, j) -= bonds_alpha[l][x] * k_1 * delta / 4.0;
                            
                            for (int y = 0; y < 3; y++)
                            {
                                dyn(i * 3 + x, j * 3 + y) -= alpha_mat[l][x][y] * k_1 / 2.0;
                            }
                        }
                    }
                }
            }

            for (int l = 0; l < zlat_gamma; l++)
            {
                b1 = a1 + order[l + zlat_alpha][0];
                b2 = a2 + order[l + zlat_alpha][1];
                b3 = a3 + order[l + zlat_alpha][2];
                
                if (b1 >= 0 && b2 >= 0 && b3 >= 0 && b1 < n_s && b2 < n_s && b3 < n_z)
                {
                    neighb_ind = b3 * n_sq + b2 * n_s + b1;
                }
                else
                {
                    neighb_ind = num_parts;
                }
                
                if (neighb_ind < num_parts)
                {
                    z_r += k_2;
                    if (i == j)
                    {
                        spin(i, j) += 2.0 * k_2 * delta * delta / 32.0;
                        
                        for (int x = 0; x < 3; x++)
                        {
                            coup(i * 3 + x, j) -= bonds_gamma[l][x] * sqrt(2.0) * k_2 * delta / 4.0;
                            
                            for (int y = 0; y < 3; y++)
                            {
                                dyn(i * 3 + x, j * 3 + y) += gamma_mat[l][x][y] * k_2 / 2.0;
                            }
                        }
                    }
                    else if (neighb_ind == j)
                    {
                        spin(i, j) += 2.0 * k_2 * delta * delta / 32.0;
                        for (int x = 0; x < 3; x++)
                        {
                            coup(i * 3 + x, j) -= bonds_gamma[l][x] * sqrt(2.0) * k_2 * delta / 4.0;
                            
                            for (int y = 0; y < 3; y++)
                            {
                                dyn(i * 3 + x, j * 3 + y) -= gamma_mat[l][x][y] * k_2 / 2.0;
                            }
                        }
                    }
                }
            }
            if (j == 0)
            {
                z_vec(i) = z_r;
            }
        }
    }
    
    double z_bar = arma::sum(z_vec);
    arma::vec h_vec = (num_parts * p_tilde * z_vec) / (2 * z_bar);
    z_vec = z_vec * delta;

    // Pin 6 degrees of freedom on the cube to eliminate global translations and rotations from the dynamical matrix
    dyn.shed_row(3*n_s*n_s*n_z-2);
    dyn.shed_col(3*n_s*n_s*n_z-2);
    dyn.shed_row(3*n_s*n_s-1);
    dyn.shed_col(3*n_s*n_s-1);
    dyn.shed_row(3*(n_s*(n_s-1)+1)-3);
    dyn.shed_col(3*(n_s*(n_s-1)+1)-3);

    dyn.shed_rows(0,2);
    dyn.shed_cols(0,2);
    
    coup.shed_row(3*n_s*n_s*n_z-2);
    coup.shed_row(3*n_s*n_s-1);
    coup.shed_row(3*(n_s*(n_s-1)+1)-3);
    coup.shed_rows(0,2);
    
    arma::mat dinvc = arma::inv(dyn) * coup;
    arma::mat v_rr = spin - 0.25 * coup.t() * dinvc;

    std::string meta_dir = "/pscratch/sd/s/soakslea/veffstore_prod/";
    v_rr.save(meta_dir + "vrr2springs_0K_arma_bin_nokmat_" + std::to_string(n_s) + "_" + std::to_string(n_z) + ".txt", arma::arma_binary);
    h_vec.save(meta_dir + "hr2springs_0K_arma_bin_nokmat_" + std::to_string(n_s) + "_" + std::to_string(n_z) + ".txt", arma::arma_binary);
    dinvc.save(meta_dir + "dinvc2springs_0K_arma_bin_nokmat_" + std::to_string(n_s) + "_" + std::to_string(n_z) + ".txt", arma::arma_binary);
    return 0;
}


