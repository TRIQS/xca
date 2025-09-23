/*******************************************************************************
 *
 * triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2025, Z. Huang, H. U.R. Strand
 *
 * triqs_soehyb is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * triqs_soehyb is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * triqs_soehyb. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once
#include "nda/nda.hpp"
#include "cppdlr/cppdlr.hpp"
#include <nda/blas/tools.hpp>
#include "strong_cpl.hpp"

using namespace cppdlr;
using namespace nda;

/**
@note n is size of hybridization matrix,i.e. impurity size (number of single-particle basis of impurity); 
@note N is size of Green's function matrix, i.e. the dimension of impurity Fock space;
@note P is number of terms in the decomposition of the hybridization function Delta
@note r is the size of the time grid, i.e. the DLR rank
* */

/**
  * @class fastdiagram
  * @brief Class responsible for fast diagram calculation of a given impurity problem using hybridization expansion.
* */
class fastdiagram {
  public:
  /** 
    * @brief Constructor for fastdiagram, construct itops and diagram topology matrices
    * @param[in] beta inverse temperature
    * @param[in] lambda DLR cutoff parameter
    * @param[in] eps DLR accuracy tolerance
    * @param[in] F impurity annihilation operator in pseudo-particle space, of size n*N*N
    * @param[in] F_dag impurity creation operator in pseudo-particle space, of size n*N*N
    * */
  fastdiagram(double beta, double lambda, imtime_ops itops, nda::array<dcomplex, 3> F, nda::array<dcomplex, 3> F_dag);

  void hyb_init(nda::array<dcomplex, 3> Deltat0, bool poledlrflag = true);

  /** 
    * @brief calculate decomposition and reflection of hybridization Deltat
    * @param[in] Deltat hybridization function in imaginary time, nda array of size r*n*n
    * @param[in] poledlrflag flag for whether to use dlr for pole expansion. True for using dlr. False has not been implemented yet. 
    * */
  void hyb_decomposition(bool poledlrflag = true, double eps = 0.0);

  nda::vector<dcomplex> get_it_actual();

  /** 
    * @brief free green's function, wrapped from free_gf of cppdlr 
    * */
  nda::array<dcomplex, 3> free_greens(double beta, nda::array<dcomplex, 2> H_S, double mu = 0.0, bool time_order = false);

  /** 
    * @brief free pseudo-particle green's function, wrapped from free_gf_ppsc
    * */
  nda::array<dcomplex, 3> free_greens_ppsc(double beta, nda::array<dcomplex, 2> H_S);

  double partition_function(nda::array<dcomplex, 3> Gt);

  /** 
    * @brief Compute pseudo-particle self energy diagram of certain order, given pseudo-particle Green's function G(t)
    * @param[in] Gt pseudo-particle Green's function G(t), of size r*N*N
    * @param[in] order diagram order: "NCA", "OCA" or "TCA"
    * @return pseudo-particle self energy diagram, r*N*N
    * */
  nda::array<dcomplex, 3> Sigma_calc(nda::array<dcomplex, 3> Gt, std::string order);

  /** 
    * @brief Compute impurity Green's function diagram of certain order, given pseudo-particle Green's function G(t)
    * @param[in] Gt pseudo-particle Green's function G(t), of size r*N*N
    * @param[in] order diagram order: "NCA", "OCA" or "TCA"
    * @return impurity Green's function diagram, r*n*n
    * */
  nda::array<dcomplex, 3> G_calc(nda::array<dcomplex, 3> Gt, std::string order);

  nda::array<dcomplex, 3> time_ordered_dyson(double beta, nda::array<dcomplex, 2> H_S, double eta_0, nda::array_const_view<dcomplex, 3> Sigma_t);

  int number_of_diagrams(int m);
  nda::array<dcomplex, 3> Sigma_calc_group(nda::array<dcomplex, 3> Gt, nda::array<int, 2> D, nda::array<int, 1> diagramindex);
  nda::array<dcomplex, 3> G_calc_group(nda::array<dcomplex, 3> Gt, nda::array<int, 2> D, nda::array<int, 1> diagramindex);

  nda::array<dcomplex, 3> Deltaiw;
  nda::array<dcomplex, 3> Deltaiw_reflect;
  nda::vector<dcomplex> dlr_if_dense;

  void copy_aaa_result(nda::vector<double> pol0, nda::array<dcomplex, 3> weights0);

  private:
  double beta;   // inverse temperature
  double lambda; // DLR cutoff parameter

  imtime_ops itops; // DLR imaginary time objects from cppdlr

  nda::array<dcomplex, 3> F;     // impurity annihilation operator in pseudo-particle space, of size n*N*N
  nda::array<dcomplex, 3> F_dag; // impurity creation operator in pseudo-particle space, of size n*N*N

  int n; // no operator flavours in expansion
  int r; // no DLR coefficients (itops.rank())
  int N; // size of local Hilbert space
  int P;

  nda::vector<double> dlr_rf;        // DLR real frequencies
  nda::vector<double> dlr_it;        // DLR imaginary time nodes
  nda::vector<double> dlr_it_actual; // DLR imaginary time nodes

  nda::array<dcomplex, 3> Deltat;         // Hybridization function in imaginary time, nda array of size r*n*n
  nda::array<dcomplex, 3> Deltat_reflect; // Delta(beta-t), of size r*n*n

  hyb_F Delta_F;         // Compression of Delta(t) and F, F_dag matrices
  hyb_F Delta_F_reflect; // Compression of Delta(-t) and F, F_dag matrices

  nda::array<int, 2> D_NCA;   // NCA diagram information
  nda::array<int, 2> D_OCA;   // OCA diagram information
  nda::array<int, 2> D_TCA_1; // TCA 1st diagram information
  nda::array<int, 2> D_TCA_2; // TCA 2nd diagram information
  nda::array<int, 2> D_TCA_3; // TCA 3rd diagram information
  nda::array<int, 2> D_TCA_4; // TCA 4th diagram information

  nda::vector<double> pol;
  nda::array<dcomplex, 3> weights;
  nda::vector<double> pol_reflect;
  nda::array<dcomplex, 3> weights_reflect;
};
