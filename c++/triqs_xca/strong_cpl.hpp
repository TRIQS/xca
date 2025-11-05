/*******************************************************************************
 *
 * triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2025, Z. Huang, H. U.R. Strand
 *
 * triqs_xca is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * triqs_xca is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * triqs_xca. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once
#include "nda/nda.hpp"
#include "cppdlr/cppdlr.hpp"
#include <nda/blas/tools.hpp>
#include <nda/linalg/matmul.hpp>

using namespace cppdlr;
using namespace nda;
using nda::linalg::matmul;

/**
@note n is size of hybridization matrix,i.e. impurity size (number of single-particle basis of impurity); 
@note N is size of Green's function matrix, i.e. the dimension of impurity Fock space;
@note P is number of terms in the decomposition of the hybridization function Delta
@note r is the size of the time grid, i.e. the DLR rank
* */

/**
  * @class hyb_decomp
  * @brief Class responsible for decomposition of a hybridization function: Delta_ab(iv) = sum_R U(R,a)V(R,b)/(iv-w(R))
* */
class hyb_decomp {
  public:
  nda::vector<double> w;   // real-valued poles, vector of length P
  nda::matrix<dcomplex> U; // P*n array: complex-valued;
  nda::matrix<dcomplex> V; // P*n array: complex-valued;

  /** 
    * @brief Constructor for hyb_decomp
    * @param[in] Matrices Matrices corresponding to poles; if use DLR, Matrices = Deltadlr (i.e., DLR coefficients for Delta(t))
    * @param[in] poles real poles; if use DLR, poles = dlr_rf (i.e., DLR real frequencies)
    * @param[in] eps SVD truncation threshold
    * */
  hyb_decomp(nda::array_const_view<dcomplex, 3> Matrices, nda::vector_const_view<double> poles, double eps = 0);

  /** 
    * @brief check accuracy of decomposition
    * @param[in] Deltat hybridization function Delta(t) evaluated at dlr_it, r*n*n
    * @param[in] dlr_it: DLR imaginary time nodes
    * */
  void check_accuracy(nda::array_const_view<dcomplex, 3> Deltat, nda::vector_const_view<double> dlr_it);
};

/**
  * @class hyb_F
  * @brief Class responsible for compressing hybridization function with F and F_dag matrices, built on top of hyb_decomp class
* */
class hyb_F {

  public:
  int N; // size of local Hilbert space
  int r; // number of DLR coefficients
  int n; // number of single particle flavours
  int P; // number of poles in compressed representation P <= n*r

  // All arrays below are allocated with P = n*r
  nda::vector<double> c;           // constant corresponding to each pole, of size P
  nda::vector<double> w;           // poles, of size P
  nda::array<dcomplex, 4> U_tilde; // compression of U and F, of size (P,r,N,N)
  nda::array<dcomplex, 4> V_tilde; // compression of F_dag and V, of size (P,r,N,N)
  nda::array<double, 2> K_matrix;  // K(w,tau) matrix with w being poles and tau in dlr_it

  /** 
    * @brief Constructor for hyb_F
    * @param[in] N size of local Hilbert space
    * @param[in] r number of DLR coefficients
    * @param[in] n number of single particle flavours
    * */
  hyb_F(int N, int r, int n);

  hyb_F() = delete; // Disable copy constructor

  /** 
    * @brief Update factorization inplace
    * @param[in] hyb_decomp hyb_decomp class
    * @param[in] dlr_rf DLR real frequencies
    * @param[in] dlr_it DLR imaginary time nodes
    * @param[in] F impurity operator in pseudo-particle space, of size n*N*N
    * @param[in] F_dag impurity operator in pseudo-particle space, of size n*N*N
    * */
  void update_inplace(const hyb_decomp &hyb_decomp, nda::vector_const_view<double> dlr_it, nda::array_const_view<dcomplex, 3> F,
                      nda::array_const_view<dcomplex, 3> F_dag);
};

/** 
* @brief Evaluate a pseudo-particle self energy diagram with given topology and forward/backward flag
* @param[in] hyb_F_self class hyb_F for Delta(t)
* @param[in] hyb_F_reflect class hyb_F for Delta(beta-t)
* @param[in] D matrix for diagram topology
* @param[in] Deltat hybridization function Delta(t), nda array of size r*n*n
* @param[in] Deltat_reflect Delta(beta-t), nda array of size r*n*n
* @param[in] Gt pseudo-particle Green's function G(t), nda array of size r*N*N
* @param[in] beta inverse temperature
* @param[in] F impurity operator in pseudo-particle space, of size n*N*N
* @param[in] F_dag impurity operator in pseudo-particle space, of size n*N*N
* @param[in] fb vector of length m, indicator of the 2-th to m-th hybridization line, whether they are forward/backward
* @param[in] backward flag for whether to include the diagram that is opposite to fb
* @return pseudo-particle self energy Sigma(t) r*N*N
* */
nda::array<dcomplex, 3> Sigma_Diagram_calc(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                           nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect,
                                           nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                           nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                                           nda::vector_const_view<int64_t> fb, bool backward = true);

/**
* @brief Evaluating impurity Green's function diagram with given topology and forward/backward flag. Input arguments are the same as Sigma_Diagram_calc
* @return impurity Green's function r*n*n
* */
nda::array<dcomplex, 3> G_Diagram_calc(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                       nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> F,
                                       nda::array_const_view<dcomplex, 3> F_dag, nda::vector_const_view<int64_t> fb);

/** 
* @brief Evaluating all pseudo-particle self energy diagram with given topology, which sum over all forward and backward choices. For explanation of input parameters, see Sigma_Diagram_calc.
* @return pseudo-particle self energy r*N*N
* */
nda::array<dcomplex, 3> Sigma_Diagram_calc_sum_all(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                                   nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect,
                                                   nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                                   nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag);

/** 
* @brief Evaluating all impurity Green's function diagram with given topology, which sum over all forward and backward choices. For explanation of input parameters, see Sigma_Diagram_calc.
* @return impurity Green's function r*n*n
* */
nda::array<dcomplex, 3> G_Diagram_calc_sum_all(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                               nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                               nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag);

/** 
* @brief Evaluating OCA pseudo-particle self energy diagram with given topology and forward/backward flag. Input arguments are the same as Sigma_Diagram_calc
* @return pseudo-particle self energy r*N*N
* */
nda::array<dcomplex, 3> Sigma_OCA_calc(hyb_F &hyb_F, nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect,
                                       nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> F,
                                       nda::array_const_view<dcomplex, 3> F_dag, bool backward = true);

/** 
* @brief Evaluating OCA impurity Green's function diagram with given topology and forward/backward flag. Input arguments are the same as Sigma_Diagram_calc
* @return impurity Green's function r*n*n
* */
nda::array<dcomplex, 3> G_OCA_calc(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                   nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                                   nda::vector_const_view<int64_t> fb);

/** 
* @brief Gt(k,_,_) = matmul(Ft(k,_,_), Gt(k,_,_))
* */
void multiplicate_onto(nda::array_const_view<dcomplex, 3> Ft, nda::array_view<dcomplex, 3> Gt);

/** 
* @brief Ft(k,_,_) = matmul(Ft(k,_,_), Gt(k,_,_))
* */
void multiplicate_onto_left(nda::array_view<dcomplex, 3> Ft, nda::array_const_view<dcomplex, 3> Gt);

/** 
* @brief cut hybridization lines
* */
void cut_hybridization(int v, int Rv, nda::array_const_view<int, 2> D, double &constant, nda::array_const_view<dcomplex, 3> U_tilde_here,
                       nda::array_const_view<dcomplex, 3> V_tilde_here, nda::array_view<dcomplex, 4> line, nda::array_view<dcomplex, 4> vertex,
                       double chere, double w_here, nda::array_const_view<double, 1> K_matrix_here, int r, int N);

/** 
* @brief summation that happens only at the vertex connected to 0-th vertex
* */
void special_summation(nda::array_view<dcomplex, 3> T, nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                       nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect, int n, int r, int N,
                       bool backward = true);

/** 
* @brief final step of impurity Green's function diagram
* */
void final_evaluation(nda::array_view<dcomplex, 3> Diagram, nda::array_const_view<dcomplex, 3> T, nda::array_const_view<dcomplex, 3> T_left,
                      nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag, int n, int r, int N, double constant);

nda::array<dcomplex, 3> evaluate_one_diagram(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                             nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect,
                                             nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                             nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                                             nda::vector_const_view<int64_t> fb, bool backward, int num0, int m, int n, int r, int N, int P);

nda::array<dcomplex, 3> eval_one_diagram_G(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                           nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Gt_reflect, imtime_ops &itops,
                                           double beta, nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                                           nda::vector_const_view<int64_t> fb, int num0, int m, int n, int r, int N, int P);

dcomplex trace_matmul(nda::array_const_view<dcomplex, 2> M1, nda::array_const_view<dcomplex, 2> M2);
