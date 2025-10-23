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

#include "utils.hpp"
#include "strong_cpl.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <nda/linalg/matmul.hpp>
#include <omp.h>

#include <mpi/mpi.hpp>

#include <triqs/utility/timer.hpp>

using namespace cppdlr;
using namespace nda;

/**
@note n is size of hybridization matrix,i.e. impurity size (number of single-particle basis of impurity); 
@note N is size of Green's function matrix, i.e. the dimension of impurity Fock space;
@note P is number of terms in the decomposition of the hybridization function Delta
@note r is the size of the time grid, i.e. the DLR rank
* */

hyb_decomp::hyb_decomp(nda::array_const_view<dcomplex, 3> Matrices, nda::vector_const_view<double> poles, double eps) {
  //This is mainly done by SVD of the Matrices

  int p = Matrices.shape(0);
  int n = Matrices.shape(1);

  // prepare for svd
  auto s_vec    = nda::array<double, 1>(n);
  auto U_local  = nda::matrix<dcomplex, F_layout>(n, n);
  auto VT_local = nda::matrix<dcomplex, F_layout>(n, n);
  auto a        = nda::array<dcomplex, 2, F_layout>(n, n);

  //obtain max number of poles
  int max_num_pole = n * p;

  //prepare places to store w, U, V
  auto U_all = nda::matrix<dcomplex>(n, max_num_pole);
  auto V_all = nda::matrix<dcomplex>(max_num_pole, n);
  auto w_all = nda::vector<double>(max_num_pole);

  //loop over all dlr frequencies, do svd, truncate singular value that are too small
  int P = 0;
  for (int i = 0; i < p; ++i) {

    a = Matrices(i, _, _); // I am not happy about making a copy here
    nda::lapack::gesvd(a, s_vec, U_local, VT_local);

    for (int d = 0; d < n; ++d) {
      if (s_vec(d) > eps) {
        w_all(P)    = poles(i);
        U_all(_, P) = U_local(_, d) * sqrt(s_vec(d));
        V_all(P, _) = VT_local(d, _) * sqrt(s_vec(d));
        P += 1;
      }
    }
  }

  w = w_all(range(P));
  U = transpose(U_all(_, range(P))); //transpose U to get the shape that we want
  V = V_all(range(P), _);
}

void hyb_decomp::check_accuracy(nda::array_const_view<dcomplex, 3> Deltat, nda::vector_const_view<double> dlr_it) {

  int r = Deltat.shape(0);
  int n = Deltat.shape(1);
  int P = U.shape(0);

  std::cout << "calculating error of the decomposition of hybridization" << std::endl;
  auto Deltat_approx = nda::array<dcomplex, 3>(r, n, n);
  Deltat_approx      = 0;

  //reconstruct approximation to Delta(t)
  for (int k = 0; k < r; ++k) {
    for (int R = 0; R < P; ++R) {
      for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) { Deltat_approx(k, a, b) += k_it(dlr_it(k), w(R)) * U(R, a) * V(R, b); }
      }
    }
  }
  std::cout << "Max error in decomposition of Delta(t): " << max_element(abs((Deltat - Deltat_approx))) << std::endl;
}

hyb_F::hyb_F(int N, int r, int n)
   : N(N),
     r(r),
     n(n),
     P(2 * r), // using P = 2*r for allocation, worst case P = r*n
     c(P),
     w(P),
     U_tilde(P, r, N, N),
     V_tilde(P, r, N, N),
     K_matrix(P, r) {}

void hyb_F::update_inplace(const hyb_decomp &hyb_decomp, nda::vector_const_view<double> dlr_it, nda::array_const_view<dcomplex, 3> F,
                           nda::array_const_view<dcomplex, 3> F_dag) {

  if (F.shape(0) != hyb_decomp.U.shape(1))
    throw std::runtime_error("F matrices not in the right format, or F matrices and hybridization shape do not match");

  if (F.shape(1) != F.shape(2)) throw std::runtime_error("F are not square matrices");

  int n_in = F.shape(0);
  int N_in = F.shape(1);
  int r_in = dlr_it.shape(0);
  int P_in = hyb_decomp.V.shape(0);

  if (N_in != N || r_in != r || n_in != n || P_in != c.shape(0)) {

    mpi::communicator comm;

    if (comm.rank() == 0)
      std::cout << "hyb_F::update_inplace resize: "
                << "N, P, r = " << N << ", " << P << ", " << r << " in (N, P, r = " << N_in << ", " << P_in << ", " << r_in << ") "
                << "c.shape(0) = " << c.shape(0) << std::endl;
    N = N_in;
    n = n_in;
    P = P_in;
    r = r_in;
    c.resize(P);
    w.resize(P);
    U_tilde.resize(P, r, N, N);
    V_tilde.resize(P, r, N, N);
    K_matrix.resize(P, r);
  }

  P = P_in;

  auto U_c = arraymult(hyb_decomp.U, F_dag);
  auto V_c = arraymult(hyb_decomp.V, F);

  for (int R = 0; R < P; ++R) {
    for (int k = 0; k < r; ++k) { K_matrix(R, k) = k_it(dlr_it(k), hyb_decomp.w(R)); }
  }

  for (int R = 0; R < P; ++R) {
    for (int k = 0; k < r; ++k) { U_tilde(R, k, _, _) = K_matrix(R, k) * U_c(R, _, _); }
    if (hyb_decomp.w(R) < 0) {
      for (int k = 0; k < r; ++k) { V_tilde(R, k, _, _) = k_it(dlr_it(k), -hyb_decomp.w(R)) * V_c(R, _, _); }
      c(R) = 1 / k_it(0, -hyb_decomp.w(R));
    } else {
      for (int k = 0; k < r; ++k) { V_tilde(R, k, _, _) = K_matrix(R, k) * V_c(R, _, _); }
      c(R) = 1 / k_it(0, hyb_decomp.w(R));
    }
  }
  w() = hyb_decomp.w;
}

nda::array<dcomplex, 3> G_Diagram_calc_sum_all(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                               nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                               nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag) {

  auto Gt_reflect           = itops.reflect(Gt);
  int r                     = Gt.shape(0);  // size of time grid
  int N                     = Gt.shape(1);  // size of G matrices
  int m                     = D.shape(0);   // order of diagram
  int P                     = hyb_F_self.P; //number of poles
  int n                     = F.shape(0);   //impurity size
  auto Diagram              = nda::array<dcomplex, 3>::zeros({r, n, n});
  auto total_num_fb_diagram = pown(2, m - 1); // total number of forward and backward choices
  auto num_diagram_per_fb   = pown(P, m - 1); //number of diagrams per fb
  auto total_num_diagram    = num_diagram_per_fb * total_num_fb_diagram;

  std::cout << "total_num_diagram = " << num_diagram_per_fb * total_num_fb_diagram << "\n";

  triqs::utility::timer timer_run;
  timer_run.start();

  //#pragma omp parallel
  //{
  //#pragma omp for
  for (int64_t num = 0; num < total_num_diagram; ++num) {
    auto fb   = nda::vector<int64_t>(m); //utility for iteration
    auto num0 = num / num_diagram_per_fb;
    for (int v = 1; v < m; ++v) {
      fb[v] = num0 % 2;
      num0  = num0 / 2;
    }
    // BUG! This accumulation has a race condition between threads! FIXME!
    Diagram = Diagram + eval_one_diagram_G(hyb_F_self, hyb_F_reflect, D, Gt, Gt_reflect, itops, beta, F, F_dag, fb, num, m, n, r, N, P);
  }
  // } #end pragma omp parallell

  std::cout << "Total time: " << timer_run << "\n";

  return Diagram;
}

nda::array<dcomplex, 3> eval_one_diagram_G(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                           nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Gt_reflect, imtime_ops &itops,
                                           double beta, nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                                           nda::vector_const_view<int64_t> fb, int num0, int m, int n, int r, int N, int P) {
  //initialize diagram
  auto Diagram = nda::array<dcomplex, 3>::zeros({r, n, n});
  if (m == 1) {
    //evaluate NCA diagram directly
    double constant = 1.0;
    final_evaluation(Diagram, Gt, Gt_reflect, F, F_dag, n, r, N, constant);
    return Diagram;
  }

  auto R      = nda::vector<int>(m);                     //utility for iteration
  auto line   = nda::array<dcomplex, 4>(2 * m, r, N, N); //used for stotring line objects
  auto vertex = nda::array<dcomplex, 4>(2 * m, r, N, N); //used for stotring vertex objects

  auto T      = nda::array<dcomplex, 3>(r, N, N); //used for storing diagram (from 0 to t)
  auto T_left = nda::array<dcomplex, 3>(r, N, N); //used for storing diagram (from t to beta)

  //obtain R2, ... , Rm, store as R[1],...,R[m-1]
  for (int v = 1; v < m; ++v) {
    R[v] = num0 % P;
    num0 = int(num0 / P);
  }

  //Phase 1: initialize line object L and point object P;
  //This is done exactly the same as in Sigma diagrams
  line = 0;
  for (int s = 1; s <= 2 * m - 1; ++s) line(s, _, _, _) = Gt; //initialize the line object
  double constant = 1;

  vertex = 0;
  //initialize vertex objects by cutting the 2th-mth hybridization line
  for (int v = 1; v < m; ++v) {
    if (fb(v) == 0)
      cut_hybridization(v, R(v), D, constant, hyb_F_self.U_tilde(R(v), _, _, _), hyb_F_self.V_tilde(R(v), _, _, _), line, vertex, hyb_F_self.c(R(v)),
                        hyb_F_self.w(R(v)), hyb_F_self.K_matrix(R(v), _), r, N);
    else
      cut_hybridization(v, R(v), D, constant, hyb_F_reflect.U_tilde(R(v), _, _, _), hyb_F_reflect.V_tilde(R(v), _, _, _), line, vertex,
                        hyb_F_reflect.c(R(v)), hyb_F_reflect.w(R(v)), hyb_F_reflect.K_matrix(R(v), _), r, N);
  }

  //Phase 2: integrate out the stuff on the right (from 0 to t)

  // first, calculate P(t1)*G(t1)
  for (int k = 0; k < r; ++k) T(k, _, _) = matmul(vertex(1, k, _, _), Gt(k, _, _));
  //integrate out the stuff on the right (from 0 to t). In each for loop, first convolution, then multiplication.
  for (int s = 1; s < D(0, 1); ++s) {
    // integrate ts out by convolution:  integral L_s(t(s+1)-ts) D(ts) dts
    T = itops.convolve(beta, Fermion, itops.vals2coefs(line(s, _, _, _)), itops.vals2coefs(T), TIME_ORDERED);
    //Then multiplication. For vertices that is not connected to zero, this is just a multiplication.
    //Do special things for the vertex connecting to 0.
    if (s + 1 != D(0, 1)) multiplicate_onto(vertex(s + 1, _, _, _), T);
  }

  //Phase 2.5: integrate out the stuff on the left (from t to beta)
  //First, change the vertex objects from v(t)->v(beta-t)
  for (int s = 2 * m - 1; s > D(0, 1); --s) vertex(s, _, _, _) = itops.reflect(vertex(s, _, _, _));

  // first, calculate P(t1)*G(t1)
  for (int k = 0; k < r; ++k) T_left(k, _, _) = matmul(Gt(k, _, _), vertex(2 * m - 1, k, _, _));

  //integrate out the stuff on the right. In each for loop, first convolution, then multiplication.
  for (int s = 2 * m - 1; s > D(0, 1); --s) {
    // integrate ts out by convolution:  integral L_s(t(s+1)-ts) D(ts) dts
    T_left = itops.convolve(beta, Fermion, itops.vals2coefs(T_left), itops.vals2coefs(line(s - 1, _, _, _)), TIME_ORDERED);
    //Then multiplication. For vertices that is not connected to zero, this is just a multiplication.
    //Do special things for the vertex connecting to 0.
    if (s - 1 != D(0, 1)) multiplicate_onto_left(T_left, vertex(s - 1, _, _, _));
  }
  //revert back from (beta-t) to t
  T_left = itops.reflect(T_left);

  //evaluating every entry of impurity Green's function from pseudoparticle information, add result to Diagram
  final_evaluation(Diagram, T, T_left, F, F_dag, n, r, N, constant);

  return Diagram;
}

nda::array<dcomplex, 3> G_Diagram_calc(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                       nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> F,
                                       nda::array_const_view<dcomplex, 3> F_dag, nda::vector_const_view<int64_t> fb) {

  //obtain basic parameters
  int r = Gt.shape(0);  // size of time grid
  int N = Gt.shape(1);  // size of G matrices
  int m = D.shape(0);   // order of diagram
  int P = hyb_F_self.P; //number of poles
  int n = F.shape(0);   //impurity size

  auto Gt_reflect = itops.reflect(Gt);

  //initialize diagram
  auto Diagram = nda::array<dcomplex, 3>(r, n, n);
  Diagram      = 0;
  if (m == 1) {
    //evaluate NCA diagram directly
    double constant = 1.0;
    final_evaluation(Diagram, Gt, Gt_reflect, F, F_dag, n, r, N, constant);
    return Diagram;
  }

  //iteration over the terms of 2, · · · , m-th hybridization. Note that 1-st hybridization is not decomposed.

  auto total_num_diagram = pown(P, m - 1);
#pragma omp parallel
  {
#pragma omp for
    for (int64_t num = 0; num < total_num_diagram; ++num) {
      Diagram = Diagram + eval_one_diagram_G(hyb_F_self, hyb_F_reflect, D, Gt, Gt_reflect, itops, beta, F, F_dag, fb, num, m, n, r, N, P);
    }
  }
  return Diagram;
}

void final_evaluation(nda::array_view<dcomplex, 3> Diagram, nda::array_const_view<dcomplex, 3> T, nda::array_const_view<dcomplex, 3> T_left,
                      nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag, int n, int r, int N, double constant) {

  for (int k = 0; k < r; ++k) {
    auto GF_left = nda::array<dcomplex, 3>(n, N, N);
    auto GF_dag  = nda::array<dcomplex, 3>(n, N, N);

    for (int a = 0; a < n; ++a) {
      GF_dag(a, _, _)  = matmul(T(k, _, _), F_dag(a, _, _));
      GF_left(a, _, _) = matmul(T_left(k, _, _), F(a, _, _));
    }

    for (int a = 0; a < n; ++a) {
      for (int b = 0; b < n; ++b) { Diagram(k, a, b) += constant * trace_matmul(GF_left(a, _, _), GF_dag(b, _, _)); }
    }
  }
}

nda::array<dcomplex, 3> G_OCA_calc(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                   nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                                   nda::vector_const_view<int64_t> fb) {

  auto D = nda::array<int, 2>{{0, 2}, {1, 3}};

  //obtain basic parameters
  int r = Gt.shape(0); // size of time grid
  int N = Gt.shape(1); // size of G matrices
  int m = D.shape(0);  // order of diagram
  int P = hyb_F_self.P;
  int n = F.shape(0);

  //initialize diagram
  auto Diagram = nda::array<dcomplex, 3>(r, n, n);
  Diagram      = 0;

  //iteration over the terms of 2, · · · , m-th hybridization. Note that 1-st hybridization is not decomposed.
  int total_num_diagram = P;

  for (int num = 0; num < total_num_diagram; ++num) {
    int num0 = num;
    //obtain R2, ... , Rm, store as R[1],...,R[m-1]
    auto R = nda::vector<int>(m);
    for (int v = 1; v < m; ++v) {
      R[v] = num0 % P;
      num0 = int(num0 / P);
    }

    //Phase 1: construct line object L and point object P;
    //This is done exactly the same as in Sigma diagrams
    auto line = nda::array<dcomplex, 4>(2 * m, r, N, N);
    line      = 0;
    for (int s = 1; s <= 2 * m - 1; ++s) line(s, _, _, _) = Gt;
    double constant = 1;
    auto vertex     = nda::array<dcomplex, 4>(2 * m, r, N, N);
    vertex          = 0;

    if (fb(1) == 0)
      cut_hybridization(1, num0, D, constant, hyb_F_self.U_tilde(num0, _, _, _), hyb_F_self.V_tilde(num0, _, _, _), line, vertex, hyb_F_self.c(num0),
                        hyb_F_self.w(num0), hyb_F_self.K_matrix(num0, _), r, N);
    else
      cut_hybridization(1, num0, D, constant, hyb_F_reflect.U_tilde(num0, _, _, _), hyb_F_reflect.V_tilde(num0, _, _, _), line, vertex,
                        hyb_F_reflect.c(num0), hyb_F_reflect.w(num0), hyb_F_reflect.K_matrix(num0, _), r, N);

    //Phase 2: integrate out the stuff on the right
    auto T = nda::array<dcomplex, 3>(r, N, N);

    // first, calculate P(t1)*G(t1)
    for (int k = 0; k < r; ++k) T(k, _, _) = matmul(vertex(1, k, _, _), Gt(k, _, _));

    //integrate out the stuff on the right. In each for loop, first convolution, then multiplication.
    // integrate ts out by convolution:  integral L_s(t(s+1)-ts) D(ts) dts
    T = itops.convolve(beta, Fermion, itops.vals2coefs(line(1, _, _, _)), itops.vals2coefs(T), TIME_ORDERED);

    //Phase 2.5: integrate out the stuff on the right
    //First, change the vertex objects from v(t)->v(beta-t)
    vertex(3, _, _, _) = itops.reflect(vertex(3, _, _, _));
    auto T_left        = nda::array<dcomplex, 3>(r, N, N);

    // first, calculate P(t1)*G(t1)
    for (int k = 0; k < r; ++k) T_left(k, _, _) = matmul(Gt(k, _, _), vertex(3, k, _, _));

    //integrate out the stuff on the right. In each for loop, first convolution, then multiplication.

    // integrate ts out by convolution:  integral L_s(t(s+1)-ts) D(ts) dts
    T_left = itops.convolve(beta, Fermion, itops.vals2coefs(T_left), itops.vals2coefs(line(2, _, _, _)), TIME_ORDERED);

    //revert back from (beta-t) to t
    T_left = itops.reflect(T_left);

    final_evaluation(Diagram, T, T_left, F, F_dag, n, r, N, constant);
  }

  return Diagram;
}

nda::array<dcomplex, 3> evaluate_one_diagram(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                             nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect,
                                             nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                             nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                                             nda::vector_const_view<int64_t> fb, bool backward, int num0, int m, int n, int r, int N, int P) {
  if (m == 1) {
    //calculate NCA diagram directly
    auto Diagram = nda::array<dcomplex, 3>(r, N, N);
    Diagram      = Gt;
    special_summation(Diagram, F, F_dag, Deltat, Deltat_reflect, n, r, N, true);
    return Diagram;
  }

  auto line   = nda::array<dcomplex, 4>(2 * m, r, N, N); //used for storing line objects
  auto vertex = nda::array<dcomplex, 4>(2 * m, r, N, N); //used for storing vertex objects
  auto T      = nda::array<dcomplex, 3>(r, N, N);        //used for storing diagrams
  auto R      = nda::vector<int>(m);                     //utility for iteration

  //obtain R2, ... , Rm, store as R[1],...,R[m-1]

  for (int v = 1; v < m; ++v) {
    R[v] = num0 % P;
    num0 = int(num0 / P);
  }

  //Phase 1: construct line object L and point object P;
  /* Construct line objects, i.e. functions of (t_s-t_{s-1}) for s =1 , ..., 2m-1
        We store these in L(r,2m,N,N). 
        We initialize them with Green's functions.
    */

  line = 0;
  for (int s = 1; s <= 2 * m - 1; ++s) line(s, _, _, _) = Gt;
  double constant = 1; // the constant term responsible for the current diagram
  //initialize vertex object

  vertex = 0;

  //Cutting 1,2, ..., (m-1)-th hybridization lines.
  //for (int v = 1;v<m;++v) cut_hybridization(v,R(v), D, constant, hyb_F_self,hyb_F_reflect, line, vertex, r, N);
  for (int v = 1; v < m; ++v) {
    if (fb(v) == 0)
      cut_hybridization(v, R(v), D, constant, hyb_F_self.U_tilde(R(v), _, _, _), hyb_F_self.V_tilde(R(v), _, _, _), line, vertex, hyb_F_self.c(R(v)),
                        hyb_F_self.w(R(v)), hyb_F_self.K_matrix(R(v), _), r, N);
    else
      cut_hybridization(v, R(v), D, constant, hyb_F_reflect.U_tilde(R(v), _, _, _), hyb_F_reflect.V_tilde(R(v), _, _, _), line, vertex,
                        hyb_F_reflect.c(R(v)), hyb_F_reflect.w(R(v)), hyb_F_reflect.K_matrix(R(v), _), r, N);
  }
  //Phase 2: integrate everything out

  // first, calculate P(t1)*G(t1)
  for (int k = 0; k < r; ++k) T(k, _, _) = matmul(vertex(1, k, _, _), Gt(k, _, _));

  //integrate out indices t1, ..., t(2m-2). In each for loop, first convolution, then multiplication.
  for (int s = 1; s <= 2 * m - 2; ++s) {
    // integrate ts out by convolution:  integral L_s(t(s+1)-ts) D(ts) dts
    T = itops.convolve(beta, Fermion, itops.vals2coefs(line(s, _, _, _)), itops.vals2coefs(T), TIME_ORDERED);
    //Then multiplication. For vertices that is not connected to zero, this is just a multiplication.
    //Do special things for the vertex connecting to 0.
    if (s + 1 != D(0, 1))
      multiplicate_onto(vertex(s + 1, _, _, _), T);
    else
      special_summation(T, F, F_dag, Deltat, Deltat_reflect, n, r, N, backward);
  }
  return make_regular(T * constant);
}

nda::array<dcomplex, 3> Sigma_Diagram_calc(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                           nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect,
                                           nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                           nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                                           nda::vector_const_view<int64_t> fb, bool backward) {

  //obtain basic parameters
  int r = Gt.shape(0);  // size of time grid
  int N = Gt.shape(1);  // size of G matrices
  int m = D.shape(0);   // order of diagram
  int P = hyb_F_self.P; //number of poles
  int n = F.shape(0);   //size of impurity

  //initialize diagram
  auto Diagram = nda::array<dcomplex, 3>::zeros({r, N, N});

  if (m == 1) {
    //calculate NCA diagram directly
    Diagram = Gt;
    special_summation(Diagram, F, F_dag, Deltat, Deltat_reflect, n, r, N, backward);
    return Diagram;
  }

  //iteration over the terms of 2, · · · , m-th hybridization. Note that 1-st hybridization is not decomposed.
  int64_t total_num_diagram = pown(P, m - 1); //number of total diagrams

  std::cout << "total_num_diagram = " << total_num_diagram << "\n";
  triqs::utility::timer timer_run;
  timer_run.start();

#pragma omp parallel
  {

#pragma omp for
    for (int64_t num = 0; num < total_num_diagram; ++num) {
      Diagram = Diagram
         + evaluate_one_diagram(hyb_F_self, hyb_F_reflect, D, Deltat, Deltat_reflect, Gt, itops, beta, F, F_dag, fb, backward, num, m, n, r, N, P);
    }
  }

  std::cout << "Total time: " << timer_run << "\n";
  return Diagram;
}

nda::array<dcomplex, 3> Sigma_Diagram_calc_sum_all(hyb_F &hyb_F_self, hyb_F &hyb_F_reflect, nda::array_const_view<int, 2> D,
                                                   nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect,
                                                   nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta,
                                                   nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag) {

  //summing over all forward and backward choices
  auto r = Gt.shape(0);  // size of time grid
  auto N = Gt.shape(1);  // size of G matrices
  auto m = D.shape(0);   // order of diagram
  auto P = hyb_F_self.P; //number of poles
  auto n = F.shape(0);   //size of impurity

  auto Diagram = nda::array<dcomplex, 3>::zeros({r, N, N});

  auto total_num_fb_diagram = pown(2, m - 1); // total number of forward and backward choices

  auto num_diagram_per_fb = pown(P, m - 1); //number of diagrams per fb
  auto total_num_diagram  = num_diagram_per_fb * total_num_fb_diagram;
  std::cout << "total_num_diagram = " << num_diagram_per_fb * total_num_fb_diagram << "\n";
  triqs::utility::timer timer_run;
  timer_run.start();

  //#pragma omp parallel
  //{
  //#pragma omp for
  for (int64_t num = 0; num < total_num_diagram; ++num) {
    auto fb   = nda::vector<int64_t>(m); //utility for iteration
    auto num0 = num / num_diagram_per_fb;
    auto num2 = num % num_diagram_per_fb;
    for (int v = 1; v < m; ++v) {
      fb[v] = num0 % 2;
      num0  = int64_t(num0 / 2);
    }
    // BUG! This accumulation has a race condition between threads! FIXME!
    Diagram =
       Diagram + evaluate_one_diagram(hyb_F_self, hyb_F_reflect, D, Deltat, Deltat_reflect, Gt, itops, beta, F, F_dag, fb, true, num2, m, n, r, N, P);
  }
  // } #end pragma omp parallell
  std::cout << "Total time: " << timer_run << "\n";
  return Diagram;
}

// nda::array<dcomplex,3> Sigma_Diagram_calc_sum_all(hyb_F &hyb_F_self,hyb_F &hyb_F_reflect,nda::array_const_view<int,2> D,nda::array_const_view<dcomplex,3> Deltat,nda::array_const_view<dcomplex,3> Deltat_reflect,nda::array_const_view<dcomplex,3> Gt,imtime_ops &itops,double beta, nda::array_const_view<dcomplex,3> F, nda::array_const_view<dcomplex,3> F_dag){
//     //summing over all forward and backward choices
//     int r = Gt.shape(0); // size of time grid
//     int N = Gt.shape(1); // size of G matrices
//     int m = D.shape(0); // order of diagram
//     auto Diagram = nda::array<dcomplex,3>(r,N,N);
//     Diagram = 0;
//     int total_num_fb_diagram = pow(2, m-1);// total number of forward and backward choices

//     std::cout << "total_num_fb_diagram = " << total_num_fb_diagram << "\n";

//     {
//     auto fb = nda::vector<int>(m); //utility for iteration

//     for (int num=0;num<total_num_fb_diagram;++num){
//         int num0 = num;

//         for (int v = 1;v<m;++v){
//             fb[v] = num0 % 2;
//             num0 = int(num0/2);
//         }
//         Diagram += Sigma_Diagram_calc(hyb_F_self,hyb_F_reflect,D,Deltat,Deltat_reflect, Gt,itops,beta, F,  F_dag,  fb, true);
//     }
//     }
//     return Diagram;
// }

nda::array<dcomplex, 3> Sigma_OCA_calc(hyb_F &hyb_F, nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect,
                                       nda::array_const_view<dcomplex, 3> Gt, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> F,
                                       nda::array_const_view<dcomplex, 3> F_dag, bool backward) {

  auto const D = nda::array<int, 2>{{0, 2}, {1, 3}};

  //obtain basic parameters
  int r = Gt.shape(0); // size of time grid
  int N = Gt.shape(1); // size of G matrices
  int P = hyb_F.P;
  int n = F.shape(0);

  //initialize diagram
  auto Diagram = nda::array<dcomplex, 3>(r, N, N);
  Diagram      = 0;
  for (int R = 0; R < P; ++R) {

    auto T = nda::array<dcomplex, 3>(r, N, N);
    T      = 0;

    // first integrate out t1
    if (hyb_F.w(R) < 0) {
      // T(t1) = Vtilde(t1)*G(t1)
      for (int k = 0; k < r; ++k) T(k, _, _) = matmul(hyb_F.V_tilde(R, k, _, _), Gt(k, _, _));
      // integrate t1 out: int G(t2-t1)* T(t1) dt1
      T = itops.convolve(beta, Fermion, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

    } else {

      // T(t2-t1) = G(t2-t1) * Vtilde(t2-t1)
      for (int k = 0; k < r; ++k) T(k, _, _) = matmul(Gt(k, _, _), hyb_F.V_tilde(R, k, _, _));
      // integrate t1 out: int T(t2-t1)* G(t1) dt1
      T = itops.convolve(beta, Fermion, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
    }
    //next, calculate T3(k) = sum_ab Delta_ab(t_k) * Fdag_a * T(k) * F_b
    special_summation(T, F, F_dag, Deltat, Deltat_reflect, n, r, N, backward);

    //finally integrate out t2
    if (hyb_F.w(R) < 0) {
      // integrate t2 out: T(t) = int G(t-t2)* T3(t2) dt2
      T = itops.convolve(beta, Fermion, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

      //T(t) = Utilde(t) * T(t)

      for (int k = 0; k < r; ++k) T(k, _, _) = matmul(hyb_F.U_tilde(R, k, _, _), T(k, _, _));
      // std::cout<<T<<std::endl;

    } else {

      // T(t-t2) = Utilde(t-t2) * Gt(t-t2)
      auto T3 = nda::array<dcomplex, 3>(r, N, N);
      for (int k = 0; k < r; ++k) T3(k, _, _) = matmul(hyb_F.U_tilde(R, k, _, _), Gt(k, _, _));
      // integrate t2 out: T(t) = int T(t-t2)* T3(t2) dt2
      T = itops.convolve(beta, Fermion, itops.vals2coefs(T3), itops.vals2coefs(T), TIME_ORDERED);
    }
    Diagram = Diagram + T * hyb_F.c(R);
  }
  return Diagram;
}

void cut_hybridization(int v, int Rv, nda::array_const_view<int, 2> D, double &constant, nda::array_const_view<dcomplex, 3> U_tilde_here,
                       nda::array_const_view<dcomplex, 3> V_tilde_here, nda::array_view<dcomplex, 4> line, nda::array_view<dcomplex, 4> vertex,
                       double chere, double w_here, nda::array_const_view<double, 1> K_matrix_here, int r, int N) {

  constant *= chere;
  //when w(R(v))>0, we need to modify the line object, and the constant. The point object is assigned to be the identity matrix.
  if (w_here > 0) {
    for (int k = 0; k < r; ++k) {
      vertex(D(v, 0), k, _, _)   = eye<dcomplex>(N);
      vertex(D(v, 1), k, _, _)   = eye<dcomplex>(N);
      line(D(v, 0), k, _, _)     = matmul(line(D(v, 0), k, _, _), V_tilde_here(k, _, _));
      line(D(v, 1) - 1, k, _, _) = matmul(U_tilde_here(k, _, _), line(D(v, 1) - 1, k, _, _));
    }
    for (int s = D(v, 0) + 1; s < D(v, 1) - 1; ++s) {
      for (int k = 0; k < r; ++k) line(s, k, _, _) *= K_matrix_here(k);
      constant *= chere;
    }
  }
  //when w(R(v))<0, we need only to modify the point object.
  else {
    vertex(D(v, 0), _, _, _) = V_tilde_here;
    vertex(D(v, 1), _, _, _) = U_tilde_here;
  }
}

void special_summation(nda::array_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag,
                       nda::array_const_view<dcomplex, 3> Deltat, nda::array_const_view<dcomplex, 3> Deltat_reflect, int n, int r, int N,
                       bool backward) {

  //This implementation reduces cost from O(2rn^2N^3) to O(rn^2N^3 + 2rnN^3)
  auto Sigmat = nda::array<dcomplex, 3>(Gt.shape());

  Sigmat() = 0;
  for (int t = 0; t < r; ++t) {
    auto temp   = nda::array<dcomplex, 3>(n, N, N);
    auto temp_D = nda::array<dcomplex, 3>(n, N, N);
    for (int b = 0; b < n; ++b) { temp(b, _, _) = matmul(Gt(t, _, _), F(b, _, _)); }
    temp_D = arraymult(Deltat(t, _, _), temp);
    for (int a = 0; a < n; ++a) { Sigmat(t, _, _) += matmul(F_dag(a, _, _), temp_D(a, _, _)); }
    if (backward) {
      auto temp2   = nda::array<dcomplex, 3>(n, N, N);
      auto tempD_2 = nda::array<dcomplex, 3>(n, N, N);
      for (int a = 0; a < n; ++a) { temp2(a, _, _) = matmul(F(a, _, _), Gt(t, _, _)); }
      tempD_2 = arraymult(Deltat_reflect(t, _, _), F_dag);
      for (int a = 0; a < n; ++a) {
        // for (int b = 0; b < n; ++b) {
        //   Sigmat(t,_,_) += Deltat_reflect(t, a, b) * matmul(temp2(a, _, _), F_dag(b, _, _));
        // }
        Sigmat(t, _, _) += matmul(temp2(a, _, _), tempD_2(a, _, _));
      }
    }

    // for (int a = 0; a < n; ++a) {
    //   for (int b = 0; b < n; ++b) {

    //   Sigmat(t, _, _) += Deltat(t, a, b) * matmul(F_dag(a, _, _), matmul(Gt(t, _, _), F(b, _, _)));
    //   if ( backward )
    //     Sigmat(t, _, _) += Deltat_reflect(t, a, b) * matmul(F(a, _, _), matmul(Gt(t, _, _), F_dag(b, _, _)));
    // }
    //   }
  }

  Gt() = Sigmat;
}

void multiplicate_onto(nda::array_const_view<dcomplex, 3> Ft, nda::array_view<dcomplex, 3> Gt) {
  int r = Gt.shape(0);
  for (int k = 0; k < r; ++k) Gt(k, _, _) = matmul(Ft(k, _, _), Gt(k, _, _));
}

void multiplicate_onto_left(nda::array_view<dcomplex, 3> Ft, nda::array_const_view<dcomplex, 3> Gt) {
  int r = Gt.shape(0);
  for (int k = 0; k < r; ++k) Ft(k, _, _) = matmul(Ft(k, _, _), Gt(k, _, _));
}

dcomplex trace_matmul(nda::array_const_view<dcomplex, 2> M1, nda::array_const_view<dcomplex, 2> M2) {
  int n        = M1.shape(0);
  auto M2_T    = transpose(M2);
  dcomplex res = 0;
  for (int i = 0; i < n; ++i) { res += linalg::dot(M1(i, _), M2_T(i, _)); }
  return res;
}
