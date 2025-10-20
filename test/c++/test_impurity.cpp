/*******************************************************************************
 *
 * triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2025, Z. Huang
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

#include "triqs_soehyb/strong_cpl.hpp"
#include "triqs_soehyb/impurity.hpp"
#include "nda/nda.hpp"
#include <cppdlr/cppdlr.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <functional>
#include <gtest/gtest.h>
#include <nda/algorithms.hpp>
#include <nda/basic_functions.hpp>
#include <nda/declarations.hpp>
#include <nda/layout/policies.hpp>
#include <chrono>
#include <nda/matrix_functions.hpp>
#include <omp.h>

using namespace cppdlr;
using namespace nda;
//export OMP_NUM_THREADS=1;
nda::array<dcomplex, 3> ppsc_free_greens_tau(nda::vector_const_view<double> tau_i, nda::array_view<dcomplex, 2> H_S, double beta);
nda::array<dcomplex, 3> NCA(nda::array_view<dcomplex, 3> Deltat, nda::array_view<dcomplex, 3> Deltat_reflect, nda::array<dcomplex, 3> G_iaa,
                            nda::array_const_view<dcomplex, 3> F, nda::array_const_view<dcomplex, 3> F_dag);
// nda::array<dcomplex,3> step(nda::array_view<dcomplex,2> G_iaa);
TEST(strong_coupling, dimer) {
  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(1); // Use 4 threads for all consecutive parallel regions
  int N       = 64;
  auto c0_dag = nda::array<dcomplex, 2>(N, N);
  c0_dag      = 0;
  auto c1_dag = nda::array<dcomplex, 2>(N, N);
  c1_dag      = 0;
  auto c2_dag = nda::array<dcomplex, 2>(N, N);
  c2_dag      = 0; //b00
  auto c3_dag = nda::array<dcomplex, 2>(N, N);
  c3_dag      = 0; //b01
  auto c4_dag = nda::array<dcomplex, 2>(N, N);
  c4_dag      = 0; //b01
  auto c5_dag = nda::array<dcomplex, 2>(N, N);
  c5_dag      = 0; //b11

  for (int i = 0; i < 32; ++i) c0_dag(i + 32, i) = 1;

  for (int i = 0; i < 16; ++i) c1_dag(i + 16, i) = 1;
  for (int i = 32; i < 48; ++i) c1_dag(i + 16, i) = -1;

  for (int i = 0; i < 8; ++i) {
    c2_dag(i + 8, i)       = 1;
    c2_dag(i + 24, i + 16) = -1;
    c2_dag(i + 40, i + 32) = -1;
    c2_dag(i + 56, i + 48) = 1;
  }
  for (int i = 0; i < 4; ++i) {
    c3_dag(i + 4, i)       = 1;
    c3_dag(i + 12, i + 8)  = -1;
    c3_dag(i + 20, i + 16) = -1;
    c3_dag(i + 28, i + 24) = 1;
    c3_dag(i + 36, i + 32) = -1;
    c3_dag(i + 44, i + 40) = 1;
    c3_dag(i + 52, i + 48) = 1;
    c3_dag(i + 60, i + 56) = -1;
  }

  for (int i = 0; i < 16; ++i) {
    if ((i == 1) or (i == 2) or (i == 4) or (i == 8) or (i == 14) or (i == 13) or (i == 11) or (i == 7)) {
      c4_dag(4 * i + 2, 4 * i)     = -1;
      c4_dag(4 * i + 3, 4 * i + 1) = -1;
    } else {
      c4_dag(4 * i + 2, 4 * i)     = 1;
      c4_dag(4 * i + 3, 4 * i + 1) = 1;
    }
  }
  for (int i = 0; i < 32; ++i) {
    if ((i == 1) or (i == 2) or (i == 4) or (i == 8) or (i == 14) or (i == 13) or (i == 11) or (i == 7))
      c5_dag(2 * i + 1, 2 * i) = -1;
    else if ((i == 16) or (i == 19) or (i == 21) or (i == 25) or (i == 22) or (i == 26) or (i == 28) or (i == 31))
      c5_dag(2 * i + 1, 2 * i) = -1;
    else
      c5_dag(2 * i + 1, 2 * i) = 1;
  }

  auto c0 = conj(transpose(c0_dag));
  auto c1 = conj(transpose(c1_dag));
  auto c2 = conj(transpose(c2_dag));
  auto c3 = conj(transpose(c3_dag));
  auto c4 = conj(transpose(c4_dag));
  auto c5 = conj(transpose(c5_dag));
  // for (int i =0;i<64;++i) std::cout<<(matmul(c0,c1_dag)+matmul(c1_dag,c0))(i,i);

  double t = 1.0;
  auto U   = 4 * t;
  auto v   = 1.5;
  auto tp  = 1.5;
  auto mu  = 0.0;

  nda::array<dcomplex, 2> H = U * matmul(matmul(c0_dag, c0), matmul(c1_dag, c1));
  H                         = H - v * (matmul(c0_dag, c1) + matmul(c1_dag, c0));
  H                         = H - mu * (matmul(c0_dag, c0) + matmul(c1_dag, c1));
  H                         = H - t * (matmul(c0_dag, c2) + matmul(c2_dag, c0)) - t * (matmul(c0_dag, c3) + matmul(c3_dag, c0));
  H                         = H - t * (matmul(c1_dag, c4) + matmul(c4_dag, c1)) - t * (matmul(c1_dag, c5) + matmul(c5_dag, c1));
  H                         = H - tp * (matmul(c2_dag, c4) + matmul(c4_dag, c2)) - tp * (matmul(c3_dag, c5) + matmul(c5_dag, c3));

  auto [eval, evec] = nda::linalg::eigenelements(H);

  // std::cout<<exp_eval;
  double beta                  = 8;
  nda::vector<double> exp_eval = exp(-beta * (eval - min_element(eval)));
  exp_eval                     = exp_eval / sum(exp_eval);
  // std::cout<<((eval));
  double lambda      = 640;
  double eps         = 1.0e-9;
  std::string order  = "OCA";
  auto dlr_rf        = build_dlr_rf(lambda, eps);  // Get DLR frequencies
  auto itops         = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();
  std::cout << "dlr rank is" << r;
  // nda::vector<double> tau_actual = itops.get_itnodes();
  // for (int k =0;k<r;++k){
  //     if (tau_actual(k)<0) {tau_actual(k) = tau_actual(k)+1;}
  // }
  // tau_actual = tau_actual*beta;

  int N_t         = 100;
  auto t_relative = nda::vector<double>(N_t);
  for (int i = 0; i < N_t; ++i) {
    t_relative(i) = (i + 0.0) / N_t;
    if (t_relative(i) > 0.5) { t_relative(i) -= 1; };
  }

  auto G00      = nda::array<dcomplex, 3>(r, 1, 1);
  G00           = 0;
  auto G01      = nda::array<dcomplex, 3>(r, 1, 1);
  G01           = 0;
  auto G00_long = nda::array<dcomplex, 3>(N_t, 1, 1);
  G00_long      = 0;
  auto G01_long = nda::array<dcomplex, 3>(N_t, 1, 1);
  G01_long      = 0;

  auto g_S_long = nda::array<dcomplex, 3>(N_t, 2, 2);
  g_S_long      = 0;

  auto V_sr_0 = matmul(conj(transpose(evec)), matmul(c0, evec));

  auto V_sr_1 = matmul(conj(transpose(evec)), matmul(c1, evec));
  for (int s = 0; s < N; ++s) {
    for (int s2 = 0; s2 < N; ++s2) {
      double w = eval(s2) - eval(s);
      for (int k = 0; k < r; ++k) {

        G00(k, 0, 0) += V_sr_0(s, s2) * conj(V_sr_0(s, s2)) * (exp_eval(s) + exp_eval(s2)) * k_it(dlr_it(k), w * beta);
        G01(k, 0, 0) += V_sr_0(s, s2) * conj(V_sr_1(s, s2)) * (exp_eval(s) + exp_eval(s2)) * k_it(dlr_it(k), w * beta);
      }
      for (int k2 = 0; k2 < N_t; ++k2) {

        G00_long(k2, 0, 0) += V_sr_0(s, s2) * conj(V_sr_0(s, s2)) * (exp_eval(s) + exp_eval(s2)) * k_it(t_relative(k2), w * beta);
        G01_long(k2, 0, 0) += V_sr_0(s, s2) * conj(V_sr_1(s, s2)) * (exp_eval(s) + exp_eval(s2)) * k_it(t_relative(k2), w * beta);
      }
    }
  }

  auto G00_dlr = itops.vals2coefs(G00);
  auto G01_dlr = itops.vals2coefs(G01);

  // std::cout<<std::setprecision(8)<<itops.coefs2eval(G00_dlr, 0)<<std::endl;
  // std::cout<<itops.coefs2eval(G01_dlr, 0)<<std::endl;

  auto h_bath  = nda::array<dcomplex, 2>(2, 2);
  h_bath       = 0;
  h_bath(0, 1) = -tp;
  h_bath(1, 0) = -tp;
  auto Deltat  = make_regular(2 * free_gf(beta, itops, h_bath) * pow(t, 2));

  int N_S        = 4;
  auto c0_S_dag  = nda::array<dcomplex, 2>(N_S, N_S);
  c0_dag         = 0;
  auto c1_S_dag  = nda::array<dcomplex, 2>(N_S, N_S);
  c1_dag         = 0;
  c0_S_dag(2, 0) = 1;
  c0_S_dag(3, 1) = 1;
  c1_S_dag(1, 0) = 1;
  c1_S_dag(3, 2) = -1;
  auto c0_S      = conj(transpose(c0_S_dag));
  auto c1_S      = conj(transpose(c1_S_dag));

  auto F         = nda::array<dcomplex, 3>(2, N_S, N_S);
  F(0, _, _)     = c0_S;
  F(1, _, _)     = c1_S;
  auto F_dag     = nda::array<dcomplex, 3>(2, N_S, N_S);
  F_dag(0, _, _) = c0_S_dag;
  F_dag(1, _, _) = c1_S_dag;

  nda::array<dcomplex, 2> H_S = U * matmul(matmul(c0_S_dag, c0_S), matmul(c1_S_dag, c1_S));
  H_S                         = H_S - v * (matmul(c0_S_dag, c1_S) + matmul(c1_S_dag, c0_S));
  H_S                         = H_S - mu * (matmul(c0_S_dag, c0_S) + matmul(c1_S_dag, c1_S));

  auto [E_HS, U_HS] = nda::linalg::eigenelements(H_S);
  auto E0_HS        = min_element(E_HS);
  E_HS -= E0_HS;
  auto Z_HS  = sum(exp(-beta * E_HS));
  auto eta_0 = E0_HS - log(Z_HS) / beta;

  auto G0_S_tau = free_gf(beta, itops, H_S, eta_0, true);

  auto impsol = fastdiagram(beta, lambda, itops, F, F_dag);
  impsol.hyb_init(Deltat);
  impsol.hyb_decomposition();
  auto tau_actual = impsol.get_it_actual();

  nda::array<dcomplex, 3> G_S_tau     = G0_S_tau;
  nda::array<dcomplex, 3> G_S_tau_old = 0.0 * G_S_tau;

  auto begin = std::chrono::high_resolution_clock::now();
  //auto Sigma_t = impsol.Sigma_calc(G0_S_tau,"TCA");
  auto end     = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t3    = elapsed.count();
  std::cout << "Time spent is " << t3 << " seconds" << std::endl;
  double eta, Z_S;
  for (int ppsc_iter = 0; ppsc_iter < 10; ++ppsc_iter) {
    begin = std::chrono::high_resolution_clock::now();
    if (max_element(abs(G_S_tau_old - G_S_tau)) < 1e-10) break;
    G_S_tau_old  = G_S_tau;
    auto Sigma_t = impsol.Sigma_calc(G_S_tau, order);

    auto G_new_tau = impsol.time_ordered_dyson(beta, H_S, eta_0, Sigma_t);

    Z_S = impsol.partition_function(G_new_tau);
    eta = log(Z_S) / beta;
    H_S += eta * nda::eye<dcomplex>(H_S.shape(0));

    for (int k = 0; k < r; ++k) { G_new_tau(k, _, _) = G_new_tau(k, _, _) * exp(-tau_actual(k) * eta); }
    G_S_tau = 1.0 * G_new_tau + 0.0 * G_S_tau_old;
    std::cout << "iter " << ppsc_iter << " , diff is " << max_element(abs(G_S_tau_old - G_S_tau)) << std::endl;
    end     = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    t3      = elapsed.count();
    std::cout << "Time spent is " << t3 / 1000 << " seconds" << std::endl;
  }
  auto G_S_dlr = itops.vals2coefs(G_S_tau);
  begin        = std::chrono::high_resolution_clock::now();
  auto g_S     = impsol.G_calc(G_S_tau, order);
  end          = std::chrono::high_resolution_clock::now();
  elapsed      = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  t3           = elapsed.count();
  std::cout << "Time spent is " << t3 / 1000 << " seconds" << std::endl;

  auto g_S_dlr = itops.vals2coefs(make_regular(g_S));

  for (int i = 0; i < r; ++i) std::cout << abs(G00(i, 0, 0) - g_S(i, 0, 0)) << " ";
  for (int i = 0; i < N_t; ++i) g_S_long(i, _, _) = itops.coefs2eval(g_S_dlr, t_relative(i));
  std::cout << std::endl;
  for (int i = 0; i < N_t; ++i) std::cout << abs(G00_long(i, 0, 0) - g_S_long(i, 0, 0)) << " ";

  std::cout << std::endl << "result" << std::endl;
  for (int i = 0; i < r; ++i) std::cout << real(G00(i, 0, 0)) << " ";
}
