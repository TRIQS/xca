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

using namespace cppdlr;
using namespace nda;

TEST(strong_coupling, dimer) {
  //parameters
  double t = 1.0;
  auto U   = 4 * t;
  auto v   = 1.5;
  auto tp  = 1.5;
  auto mu  = 0.0;

  double beta = 10;

  double lambda = 640;
  double eps    = 1.0e-12;

  //preparing itops
  auto dlr_rf = build_dlr_rf(lambda, eps);  // Get DLR frequencies
  auto itops  = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
  int r       = itops.rank();

  std::cout << "rank is " << r << std::endl;

  //preparing Delta(t)
  auto h_bath  = nda::array<dcomplex, 2>(2, 2);
  h_bath       = 0;
  h_bath(0, 1) = -tp;
  h_bath(1, 0) = -tp;
  auto Deltat  = make_regular(2 * free_gf(beta, itops, h_bath) * pow(t, 2));

  // preparing G0_S(tau)
  int N_S        = 4;
  auto c0_S_dag  = nda::array<dcomplex, 2>(N_S, N_S);
  c0_S_dag       = 0;
  auto c1_S_dag  = nda::array<dcomplex, 2>(N_S, N_S);
  c1_S_dag       = 0;
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

  auto G0_S_tau = free_gf(beta, itops, H_S, 0.0, true);

  //preparing impurity solver (hyb decomposition)
  auto impsol = fastdiagram(beta, lambda, itops, F, F_dag);
  impsol.hyb_init(Deltat);
  impsol.hyb_decomposition();

  //tca topology
  auto Dt1 = nda::array<int, 2>{{0, 2}, {1, 4}, {3, 5}};

  //Evaluating diagrams
  auto begin = std::chrono::high_resolution_clock::now();
  //Evaluating the first Nd tca diagram of this topology
  int Nd           = 1000;
  auto Sigma_tca_1 = impsol.Sigma_calc_group(G0_S_tau, Dt1, arange(Nd));
  auto end         = std::chrono::high_resolution_clock::now();
  auto elapsed     = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t3        = elapsed.count();
  std::cout << "Time spent is " << t3 << " ms" << std::endl;
}
