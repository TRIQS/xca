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
TEST(OCA, G) {
  double beta        = 1.0;
  double lambda      = 100.0;
  double eps         = 1.0e-13;
  auto dlr_rf        = build_dlr_rf(lambda, eps);  // Get DLR frequencies
  auto itops         = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();

  nda::vector<double> dlr_it_actual = dlr_it;
  for (int i = 0; i < r; ++i) {
    if (dlr_it(i) < 0) dlr_it_actual(i) = dlr_it(i) + 1;
  }

  double D        = -2.0;
  auto Deltat     = nda::array<dcomplex, 3>(r, 1, 1);
  Deltat(_, 0, 0) = exp(-D * dlr_it_actual * beta);
  auto Deltadlr   = itops.vals2coefs(Deltat);
  //reflect Deltat
  auto Deltat_reflect   = itops.reflect(Deltat);
  auto Deltadlr_reflect = itops.vals2coefs(Deltat_reflect);

  //decomposition of hybridization

  double g    = 3.0;
  auto Gt     = nda::array<dcomplex, 3>(r, 1, 1);
  Gt(_, 0, 0) = exp(-g * dlr_it_actual * beta);
  auto G_xaa  = itops.vals2coefs(Gt);

  auto I_aa = nda::eye<dcomplex>(1);

  auto D2  = nda::array<int, 2>{{0, 2}, {1, 3}};
  auto fb2 = nda::vector<int>(2);
  fb2      = 0;

  auto F     = nda::array<dcomplex, 3>(1, 1, 1);
  F(0, 0, 0) = 1.0;
  nda::array<double, 1> pol(1);
  pol(0)                      = D * beta;
  nda::array<double, 1> pol_r = -pol;
  nda::array<dcomplex, 3> A(1, 1, 1);
  A(0, _, _) = eye<dcomplex>(1) / k_it(0, D * beta);

  // auto Delta_decomp = hyb_decomp(Deltadlr,dlr_rf);
  // auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect,dlr_rf);
  auto Delta_decomp         = hyb_decomp(A, pol);
  auto Delta_decomp_reflect = hyb_decomp(A, pol_r);
  Delta_decomp.check_accuracy(Deltat, dlr_it);

  Delta_decomp_reflect.check_accuracy(Deltat_reflect, dlr_it);

  hyb_F Delta_F(1, r, 1);
  hyb_F Delta_F_reflect(1, r, 1);

  Delta_F.update_inplace(Delta_decomp, dlr_it, F, F);
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F, F);

  //calculating diagrams
  auto G_OCAdiagram = G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D2, Gt, itops, beta, F, F);
  std::cout << G_OCAdiagram << std::endl;
  if (abs(D) < 1e-8) {
    for (int i = 0; i < r; ++i) std::cout << " " << make_regular((exp(-g * beta) * beta * beta * (1 - dlr_it_actual(i)) * dlr_it_actual(i)));
  } else {

    for (int i = 0; i < r; ++i)
      std::cout << " " << make_regular(exp(-(g + D) * beta) * (1 - exp(D * (1 - dlr_it_actual(i)))) * (1 - exp(D * dlr_it_actual(i))) / pow(D, 2));
    std::cout << std::endl;
    for (int i = 0; i < r; ++i)
      std::cout << " "
                << make_regular(exp(-D * beta) * exp(-(g - D) * beta) * (1 - exp(-D * (1 - dlr_it_actual(i)))) * (1 - exp(-D * dlr_it_actual(i)))
                                / pow(D, 2));
  }
}
