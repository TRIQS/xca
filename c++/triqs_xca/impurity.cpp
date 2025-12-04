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
#include "impurity.hpp"
#include "dlr_dyson_ppsc.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <nda/linalg/matmul.hpp>

fastdiagram::fastdiagram(double beta, double lambda, imtime_ops itops, nda::array<dcomplex, 3> F, nda::array<dcomplex, 3> F_dag)
   : beta(beta),
     lambda(lambda),
     itops(itops),
     F(F),
     F_dag(F_dag),
     n(F.shape(0)),
     r(itops.rank()),
     N(F.shape(1)),
     dlr_rf(itops.get_rfnodes()),
     dlr_it(itops.get_itnodes()),
     dlr_it_actual(dlr_it),
     Deltat(r, n, n),
     Deltat_reflect(r, n, n),
     Delta_F(N, r, n),
     Delta_F_reflect(N, r, n),

     // -- Diagram topologies (not in use?, remove?)

     D_NCA{{0, 1}},                   // NCA
     D_OCA{{0, 2}, {1, 3}},           // OCA
     D_TCA_1{{0, 2}, {1, 4}, {3, 5}}, // TCA 1st
     D_TCA_2{{0, 3}, {1, 5}, {2, 4}}, // TCA 2nd
     D_TCA_3{{0, 4}, {1, 3}, {2, 5}}, // TCA 3rd
     D_TCA_4{{0, 3}, {1, 4}, {2, 5}}  // TCA 4th

{

  // Change from tau in the [-0.5, 0.5] interval to [0, 1]
  for (int k = 0; k < r; ++k) {
    if (dlr_it_actual(k) < 0) { dlr_it_actual(k) = 1.0 + dlr_it_actual(k); }
  }

  dlr_it_actual = dlr_it_actual * beta; // Rescale from [0, 1] to [0, beta]
}

nda::vector<dcomplex> fastdiagram::get_it_actual() { return dlr_it_actual; }

nda::array<dcomplex, 3> fastdiagram::free_greens(double beta, nda::array<dcomplex, 2> H_S, double mu, bool time_order) {
  return free_gf(beta, itops, H_S, mu, time_order);
}

nda::array<dcomplex, 3> fastdiagram::free_greens_ppsc(double beta, nda::array<dcomplex, 2> H_S) { return free_gf_ppsc(beta, itops, H_S); }

void fastdiagram::hyb_init(nda::array<dcomplex, 3> Deltat0, bool poledlrflag) {

  Deltat                           = Deltat0;
  auto Deltat_reflect_intermediate = itops.reflect(Deltat); // obtain Delta(-t) from Delta(t)
  Deltat_reflect                   = Deltat_reflect_intermediate;

  for (int i = 0; i < r; ++i) Deltat_reflect(i, _, _) = transpose(Deltat_reflect_intermediate(i, _, _));

  if (poledlrflag == false) {
    auto ifops = imfreq_ops(lambda, dlr_rf, Fermion);
    int Nwmax  = int(max_element(abs((2.0 * ifops.get_ifnodes()) + 1.0)));
    if (Nwmax % 2 == 0) Nwmax += 1;
    dlr_if_dense = nda::arange(-Nwmax, Nwmax + 1, 2) * std::atan2(0, -1) / beta;
  }
}

void fastdiagram::copy_aaa_result(nda::vector<double> pol0, nda::array<dcomplex, 3> weights0) {

  pol             = pol0 * beta;
  pol_reflect     = -1.0 * pol;
  weights         = weights0;
  weights_reflect = weights0;

  for (int i = 0; i < weights.shape(0); ++i) { weights_reflect(i, _, _) = transpose(weights(i, _, _)); }
}

void fastdiagram::hyb_decomposition(bool poledlrflag, double eps) {
  // this->hyb_init(Deltat0,poledlrflag);

  if (poledlrflag == false) {
    auto Delta_decomp         = hyb_decomp(weights, pol, eps);                 // decomposition of Delta(t) using DLR coefficient
    auto Delta_decomp_reflect = hyb_decomp(weights_reflect, pol_reflect, eps); // decomposition of Delta(-t) using DLR coefficient

    Delta_F.update_inplace(Delta_decomp, dlr_it, F, F_dag);                 // Compression of Delta(t) and F, F_dag matrices
    Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dag, F); // Compression of Delta(-t) and F, F_dag matrices

  } else {

    auto Deltadlr = itops.vals2coefs(Deltat); //obtain dlr coefficient of Delta(t)

    nda::vector<double> dlr_rf_reflect       = -dlr_rf;
    nda::array<dcomplex, 3> Deltadlr_reflect = Deltadlr * 1.0;

    for (int i = 0; i < Deltadlr.shape(0); ++i) { Deltadlr_reflect(i, _, _) = transpose(Deltadlr(i, _, _)); }

    auto Delta_decomp         = hyb_decomp(Deltadlr, dlr_rf, eps);                 // decomposition of Delta(t) using DLR coefficient
    auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect, dlr_rf_reflect, eps); // decomposition of Delta(-t) using DLR coefficient

    Delta_F.update_inplace(Delta_decomp, dlr_it, F, F_dag);                 // Compression of Delta(t) and F, F_dag matrices
    Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dag, F); // Compression of Delta(-t) and F, F_dag matrices
  }

  P = Delta_F.P;
}

int64_t fastdiagram::number_of_diagrams(int m) { return pown(static_cast<int64_t>(P), m - 1) * pown(static_cast<int64_t>(2), m - 1); }

nda::array<dcomplex, 3> fastdiagram::Sigma_calc_group(nda::array<dcomplex, 3> Gt, nda::array<int, 2> D, nda::array<int, 1> diagramindex) {

  auto N                  = Gt.shape(1);
  auto Nd                 = diagramindex.shape(0);
  auto m                  = D.shape(0);
  auto num_diagram_per_fb = pown(static_cast<int64_t>(P), m - 1);
  auto Diagram            = nda::array<dcomplex, 3>::zeros({r, N, N});

  for (int id = 0; id < Nd; ++id) {

    auto fb   = nda::vector<int64_t>(m); //utility for iteration
    auto num  = diagramindex(id);
    auto num0 = num / num_diagram_per_fb;
    auto num2 = num % num_diagram_per_fb;

    for (int v = 1; v < m; ++v) {
      fb[v] = num0 % 2;
      num0  = int64_t(num0 / 2);
    }

    Diagram =
       Diagram + evaluate_one_diagram(Delta_F, Delta_F_reflect, D, Deltat, Deltat_reflect, Gt, itops, beta, F, F_dag, fb, true, num2, m, n, r, N, P);
  }

  return Diagram;
}

nda::array<dcomplex, 3> fastdiagram::Sigma_calc(nda::array<dcomplex, 3> Gt, std::string order) {

  if (order.compare("NCA") != 0 && order.compare("OCA") != 0 && order.compare("TCA") != 0)
    throw std::runtime_error("order needs to be NCA, OCA or TCA\n");

  // First do NCA calculation
  std::cout << "S-NCA: start\n";
  nda::array<dcomplex, 3> Sigma_NCA = -Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_NCA, Deltat, Deltat_reflect, Gt, itops, beta, F, F_dag);
  std::cout << "S-NCA: done\n";

  if (order.compare("NCA") == 0) return Sigma_NCA;

  // Do OCA calculation
  std::cout << "S-OCA: start\n";
  auto m               = D_OCA.shape(0);
  auto Num_diagram_oca = this->number_of_diagrams(m);

  std::cout << "number of OCA diagrams = " << Num_diagram_oca << "\n";

  nda::array<dcomplex, 3> Sigma_OCA = -Sigma_calc_group(Gt, D_OCA, arange(Num_diagram_oca));
  std::cout << "S-OCA: done\n";

  if (order.compare("OCA") == 0) return make_regular(Sigma_NCA + Sigma_OCA);

  // Do TCA calculation
  std::cout << "S-TCA: start\n";
  nda::array<dcomplex, 3> Sigma_TCA =
     -Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_1, Deltat, Deltat_reflect, Gt, itops, beta, F, F_dag)
     - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_2, Deltat, Deltat_reflect, Gt, itops, beta, F, F_dag)
     - Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_3, Deltat, Deltat_reflect, Gt, itops, beta, F, F_dag)
     + Sigma_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_4, Deltat, Deltat_reflect, Gt, itops, beta, F, F_dag);
  std::cout << "S-TCA: done\n";

  return make_regular(Sigma_NCA + Sigma_OCA + Sigma_TCA);
}

nda::array<dcomplex, 3> fastdiagram::G_calc_group(nda::array<dcomplex, 3> Gt, nda::array<int, 2> D, nda::array<int, 1> diagramindex) {

  auto N                  = Gt.shape(1);
  auto Nd                 = diagramindex.shape(0);
  auto m                  = D.shape(0);
  auto Diagram            = nda::array<dcomplex, 3>::zeros({r, n, n});
  auto num_diagram_per_fb = pown(static_cast<int64_t>(P), m - 1);
  auto Gt_reflect         = itops.reflect(Gt);

  // for (int i=0; i<r; ++i) Gt_reflect(i,_,_) = transpose(Gt_reflect(i,_,_));

  for (int id = 0; id < Nd; ++id) {
    auto fb   = nda::vector<int64_t>(m); //utility for iteration
    auto num  = diagramindex(id);
    auto num0 = num / num_diagram_per_fb;

    for (int v = 1; v < m; ++v) {
      fb[v] = num0 % 2;
      num0  = int64_t(num0 / 2);
    }

    Diagram = Diagram + eval_one_diagram_G(Delta_F, Delta_F_reflect, D, Gt, Gt_reflect, itops, beta, F, F_dag, fb, num, m, n, r, N, P);
  }

  return Diagram;
}

nda::array<dcomplex, 3> fastdiagram::G_calc(nda::array<dcomplex, 3> Gt, std::string order) {

  if (order.compare("NCA") != 0 && order.compare("OCA") != 0 && order.compare("TCA") != 0)
    throw std::runtime_error("order needs to be NCA, OCA or TCA\n");

  // First do NCA calculation
  std::cout << "G-NCA: start\n";
  nda::array<dcomplex, 3> g_NCA = -G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_NCA, Gt, itops, beta, F, F_dag);
  std::cout << "G-NCA: done\n";
  if (order.compare("NCA") == 0) return g_NCA;

  // Do OCA calculation
  std::cout << "G-OCA: start\n";
  auto m               = D_OCA.shape(0);
  auto Num_diagram_oca = this->number_of_diagrams(m);
  std::cout << "number of OCA diagrams = " << Num_diagram_oca << "\n";
  nda::array<dcomplex, 3> g_OCA = -G_calc_group(Gt, D_OCA, arange(Num_diagram_oca));

  std::cout << "G-OCA done\n";
  if (order.compare("OCA") == 0) return make_regular(g_NCA + g_OCA);

  // Do TCA calculation
  std::cout << "G-TCA: start\n";
  nda::array<dcomplex, 3> g_TCA = -G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_1, Gt, itops, beta, F, F_dag)
     - G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_2, Gt, itops, beta, F, F_dag)
     - G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_3, Gt, itops, beta, F, F_dag)
     + G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_TCA_4, Gt, itops, beta, F, F_dag);
  std::cout << "G-TCA done\n";
  return make_regular(g_NCA + g_OCA + g_TCA);
}

nda::array<dcomplex, 3> fastdiagram::time_ordered_dyson(double beta, nda::array<dcomplex, 2> H_S, double eta_0,
                                                        nda::array_const_view<dcomplex, 3> Sigma_t) {
  auto dys = dyson_it_ppsc(beta, itops, H_S);
  return dys.solve(Sigma_t, eta_0);
}

double fastdiagram::partition_function(nda::array<dcomplex, 3> Gt) {
  auto G_dlr = itops.vals2coefs(Gt);
  return -real(trace(itops.coefs2eval(G_dlr, 1.0)));
}
