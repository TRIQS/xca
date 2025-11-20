#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual_gf.hpp>
#include <triqs_xca/backbone.hpp>
#include <triqs_xca/dense_backbone.hpp>

TEST(DenseGFBackbone, print) {
  nda::array<int, 2> topology = {{0, 3}, {1, 5}, {2, 4}};
  int n                       = 4;
  auto CB                     = CorrelatorBackbone(topology, n);
  nda::vector<int> fb{0, 1};
  CB.set_directions(fb);
  nda::vector<int> pole_inds{2, 5};
  nda::vector<double> dlr_rf{-10.0, -5.0, -1.0, 1.0, 5.0, 10.0, 20.0};
  CB.set_pole_inds(pole_inds, dlr_rf);
  nda::vector<int> orb_inds{0, 1, 2, 0, 2, 1};
  CB.set_orb_inds(orb_inds);
  std::cout << CB << std::endl;
}

TEST(DenseGFBackbone, OCA) {
  int n         = 4;
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // load in functions from two_band.py
  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  // compute Fbars and Fdagbars and store in Fset
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = Deltat;
  auto hyb_refl_coeffs = hyb_coeffs;
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // initialize Backbone and DiagramEvaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, n);
  auto C                      = CorrelatorDiagramEvaluator(beta, itops, Deltat, hyb_refl, dlr_rf, Gt_dense, Fset);
  auto gf                     = C.eval_diagram_dense(B, Fs_dense, F_dags_dense);

  // compare against manually-computed OCA result
  auto OCA_gf_dense_result = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, dlr_rf, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(gf - OCA_gf_dense_result)), eps);
}