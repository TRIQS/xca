#include <gtest/gtest.h>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using triqs::operators::n;

using nda::range;

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::dense::DenseDiagramEvaluator;

using triqs_xca::block_sparse::aaa_coefs2vals;

using triqs_xca::block_sparse::DiagramEvaluator;

using triqs_xca::atom_diag::get_operators_dense;
using triqs_xca::atom_diag::ad_to_atom_prop;

TEST(two_fermions, one_hyb_pole) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;

  // set up Hamiltonian
  triqs::operators::many_body_operator_real H;
  triqs::atom_diag::fundamental_operator_set fop_set;
  double mu = 0.0;
  double U  = 3.0;
  auto N0   = n("0", 0);
  auto N1   = n("1", 0);
  auto Nop  = N0 + N1;
  H         = -mu * Nop + U * N0 * N1;
  fop_set.insert("0", 0);
  fop_set.insert("1", 0);

  // conserved operators
  std::vector<triqs::operators::many_body_operator_real> sym_ops = {Nop};
  // std::vector<triqs::operators::many_body_operator_real> sym_ops = {N0, N1};

  // atom_diag object
  triqs::atom_diag::atom_diag<false> ad(H, fop_set, sym_ops);

  // hybridization with one pole
  int p    = 1;
  int norb = 2;
  nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
  hyb_coeffs(0, _, _) = nda::eye(2);
  // double r0 = 0.5;
  // hyb_coeffs(0, 0, 1) = r0;
  // hyb_coeffs(0, 1, 0) = r0;
  nda::vector<double> hyb_poles(p);
  hyb_poles = -1.5;

  // compute single-particle Green's function for the two-fermion system with one hybridization pole
  auto G_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
  DiagramEvaluator D(beta, Lambda, eps, hyb_poles, hyb_coeffs, G_ppsc, ad);
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto spgf                   = D.compute_single_ptcle_gf(topology);

  // compare to call to dense code
  auto hyb          = aaa_coefs2vals(beta, Lambda, eps, hyb_coeffs, hyb_poles);
  auto dlr_rf       = build_dlr_rf(Lambda, eps);
  auto itops        = imtime_ops(Lambda, dlr_rf);
  auto hyb_refl     = itops.reflect(hyb);
  auto G_ppsc_dense = nda::zeros<dcomplex>(itops.rank(), ad.get_full_hilbert_space_dim(), ad.get_full_hilbert_space_dim());
  int s0            = 0;
  int s1            = 0;
  for (int s = 0; s < ad.n_subspaces(); ++s) {
    s1 += ad.get_fock_states(s).size();
    G_ppsc_dense(_, range(s0, s1), range(s0, s1)) = G_ppsc[s].data();
    s0                                            = s1;
  }
  auto Fset = get_operators_dense(ad, hyb_coeffs);
  hyb_poles = nda::make_regular(beta * hyb_poles);
  DenseDiagramEvaluator D_dense(beta, itops, hyb, hyb_refl, hyb_poles, G_ppsc_dense, Fset);
  auto mu_ops  = Fset.Fs;
  auto kap_ops = Fset.F_dags;
  CorrelatorBackbone B(topology, norb);
  auto spgf_dense = D_dense.eval_correlator(B, mu_ops, kap_ops);
  ASSERT_LE(nda::max_element(nda::abs(spgf - spgf_dense)), 1.0e-15);
}
