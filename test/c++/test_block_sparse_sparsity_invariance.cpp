#include <algorithm>
#include <limits>

#include <gtest/gtest.h>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/hyb.hpp>

#include "block_sparse_utils.hpp"

using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::dense::DenseDiagramEvaluator;
using triqs_xca::dense::DenseFSet;

using triqs_xca::block_sparse::DiagramEvaluator;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;

/**
 * @file test_block_sparse_sparsity_invariance.cpp
 *
 * @brief Tests that the block-sparse evaluator is indexed by orbital independently of how the orbitals are grouped into symmetry sets
 */

/**
 * @brief Check that the single-particle Green's function is indexed by orbital, not by symmetry set
 *
 * @details The operator lists built by setup_{mu,kap}_ops_for_single_ptcle_gf() must be ordered by orbital. Iterating the symmetry sets in the
 * outer loop orders them by (symmetry set, orbital) instead, which permutes the rows and columns of the returned correlator whenever the
 * symmetry set labels are not non-decreasing in the orbital index, as for the spin-flip model with autopartitioning, labels {0, 1, 0, 1}.
 */
TEST(SparsityInvariance, spgf_is_indexed_by_orbital) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;

  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  int norb             = 2;
  int nn               = 2 * norb;
  auto [hyb, hyb_refl] = discrete_bath_spin_flip_helper(beta, Lambda, eps, nn);
  auto hyb_coeffs      = itops.vals2coefs(hyb);

  // Break the flavour permutation symmetry of the spin-flip fixture by the congruence hyb -> diag(s) hyb diag(s), which leaves the
  // sparsity pattern unchanged
  nda::vector<double> flavour_scale = {1.0, 1.3, 0.7, 1.9};
  for (int i = 0; i < nn; ++i) {
    for (int j = 0; j < nn; ++j) { hyb_coeffs(nda::range::all, i, j) *= flavour_scale(i) * flavour_scale(j); }
  }

  auto ad = spin_flip_atom_diag_helper(norb, false); // autopartitioning -> several symmetry sets

  auto dlr_it_abs = cppdlr::rel2abs(itops.get_itnodes());
  auto Gt         = ad_to_atom_prop(ad, beta, itops);

  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // check that the set-major enumeration differs from the orbital order
  std::vector<int> set_major_order;
  for (int q_ix = 0; q_ix < nda::max_element(Fq.sym_set_labels) + 1; ++q_ix) {
    for (int i = 0; i < Fq.sym_set_sizes(q_ix); ++i) { set_major_order.push_back(static_cast<int>(Fq.sym_set_to_orb(q_ix, i))); }
  }
  ASSERT_EQ(static_cast<int>(set_major_order.size()), nn);
  bool interleaves = false;
  for (int i = 0; i < nn; ++i) interleaves = interleaves || (set_major_order[i] != i);
  ASSERT_TRUE(interleaves) << "vacuous test: the symmetry sets of this model do not interleave with the orbital order";

  nda::array<int, 2> topology = {{0, 2}, {1, 3}}; // crossing: gives the largest permutation gap of the order-2 topologies
  auto B                      = CorrelatorBackbone(topology, nn);

  // dense reference, ordered per orbital by construction
  auto Gt_dense = Hmat_to_Gtmat(get_full_h_atomic(ad), beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, hyb_coeffs);
  DenseDiagramEvaluator D_dense(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  auto ref_dense = D_dense.eval_correlator(Gt_dense, B, Fset.Fs, Fset.F_dags);

  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);

  // second reference, the block-sparse evaluator with the operator list built per orbital by make_correlator_ops()
  auto [mu_ops, kap_ops] = make_correlator_ops(Fq, nn);
  auto ref_bs            = D.eval_correlator(Gt, B, mu_ops, kap_ops);

  double scale = nda::max_element(nda::abs(ref_dense));
  ASSERT_GT(scale, 1.0e-6) << "vacuous test: the correlator is zero";
  ASSERT_LE(nda::max_element(nda::abs(ref_bs - ref_dense)), 1.0e-13 * scale) << "the two references disagree with each other";

  // check that no non-identity flavour permutation leaves the correlator unchanged
  auto permutation_gap = [&](std::vector<int> const &perm) {
    double gap = 0.0;
    for (int t = 0; t < ref_dense.extent(0); ++t) {
      for (int i = 0; i < nn; ++i) {
        for (int j = 0; j < nn; ++j) { gap = std::max(gap, std::abs(ref_dense(t, i, j) - ref_dense(t, perm[i], perm[j]))); }
      }
    }
    return gap;
  };

  std::vector<int> perm(nn);
  for (int i = 0; i < nn; ++i) perm[i] = i;
  double smallest_gap = std::numeric_limits<double>::max();
  while (std::next_permutation(perm.begin(), perm.end())) { smallest_gap = std::min(smallest_gap, permutation_gap(perm)); }
  ASSERT_GT(smallest_gap, 1.0e-2 * scale) << "vacuous test: some flavour permutation leaves the correlator invariant, "
                                          << "so a misordered spgf would be indistinguishable from a correct one";
  ASSERT_GT(permutation_gap(set_major_order), 1.0e-2 * scale) << "vacuous test: the set-major permutation is undetectable";

  // the routine under test, all three overloads
  auto spgf = D.compute_single_ptcle_gf(Gt, topology);
  EXPECT_LE(nda::max_element(nda::abs(spgf - ref_dense)), 1.0e-13 * scale) << "compute_single_ptcle_gf disagrees with the dense evaluator";
  EXPECT_LE(nda::max_element(nda::abs(spgf - ref_bs)), 1.0e-13 * scale)
     << "compute_single_ptcle_gf disagrees with eval_correlator fed make_correlator_ops";

  int n_backbones = D.get_num_single_ptcle_gf_backbones(topology);
  ASSERT_GT(n_backbones, 0);

  auto spgf_single = nda::make_regular(0 * ref_dense);
  for (int f_ix = 0; f_ix < n_backbones; ++f_ix) spgf_single += D.compute_single_ptcle_gf(Gt, topology, f_ix);
  EXPECT_LE(nda::max_element(nda::abs(spgf_single - ref_dense)), 1.0e-13 * scale)
     << "compute_single_ptcle_gf(Gt, topology, f_ix) disagrees with the dense evaluator";

  // the f_ix_vec overload is the one driven by the solver and takes a block_gf, so the propagator is rebuilt in that form
  auto G_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
  nda::vector<int> f_ix_vec(n_backbones);
  for (int f_ix = 0; f_ix < n_backbones; ++f_ix) f_ix_vec(f_ix) = f_ix;
  auto spgf_vec = D.compute_single_ptcle_gf(G_ppsc, topology, f_ix_vec);
  EXPECT_LE(nda::max_element(nda::abs(spgf_vec - ref_dense)), 1.0e-13 * scale)
     << "compute_single_ptcle_gf(G_ppsc, topology, f_ix_vec) disagrees with the dense evaluator";
}

/**
 * @brief Check that the [beta, tau] half of the correlator survives a topology whose vertex 0 pairs with vertex 2m-1
 *
 * @details eval_correlator() assembles the diagram in two block loops, and the walk of the [beta, tau] loop over w = vct0 + 1 .. 2m-1 is empty
 * exactly when vct0 == 2m-1. A broken-path flag shared between the loops then keeps the value left by the last block of the first loop, and
 * on a model with an absent block path the whole [beta, tau] half is dropped. This is a partition dependence, the single-block twin of the
 * same model comes out right either way. Such topologies are disconnected for m >= 2 and never produced by the solver's enumeration, so
 * they are written out by hand here.
 */
TEST(SparsityInvariance, correlator_keeps_the_beta_tau_side_when_vertex_zero_pairs_with_the_last_vertex) {
  double beta   = 2.0;
  double Lambda = 40.0;
  double eps    = 1.0e-10;
  int p_poles   = 2;

  // the partitioned model and its single-subspace twin, the correlator is indexed by orbital and needs no basis bridge
  auto ad                       = unequal_sym_set_model(true);
  auto ad_flat                  = unequal_sym_set_model(false);
  int nflav                     = static_cast<int>(ad.get_fops().size());
  auto labels                   = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, nflav, nflav)));
  auto hyb_coeffs               = sym_set_diagonal_hyb(labels, p_poles);
  nda::vector<double> hyb_poles = {1.3, -0.8};

  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  auto Gt     = ad_to_atom_prop(ad, beta, itops);
  auto G_flat = ad_to_atom_prop(ad_flat, beta, Lambda, eps);
  auto Fq     = std::get<0>(get_operators(ad, hyb_coeffs));
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad);
  auto [mu_ops, kap_ops] = make_correlator_ops(Fq, nflav);

  // The dense reference, and the one-block block-sparse twin of the same model.
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat);
  auto Gt_dense                 = Hmat_to_Gtmat(get_full_h_atomic(ad_flat), beta, cppdlr::rel2abs(itops.get_itnodes()));
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad_flat);
  auto [Gt_triv, Fq_triv]       = trivial_sparsity_helper(Gt_dense, Fs_dense, F_dags_dense, hyb_coeffs, nflav);
  DiagramEvaluator D_triv(beta, Lambda, eps, hyb_poles, hyb_coeffs, Fq_triv);
  auto [mu_triv, kap_triv] = make_correlator_ops(Fq_triv, nflav);

  // check that the partitioned side has several blocks and an absent block path, without which the bug is invisible
  ASSERT_GT(Gt.get_num_block_cols(), 1) << "vacuous test: the partitioned side has a single block";
  ASSERT_EQ(Gt_triv.get_num_block_cols(), 1);
  int absent = 0;
  for (auto const &F : Fq.Fs)
    for (int j = 0; j < F.get_num_block_cols(); ++j)
      if (F.get_block_index(j) == -1) ++absent;
  ASSERT_GT(absent, 0) << "vacuous test: every block path in this model is complete, so no block loop can leave a "
                          "broken-path flag set for the next one to inherit";

  int n_red_drivers = 0;
  for (auto topology : {nda::array<int, 2>{{0, 3}, {1, 2}}, nda::array<int, 2>{{0, 5}, {1, 2}, {3, 4}}, nda::array<int, 2>{{0, 2}, {1, 3}},
                        nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}}) {

    int two_m       = 2 * topology.extent(0);
    bool empty_left = (topology(0, 1) == two_m - 1); // the [beta, tau] walk has no iteration in which to reset
    if (empty_left) ++n_red_drivers;
    SCOPED_TRACE("topology " + [&] {
      std::ostringstream o;
      o << topology;
      return o.str();
    }() + (empty_left ? " (red driver, vct0 == 2m-1)" : " (control, vct0 < 2m-1)"));

    auto ref     = D_dense.compute_single_ptcle_gf(G_flat, topology);
    double scale = nda::max_element(nda::abs(ref));
    ASSERT_GT(scale, 0.02) << "vacuous test: the correlator is too small to resolve a dropped half";

    CorrelatorBackbone B(topology, nflav);
    auto bs = D.eval_correlator(Gt, B, mu_ops, kap_ops);

    CorrelatorBackbone B_triv(topology, nflav);
    auto triv = D_triv.eval_correlator(Gt_triv, B_triv, mu_triv, kap_triv);

    // the single-block control has no absent path and is insensitive to the bug
    EXPECT_LE(nda::max_element(nda::abs(triv - ref)), 1.0e-13 * scale)
       << "the trivial-sparsity block-sparse correlator disagrees with dense - the harness itself is broken, not the "
          "block bookkeeping";

    EXPECT_LE(nda::max_element(nda::abs(bs - ref)), 1.0e-13 * scale) << "max|dense| = " << scale
                                                                     << "; the partitioned block-sparse correlator disagrees with dense while the "
                                                                        "single-block one agrees, i.e. the result depends on the partition";
  }

  // check that at least one topology pairs vertex 0 with vertex 2m-1
  ASSERT_GT(n_red_drivers, 0) << "vacuous test: no topology here pairs vertex 0 with vertex 2m-1, which is the only "
                                 "case in which the [beta, tau] block walk is empty";
}
