#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>

#include "test_utils/block_sparse.hpp"

using nda::dcomplex;

using triqs::operators::many_body_operator_real;
using triqs::operators::n;

using mesh_t = triqs::mesh::dlr_imtime;
using bgf_t  = triqs::gfs::block_gf<mesh_t>;

namespace block_sparse = triqs_xca::block_sparse;
namespace dense        = triqs_xca::dense;
namespace test_utils   = triqs_xca::test_utils;

/**
 * @file propagator_layout.cpp
 *
 * @brief Both diagram evaluators reject a pseudo-particle propagator that is not in their layout
 *
 * @details The dense evaluator takes one (r, N, N) block over the full Hilbert space, the block-sparse evaluator one (r, d, d) block per
 * invariant subspace of its atom_diag. The layout of the other evaluator, an empty block_gf, a block of the wrong dimension and a propagator
 * on a DLR mesh of another rank are rejected with std::invalid_argument by every block_gf entry point. The one-time correlator also rejects
 * an atom_diag other than the one the evaluator was built on.
 */

/// @brief Check every wrong layout against every block_gf entry point of both evaluators
TEST(PropagatorLayout, wrong_layouts_are_rejected) {
  double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  int p = 2;

  auto ad         = test_utils::unequal_sym_set_model(true);
  auto ad_flat    = test_utils::unequal_sym_set_model(false);
  int n_hyb       = static_cast<int>(ad.get_fops().size());
  auto labels     = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
  auto hyb_coeffs = test_utils::sym_set_diagonal_hyb(labels, p);
  auto hyb_poles  = nda::vector<double>{1.3, -0.8};
  auto G_bs       = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);
  auto G_flat     = test_utils::ad_to_atom_prop(ad_flat, beta, Lambda, eps);
  auto ops        = std::vector<many_body_operator_real>{n("A", 0)};
  auto topology   = nda::array<int, 2>{{0, 1}};
  auto f_ix_vec   = nda::vector<int>{0};
  auto D_dense    = dense::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat);
  auto D_bs       = block_sparse::DiagramEvaluator(hyb_poles, hyb_coeffs, G_bs[0].mesh(), ad);

  // the evaluators accept their own layout
  ASSERT_GT(ad.n_subspaces(), 1);
  ASSERT_NO_THROW(D_dense.compute_self_energy(G_flat, topology));
  ASSERT_NO_THROW(D_bs.compute_self_energy(G_bs, topology));

  // the block-sparse propagator with one block of dimension > 1 replaced by a 1x1 block
  auto blocks = std::vector<triqs::gfs::gf<mesh_t>>{};
  for (int b = 0; b < G_bs.size(); ++b) blocks.emplace_back(G_bs[b]);
  int b_big = 0;
  while (b_big < G_bs.size() && G_bs[b_big].target_shape()[0] < 2) ++b_big;
  ASSERT_LT(b_big, G_bs.size()) << "vacuous test: no subspace of dimension > 1";
  blocks[b_big]       = triqs::gfs::gf<mesh_t>(G_bs[0].mesh(), nda::zeros<dcomplex>(G_bs[0].mesh().size(), 1, 1));
  auto G_bs_wrong_dim = bgf_t{blocks};

  auto G_empty     = bgf_t{std::vector<triqs::gfs::gf<mesh_t>>{}};
  auto G_flat_rank = test_utils::ad_to_atom_prop(ad_flat, beta, Lambda, 1.0e-6);
  auto G_bs_rank   = test_utils::ad_to_atom_prop(ad, beta, Lambda, 1.0e-6);
  ASSERT_NE(G_flat_rank[0].mesh().size(), G_flat[0].mesh().size());

  auto wrong_for_dense = std::vector<std::pair<std::string, bgf_t>>{{"block-sparse layout", G_bs}, {"empty", G_empty}, {"other rank", G_flat_rank}};
  for (auto &[what, G] : wrong_for_dense) {
    SCOPED_TRACE("dense, " + what);
    EXPECT_THROW(D_dense.compute_self_energy(G, topology), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_self_energy(G, topology, 0), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_self_energy(G, topology, f_ix_vec), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_self_energy_by_pairs(G, topology), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_self_energy_by_pairs(G, topology, 0), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_self_energy_by_pairs(G, topology, f_ix_vec), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_single_ptcle_gf(G, topology), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_single_ptcle_gf(G, topology, 0), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_single_ptcle_gf(G, topology, f_ix_vec), std::invalid_argument);
    EXPECT_THROW(D_dense.compute_one_time_correlator(G, ops, ops, ad_flat, topology, f_ix_vec), std::invalid_argument);
  }
  auto wrong_for_bs = std::vector<std::pair<std::string, bgf_t>>{
     {"dense layout", G_flat}, {"empty", G_empty}, {"wrong block dimension", G_bs_wrong_dim}, {"other rank", G_bs_rank}};
  for (auto &[what, G] : wrong_for_bs) {
    SCOPED_TRACE("block-sparse, " + what);
    EXPECT_THROW(D_bs.compute_self_energy(G, topology), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_self_energy(G, topology, 0), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_self_energy(G, topology, f_ix_vec), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(G, topology), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(G, topology, 0), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(G, topology, f_ix_vec), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_one_time_correlator(G, ops, ops, ad, topology, f_ix_vec), std::invalid_argument);
  }

  // the one-time correlator with an atom_diag of another Hilbert space (dense) or other subspaces (block-sparse)
  auto ad_other = test_utils::sz_resolved_atom_diag_helper(1, false);
  ASSERT_NE(ad_other.get_full_hilbert_space_dim(), ad_flat.get_full_hilbert_space_dim());
  EXPECT_THROW(D_dense.compute_one_time_correlator(G_flat, ops, ops, ad_other, topology, f_ix_vec), std::invalid_argument);
  EXPECT_THROW(D_bs.compute_one_time_correlator(G_bs, ops, ops, ad_flat, topology, f_ix_vec), std::invalid_argument);
}
