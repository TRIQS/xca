#include <gtest/gtest.h>

#include <nda/algorithms.hpp>
#include <triqs/operators/many_body_operator.hpp>
#include <triqs/atom_diag/functions.hpp>

#include <triqs_xca/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse.hpp>

#include "block_sparse_utils.hpp"

using nda::linalg::matmul;

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;
using cppdlr::rel2abs;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace;

using triqs_xca::block_sparse::atom_prop_from_eigensystem;
using triqs_xca::block_sparse::trace;

/**
 * @file test_block_sparse_atom_prop.cpp
 *
 * @brief Tests of the atomic (pseudo-particle) propagator generators, independently of any diagram
 *
 * @details Every solve starts by building G0 with ad_to_atom_prop, which delegates the exponentiation to
 * atom_prop_from_eigensystem. Both routines appear throughout the suite only as fixtures feeding a diagram
 * evaluator, so a normalization or basis-ordering error in either would surface as a diagram mismatch rather
 * than as a localized failure. The tests here pin them down directly:
 *
 * - atom_prop_from_eigensystem against a hand-built eigensystem, including the eta_0 = log(Z)/beta shift and
 *   the invariant that no block of the propagator is ever flagged zero;
 * - the normalization Tr G(beta) = -1, which is what fixes eta_0 and which no other test asserts;
 * - the per-block result against an independent dense construction from the full Hamiltonian, which is what
 *   catches a mismatch between the subspace ordering of the eigensystems and the block ordering of the output;
 * - agreement between the BlockDiagOpFun and block_gf overloads, and of the latter's DLR mesh with the
 *   (Lambda, eps) grid its data was sampled on.
 */

namespace {

  // Largest deviation between two arrays of equal shape
  template <typename A, typename B> double max_dev(A const &a, B const &b) { return nda::max_element(nda::abs(nda::make_regular(a - b))); }

  // Reference propagator block -exp(-tau (H_B + eta_0)) built from an eigendecomposition, for tau = beta * t_rel
  nda::array<dcomplex, 3> ref_prop_block(nda::array<double, 1> const &evals, nda::array<dcomplex, 2> const &evecs, double eta_0, double beta,
                                         nda::vector_const_view<double> t_rel) {
    int n     = evals.size();
    int r     = t_rel.size();
    auto out  = nda::zeros<dcomplex>(r, n, n);
    auto Udag = nda::make_regular(nda::dagger(evecs));
    auto diag = nda::zeros<dcomplex>(n, n);
    for (int t = 0; t < r; ++t) {
      for (int j = 0; j < n; ++j) { diag(j, j) = -std::exp(-beta * t_rel(t) * (evals(j) + eta_0)); }
      out(t, _, _) = matmul(evecs, matmul(diag, Udag));
    }
    return out;
  }

} // namespace

/**
 * @brief Check atom_prop_from_eigensystem against a hand-built eigensystem
 *
 * @details The eigensystem has a 1x1 block, a 2x2 block whose eigenvectors are a non-trivial rotation, and a block whose Hamiltonian is zero.
 * Imaginary times are supplied directly rather than through a DLR grid so that tau = beta is available exactly and the normalization can be read off
 * without interpolation.
 */
TEST(AtomProp, from_eigensystem) {
  double beta = 2.0;

  // absolute imaginary times, including both endpoints
  nda::vector<double> t_abs = {0.0, 0.25, 0.5, 1.0};

  std::vector<nda::array<double, 1>> evals(3);
  std::vector<nda::array<dcomplex, 2>> evecs(3);

  evals[0] = nda::array<double, 1>{0.3};
  evecs[0] = nda::array<dcomplex, 2>{{1.0}};

  // 2x2 block with a non-trivial (real orthogonal) eigenbasis
  double theta = 0.7;
  evals[1]     = nda::array<double, 1>{0.0, 1.3};
  evecs[1]     = nda::array<dcomplex, 2>{{std::cos(theta), -std::sin(theta)}, {std::sin(theta), std::cos(theta)}};

  // block with H = 0: the propagator is still -exp(-tau eta_0) I, so it must not be flagged zero
  evals[2] = nda::array<double, 1>{0.0, 0.0};
  evecs[2] = nda::eye<dcomplex>(2);

  double Z = 0;
  for (auto const &e : evals) Z += nda::sum(nda::exp(-beta * e));
  double eta_0 = std::log(Z) / beta;

  auto ap = atom_prop_from_eigensystem(evals, evecs, Z, beta, t_abs);

  ASSERT_EQ(ap.get_num_block_cols(), 3);
  for (int b = 0; b < 3; ++b) {
    SCOPED_TRACE("block " + std::to_string(b));
    ASSERT_EQ(ap.get_block_size(b), evals[b].size());
    // no block of the propagator is ever zero, including the one whose Hamiltonian vanishes
    EXPECT_EQ(ap.get_zero_block_index(b), 0);
    EXPECT_LE(max_dev(ap.get_block(b), ref_prop_block(evals[b], evecs[b], eta_0, beta, t_abs)), 1e-14);
  }

  // G(0) = -I on every block
  for (int b = 0; b < 3; ++b) {
    SCOPED_TRACE("block " + std::to_string(b));
    EXPECT_LE(max_dev(ap.get_block(b)(0, _, _), -1.0 * nda::eye<dcomplex>(ap.get_block_size(b))), 1e-14);
  }

  // Tr G(beta) = -1. This is what fixes eta_0: with the shift dropped the trace would be -Z instead.
  dcomplex tr_beta = 0;
  for (int b = 0; b < 3; ++b) tr_beta += nda::trace(nda::make_regular(ap.get_block(b)(3, _, _)));
  EXPECT_LE(std::abs(tr_beta + 1.0), 1e-14);
}

/**
 * @brief Check that ad_to_atom_prop produces a correctly normalized propagator, Tr G(beta) = -1
 */
TEST(AtomProp, normalization) {
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1e-10;

  {
    SCOPED_TRACE("two-fermion model");
    auto model = two_fermion_model_helper(beta, Lambda, eps);
    EXPECT_LE(std::abs(trace(model.G_ppsc) + 1.0), 1e-10);
  }

  {
    SCOPED_TRACE("spin-flip model, particle-number symmetry");
    auto ad     = spin_flip_atom_diag_helper(2, true);
    auto G_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
    EXPECT_LE(std::abs(trace(G_ppsc) + 1.0), 1e-10);
  }

  {
    SCOPED_TRACE("spin-flip model, autopartitioned");
    auto ad     = spin_flip_atom_diag_helper(2, false);
    auto G_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
    EXPECT_LE(std::abs(trace(G_ppsc) + 1.0), 1e-10);
  }
}

/**
 * @brief Check ad_to_atom_prop block by block against a dense construction
 *
 * @details The reference goes through get_full_h_atomic and Hmat_to_Gtmat, which reconstruct the unshifted
 * Hamiltonian over the whole Hilbert space and then subtract the ground state energy and compute Z themselves --
 * a different code path from ad_to_atom_prop, which takes both from atom_diag. Projecting the dense result into
 * subspace b with get_tensor_in_atom_diag_subspace and comparing against block b is what pins the correspondence
 * between the subspace ordering of the eigensystems and the block ordering of the output.
 */
TEST(AtomProp, blocks_match_dense) {
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1e-10;

  auto dlr_rf     = build_dlr_rf(Lambda, eps);
  auto itops      = imtime_ops(Lambda, dlr_rf);
  auto dlr_it_abs = rel2abs(itops.get_itnodes());

  for (bool use_particle_number_sym : {true, false}) {
    SCOPED_TRACE(use_particle_number_sym ? "particle-number symmetry" : "autopartitioned");

    auto ad = spin_flip_atom_diag_helper(2, use_particle_number_sym);
    auto ap = ad_to_atom_prop(ad, beta, itops);

    auto Gt_dense = Hmat_to_Gtmat(get_full_h_atomic(ad), beta, dlr_it_abs);

    ASSERT_EQ(ap.get_num_block_cols(), ad.n_subspaces());
    for (int b = 0; b < ad.n_subspaces(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      auto dense_block = get_tensor_in_atom_diag_subspace(Gt_dense, b, ad);
      ASSERT_EQ(ap.get_block_size(b), dense_block.extent(1));
      EXPECT_LE(max_dev(ap.get_block(b), dense_block), 1e-12);
    }
  }
}

/**
 * @brief Check that the two ad_to_atom_prop overloads agree, and that the block_gf overload's mesh matches its data
 *
 * @details triqs::mesh::dlr_imtime takes w_max = Lambda / beta and rebuilds the DLR grid from w_max * beta, so the
 * mesh the block_gf overload attaches is only consistent with the (Lambda, eps) grid its data was sampled on if it
 * is constructed with Lambda / beta. Asserting w_max directly documents the convention; evaluating the DLR
 * expansion away from the sample nodes checks it, since a mesh built on the wrong cutoff would interpolate data
 * sampled on a different grid and miss the closed form.
 */
TEST(AtomProp, overloads_agree) {
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1e-10;

  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  auto ad  = spin_flip_atom_diag_helper(2, true);
  auto ap  = ad_to_atom_prop(ad, beta, itops);
  auto bgf = ad_to_atom_prop(ad, beta, Lambda, eps);

  // the mesh cutoff is w_max = Lambda / beta, matching the grid the data was sampled on
  EXPECT_EQ(bgf.size(), ap.get_num_block_cols());
  EXPECT_NEAR(bgf[0].mesh().w_max(), Lambda / beta, 1e-12);
  EXPECT_NEAR(bgf[0].mesh().beta(), beta, 1e-12);
  EXPECT_EQ(bgf[0].mesh().size(), itops.rank());

  // same data, block for block
  for (int b = 0; b < ap.get_num_block_cols(); ++b) {
    SCOPED_TRACE("block " + std::to_string(b));
    EXPECT_LE(max_dev(bgf[b].data(), ap.get_block(b)), 1e-14);
  }

  // Evaluating off the sample nodes exercises the mesh: compare against -exp(-tau (E + eta_0)) in the eigenbasis.
  double Z                 = triqs::atom_diag::partition_function(ad, beta);
  double eta_0             = std::log(Z) / beta;
  auto const &eigensystems = ad.get_eigensystems();

  for (double tau : {0.3 * beta, 0.5 * beta, beta}) {
    SCOPED_TRACE("tau / beta = " + std::to_string(tau / beta));
    for (int b = 0; b < ad.n_subspaces(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      auto g_dlr    = make_gf_dlr(bgf[b]);
      auto const &U = eigensystems[b].unitary_matrix;
      auto evals    = eigensystems[b].eigenvalues;

      int n     = evals.size();
      auto diag = nda::zeros<dcomplex>(n, n);
      for (int j = 0; j < n; ++j) { diag(j, j) = -std::exp(-tau * (evals(j) + eta_0)); }
      auto expected = matmul(U, matmul(diag, nda::make_regular(nda::dagger(U))));

      EXPECT_LE(max_dev(g_dlr(tau), expected), 1e-9);
    }
  }
}
