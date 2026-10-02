#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/dense/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>
#include <triqs_xca/dense/fset.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/dense/dynint.hpp>
#include <triqs_xca/block_sparse/dynint.hpp>

#include "block_sparse_utils.hpp"
#include "parallel_atom_diag_check.hpp"

using nda::dcomplex;

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::many_body_operator_real;
using triqs::operators::n;

using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;
using triqs_xca::block_sparse::BlockOpSymQuartet;
using triqs_xca::dynint::get_operators_and_interactions;
using triqs_xca::dynint::get_operators_and_interactions_dense;

/**
 * @file test_block_sparse_dynint_sets.cpp
 *
 * @brief Tests of the symmetry-set construction with dynamical interactions, dynint::get_operators_and_interactions()
 *
 * @details The interaction operators are grouped into their own symmetry sets by identical connection row, never merged with a fermionic set,
 * and labeled contiguously after the fermionic ones. The extended coefficients and the concatenated labels go to the BlockOpSymQuartet
 * constructor, whose cross-set guard validates the fermionic and the interaction sets uniformly, and the resulting barred operators must equal
 * the dense ones. Since the dense path requires a single atom_diag subspace, the dense reference is built on a second atom_diag of the same
 * Hamiltonian with sym_ops = {}, see parallel_atom_diag_check.hpp. The comparison loop compare_bars_with_dense() only visits the entries the
 * block-sparse object stores, so uncovered_dense_weight() checks that the dense bars carry no weight outside the declared block pattern.
 */

namespace {

  // tolerance for the comparison with the dense reference, the measured agreement is ~1e-15
  constexpr double bar_tol = 1.0e-13;

  /// max |block-sparse bar - dense bar| over every entry the block-sparse object stores.
  std::pair<double, double> compare_bars_with_dense(BlockOpSymQuartet const &Fq, triqs::atom_diag::atom_diag<true> const &ad,
                                                    triqs_xca::dense::FSet const &Fset, int p) {
    int n_sets    = static_cast<int>(nda::max_element(Fq.sym_set_labels) + 1);
    double da_err = 0.0, fb_err = 0.0;
    for (int q_ix = 0; q_ix < n_sets; ++q_ix) {
      auto const &F_dag_bar  = Fq.F_dag_bars[q_ix];
      auto const &F_bar_refl = Fq.F_bars_refl[q_ix];
      for (int i = 0; i < Fq.sym_set_sizes(q_ix); ++i) {
        long orb = Fq.sym_set_to_orb(q_ix, i);
        for (int b = 0; b < ad.n_subspaces(); ++b) {
          auto const &fock_col = ad.get_fock_states(b);

          int bc_dag = F_dag_bar.get_block_index(b);
          if (bc_dag != -1) {
            auto const &fock_row = ad.get_fock_states(bc_dag);
            for (int l = 0; l < p; ++l)
              for (size_t ri = 0; ri < fock_row.size(); ++ri)
                for (size_t cj = 0; cj < fock_col.size(); ++cj)
                  da_err = std::max(da_err, std::abs(F_dag_bar.get_block(b)(i, l, ri, cj) - Fset.F_dag_bars(orb, l, fock_row[ri], fock_col[cj])));
          }

          int bc = F_bar_refl.get_block_index(b);
          if (bc != -1) {
            auto const &fock_row = ad.get_fock_states(bc);
            for (int l = 0; l < p; ++l)
              for (size_t ri = 0; ri < fock_row.size(); ++ri)
                for (size_t cj = 0; cj < fock_col.size(); ++cj)
                  fb_err = std::max(fb_err, std::abs(F_bar_refl.get_block(b)(i, l, ri, cj) - Fset.F_bars_refl(orb, l, fock_row[ri], fock_col[cj])));
          }
        }
      }
    }
    return {da_err, fb_err};
  }

  /// max |dense bar| over every entry the block-sparse object does not store
  std::pair<double, double> uncovered_dense_weight(BlockOpSymQuartet const &Fq, triqs::atom_diag::atom_diag<true> const &ad,
                                                   triqs_xca::dense::FSet const &Fset, int p) {
    int N        = ad.get_full_hilbert_space_dim();
    int n_ext    = static_cast<int>(Fq.sym_set_labels.size());
    int n_sets   = static_cast<int>(nda::max_element(Fq.sym_set_labels) + 1);
    auto cov_dag = nda::zeros<int>(n_ext, N, N), cov_refl = nda::zeros<int>(n_ext, N, N);

    for (int q_ix = 0; q_ix < n_sets; ++q_ix) {
      for (int i = 0; i < Fq.sym_set_sizes(q_ix); ++i) {
        long orb = Fq.sym_set_to_orb(q_ix, i);
        for (int b = 0; b < ad.n_subspaces(); ++b) {
          auto const &fock_col = ad.get_fock_states(b);
          int bc_dag           = Fq.F_dag_bars[q_ix].get_block_index(b);
          if (bc_dag != -1) {
            auto const &fock_row = ad.get_fock_states(bc_dag);
            for (auto r : fock_row)
              for (auto cc : fock_col) cov_dag(orb, r, cc) = 1;
          }
          int bc = Fq.F_bars_refl[q_ix].get_block_index(b);
          if (bc != -1) {
            auto const &fock_row = ad.get_fock_states(bc);
            for (auto r : fock_row)
              for (auto cc : fock_col) cov_refl(orb, r, cc) = 1;
          }
        }
      }
    }

    double ud = 0.0, uf = 0.0;
    for (int o = 0; o < n_ext; ++o)
      for (int l = 0; l < p; ++l)
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j) {
            if (cov_dag(o, i, j) == 0) ud = std::max(ud, std::abs(Fset.F_dag_bars(o, l, i, j)));
            if (cov_refl(o, i, j) == 0) uf = std::max(uf, std::abs(Fset.F_bars_refl(o, l, i, j)));
          }
    return {ud, uf};
  }

  /// dynint coefficients that are block diagonal w.r.t. the dynint groups given by `group`.
  nda::array<dcomplex, 3> group_diagonal_dynint_coeffs(std::vector<int> const &group, int p) {
    int n_int = static_cast<int>(group.size());
    auto d    = nda::zeros<dcomplex>(p, n_int, n_int);
    for (int l = 0; l < p; ++l)
      for (int i = 0; i < n_int; ++i)
        for (int j = 0; j < n_int; ++j)
          if (group[i] == group[j]) d(l, i, j) = 0.4 + 0.1 * l + 0.07 * i - 0.03 * j;
    return d;
  }

  many_body_operator_real S_plus(int norb) {
    many_body_operator_real S;
    for (int i = 0; i < norb; ++i) S += c_dag("up", i) * c("do", i);
    return S;
  }

} // namespace

/**
 * @brief Check that the fixtures have the connection structure the tests below rely on
 */
TEST(BlockSparseDynintSets, fixtures_have_the_expected_connection_structure) {

  // (a) unequal_sym_set_model: two fermionic sets of unequal size {2, 1}; N_A is block diagonal.
  {
    auto ad     = unequal_sym_set_model();
    auto labels = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(2, 3, 3)));
    ASSERT_EQ(labels.size(), 3);
    EXPECT_EQ(nda::max_element(labels) + 1, 2);

    auto conn           = ad.get_op_mat(many_body_operator_real(n("A", 0) + n("A", 1))).connection;
    bool block_diagonal = true;
    for (int b = 0; b < ad.n_subspaces(); ++b)
      if (conn(b) >= 0 && conn(b) != b) block_diagonal = false;
    EXPECT_TRUE(block_diagonal) << "N_A should be block diagonal on this partition, got " << conn;
  }

  // (b) spin_flip_atom_diag_helper(2, false): interleaved fermionic labels {0,1,0,1}, and the four n-type operators fall into two dynint
  //     groups of size two, {n_up0, n_do0} and {n_up1, n_do1}
  {
    auto ad     = spin_flip_atom_diag_helper(2, false);
    auto labels = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(2, 4, 4)));
    ASSERT_EQ(labels.size(), 4);
    EXPECT_EQ(labels, (nda::vector<int>{0, 1, 0, 1})) << "expected interleaved fermionic sets, got " << labels;

    auto up0 = ad.get_op_mat(many_body_operator_real(n("up", 0))).connection;
    auto do0 = ad.get_op_mat(many_body_operator_real(n("do", 0))).connection;
    auto up1 = ad.get_op_mat(many_body_operator_real(n("up", 1))).connection;
    EXPECT_EQ(up0, do0) << "n(up,0) and n(do,0) must share a connection row to form one dynint set";
    EXPECT_NE(up0, up1) << "n(up,0) and n(up,1) must differ, or there is only one dynint set";
  }

  // (c) sz_resolved_atom_diag_helper(2): S^+ and S^- have distinct, off-block-diagonal, injective connection maps that are each other's inverse
  {
    auto ad = sz_resolved_atom_diag_helper(2);
    auto Sp = S_plus(2);
    auto Sm = many_body_operator_real(dagger(Sp));

    auto cp = ad.get_op_mat(Sp).connection;
    auto cm = ad.get_op_mat(Sm).connection;
    EXPECT_NE(cp, cm) << "S^+ and S^- must land in different symmetry sets, but their connection rows agree: " << cp;
    // mutually inverse connection maps
    for (int b = 0; b < ad.n_subspaces(); ++b) {
      if (cp(b) != -1) EXPECT_EQ(cm(cp(b)), b) << "S^- does not invert S^+ at subspace " << b;
      if (cm(b) != -1) EXPECT_EQ(cp(cm(b)), b) << "S^+ does not invert S^- at subspace " << b;
    }

    int n_offdiag = 0, n_live = 0;
    for (int b = 0; b < ad.n_subspaces(); ++b) {
      if (cp(b) < 0) continue;
      ++n_live;
      if (cp(b) != b) ++n_offdiag;
      for (int b2 = b + 1; b2 < ad.n_subspaces(); ++b2) ASSERT_NE(cp(b), cp(b2)) << "S^+ connection map is non-injective: " << cp;
    }
    ASSERT_GT(n_live, 1) << "S^+ must connect more than one subspace, or the fixture is degenerate";
    EXPECT_EQ(n_offdiag, n_live) << "S^+ must be strictly off-block-diagonal here, got " << cp;
  }
}

/**
 * @brief Check the labeling and the sizes of the interaction symmetry sets
 */
TEST(BlockSparseDynintSets, dynint_ops_get_their_own_contiguous_symmetry_sets) {

  int p = 2;

  // (a) unequal_sym_set_model + a single block-diagonal interaction operator.
  {
    auto ad       = unequal_sym_set_model();
    int n_hyb     = static_cast<int>(ad.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
    int n_sym_hyb = static_cast<int>(nda::max_element(labels_f) + 1);
    auto hyb      = sym_set_diagonal_hyb(labels_f, p);

    std::vector<many_body_operator_real> ops = {many_body_operator_real(n("A", 0) + n("A", 1))};
    auto dynint_coeffs                       = group_diagonal_dynint_coeffs({0}, p);

    auto [Fq, labels] = get_operators_and_interactions(ad, hyb, dynint_coeffs, ops);

    ASSERT_EQ(labels.size(), n_hyb + static_cast<int>(ops.size()));
    for (int i = 0; i < n_hyb; ++i) EXPECT_EQ(labels(i), labels_f(i)) << "the fermionic labels must be unchanged by the dynint extension";
    for (size_t i = 0; i < ops.size(); ++i)
      EXPECT_GE(labels(n_hyb + i), n_sym_hyb) << "dynint operator " << i << " was merged into a fermionic symmetry set";
    EXPECT_EQ(nda::max_element(labels) + 1, static_cast<long>(Fq.Fs.size())) << "labels must be contiguous: " << labels;
    EXPECT_EQ(Fq.Fs.size(), Fq.F_dags.size());
    EXPECT_EQ(nda::sum(Fq.sym_set_sizes), labels.size());
    EXPECT_EQ(labels, (nda::vector<long>{0, 0, 1, 2}));
    EXPECT_EQ(Fq.sym_set_sizes, (nda::vector<long>{2, 1, 1}));
  }

  // (b) spin-flip model + four n-type operators forming two dynint groups of size two.
  {
    auto ad       = spin_flip_atom_diag_helper(2, false);
    int n_hyb     = static_cast<int>(ad.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
    int n_sym_hyb = static_cast<int>(nda::max_element(labels_f) + 1);
    auto hyb      = sym_set_diagonal_hyb(labels_f, p);

    std::vector<many_body_operator_real> ops = {many_body_operator_real(n("up", 0)), many_body_operator_real(n("do", 0)),
                                                many_body_operator_real(n("up", 1)), many_body_operator_real(n("do", 1))};
    auto dynint_coeffs                       = group_diagonal_dynint_coeffs({0, 0, 1, 1}, p);

    auto [Fq, labels] = get_operators_and_interactions(ad, hyb, dynint_coeffs, ops);

    EXPECT_EQ(labels, (nda::vector<long>{0, 1, 0, 1, 2, 2, 3, 3})) << "got " << labels;
    EXPECT_EQ(Fq.sym_set_sizes, (nda::vector<long>{2, 2, 2, 2}));
    EXPECT_EQ(nda::max_element(labels) + 1, static_cast<long>(Fq.Fs.size()));
    for (size_t i = 0; i < ops.size(); ++i) EXPECT_GE(labels(n_hyb + i), n_sym_hyb);

    // check that at least one dynint set holds more than one operator
    bool has_multi = false;
    for (int s = n_sym_hyb; s < static_cast<int>(Fq.sym_set_sizes.size()); ++s) has_multi = has_multi || (Fq.sym_set_sizes(s) > 1);
    ASSERT_TRUE(has_multi) << "every dynint set is a singleton; the within-set dynint contraction is untested";
  }
}

/**
 * @brief Compare the barred operators to the dense reference built on a single-subspace atom_diag of the same Hamiltonian
 */
TEST(BlockSparseDynintSets, bars_match_dense_with_dynamical_interactions) {

  int p = 2;

  // (a) unequal_sym_set_model + N_A
  {
    auto ad_bs   = unequal_sym_set_model(true);
    auto ad_flat = unequal_sym_set_model(false);
    ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(ad_bs, ad_flat));

    int n_hyb     = static_cast<int>(ad_bs.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad_bs, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
    auto hyb      = sym_set_diagonal_hyb(labels_f, p);

    std::vector<many_body_operator_real> ops = {many_body_operator_real(n("A", 0) + n("A", 1))};
    auto dynint_coeffs                       = group_diagonal_dynint_coeffs({0}, p);

    auto Fq   = std::get<0>(get_operators_and_interactions(ad_bs, hyb, dynint_coeffs, ops));
    auto Fset = get_operators_and_interactions_dense(ad_flat, hyb, dynint_coeffs, ops);

    ASSERT_EQ(Fset.Fs.extent(0), n_hyb + static_cast<int>(ops.size()));
    ASSERT_GT(nda::max_element(nda::abs(Fset.F_dag_bars)), 0.1) << "non-vacuity: the dense reference bars are all zero";

    auto [da_err, fb_err] = compare_bars_with_dense(Fq, ad_bs, Fset, p);
    EXPECT_LE(da_err, bar_tol) << "block-sparse F_dag_bars disagree with the dense dynint reference";
    EXPECT_LE(fb_err, bar_tol) << "block-sparse F_bars_refl disagree with the dense dynint reference";

    auto [ud, uf] = uncovered_dense_weight(Fq, ad_bs, Fset, p);
    EXPECT_LE(ud, bar_tol) << "the dense F_dag_bars carry weight outside the declared block pattern";
    EXPECT_LE(uf, bar_tol) << "the dense F_bars_refl carry weight outside the declared block pattern";
  }

  // (b) spin-flip model + four n-type operators in two dynint sets of size two
  {
    auto ad_bs   = spin_flip_atom_diag_helper(2, false);
    auto ad_flat = spin_flip_atom_diag_helper_single_subspace(2);
    ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(ad_bs, ad_flat));

    int n_hyb     = static_cast<int>(ad_bs.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad_bs, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
    auto hyb      = sym_set_diagonal_hyb(labels_f, p);

    std::vector<many_body_operator_real> ops = {many_body_operator_real(n("up", 0)), many_body_operator_real(n("do", 0)),
                                                many_body_operator_real(n("up", 1)), many_body_operator_real(n("do", 1))};
    auto dynint_coeffs                       = group_diagonal_dynint_coeffs({0, 0, 1, 1}, p);

    // check that the dynint coefficients are non-diagonal and asymmetric within a set, so that the operator order inside a set is observable
    ASSERT_GT(std::abs(dynint_coeffs(0, 0, 1)), 0.1);
    ASSERT_GT(std::abs(dynint_coeffs(0, 0, 1) - dynint_coeffs(0, 1, 0)), 0.05) << "the within-set block is symmetric, so the operator order inside "
                                                                                  "a dynint set would be unobservable";

    auto Fq   = std::get<0>(get_operators_and_interactions(ad_bs, hyb, dynint_coeffs, ops));
    auto Fset = get_operators_and_interactions_dense(ad_flat, hyb, dynint_coeffs, ops);

    ASSERT_GT(nda::max_element(nda::abs(Fset.F_dag_bars)), 0.1) << "non-vacuity: the dense reference bars are all zero";

    auto [da_err, fb_err] = compare_bars_with_dense(Fq, ad_bs, Fset, p);
    EXPECT_LE(da_err, bar_tol);
    EXPECT_LE(fb_err, bar_tol);
    auto [ud, uf] = uncovered_dense_weight(Fq, ad_bs, Fset, p);
    EXPECT_LE(ud, bar_tol);
    EXPECT_LE(uf, bar_tol);
  }

  // (c) dynint sets of size two whose operators are off-block-diagonal and non-hermitian, so that blocks_dag and the adjoint connection
  //     row are exercised within a set
  {
    auto ad_bs   = sz_resolved_atom_diag_helper(2, true);
    auto ad_flat = sz_resolved_atom_diag_helper(2, false);
    ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(ad_bs, ad_flat));

    int n_hyb     = static_cast<int>(ad_bs.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad_bs, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
    auto hyb      = sym_set_diagonal_hyb(labels_f, p);

    many_body_operator_real A0               = c_dag("up", 0) * c("do", 0);
    many_body_operator_real A1               = c_dag("up", 1) * c("do", 1);
    std::vector<many_body_operator_real> ops = {A0, A1, many_body_operator_real(dagger(A0)), many_body_operator_real(dagger(A1))};
    auto dynint_coeffs                       = group_diagonal_dynint_coeffs({0, 0, 1, 1}, p);

    auto Fq   = std::get<0>(get_operators_and_interactions(ad_bs, hyb, dynint_coeffs, ops));
    auto Fset = get_operators_and_interactions_dense(ad_flat, hyb, dynint_coeffs, ops);

    // check that there are two dynint sets of size two with off-diagonal blocks
    ASSERT_EQ(nda::max_element(Fq.sym_set_labels) + 1, 4);
    ASSERT_EQ(Fq.sym_set_sizes(2), 2);
    ASSERT_EQ(Fq.sym_set_sizes(3), 2);
    int n_offdiag = 0;
    for (int b = 0; b < ad_bs.n_subspaces(); ++b) {
      if (Fq.Fs[2].get_block_index(b) != -1 && Fq.Fs[2].get_block_index(b) != b) ++n_offdiag;
    }
    ASSERT_GT(n_offdiag, 0) << "non-vacuity: the interaction operators are block diagonal after all";
    ASSERT_GT(nda::max_element(nda::abs(Fset.F_dag_bars)), 0.1) << "non-vacuity: the dense reference bars are all zero";

    auto [da_err, fb_err] = compare_bars_with_dense(Fq, ad_bs, Fset, p);
    EXPECT_LE(da_err, bar_tol);
    EXPECT_LE(fb_err, bar_tol);
    auto [ud, uf] = uncovered_dense_weight(Fq, ad_bs, Fset, p);
    EXPECT_LE(ud, bar_tol);
    EXPECT_LE(uf, bar_tol);
  }

  // (d) no interaction operators, the extended construction must reduce exactly to get_operators()
  {
    auto ad       = unequal_sym_set_model();
    int n_hyb     = static_cast<int>(ad.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
    auto hyb      = sym_set_diagonal_hyb(labels_f, p);

    auto [Fq_plain, labels_plain] = get_operators(ad, hyb);
    // note the shape: a default-constructed (0,0,0) array is rejected by the shared-pole-count check
    auto [Fq_ext, labels_ext] = get_operators_and_interactions(ad, hyb, nda::zeros<dcomplex>(p, 0, 0), {});

    ASSERT_EQ(labels_ext.size(), labels_plain.size());
    for (long i = 0; i < labels_ext.size(); ++i) ASSERT_EQ(labels_ext(i), labels_plain(i));
    ASSERT_EQ(Fq_ext.Fs.size(), Fq_plain.Fs.size());

    double err = 0.0;
    for (size_t q_ix = 0; q_ix < Fq_ext.Fs.size(); ++q_ix) {
      for (int b = 0; b < ad.n_subspaces(); ++b) {
        if (Fq_plain.F_dag_bars[q_ix].get_block_index(b) == -1) continue;
        err = std::max(err, nda::max_element(nda::abs(Fq_ext.F_dag_bars[q_ix].get_block(b) - Fq_plain.F_dag_bars[q_ix].get_block(b))));
        err = std::max(err, nda::max_element(nda::abs(Fq_ext.F_bars_refl[q_ix].get_block(b) - Fq_plain.F_bars_refl[q_ix].get_block(b))));
      }
    }
    EXPECT_EQ(err, 0.0) << "with no interaction operators the extended construction must be bit-identical to get_operators()";
  }
}

/**
 * @brief Check that a dynint coefficient coupling two operators with different connection rows is rejected
 */
TEST(BlockSparseDynintSets, rejects_coefficients_coupling_different_dynint_sets) {

  int p         = 2;
  auto ad       = unequal_sym_set_model();
  int n_hyb     = static_cast<int>(ad.get_fops().size());
  auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
  auto hyb      = sym_set_diagonal_hyb(labels_f, p);

  // n(A,0) is block diagonal; c^dag_A0 c_B0 moves (N_A, N_B) -> (N_A+1, N_B-1). Different rows.
  std::vector<many_body_operator_real> ops = {many_body_operator_real(n("A", 0)), many_body_operator_real(c_dag("A", 0) * c("B", 0))};
  auto dynint_coeffs                       = group_diagonal_dynint_coeffs({0, 1}, p);

  // check that the two operators land in different sets
  auto labels = std::get<1>(get_operators_and_interactions(ad, hyb, dynint_coeffs, ops));
  ASSERT_NE(labels(n_hyb), labels(n_hyb + 1)) << "the two dynint operators were merged into one set, got labels " << labels;

  dynint_coeffs(0, 0, 1) = 0.25;
  dynint_coeffs(0, 1, 0) = 0.25;

  // match the message, other std::invalid_argument checks sit earlier in the constructor
  try {
    get_operators_and_interactions(ad, hyb, dynint_coeffs, ops);
    FAIL() << "a dynint coefficient coupling two symmetry sets must be rejected";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string(e.what()).find("couples different symmetry sets"), std::string::npos) << e.what();
  }

  // a hyb <-> dynint coefficient is rejected by the same guard
  auto clean = group_diagonal_dynint_coeffs({0, 1}, p);
  auto ext   = triqs_xca::hyb::get_extended_coefficients(hyb, clean);
  ASSERT_EQ(ext(0, 0, n_hyb), 0.0) << "get_extended_coefficients must zero-fill the cross blocks";
  ext(0, 0, n_hyb) = 0.3;
  auto Fq_ok       = std::get<0>(get_operators_and_interactions(ad, hyb, clean, ops));
  try {
    BlockOpSymQuartet(Fq_ok.Fs, Fq_ok.F_dags, ext, Fq_ok.sym_set_labels);
    FAIL() << "a coefficient coupling a hybridization flavour to an interaction flavour must be rejected";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string(e.what()).find("couples different symmetry sets"), std::string::npos) << e.what();
  }
}

/**
 * @brief Check that the non-block-diagonal operators S^+ and S^- form distinct symmetry sets
 *
 * @details dynint_ops = {S^-, S^+} with the diagonal coefficients diag(D_{+-}, D_{-+}) encodes the term S^+ D_{+-} S^- + S^- D_{-+} S^+, since a
 * coefficient entry pairs F_dags(i) with Fs(j).
 */
TEST(BlockSparseDynintSets, non_block_diagonal_ops_form_distinct_symmetry_sets) {

  int p        = 2;
  auto ad_bs   = sz_resolved_atom_diag_helper(2, true);
  auto ad_flat = sz_resolved_atom_diag_helper(2, false);
  ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(ad_bs, ad_flat));

  int n_hyb     = static_cast<int>(ad_bs.get_fops().size());
  auto labels_f = std::get<1>(get_operators(ad_bs, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
  int n_sym_hyb = static_cast<int>(nda::max_element(labels_f) + 1);
  auto hyb      = sym_set_diagonal_hyb(labels_f, p);

  auto Sp                                  = S_plus(2);
  auto Sm                                  = many_body_operator_real(dagger(Sp));
  std::vector<many_body_operator_real> ops = {Sm, Sp};

  auto dynint_coeffs = nda::zeros<dcomplex>(p, 2, 2);
  for (int l = 0; l < p; ++l) {
    dynint_coeffs(l, 0, 0) = 0.61 - 0.07 * l; // D_{+-}, paired with (S^-)^dag = S^+
    dynint_coeffs(l, 1, 1) = 0.53 + 0.09 * l; // D_{-+}
  }

  auto [Fq, labels] = get_operators_and_interactions(ad_bs, hyb, dynint_coeffs, ops);

  // two distinct singleton sets, after the fermionic ones
  ASSERT_EQ(labels.size(), n_hyb + 2);
  EXPECT_GE(labels(n_hyb), n_sym_hyb);
  EXPECT_GE(labels(n_hyb + 1), n_sym_hyb);
  EXPECT_NE(labels(n_hyb), labels(n_hyb + 1)) << "S^- and S^+ must not share a symmetry set, got labels " << labels;
  EXPECT_EQ(Fq.sym_set_sizes(labels(n_hyb)), 1);
  EXPECT_EQ(Fq.sym_set_sizes(labels(n_hyb + 1)), 1);
  EXPECT_EQ(nda::max_element(labels) + 1, static_cast<long>(Fq.Fs.size()));

  // permutation-type maps: each set's F block indices are the other's F_dag block indices
  auto const &set_m = Fq.Fs[labels(n_hyb)];
  auto const &set_p = Fq.Fs[labels(n_hyb + 1)];
  EXPECT_EQ(set_m.get_block_indices(), Fq.F_dags[labels(n_hyb + 1)].get_block_indices())
     << "the S^- set and the S^+ set are not each other's adjoint";
  EXPECT_EQ(set_p.get_block_indices(), Fq.F_dags[labels(n_hyb)].get_block_indices());

  int n_offdiag = 0;
  for (int b = 0; b < ad_bs.n_subspaces(); ++b)
    if (set_p.get_block_index(b) >= 0 && set_p.get_block_index(b) != b) ++n_offdiag;
  ASSERT_GT(n_offdiag, 0) << "the S^+ set is block diagonal, so this test is not about non-block-diagonal operators at all";

  // the diagonal coefficients are accepted, and the bars match the dense reference
  auto Fset = get_operators_and_interactions_dense(ad_flat, hyb, dynint_coeffs, ops);
  ASSERT_GT(nda::max_element(nda::abs(Fset.F_dag_bars)), 0.1);

  auto [da_err, fb_err] = compare_bars_with_dense(Fq, ad_bs, Fset, p);
  EXPECT_LE(da_err, bar_tol) << "block-sparse F_dag_bars disagree with the dense reference for S^+/S^-";
  EXPECT_LE(fb_err, bar_tol) << "block-sparse F_bars_refl disagree with the dense reference for S^+/S^-";
  auto [ud, uf] = uncovered_dense_weight(Fq, ad_bs, Fset, p);
  EXPECT_LE(ud, bar_tol);
  EXPECT_LE(uf, bar_tol);

  // the off-diagonal entries, which would encode a non-hermitian S^- D S^- term, are rejected
  dynint_coeffs(0, 0, 1) = 0.2;
  try {
    get_operators_and_interactions(ad_bs, hyb, dynint_coeffs, ops);
    FAIL() << "an S^- <-> S^+ off-diagonal coefficient must be rejected";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string(e.what()).find("couples different symmetry sets"), std::string::npos) << e.what();
  }
}

/**
 * @brief Check that a non-injective connection map is rejected with a message naming the collision
 *
 * @details Injectivity is the condition for the adjoint to be a single-target operator, and the check has to run before the adjoint is fetched
 * from TRIQS. On H = 0 with two flavours every Fock state is its own subspace, and O = c^dag_0 + c_1 n_0 maps both |00> and |11> to |10>.
 */
TEST(BlockSparseDynintSets, rejects_non_injective_connection_map) {

  int p = 2;
  triqs::atom_diag::fundamental_operator_set fop_set;
  fop_set.insert("x", 0);
  fop_set.insert("x", 1);
  many_body_operator_complex H;
  H += 0.0 * n("x", 0);
  auto ad = triqs::atom_diag::atom_diag<true>(H, fop_set);

  many_body_operator_real O = c_dag("x", 0) + c("x", 1) * n("x", 0);

  // check that the map is single-target but non-injective
  auto conn        = ad.get_op_mat(O).connection;
  int n_collisions = 0;
  for (int b1 = 0; b1 < ad.n_subspaces(); ++b1)
    for (int b2 = b1 + 1; b2 < ad.n_subspaces(); ++b2)
      if (conn(b1) >= 0 && conn(b1) == conn(b2)) ++n_collisions;
  ASSERT_GT(n_collisions, 0) << "the operator's connection map " << conn << " is injective, so this test proves nothing";
  ASSERT_ANY_THROW(ad.get_op_mat(many_body_operator_real(dagger(O)))) << "the adjoint was expected to be multi-target";

  int n_hyb          = static_cast<int>(ad.get_fops().size());
  auto labels_f      = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
  auto hyb           = sym_set_diagonal_hyb(labels_f, p);
  auto dynint_coeffs = group_diagonal_dynint_coeffs({0}, p);

  try {
    get_operators_and_interactions(ad, hyb, dynint_coeffs, {O});
    FAIL() << "a non-injective connection map must be rejected";
  } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("non-injective"), std::string::npos) << e.what(); }
}
