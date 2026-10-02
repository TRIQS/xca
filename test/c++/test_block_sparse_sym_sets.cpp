#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>
#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/dense/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>
#include <triqs_xca/dense/fset.hpp>

#include "block_sparse_utils.hpp"

using nda::dcomplex;

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;

/**
 * @file test_block_sparse_sym_sets.cpp
 *
 * @brief Tests of the barred-operator construction in the BlockOpSymQuartet constructor
 *
 * @details The contraction of the hybridization coefficients with the operators of a symmetry set is only defined within one set, as in the dense
 * reference FSet::update_hybridization(). A hybridization coupling two different symmetry sets can not be represented by the block-sparse
 * storage and must be rejected by the constructor, with the rejection threshold sym_set_coupling_tol relative to max|hyb_coeffs| pinned from both
 * sides. With symmetry sets of unequal size, an index mix-up between the sets is an out-of-bounds read that only shows up in a bounds-checked or
 * sanitized build (-DASAN=ON or -DNDA_ENFORCE_BOUNDCHECK), so the barred operators are also compared to the dense reference, which is defined
 * for any coefficient matrix.
 */

/**
 * @brief Check that the model has two symmetry sets of unequal size, so that the tests below are not vacuous
 */
TEST(BlockSparseSymSets, model_has_unequal_symmetry_sets) {

  auto ad = unequal_sym_set_model();
  int p   = 2;

  auto zero_coeffs  = nda::zeros<dcomplex>(p, 3, 3);
  auto [Fq, labels] = get_operators(ad, zero_coeffs);

  ASSERT_EQ(labels.size(), 3);
  ASSERT_EQ(nda::max_element(Fq.sym_set_labels) + 1, 2) << "expected exactly two symmetry sets, got labels " << labels;
  ASSERT_EQ(nda::sum(Fq.sym_set_sizes), 3);
  ASSERT_NE(Fq.sym_set_sizes(0), Fq.sym_set_sizes(1)) << "expected symmetry sets of unequal size, got " << Fq.sym_set_sizes;
}

/**
 * @brief Check that a hybridization coupling two symmetry sets is rejected by the BlockOpSymQuartet constructor
 */
TEST(BlockSparseSymSets, bars_reject_symmetry_set_coupling) {

  auto ad = unequal_sym_set_model();
  int p   = 2;

  auto labels = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p, 3, 3)));

  // Find a pair of orbitals living in different symmetry sets.
  int i_cross = -1, j_cross = -1;
  for (int i = 0; i < labels.size() && i_cross < 0; ++i) {
    for (int j = 0; j < labels.size(); ++j) {
      if (labels(i) != labels(j)) {
        i_cross = i;
        j_cross = j;
        break;
      }
    }
  }
  ASSERT_GE(i_cross, 0);

  auto hyb_coeffs = sym_set_diagonal_hyb(labels, p);

  // couple the two symmetry sets
  hyb_coeffs(0, i_cross, j_cross) = 0.42;
  hyb_coeffs(0, j_cross, i_cross) = 0.42;

  // match the message, other std::invalid_argument checks sit earlier in the same constructor
  try {
    get_operators(ad, hyb_coeffs);
    FAIL() << "hyb_coeffs couples orbitals " << i_cross << " and " << j_cross
           << ", which are in different symmetry sets; the BlockOpSymQuartet constructor must reject that";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string(e.what()).find("couples different symmetry sets"), std::string::npos) << e.what();
  }
}

/**
 * @brief Bracket the rejection threshold sym_set_coupling_tol from both sides
 *
 * @details Cross-set entries perturbed by 0.5 * tol * max|hyb_coeffs| must be accepted as round-off, and by 2 * tol * max|hyb_coeffs| rejected.
 * The unperturbed fixture has exactly zero cross-set entries, so only injected noise exercises the threshold.
 */
TEST(BlockSparseSymSets, bars_bracket_symmetry_set_coupling_tolerance) {

  using triqs_xca::block_sparse::sym_set_coupling_tol;

  // the bracket below scales with the tolerance, so pin its magnitude directly
  static_assert(sym_set_coupling_tol > 0.0 && sym_set_coupling_tol <= 1.0e-10,
                "the guard is only meaningful at a round-off-scale relative tolerance");

  auto ad = unequal_sym_set_model();
  int p   = 2;

  auto labels = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p, 3, 3)));
  // scaled away from 1 to distinguish a relative threshold from an absolute one
  auto hyb_coeffs = nda::make_regular(1.0e6 * sym_set_diagonal_hyb(labels, p));
  int norb        = static_cast<int>(labels.size());

  // check that the fixture has cross-set entries and that they are exactly zero
  ASSERT_GE(nda::max_element(labels) + 1, 2) << "need at least two symmetry sets, got labels " << labels;

  double max_abs = 0.0;
  int n_cross    = 0;
  for (int l = 0; l < p; ++l) {
    for (int i = 0; i < norb; ++i) {
      for (int j = 0; j < norb; ++j) {
        max_abs = std::max(max_abs, std::abs(hyb_coeffs(l, i, j)));
        if (labels(i) != labels(j)) {
          ++n_cross;
          ASSERT_EQ(hyb_coeffs(l, i, j), 0.0) << "the fixture must start from exactly zero cross-set entries";
        }
      }
    }
  }
  ASSERT_GT(n_cross, 0) << "no cross-set entries to perturb";
  ASSERT_GT(max_abs, 0.0) << "all-zero coefficients make a relative threshold vacuous";

  // perturb every cross-set entry by +-delta with alternating sign
  auto perturbed = [&](double delta) {
    auto coeffs = nda::array<dcomplex, 3>(hyb_coeffs);
    double sgn  = 1.0;
    for (int l = 0; l < p; ++l) {
      for (int i = 0; i < norb; ++i) {
        for (int j = 0; j < norb; ++j) {
          if (labels(i) != labels(j)) {
            coeffs(l, i, j) += sgn * delta;
            sgn = -sgn;
          }
        }
      }
    }
    return coeffs;
  };

  EXPECT_NO_THROW(get_operators(ad, perturbed(0.5 * sym_set_coupling_tol * max_abs)))
     << "cross-set entries at 0.5 * tol * max|hyb_coeffs| must be accepted as round-off";

  EXPECT_THROW(get_operators(ad, perturbed(2.0 * sym_set_coupling_tol * max_abs)), std::invalid_argument)
     << "cross-set entries at 2 * tol * max|hyb_coeffs| must be rejected as a structural violation";

  // the unperturbed fixture must pass
  EXPECT_NO_THROW(get_operators(ad, hyb_coeffs));
}

/**
 * @brief Compare the barred operators of symmetry sets of unequal size to the dense reference
 */
TEST(BlockSparseSymSets, bars_match_dense_for_unequal_sym_sets) {

  auto ad = unequal_sym_set_model();
  int p   = 2;

  auto labels     = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p, 3, 3)));
  auto hyb_coeffs = sym_set_diagonal_hyb(labels, p);

  auto [Fq, labels2] = get_operators(ad, hyb_coeffs);
  auto Fset          = get_operators_dense(ad, hyb_coeffs); // dense reference, no symmetry sets involved

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
          for (int l = 0; l < p; ++l) {
            for (int ri = 0; ri < fock_row.size(); ++ri) {
              for (int cj = 0; cj < fock_col.size(); ++cj) {
                auto bs    = F_dag_bar.get_block(b)(i, l, ri, cj);
                auto dense = Fset.F_dag_bars(orb, l, fock_row[ri], fock_col[cj]);
                da_err     = std::max(da_err, std::abs(bs - dense));
              }
            }
          }
        }

        int bc = F_bar_refl.get_block_index(b);
        if (bc != -1) {
          auto const &fock_row = ad.get_fock_states(bc);
          for (int l = 0; l < p; ++l) {
            for (int ri = 0; ri < fock_row.size(); ++ri) {
              for (int cj = 0; cj < fock_col.size(); ++cj) {
                auto bs    = F_bar_refl.get_block(b)(i, l, ri, cj);
                auto dense = Fset.F_bars_refl(orb, l, fock_row[ri], fock_col[cj]);
                fb_err     = std::max(fb_err, std::abs(bs - dense));
              }
            }
          }
        }
      }
    }
  }

  EXPECT_LE(da_err, 1.0e-14) << "block-sparse F_dag_bars disagree with the dense reference";
  EXPECT_LE(fb_err, 1.0e-14) << "block-sparse F_bars_refl disagree with the dense reference";
}
