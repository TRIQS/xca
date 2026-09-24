#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>
#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/dense.hpp>

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
 * @details The contraction of the hybridization coefficients with the operators of a symmetry set is only defined within one set. A hybridization
 * coupling two different symmetry sets can not be represented by the block-sparse storage and must be rejected by the constructor. With symmetry
 * sets of unequal size, an index mix-up between the sets is an out-of-bounds read that only shows up in a bounds-checked or sanitized build
 * (-DASAN=ON or -DNDA_ENFORCE_BOUNDCHECK), so the barred operators are also compared to the dense reference, which is defined for any coefficient
 * matrix.
 */

namespace {

  /**
   * @brief Model with two symmetry sets of unequal size: orbitals A0 and A1 are mixed by a hopping term and share the conserved N_A, while the
   * decoupled orbital B0 carries its own conserved N_B.
   */
  triqs::atom_diag::atom_diag<true> unequal_sym_set_model() {

    many_body_operator_complex NA = n("A", 0) + n("A", 1);
    many_body_operator_complex NB = n("B", 0);

    many_body_operator_complex H;
    H += 0.3 * NA - 0.7 * NB;
    H += 0.4 * (c_dag("A", 0) * c("A", 1) + c_dag("A", 1) * c("A", 0));
    H += 1.1 * n("A", 0) * n("A", 1);
    H += 0.9 * n("A", 0) * n("B", 0) + 0.5 * n("A", 1) * n("B", 0);

    triqs::atom_diag::fundamental_operator_set fop_set;
    fop_set.insert("A", 0);
    fop_set.insert("A", 1);
    fop_set.insert("B", 0);

    std::vector<many_body_operator_complex> sym_ops = {NA, NB};
    return {H, fop_set, sym_ops};
  }

  /**
   * @brief Hybridization coefficients that are block diagonal with respect to the symmetry sets
   */
  nda::array<dcomplex, 3> sym_set_diagonal_hyb(nda::vector_const_view<int> labels, int p) {
    int norb        = static_cast<int>(labels.size());
    auto hyb_coeffs = nda::zeros<dcomplex>(p, norb, norb);
    for (int l = 0; l < p; ++l) {
      for (int i = 0; i < norb; ++i) {
        for (int j = 0; j < norb; ++j) {
          if (labels(i) == labels(j)) hyb_coeffs(l, i, j) = 0.3 + 0.1 * l + 0.2 * i - 0.05 * j;
        }
      }
    }
    return hyb_coeffs;
  }

} // namespace

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

  EXPECT_THROW(get_operators(ad, hyb_coeffs), std::exception)
     << "hyb_coeffs couples orbitals " << i_cross << " and " << j_cross
     << ", which are in different symmetry sets; the BlockOpSymQuartet constructor drops these terms silently";
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
