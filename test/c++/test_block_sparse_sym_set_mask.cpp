#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>

#include "block_sparse_utils.hpp"

using nda::dcomplex;

using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_operators;

using triqs_xca::block_sparse::BlockDiagOpFun;
using triqs_xca::block_sparse::BlockOpSymQuartet;
using triqs_xca::block_sparse::DiagramEvaluator;
using triqs_xca::block_sparse::sym_set_coupling_tol;

/**
 * @file test_block_sparse_sym_set_mask.cpp
 *
 * @brief Tests of the symmetry-set mask on the zero vertex of the self-energy, and of the input validation of the Fq-only constructor
 *
 * @details The BlockOpSymQuartet constructor rejects a cross-symmetry-set coefficient only above sym_set_coupling_tol * max|hyb_coeffs|, and
 * the accepted remainder has to be made harmless downstream. The absolute 1e-16 guard on the assembled T_out in find_path_self_energy() does
 * not do that, since T_out also carries the propagators, the DLR kernel and the operator norms, and a cross-set (p_kap, p_mu) path need not
 * return to block b_ix, so its contribution is accumulated into a block of the wrong shape and the error is independent of the size of the
 * coefficient. Restricting the (p_kap, p_mu) loop to p_mu == p_kap removes these contributions, and this file checks that the accepted
 * remainder is inert. The rejection side of the threshold is pinned by BlockSparseSymSets.bars_bracket_symmetry_set_coupling_tolerance.
 */

namespace {

  constexpr double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  constexpr int p_poles = 2;

  /// Where the perturbation goes: the cross-set entries, or the within-set off-diagonals used as the control
  enum class Slot { cross, within };

  /// The symmetry-set labels the quartet assigns to this model's flavours.
  nda::vector<int> sym_set_labels_of(triqs::atom_diag::atom_diag<true> const &ad) {
    int n_hyb   = static_cast<int>(ad.get_fops().size());
    auto labels = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
    nda::vector<int> lab(labels.size());
    for (long i = 0; i < labels.size(); ++i) lab(i) = static_cast<int>(labels(i));
    return lab;
  }

  /**
   * @brief sym_set_diagonal_hyb with every entry of slot shifted by +-delta, with alternating sign so that the perturbations sum to zero
   */
  nda::array<dcomplex, 3> perturbed_hyb(nda::vector_const_view<int> lab, Slot slot, double delta) {
    int n_hyb  = static_cast<int>(lab.size());
    auto c     = nda::array<dcomplex, 3>(sym_set_diagonal_hyb(lab, p_poles));
    double sgn = 1.0;
    for (int l = 0; l < p_poles; ++l)
      for (int i = 0; i < n_hyb; ++i)
        for (int j = 0; j < n_hyb; ++j) {
          bool hit = (slot == Slot::cross) ? (lab(i) != lab(j)) : (lab(i) == lab(j) && i != j);
          if (hit) {
            c(l, i, j) += sgn * delta;
            sgn = -sgn;
          }
        }
    return c;
  }

  /// Sigma at the crossing OCA topology, from coefficients the quartet is also built from
  BlockDiagOpFun sigma_from(triqs::atom_diag::atom_diag<true> const &ad, nda::array<dcomplex, 3> const &coeffs) {
    nda::vector<double> hyb_poles = {1.3, -0.8};
    nda::array<int, 2> topology   = {{0, 2}, {1, 3}};

    auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
    auto Fq    = std::get<0>(get_operators(ad, coeffs));
    DiagramEvaluator D(beta, Lambda, eps, hyb_poles, coeffs, Fq);
    auto Gt = ad_to_atom_prop(ad, beta, itops);
    return BlockDiagOpFun(D.compute_self_energy(Gt, topology));
  }

  double max_block_diff(BlockDiagOpFun const &A, BlockDiagOpFun const &B) {
    double d = 0.0;
    for (int b = 0; b < A.get_num_block_cols(); ++b) d = std::max(d, nda::max_element(nda::abs(A.get_block(b) - B.get_block(b))));
    return d;
  }

  double max_block_abs(BlockDiagOpFun const &A) {
    double d = 0.0;
    for (int b = 0; b < A.get_num_block_cols(); ++b) d = std::max(d, nda::max_element(nda::abs(A.get_block(b))));
    return d;
  }

} // namespace

/**
 * @brief Check that a cross-set coefficient accepted by the quartet constructor does not reach the self-energy
 */
TEST(BlockSparseSymSetMask, accepted_cross_set_coefficients_leave_sigma_unchanged) {

  // the perturbation is a fraction of the tolerance, so pin its magnitude from both sides
  static_assert(sym_set_coupling_tol >= 1.0e-14 && sym_set_coupling_tol <= 1.0e-10,
                "the guard is only meaningful at a round-off-scale relative tolerance, and this test's "
                "perturbation is a fraction of it");

  // Order 2 and crossing. Order 1 has a single vertex pair and cannot separate p_kap from p_mu.
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  struct Fixture {
    char const *name;
    triqs::atom_diag::atom_diag<true> ad;
  };
  std::vector<Fixture> fixtures;
  fixtures.push_back({"unequal_sym_set_model(true)", unequal_sym_set_model(true)});                   // sets {2, 1}
  fixtures.push_back({"spin_flip_atom_diag_helper(2, false)", spin_flip_atom_diag_helper(2, false)}); // interleaved sets
  fixtures.push_back({"sz_resolved_atom_diag_helper(2)", sz_resolved_atom_diag_helper(2, true)});     // contiguous sets

  for (auto const &f : fixtures) {
    SCOPED_TRACE(f.name);

    int n_hyb = static_cast<int>(f.ad.get_fops().size());
    auto lab  = sym_set_labels_of(f.ad);

    // check that the fixture has cross-set entries to perturb and within-set off-diagonals for the control
    ASSERT_GE(nda::max_element(lab) + 1, 2) << "need at least two symmetry sets, got labels " << lab;
    int n_cross = 0, n_within = 0;
    for (int i = 0; i < n_hyb; ++i)
      for (int j = 0; j < n_hyb; ++j) {
        if (lab(i) != lab(j)) ++n_cross;
        if (lab(i) == lab(j) && i != j) ++n_within;
      }
    ASSERT_GT(n_cross, 0) << "no cross-set entries to perturb";
    ASSERT_GT(n_within, 0) << "no within-set off-diagonal entries: the control below would be vacuous";

    auto clean     = perturbed_hyb(lab, Slot::cross, 0.0);
    double max_abs = nda::max_element(nda::abs(clean));
    ASSERT_GT(max_abs, 0.0) << "all-zero coefficients make a relative threshold vacuous";
    double delta = 0.9 * sym_set_coupling_tol * max_abs;

    // the entry must be on the accepted side of the threshold
    auto cross = perturbed_hyb(lab, Slot::cross, delta);
    ASSERT_NO_THROW(get_operators(f.ad, cross))
       << "premise broken: a cross-set entry at 0.9 * sym_set_coupling_tol * max|hyb_coeffs| must be accepted";

    auto Sigma_0     = sigma_from(f.ad, clean);
    auto Sigma_cross = sigma_from(f.ad, cross);
    ASSERT_GT(max_block_abs(Sigma_0), 0.1) << "vacuous test: the self-energy is too small to be sensitive to anything";

    // exact equality, under the mask the cross-set coefficients are never read on the Sigma path
    EXPECT_EQ(max_block_diff(Sigma_cross, Sigma_0), 0.0)
       << "a cross-set coefficient at 0.9 * sym_set_coupling_tol * max|hyb_coeffs| = " << delta << " changed Sigma by "
       << max_block_diff(Sigma_cross, Sigma_0) << " against max|Sigma| = " << max_block_abs(Sigma_0)
       << ".\nThe (p_kap, p_mu) loop of find_path_self_energy must be restricted to p_mu == p_kap. The 1e-16 guard "
          "does not catch this: the cross-set T_out reaches it one to two orders above 1e-16, and is then added to a "
          "block of the wrong shape.";

    // check that the same perturbation within a set does move Sigma
    auto Sigma_within = sigma_from(f.ad, perturbed_hyb(lab, Slot::within, delta));
    EXPECT_GT(max_block_diff(Sigma_within, Sigma_0), 0.0) << "vacuous test: a within-set perturbation of the same size " << delta
                                                          << " does not move Sigma either, so the "
                                                             "exact equality above says nothing about the mask";
  }
}

/**
 * @brief Check that the Fq-only constructor rejects coefficients that couple different symmetry sets
 *
 * @details The atom_diag constructors build the quartet and the hybridization from the same array, while the Fq-only constructor takes them
 * separately and the quartet stores no coefficients to check against.
 */
TEST(BlockSparseSymSetMask, Fq_only_constructor_rejects_cross_set_coefficients) {
  auto ad   = unequal_sym_set_model(true);
  auto lab  = sym_set_labels_of(ad);
  auto good = perturbed_hyb(lab, Slot::cross, 0.0);
  auto Fq   = std::get<0>(get_operators(ad, good));

  nda::vector<double> hyb_poles = {1.3, -0.8};
  ASSERT_NO_THROW(DiagramEvaluator(beta, Lambda, eps, hyb_poles, good, Fq)) << "the clean pair must still construct";

  // Well above sym_set_coupling_tol, i.e. the case the quartet itself would have rejected.
  auto bad = perturbed_hyb(lab, Slot::cross, 0.25 * nda::max_element(nda::abs(good)));
  ASSERT_THROW(std::get<0>(get_operators(ad, bad)), std::invalid_argument) << "premise: the quartet rejects this array";

  try {
    DiagramEvaluator D(beta, Lambda, eps, hyb_poles, bad, Fq);
    FAIL() << "the Fq-only constructor accepted hyb_coeffs that couple different symmetry sets, which the "
              "quartet it was handed cannot represent";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string(e.what()).find("symmetry set"), std::string::npos)
       << "wrong throw - the message should name the symmetry-set coupling: " << e.what();
  }
}

/**
 * @brief Check that the Fq-only constructor rejects coefficients whose extents disagree with the quartet or the poles
 */
TEST(BlockSparseSymSetMask, Fq_only_constructor_rejects_mismatched_coefficient_extents) {
  auto ad  = unequal_sym_set_model(true);
  auto lab = sym_set_labels_of(ad);
  auto Fq  = std::get<0>(get_operators(ad, perturbed_hyb(lab, Slot::cross, 0.0)));

  int n = static_cast<int>(nda::sum(Fq.sym_set_sizes));
  ASSERT_GT(n, 2) << "vacuous test: the truncated array below must be smaller than the quartet";

  auto too_small                = nda::zeros<dcomplex>(p_poles, n - 1, n - 1);
  nda::vector<double> hyb_poles = {1.3, -0.8};

  // match on "shape": with the extent check deleted, check_sym_set_block_diagonal would read past the smaller array and throw the
  // cross-set message instead
  try {
    DiagramEvaluator D(beta, Lambda, eps, hyb_poles, too_small, Fq);
    FAIL() << "the Fq-only constructor accepted a (" << p_poles << ", " << n - 1 << ", " << n - 1 << ") coefficient array for a quartet holding " << n
           << " orbitals";
  } catch (std::invalid_argument const &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("shape"), std::string::npos)
       << "wrong throw - this must be the extent check, not the cross-set check reading past the array: " << msg;
    EXPECT_NE(msg.find("sum(Fq.sym_set_sizes)"), std::string::npos) << "the message should say what n was compared against: " << msg;
  }

  // the pole extent too, a wrong p is silently reinterpreted by the reshape in coefs2vals
  auto wrong_poles = nda::zeros<dcomplex>(p_poles + 1, n, n);
  try {
    DiagramEvaluator D(beta, Lambda, eps, hyb_poles, wrong_poles, Fq);
    FAIL() << "the Fq-only constructor accepted a (" << p_poles + 1 << ", " << n << ", " << n << ") coefficient array against " << hyb_poles.size()
           << " hybridization poles";
  } catch (std::invalid_argument const &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("shape"), std::string::npos) << "wrong throw - this must be the extent check: " << msg;
    EXPECT_NE(msg.find("hyb_poles.size()"), std::string::npos) << "the message should name hyb_poles as the thing that disagrees: " << msg;
  }
}
