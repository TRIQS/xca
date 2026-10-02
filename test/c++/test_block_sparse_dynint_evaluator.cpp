#include <algorithm>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/block_sparse/dynint.hpp>
#include <triqs_xca/topology.hpp>

#include "block_sparse_utils.hpp"
#include "dense_comparison.hpp"
#include "parallel_atom_diag_check.hpp"

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;
using nda::dcomplex;

using triqs::operators::many_body_operator_real;
using triqs::operators::n;

using triqs_xca::block_sparse::atom_diag::get_operators;
using triqs_xca::block_sparse::BlockDiagOpFun;
using triqs_xca::block_sparse::BlockOpSymQuartet;
using triqs_xca::block_sparse::DiagramEvaluator;
namespace dense = triqs_xca::dense;
using triqs_xca::hyb::get_extended_coefficients;
using triqs_xca::block_sparse::dynint::get_operators_and_interactions;

/**
 * @file test_block_sparse_dynint_evaluator.cpp
 *
 * @brief Tests of the block-sparse DiagramEvaluator with dynamical interactions: the constructors, n / n_hyb / n_int and Nmax
 *
 * @details The self-energy is compared to the dense evaluator with dynamical interactions, which runs on the single-subspace twin of the
 * partitioned atom_diag, with the two eigenbases bridged by get_tensor_in_atom_diag_subspace(). The comparison on the crossing topology
 * requires n_int to reach every Backbone constructed in block_sparse/diagram_evaluator.cpp, since the permutation parity otherwise treats the
 * interaction line as fermionic, while the non-crossing topologies are insensitive to this and pin the rest of the construction.
 */

namespace {

  // tolerance for the comparison with the dense reference, the measured agreement is ~1e-16
  constexpr double sigma_tol = 1.0e-13;

  constexpr double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  constexpr int p_poles = 2;

  struct DynintModel {
    triqs::atom_diag::atom_diag<true> ad;      // partitioned: the block-sparse side
    triqs::atom_diag::atom_diag<true> ad_flat; // the sym_ops = {} twin: the dense side
    nda::vector<double> hyb_poles;
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::array<dcomplex, 3> dynint_coeffs;
    std::vector<many_body_operator_real> dynint_ops;
  };

  /**
   * @brief unequal_sym_set_model and its single-subspace twin, plus n_int density interactions
   *
   * @details The symmetry sets have unequal sizes {2, 1} and the largest subspace has dimension 2, so the Nmax tests are not vacuous.
   */
  DynintModel dynint_model(int n_int) {
    auto ad       = unequal_sym_set_model(true);
    auto ad_flat  = unequal_sym_set_model(false);
    int n_hyb     = static_cast<int>(ad.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
    auto hyb      = sym_set_diagonal_hyb(labels_f, p_poles);

    std::vector<many_body_operator_real> ops;
    if (n_int > 0) ops.emplace_back(n("A", 0));
    // n_A0 and n_B0 have different connection rows and land in two singleton dynint sets
    if (n_int > 1) ops.emplace_back(n("B", 0));
    EXPECT_EQ(static_cast<int>(ops.size()), n_int) << "dynint_model only knows two interaction operators";

    // l- and i-dependent, so that a wrong pole or interaction flavour shows up
    auto d = nda::zeros<dcomplex>(p_poles, n_int, n_int);
    for (int l = 0; l < p_poles; ++l)
      for (int i = 0; i < n_int; ++i) d(l, i, i) = 0.61 - 0.07 * l + 0.05 * i;

    nda::vector<double> hyb_poles = {1.3, -0.8};
    return {ad, ad_flat, hyb_poles, hyb, d, ops};
  }

  /// Same quartet with symmetry set s moved to position 0
  BlockOpSymQuartet permute_set_to_front(BlockOpSymQuartet const &Fq, long s, nda::array_const_view<dcomplex, 3> coeffs) {
    long nsets = nda::max_element(Fq.sym_set_labels) + 1;
    std::vector<long> order{s};
    for (long i = 0; i < nsets; ++i)
      if (i != s) order.push_back(i);
    std::vector<long> pos(nsets);
    for (long i = 0; i < nsets; ++i) pos[order[i]] = i;

    std::vector<triqs_xca::block_sparse::BlockOpSymSet> Fs, F_dags;
    for (long i = 0; i < nsets; ++i) {
      Fs.push_back(Fq.Fs[order[i]]);
      F_dags.push_back(Fq.F_dags[order[i]]);
    }
    auto labels = nda::vector<long>(Fq.sym_set_labels.size());
    for (long i = 0; i < Fq.sym_set_labels.size(); ++i) labels(i) = pos[Fq.sym_set_labels(i)];
    return BlockOpSymQuartet(Fs, F_dags, coeffs, labels);
  }

  int max_subspace_dim(triqs::atom_diag::atom_diag<true> const &ad) {
    auto d = ad.get_subspace_dims();
    return *std::max_element(d.begin(), d.end());
  }

} // namespace

/**
 * @brief Compare the self-energy on the crossing topology to the dense evaluator with dynamical interactions
 */
TEST(BlockSparseDynintEvaluator, self_energy_matches_dense_with_dynamical_interactions) {

  auto m = dynint_model(/*n_int=*/1);
  ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(m.ad, m.ad_flat));

  // only a crossing topology exposes the parity error
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  int n_hyb = static_cast<int>(m.ad.get_fops().size());
  int n_int = static_cast<int>(m.dynint_ops.size());

  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = ad_to_atom_prop(m.ad, beta, itops);

  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);

  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  ASSERT_EQ(D.n_hyb, n_hyb);
  ASSERT_EQ(D.n_int, n_int);
  ASSERT_EQ(D.n, n_hyb + n_int);

  // check that some backbones carry the interaction operator on the internal line, which does not hold at order 1
  auto Fq_ferm = std::get<0>(get_operators(m.ad, m.hyb_coeffs));
  DiagramEvaluator D_ferm(beta, Lambda, eps, m.hyb_poles, m.hyb_coeffs, Fq_ferm);
  ASSERT_GT(D.get_num_self_energy_backbones(topology), D_ferm.get_num_self_energy_backbones(topology))
     << "vacuous test: the interaction flavour is not in the backbone enumeration (this compares two evaluators, so it asserts n_ext > n_hyb rather than that any particular backbone puts the interaction on an internal line)";

  auto Sigma = BlockDiagOpFun(D.compute_self_energy(Gt, topology));

  // dense reference on the single-subspace twin
  auto G_flat = ad_to_atom_prop(m.ad_flat, beta, Lambda, eps);
  dense::DiagramEvaluator D_dense(m.hyb_poles, m.hyb_coeffs, G_flat[0].mesh(), m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  ASSERT_EQ(D_dense.n_hyb, n_hyb);
  ASSERT_EQ(D_dense.n_int, n_int);
  auto Sigma_dense = D_dense.compute_self_energy(G_flat, topology);

  auto [err, scale] = compare_sigma_with_dense(Sigma, Sigma_dense, m.ad);

  // the comparison is absolute, so require a non-negligible self-energy
  ASSERT_GT(scale, 0.1) << "vacuous test: the self-energy is too small for an absolute tolerance";

  EXPECT_LE(err, sigma_tol) << "max|Sigma_dense| = " << scale << ", max|bs - dense| = " << err
                            << " -- the interaction line is being given fermionic parity";
}

/**
 * @brief Check the comparison harness without interaction operators, where the block-sparse and the dense evaluator must agree
 */
TEST(BlockSparseDynintEvaluator, no_dynamical_interaction_matches_dense) {

  auto m = dynint_model(/*n_int=*/0);
  ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(m.ad, m.ad_flat));

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto itops                  = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt                     = ad_to_atom_prop(m.ad, beta, itops);

  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, nda::zeros<dcomplex>(p_poles, 0, 0), {}));
  auto ext = get_extended_coefficients(m.hyb_coeffs, nda::zeros<dcomplex>(p_poles, 0, 0));
  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, /*n_int=*/0);
  auto Sigma = BlockDiagOpFun(D.compute_self_energy(Gt, topology));

  auto G_flat = ad_to_atom_prop(m.ad_flat, beta, Lambda, eps);
  dense::DiagramEvaluator D_dense(m.hyb_poles, m.hyb_coeffs, G_flat[0].mesh(), m.ad_flat);
  auto Sigma_dense = D_dense.compute_self_energy(G_flat, topology);

  auto [err, scale] = compare_sigma_with_dense(Sigma, Sigma_dense, m.ad);
  ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is zero"; // measured 0.134
  EXPECT_LE(err, sigma_tol) << "max|bs - dense| = " << err;
}

/**
 * @brief Compare the self-energy on the non-crossing topologies, which are insensitive to the parity of the interaction line
 */
TEST(BlockSparseDynintEvaluator, self_energy_matches_dense_on_non_crossing_topologies) {

  auto m = dynint_model(/*n_int=*/1);
  ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(m.ad, m.ad_flat));
  int n_int = static_cast<int>(m.dynint_ops.size());

  auto itops  = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt     = ad_to_atom_prop(m.ad, beta, itops);
  auto Fq     = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext    = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  auto G_flat = ad_to_atom_prop(m.ad_flat, beta, Lambda, eps);

  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  dense::DiagramEvaluator D_dense(m.hyb_poles, m.hyb_coeffs, G_flat[0].mesh(), m.ad_flat, m.dynint_ops, m.dynint_coeffs);

  for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 3}, {1, 2}}}) {
    SCOPED_TRACE("topology " + [&] {
      std::ostringstream o;
      o << topology;
      return o.str();
    }());
    ASSERT_EQ(triqs_xca::topology::topology_parity(topology), 1) << "this test is about non-crossing topologies";

    auto Sigma        = BlockDiagOpFun(D.compute_self_energy(Gt, topology));
    auto Sigma_dense  = D_dense.compute_self_energy(G_flat, topology);
    auto [err, scale] = compare_sigma_with_dense(Sigma, Sigma_dense, m.ad);
    ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is zero"; // measured 2.68 and 0.68
    EXPECT_LE(err, sigma_tol) << "max|bs - dense| = " << err;
  }
}

/**
 * @brief Check the atom_diag constructor with dynamical interactions
 *
 * @details hyb_coeffs must cover every fundamental operator of the atom_diag, which is stricter than get_operators().
 */
TEST(BlockSparseDynintEvaluator, dynint_constructor_contract) {

  auto m    = dynint_model(/*n_int=*/2); // two interaction flavours, in two singleton dynint sets
  int n_hyb = static_cast<int>(m.ad.get_fops().size());
  int n_int = static_cast<int>(m.dynint_ops.size());

  auto G = ad_to_atom_prop(m.ad, beta, Lambda, eps);
  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);

  EXPECT_EQ(D.n_hyb, n_hyb);
  EXPECT_EQ(D.n_int, n_int);
  EXPECT_EQ(D.n, n_hyb + n_int);
  ASSERT_GT(n_int, 0) << "vacuous test: no interaction operator";

  // the evaluator built from the quartet must give the same Sigma exactly
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto Fq                     = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext                    = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  EXPECT_EQ(D.q, static_cast<int>(nda::max_element(Fq.sym_set_labels) + 1));

  DiagramEvaluator D_fq(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = ad_to_atom_prop(m.ad, beta, itops);

  auto Sa = BlockDiagOpFun(D.compute_self_energy(Gt, topology));
  auto Sb = BlockDiagOpFun(D_fq.compute_self_energy(Gt, topology));
  for (int b = 0; b < Sa.get_num_block_cols(); ++b) EXPECT_EQ(nda::max_element(nda::abs(Sa.get_block(b) - Sb.get_block(b))), 0.0) << "block " << b;

  // match the message, other std::invalid_argument checks sit earlier in the same routine
  auto short_hyb = nda::zeros<dcomplex>(p_poles, n_hyb - 1, n_hyb - 1);
  try {
    DiagramEvaluator D_bad(m.hyb_poles, short_hyb, G[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);
    (void)D_bad;
    FAIL() << "expected std::invalid_argument for a hyb_coeffs that does not cover every fundamental operator";
  } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("fundamental operators"), std::string::npos) << e.what(); }
}

/**
 * @brief Check that the constructor with empty dynint_ops reproduces the plain constructor exactly
 *
 * @details The empty coefficient array has to have shape (p, 0, 0), a default-constructed (0, 0, 0) array is rejected by the shared-pole-count
 * check.
 */
TEST(BlockSparseDynintEvaluator, empty_dynint_ops_reproduce_the_plain_constructor) {

  auto m                      = dynint_model(/*n_int=*/0);
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto G                      = ad_to_atom_prop(m.ad, beta, Lambda, eps);

  DiagramEvaluator D0(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad);
  DiagramEvaluator D1(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad, {}, nda::zeros<dcomplex>(p_poles, 0, 0));

  EXPECT_EQ(D1.n, D0.n);
  EXPECT_EQ(D1.n_hyb, D0.n_hyb);
  EXPECT_EQ(D1.n_int, 0);
  EXPECT_EQ(D1.q, D0.q);
  EXPECT_EQ(D1.Nmax, D0.Nmax);

  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = ad_to_atom_prop(m.ad, beta, itops);
  auto S0    = BlockDiagOpFun(D0.compute_self_energy(Gt, topology));
  auto S1    = BlockDiagOpFun(D1.compute_self_energy(Gt, topology));

  double scale = 0.0;
  for (int b = 0; b < S0.get_num_block_cols(); ++b) {
    scale = std::max(scale, nda::max_element(nda::abs(S0.get_block(b))));
    EXPECT_EQ(nda::max_element(nda::abs(S0.get_block(b) - S1.get_block(b))), 0.0)
       << "block " << b << ": with no interaction operators the two constructors must be bit-identical";
  }
  ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is zero"; // measured 0.134
}

/**
 * @brief Check that the single-particle Green's function has shape (r, n_hyb, n_hyb), i.e. the interaction flavours are not external legs
 */
TEST(BlockSparseDynintEvaluator, spgf_excludes_the_interaction_flavours) {

  auto m    = dynint_model(/*n_int=*/2);
  int n_hyb = static_cast<int>(m.ad.get_fops().size());
  int n_int = static_cast<int>(m.dynint_ops.size());
  ASSERT_GT(n_int, 0) << "vacuous test: without interaction flavours n_hyb == n";

  auto G     = ad_to_atom_prop(m.ad, beta, Lambda, eps);
  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = ad_to_atom_prop(m.ad, beta, itops);

  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);
  ASSERT_EQ(D.n, n_hyb + n_int);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto spgf                   = D.compute_single_ptcle_gf(Gt, topology);

  EXPECT_EQ(spgf.extent(1), n_hyb) << "the spgf has " << spgf.extent(1) << " rows, i.e. the interaction flavours are being emitted as external legs";
  EXPECT_EQ(spgf.extent(2), n_hyb) << "the spgf has " << spgf.extent(2)
                                   << " columns, i.e. the interaction flavours are being emitted as external legs";
  EXPECT_EQ(spgf.extent(0), D.r);
}

/**
 * @brief Check that Nmax covers the largest atom_diag subspace, which sizes the evaluation buffers
 */
TEST(BlockSparseDynintEvaluator, Nmax_covers_the_largest_subspace) {

  auto m    = dynint_model(/*n_int=*/1);
  int n_int = static_cast<int>(m.dynint_ops.size());
  int msd   = max_subspace_dim(m.ad);
  ASSERT_GT(msd, 1) << "vacuous test: every subspace of this fixture is one-dimensional"; // measured 2

  auto G = ad_to_atom_prop(m.ad, beta, Lambda, eps);
  DiagramEvaluator D_ad(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);
  EXPECT_GE(D_ad.Nmax, msd);

  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  DiagramEvaluator D_fq(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  EXPECT_GE(D_fq.Nmax, msd);
}

/**
 * @brief Check that Nmax is derived from the whole quartet and not from symmetry set 0
 *
 * @details The symmetry set order is arbitrary, so set 0 may be a projector-like interaction operator with a single 1x1 block. Only the
 * constructor runs here, since evaluating with an undersized Nmax would write out of bounds.
 */
TEST(BlockSparseDynintEvaluator, Nmax_does_not_assume_symmetry_set_zero_covers_the_space) {

  auto ad       = unequal_sym_set_model(true);
  int n_hyb     = static_cast<int>(ad.get_fops().size());
  auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
  auto hyb      = sym_set_diagonal_hyb(labels_f, p_poles);

  // A projector onto a single Fock state: block diagonal, and its only block is 1x1.
  std::vector<many_body_operator_real> ops = {many_body_operator_real(n("A", 0) * n("A", 1) * n("B", 0))};
  auto d                                   = nda::zeros<dcomplex>(p_poles, 1, 1);
  d(0, 0, 0)                               = 0.61;
  d(1, 0, 0)                               = 0.54;

  auto Fq  = std::get<0>(get_operators_and_interactions(ad, hyb, d, ops));
  auto ext = get_extended_coefficients(hyb, d);

  long s_int = Fq.sym_set_labels(Fq.sym_set_labels.size() - 1); // the projector's own set
  ASSERT_EQ(Fq.sym_set_sizes(s_int), 1);
  auto Fq_perm = permute_set_to_front(Fq, s_int, ext);

  // check that the permuted set 0 misses the largest block
  int msd = max_subspace_dim(ad);
  ASSERT_LT(nda::max_element(Fq_perm.Fs[0].get_block_sizes()), msd)
     << "vacuous test: symmetry set 0 of the permuted quartet already covers the largest subspace";

  nda::vector<double> hyb_poles = {1.3, -0.8};
  DiagramEvaluator D(beta, Lambda, eps, hyb_poles, ext, Fq_perm, /*n_int=*/1);
  EXPECT_GE(D.Nmax, msd) << "Nmax must be the max block dimension over the WHOLE quartet, not over Fs[0]";
}

/**
 * @brief Check that the symmetry set order is a pure relabeling that does not change the self-energy
 */
TEST(BlockSparseDynintEvaluator, symmetry_set_order_does_not_change_sigma) {

  auto ad       = unequal_sym_set_model(true);
  int n_hyb     = static_cast<int>(ad.get_fops().size());
  auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
  auto hyb      = sym_set_diagonal_hyb(labels_f, p_poles);

  auto Fq = std::get<0>(get_operators(ad, hyb));
  ASSERT_GT(nda::max_element(Fq.sym_set_labels), 0) << "vacuous test: this model has only one symmetry set";
  auto Fq_perm = permute_set_to_front(Fq, 1, hyb); // labels [0,0,1] -> [1,1,0]

  nda::vector<double> hyb_poles = {1.3, -0.8};
  nda::array<int, 2> topology   = {{0, 2}, {1, 3}};
  auto itops                    = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt                       = ad_to_atom_prop(ad, beta, itops);

  DiagramEvaluator Da(beta, Lambda, eps, hyb_poles, hyb, Fq);
  DiagramEvaluator Db(beta, Lambda, eps, hyb_poles, hyb, Fq_perm);
  auto Sa = BlockDiagOpFun(Da.compute_self_energy(Gt, topology));
  auto Sb = BlockDiagOpFun(Db.compute_self_energy(Gt, topology));

  double err = 0.0, scale = 0.0;
  for (int b = 0; b < Sa.get_num_block_cols(); ++b) {
    err   = std::max(err, nda::max_element(nda::abs(Sa.get_block(b) - Sb.get_block(b))));
    scale = std::max(scale, nda::max_element(nda::abs(Sa.get_block(b))));
  }
  ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is zero";
  EXPECT_LE(err, 1.0e-14) << "max|Sigma_original - Sigma_permuted| = " << err;
}

// ========== Correlators with dynamical interactions ==========

namespace {

  // tolerance for the comparison with the dense reference, the measured agreement is ~1e-16
  constexpr double corr_tol = 1.0e-13;

  /**
   * @brief Parity of a correlator diagram with a bosonic external operator pair, i.e. of the topology with the pair (0, vct0) dropped
   *
   * @details This is the property that decides whether a bosonic correlator can see a wrong interaction-line statistics. It is exact only up to
   * order 3, where the reduced topology has at most two pairs. At order 2 the reduced topology is a single pair, so the crossing topology
   * {{0,2},{1,3}} is blind here and order 3 is the smallest order that is not.
   */
  int external_bosonic_parity(nda::array_const_view<int, 2> topology) {
    auto is_ferm            = nda::array<bool, 1>(2 * topology.extent(0));
    is_ferm()               = true;
    is_ferm(0)              = false;
    is_ferm(topology(0, 1)) = false;
    return triqs_xca::topology::topology_parity(triqs_xca::topology::fermionic_topology(topology, is_ferm));
  }

} // namespace

/**
 * @brief Compare the single-particle Green's function with a dynamical interaction to the dense evaluator on the crossing order-2 topology
 *
 * @details A backbone whose internal line carries the interaction operator drops only that pair, turning the crossing -1 into +1. All three
 * overloads are compared. The dense spgf emits n_ext external legs, so only the leading n_hyb x n_hyb window is comparable.
 */
TEST(BlockSparseDynintEvaluator, spgf_matches_dense_with_dynamical_interactions) {

  auto m = dynint_model(/*n_int=*/1);
  ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(m.ad, m.ad_flat));

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: a fermionic external pair needs a crossing topology here";

  int n_hyb = static_cast<int>(m.ad.get_fops().size());
  int n_int = static_cast<int>(m.dynint_ops.size());
  ASSERT_GT(n_int, 0) << "vacuous test: no interaction operator";

  auto G_bs = ad_to_atom_prop(m.ad, beta, Lambda, eps);
  auto G_fl = ad_to_atom_prop(m.ad_flat, beta, Lambda, eps);

  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  dense::DiagramEvaluator D_dense(m.hyb_poles, m.hyb_coeffs, G_fl[0].mesh(), m.ad_flat, m.dynint_ops, m.dynint_coeffs);

  // check that the interaction flavour is in the backbone enumeration
  auto Fq_ferm = std::get<0>(get_operators(m.ad, m.hyb_coeffs));
  DiagramEvaluator D_ferm(beta, Lambda, eps, m.hyb_poles, m.hyb_coeffs, Fq_ferm);
  int nb = D.get_num_single_ptcle_gf_backbones(topology);
  ASSERT_GT(nb, D_ferm.get_num_single_ptcle_gf_backbones(topology))
     << "vacuous test: the interaction flavour is not in the backbone enumeration (this compares two evaluators, so it asserts n_ext > n_hyb rather than that any particular backbone puts the interaction on an internal line)";
  ASSERT_EQ(nb, D_dense.get_num_single_ptcle_gf_backbones(topology)) << "the two evaluators do not enumerate the same backbones";

  auto f_ix_vec = nda::array<int, 1>(nb);
  for (int i = 0; i < nb; ++i) f_ix_vec(i) = i;

  auto spgf_dense = D_dense.compute_single_ptcle_gf(G_fl, topology);
  ASSERT_GE(spgf_dense.extent(1), n_hyb);

  // (a) the whole-topology overload
  {
    SCOPED_TRACE("compute_single_ptcle_gf(G, topology)");
    auto [err, scale] = compare_leading_block(D.compute_single_ptcle_gf(G_bs, topology), spgf_dense, n_hyb);
    ASSERT_GT(scale, 0.05) << "vacuous test: the single-particle Green's function is too small for an absolute tolerance";
    EXPECT_LE(err, corr_tol) << "max|dense| = " << scale << ", max|bs - dense| = " << err;
  }
  // (b) the flat-index overload, summed by hand
  {
    SCOPED_TRACE("compute_single_ptcle_gf(G, topology, f_ix)");
    auto acc = nda::zeros<dcomplex>(D.r, n_hyb, n_hyb);
    for (int f = 0; f < nb; ++f) acc += D.compute_single_ptcle_gf(G_bs, topology, f);
    auto [err, scale] = compare_leading_block(acc, spgf_dense, n_hyb);
    EXPECT_LE(err, corr_tol) << "max|dense| = " << scale << ", max|bs - dense| = " << err;
  }
  // (c) the flat-index-vector overload
  {
    SCOPED_TRACE("compute_single_ptcle_gf(G, topology, f_ix_vec)");
    auto [err, scale] = compare_leading_block(D.compute_single_ptcle_gf(G_bs, topology, f_ix_vec), spgf_dense, n_hyb);
    EXPECT_LE(err, corr_tol) << "max|dense| = " << scale << ", max|bs - dense| = " << err;
  }
}

/**
 * @brief Compare the one-time correlator of a density operator with a dynamical interaction to the dense evaluator
 *
 * @details With a bosonic external operator get_parity() first drops the pair (0, vct0), so the diagram carries external_bosonic_parity() rather
 * than topology_parity(). At order 2 that leaves a single pair, so the crossing order-2 topology is blind to the interaction-line statistics
 * and serves as a control, while the maximally crossing order-3 topology is the driver. The density operator commutes with itself, so the
 * statistics classification takes the bosonic branch, which is pinned below.
 */
TEST(BlockSparseDynintEvaluator, one_time_correlator_matches_dense_with_dynamical_interactions) {

  auto m = dynint_model(/*n_int=*/1);
  ASSERT_NO_FATAL_FAILURE(assert_parallel_atom_diags(m.ad, m.ad_flat));

  int n_hyb = static_cast<int>(m.ad.get_fops().size());
  int n_int = static_cast<int>(m.dynint_ops.size());
  int n_ext = n_hyb + n_int;
  ASSERT_EQ(n_int, 1) << "this test reads correlator component (n_hyb, n_hyb)";

  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = ad_to_atom_prop(m.ad, beta, itops);
  auto G_bs  = ad_to_atom_prop(m.ad, beta, Lambda, eps);
  auto G_fl  = ad_to_atom_prop(m.ad_flat, beta, Lambda, eps);

  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  dense::DiagramEvaluator D_dense(m.hyb_poles, m.hyb_coeffs, G_fl[0].mesh(), m.ad_flat, m.dynint_ops, m.dynint_coeffs);

  auto [mu_ops, kap_ops] = make_correlator_ops(Fq, n_ext);
  ASSERT_EQ(static_cast<int>(mu_ops.size()), n_ext);

  auto Fq_ferm = std::get<0>(get_operators(m.ad, m.hyb_coeffs));
  DiagramEvaluator D_ferm(beta, Lambda, eps, m.hyb_poles, m.hyb_coeffs, Fq_ferm);

  // the maximally crossing order-3 topology is the driver, the crossing order-2 one the parity-blind control
  int n_red_drivers = 0;
  for (auto topology : {nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}, nda::array<int, 2>{{0, 2}, {1, 3}}}) {

    int two_m       = 2 * topology.extent(0);
    bool can_see_it = (external_bosonic_parity(topology) == -1);
    if (can_see_it) ++n_red_drivers;
    SCOPED_TRACE("topology " + [&] {
      std::ostringstream o;
      o << topology;
      return o.str();
    }() + (can_see_it ? " (red driver)" : " (parity-blind control)"));

    // vertex 0 pairing with vertex 2m-1 is covered by SparsityInvariance.correlator_keeps_the_beta_tau_side_when_vertex_zero_pairs_with_the_last_vertex
    // on a purely fermionic model, so keep this test off that case
    ASSERT_NE(topology(0, 1), two_m - 1) << "covered by the sparsity-invariance correlator test, not here";

    int nb = D.get_num_single_ptcle_gf_backbones(topology);
    ASSERT_EQ(nb, D_dense.get_num_single_ptcle_gf_backbones(topology)) << "the two evaluators do not enumerate the same backbones";
    ASSERT_GT(nb, D_ferm.get_num_single_ptcle_gf_backbones(topology))
       << "vacuous test: the interaction flavour is not in the backbone enumeration (this compares two evaluators, so it asserts n_ext > n_hyb rather than that any particular backbone puts the interaction on an internal line)";
    auto f_ix_vec = nda::array<int, 1>(nb);
    for (int i = 0; i < nb; ++i) f_ix_vec(i) = i;

    // dense reference on the single-subspace twin, the one-time correlator is a trace and needs no basis bridge
    auto ref = nda::array<dcomplex, 1>(
       D_dense.compute_one_time_correlator(G_fl, m.dynint_ops, m.dynint_ops, m.ad_flat, topology, f_ix_vec)(nda::range::all, 0, 0));
    double scale = nda::max_element(nda::abs(ref));
    ASSERT_GT(scale, 0.01) << "vacuous test: the correlator is too small for an absolute tolerance";

    // (1) the routine under test
    auto bs =
       nda::array<dcomplex, 1>(D.compute_one_time_correlator(G_bs, m.dynint_ops, m.dynint_ops, m.ad, topology, f_ix_vec)(nda::range::all, 0, 0));
    double err_bs = nda::max_element(nda::abs(nda::make_regular(bs - ref)));

    // (2) the same quantity through eval_correlator with a CorrelatorBackbone built here, component (n_hyb, n_hyb) is the interaction
    //     operator against its own dagger
    CorrelatorBackbone bb(topology, n_ext, n_int);
    auto direct       = nda::array<dcomplex, 1>(D.eval_correlator(Gt, bb, mu_ops, kap_ops, /*is_fermionic=*/false)(nda::range::all, n_hyb, n_hyb));
    double err_direct = nda::max_element(nda::abs(nda::make_regular(direct - ref)));

    // (3) check that the bosonic branch differs from the fermionic one
    CorrelatorBackbone bb_f(topology, n_ext, n_int);
    auto direct_ferm = nda::array<dcomplex, 1>(D.eval_correlator(Gt, bb_f, mu_ops, kap_ops, /*is_fermionic=*/true)(nda::range::all, n_hyb, n_hyb));
    if (can_see_it)
      ASSERT_GT(nda::max_element(nda::abs(nda::make_regular(direct_ferm - direct))), 0.01)
         << "vacuous test: the external statistics flag does not change this correlator";

    EXPECT_LE(err_direct, corr_tol) << "eval_correlator with a correctly built CorrelatorBackbone disagrees with dense by " << err_direct
                                    << " (max|dense| = " << scale << ") -- the engine is at fault";
    if (can_see_it) {
      EXPECT_LE(err_bs, corr_tol) << "max|dense| = " << scale << ", max|bs - dense| = " << err_bs
                                  << " -- compute_one_time_correlator builds its CorrelatorBackbone without n_int, so the interaction line "
                                     "is given fermionic parity";
    } else {
      // the parity-blind control
      EXPECT_LE(err_bs, corr_tol) << "max|bs - dense| = " << err_bs;
      EXPECT_LE(nda::max_element(nda::abs(nda::make_regular(direct - bs))), corr_tol) << "the two block-sparse correlator routes disagree";
    }
  }

  // check that at least one topology can see the interaction-line statistics
  ASSERT_GT(n_red_drivers, 0) << "vacuous test: none of the topologies can see a wrong interaction-line statistics. "
                                 "A bosonic external operator needs external_bosonic_parity(topology) == -1, which no "
                                 "order-2 topology has - the reduced matching is a single pair and a single pair is even.";
}

/**
 * @brief Check that the three compute_self_energy() overloads agree with dynamical interactions, which fails on partial n_int plumbing
 */
TEST(BlockSparseDynintEvaluator, self_energy_flat_index_overloads_agree) {

  auto m    = dynint_model(/*n_int=*/1);
  int n_int = static_cast<int>(m.dynint_ops.size());
  ASSERT_GT(n_int, 0) << "vacuous test: no interaction operator";

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  auto G_bs = ad_to_atom_prop(m.ad, beta, Lambda, eps);
  auto Fq   = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext  = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);

  auto Fq_ferm = std::get<0>(get_operators(m.ad, m.hyb_coeffs));
  DiagramEvaluator D_ferm(beta, Lambda, eps, m.hyb_poles, m.hyb_coeffs, Fq_ferm);
  int nb = D.get_num_self_energy_backbones(topology);
  ASSERT_GT(nb, D_ferm.get_num_self_energy_backbones(topology))
     << "vacuous test: the interaction flavour is not in the backbone enumeration (this compares two evaluators, so it asserts n_ext > n_hyb rather than that any particular backbone puts the interaction on an internal line)";

  auto f_ix_vec = nda::array<int, 1>(nb);
  for (int i = 0; i < nb; ++i) f_ix_vec(i) = i;

  auto S_all = BlockDiagOpFun(D.compute_self_energy(G_bs, topology));
  auto S_vec = BlockDiagOpFun(D.compute_self_energy(G_bs, topology, f_ix_vec)); // the solver's entry point
  auto S_one = BlockDiagOpFun(D.compute_self_energy(G_bs, topology, 0));
  for (int f = 1; f < nb; ++f) {
    auto S_f = BlockDiagOpFun(D.compute_self_energy(G_bs, topology, f));
    for (int b = 0; b < S_one.get_num_block_cols(); ++b) S_one.set_block(b, nda::make_regular(S_one.get_block(b) + S_f.get_block(b)));
  }

  double scale = 0.0, e_vec = 0.0, e_one = 0.0;
  for (int b = 0; b < S_all.get_num_block_cols(); ++b) {
    scale = std::max(scale, nda::max_element(nda::abs(S_all.get_block(b))));
    e_vec = std::max(e_vec, nda::max_element(nda::abs(S_all.get_block(b) - S_vec.get_block(b))));
    e_one = std::max(e_one, nda::max_element(nda::abs(S_all.get_block(b) - S_one.get_block(b))));
  }
  ASSERT_GT(scale, 0.1) << "vacuous test: the self-energy is too small for an absolute tolerance";
  EXPECT_LE(e_vec, sigma_tol) << "the f_ix_vec overload disagrees with the whole-topology one by " << e_vec;
  EXPECT_LE(e_one, sigma_tol) << "the single-f_ix overload disagrees with the whole-topology one by " << e_one;
}

/**
 * @brief Check that the three compute_single_ptcle_gf() overloads agree with dynamical interactions, which fails on partial n_int plumbing
 *
 * @details The bound of the per-component loop is n_hyb, since the block-sparse spgf has shape (r, n_hyb, n_hyb).
 */
TEST(BlockSparseDynintEvaluator, spgf_flat_index_overloads_agree) {

  auto m    = dynint_model(/*n_int=*/1);
  int n_hyb = static_cast<int>(m.ad.get_fops().size());
  int n_int = static_cast<int>(m.dynint_ops.size());

  // only a crossing topology separates the two parities
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  auto G   = ad_to_atom_prop(m.ad, beta, Lambda, eps);
  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  ASSERT_EQ(D.n_hyb, n_hyb);
  ASSERT_EQ(D.n_int, n_int);

  // check that the interaction flavour is in the backbone enumeration
  auto Fq_ferm = std::get<0>(get_operators(m.ad, m.hyb_coeffs));
  DiagramEvaluator D_ferm(beta, Lambda, eps, m.hyb_poles, m.hyb_coeffs, Fq_ferm);
  int n_backbones = D.get_num_single_ptcle_gf_backbones(topology);
  ASSERT_GT(n_backbones, D_ferm.get_num_single_ptcle_gf_backbones(topology))
     << "vacuous test: the interaction flavour is not in the backbone enumeration (this compares two evaluators, so it asserts n_ext > n_hyb rather than that any particular backbone puts the interaction on an internal line)";

  // Route 1: sums every backbone internally
  auto spgf_all = D.compute_single_ptcle_gf(G, topology);
  ASSERT_EQ(spgf_all.extent(1), n_hyb) << "the per-component loop below assumes the (r, n_hyb, n_hyb) shape";
  ASSERT_EQ(spgf_all.extent(2), n_hyb);

  // Route 2: the vector overload, driven by the solver for the MPI split
  nda::vector<int> f_ix_vec(n_backbones);
  for (int f_ix = 0; f_ix < n_backbones; ++f_ix) f_ix_vec(f_ix) = f_ix;
  auto spgf_vec = D.compute_single_ptcle_gf(G, topology, f_ix_vec);

  // Route 3: accumulating the single flat index overload
  auto spgf_single = nda::make_regular(0 * spgf_all);
  for (int f_ix = 0; f_ix < n_backbones; ++f_ix) spgf_single += D.compute_single_ptcle_gf(G, topology, f_ix);

  // the comparisons are absolute, so require a non-negligible Green's function
  double scale = nda::max_element(nda::abs(spgf_all));
  ASSERT_GT(scale, 0.01) << "vacuous test: the single-particle Green's function is zero";

  // On failure, report which components disagree.
  for (int mu = 0; mu < n_hyb; ++mu) {
    for (int kap = 0; kap < n_hyb; ++kap) {
      double e_vec = nda::max_element(nda::abs(spgf_all(_, mu, kap) - spgf_vec(_, mu, kap)));
      double e_one = nda::max_element(nda::abs(spgf_all(_, mu, kap) - spgf_single(_, mu, kap)));
      if (std::max(e_vec, e_one) > 1.0e-12)
        std::cout << "component (" << mu << ", " << kap << "): max|all - vec| = " << e_vec << ", max|all - single| = " << e_one
                  << ", max|all| = " << nda::max_element(nda::abs(spgf_all(_, mu, kap))) << "\n";
    }
  }

  EXPECT_LE(nda::max_element(nda::abs(spgf_all - spgf_vec)), 1.0e-12)
     << "max|spgf| = " << scale
     << ": compute_single_ptcle_gf(G, topology, f_ix_vec) disagrees with compute_single_ptcle_gf(G, topology). n_int is reaching the "
        "CorrelatorBackbone in some of the overloads and not the others.";

  EXPECT_LE(nda::max_element(nda::abs(spgf_all - spgf_single)), 1.0e-12)
     << "max|spgf| = " << scale << ": compute_single_ptcle_gf(G, topology, f_ix) disagrees with compute_single_ptcle_gf(G, topology). Same cause.";
}

/**
 * @brief Compare Sigma and the spgf to the dense evaluator on the interleaved spin-flip model with a dimension-4 subspace
 *
 * @details The other comparisons in this file run on unequal_sym_set_model, with a largest subspace of dimension 2 and contiguous labels.
 * The spin-flip model with autopartitioning has 9 subspaces including one of dimension 4, interleaved symmetry-set labels {0,1,0,1}, and a
 * dynint symmetry set of size 2, since n_up0 and n_do0 share a connection row.
 */
TEST(BlockSparseDynintEvaluator, matches_dense_on_the_interleaved_deep_subspace_model) {
  auto ad      = spin_flip_atom_diag_helper(2, false);
  auto ad_flat = spin_flip_atom_diag_helper_single_subspace(2);
  assert_parallel_atom_diags(ad, ad_flat);

  int n_hyb     = static_cast<int>(ad.get_fops().size());
  auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
  auto hyb      = sym_set_diagonal_hyb(labels_f, p_poles);

  // two density operators on the same orbital share a connection row, hence one dynint set of size 2
  std::vector<many_body_operator_real> ops{n("up", 0), n("do", 0)};
  int n_int = static_cast<int>(ops.size());
  auto d    = nda::zeros<dcomplex>(p_poles, n_int, n_int);
  for (int l = 0; l < p_poles; ++l)
    for (int i = 0; i < n_int; ++i) d(l, i, i) = 0.53 - 0.06 * l + 0.04 * i;

  nda::vector<double> hyb_poles = {1.3, -0.8};
  auto [Fq, labels]             = get_operators_and_interactions(ad, hyb, d, ops);
  auto ext                      = get_extended_coefficients(hyb, d);

  ASSERT_EQ(nda::max_element(Fq.sym_set_sizes), 2) << "every symmetry set is a singleton: the within-set contraction is not exercised";
  ASSERT_EQ(Fq.sym_set_sizes(nda::max_element(Fq.sym_set_labels)), 2)
     << "the dynint operators did not land in one shared set of size 2, so this test duplicates dynint_model. "
        "sym_set_sizes = "
     << Fq.sym_set_sizes;
  {
    auto lab = nda::vector<long>(labels_f);
    std::sort(lab.begin(), lab.end());
    bool interleaved = !std::equal(lab.begin(), lab.end(), labels_f.begin());
    ASSERT_TRUE(interleaved) << "vacuous test: the fermionic labels " << labels_f
                             << " are already sorted, so set-major and "
                                "orbital order coincide and an ordering bug would be invisible";
  }

  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = ad_to_atom_prop(ad, beta, itops);
  DiagramEvaluator D(beta, Lambda, eps, hyb_poles, ext, Fq, n_int);

  auto G_flat = ad_to_atom_prop(ad_flat, beta, Lambda, eps);
  dense::DiagramEvaluator D_dense(hyb_poles, hyb, G_flat[0].mesh(), ad_flat, ops, d);

  int max_dim = 0;
  for (auto dim : ad.get_subspace_dims()) max_dim = std::max(max_dim, static_cast<int>(dim));
  ASSERT_GE(max_dim, 4) << "vacuous test: this fixture is supposed to be the deep-subspace one, got max dim " << max_dim;
  ASSERT_EQ(D.Nmax, max_dim);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  // Sigma, per block, through the basis bridge.
  {
    SCOPED_TRACE("self-energy");
    auto Sigma        = BlockDiagOpFun(D.compute_self_energy(Gt, topology));
    auto Sigma_dense  = D_dense.compute_self_energy(G_flat, topology);
    auto [err, scale] = compare_sigma_with_dense(Sigma, Sigma_dense, ad);
    ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is too small for an absolute tolerance";
    EXPECT_LE(err, sigma_tol * std::max(1.0, scale)) << "max|Sigma_dense| = " << scale << ", max|bs - dense| = " << err;
  }

  // spgf, a trace quantity without bridge, compared on the leading n_hyb window since the dense side emits all n_ext legs
  {
    SCOPED_TRACE("single-particle Green's function");
    auto spgf_dense   = D_dense.compute_single_ptcle_gf(G_flat, topology);
    auto [err, scale] = compare_leading_block(D.compute_single_ptcle_gf(Gt, topology), spgf_dense, n_hyb);
    ASSERT_GT(scale, 1.0e-3) << "vacuous test: the spgf is too small to compare";
    EXPECT_LE(err, sigma_tol * std::max(1.0, scale)) << "max|spgf_dense| = " << scale << ", max|bs - dense| = " << err;
  }
}
