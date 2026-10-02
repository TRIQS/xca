#include <stdexcept>

#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>
#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/topology.hpp>

using nda::dcomplex;

using cppdlr::_;
using cppdlr::build_dlr_rf;

using triqs::operators::many_body_operator_complex;
using triqs::operators::many_body_operator_real;
using triqs::operators::n;

using triqs_xca::block_sparse::atom_diag::ad_to_atom_prop;
using triqs_xca::dense::DiagramEvaluator;

/**
 * @file test_dense_dynint_parity.cpp
 *
 * @brief Regression tests of the dense diagram evaluator with dynamical interactions
 *
 * @details Statistics enters the diagram evaluation only through the fermionic permutation parity, where Backbone::get_parity() treats a vertex
 * as fermionic iff its orbital index is smaller than n_hyb = n - n_int. Every Backbone and CorrelatorBackbone therefore has to be constructed
 * with n_int, since the default n_int = 0 classifies the interaction lines as fermionic. Only a crossing topology exposes the parity error, so
 * the tests use the order-2 topology {{0,2},{1,3}}, which is odd while all of its sub-matchings are even. The tests of the single-subspace
 * requirement cover both the dense dynamical interaction operator construction and compute_one_time_correlator(), which read the operator
 * matrices from subspace 0 only.
 */

namespace {

  struct Model {
    triqs::atom_diag::atom_diag<true> ad;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_ppsc;
    nda::vector<double> hyb_poles;
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::array<dcomplex, 3> dynint_coeffs;
    std::vector<many_body_operator_real> dynint_ops;
  };

  /**
   * @brief Single spinless level with a retarded interaction D(tau) coupled to the density
   *
   * @details With partition = false the atom_diag has a single subspace, as required by the dense dynamical interaction path. With partition = true
   * the particle number is conserved and the atom_diag has two subspaces of dimension 1.
   */
  Model dynint_model(double beta, double Lambda, double eps, bool partition = false) {

    many_body_operator_complex H = -0.3 * n("0", 0);

    triqs::atom_diag::fundamental_operator_set fop_set;
    fop_set.insert("0", 0);

    many_body_operator_complex Nop;
    Nop = n("0", 0);

    std::vector<many_body_operator_complex> sym_ops = {}; // empty -> one subspace
    if (partition) sym_ops.push_back(Nop);                // conserving N -> two subspaces of dim 1

    auto ad = triqs::atom_diag::atom_diag<true>(H, fop_set, sym_ops);

    auto G_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);

    // Two-pole representation shared by the hybridization and the retarded interaction,
    // as produced by the joint adapol/DLR fit in the solver.
    int p = 2;
    nda::vector<double> hyb_poles(p);
    hyb_poles(0) = 1.3;
    hyb_poles(1) = -0.8;

    auto hyb_coeffs = nda::zeros<dcomplex>(p, 1, 1);
    hyb_coeffs(0, 0, 0) = 0.7;
    hyb_coeffs(1, 0, 0) = 0.5;

    auto dynint_coeffs = nda::zeros<dcomplex>(p, 1, 1);
    dynint_coeffs(0, 0, 0) = 0.9;
    dynint_coeffs(1, 0, 0) = 0.6;

    std::vector<many_body_operator_real> dynint_ops = {n<double>("0", 0)};

    return {.ad            = ad,
            .G_ppsc        = G_ppsc,
            .hyb_poles     = hyb_poles,
            .hyb_coeffs    = hyb_coeffs,
            .dynint_coeffs = dynint_coeffs,
            .dynint_ops    = dynint_ops};
  }

} // namespace

TEST(DenseDynint, spgf_flat_index_overloads_agree) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}}; // second order: one internal line

  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);

  ASSERT_EQ(D.n_hyb, 1);
  ASSERT_EQ(D.n_int, 1);
  ASSERT_EQ(D.n, 2);

  // Reference: sums all backbones internally, with n_int passed to the CorrelatorBackbone.
  auto spgf_all = D.compute_single_ptcle_gf(m.G_ppsc, topology);

  // Guard against a vacuous test: there must be backbones that put the interaction
  // operator on the internal line, i.e. bosonic vertices whose parity is at stake. (Note
  // that the total contribution of those backbones may well cancel - what this test
  // probes is the parity assigned to each of them individually.)
  DiagramEvaluator D_no_dynint(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad);

  int n_backbones          = D.get_num_single_ptcle_gf_backbones(topology);
  int n_backbones_fermonic = D_no_dynint.get_num_single_ptcle_gf_backbones(topology);
  ASSERT_GT(n_backbones, n_backbones_fermonic) << "vacuous test: no backbone carries the interaction operator";

  // Path 1: the vector overload, used by the solver for the MPI distributed evaluation.
  nda::vector<int> f_ix_vec(n_backbones);
  for (int f_ix = 0; f_ix < n_backbones; ++f_ix) f_ix_vec(f_ix) = f_ix;
  auto spgf_vec = D.compute_single_ptcle_gf(m.G_ppsc, topology, f_ix_vec);

  // Path 2: accumulating the single flat index overload.
  auto spgf_single = nda::make_regular(0 * spgf_all);
  for (int f_ix = 0; f_ix < n_backbones; ++f_ix) spgf_single += D.compute_single_ptcle_gf(m.G_ppsc, topology, f_ix);

  // On failure, report which components disagree: the interaction components are the ones
  // at risk, since only they can put a bosonic vertex on an internal line.
  for (int mu = 0; mu < D.n; ++mu) {
    for (int kap = 0; kap < D.n; ++kap) {
      double err = nda::max_element(nda::abs(spgf_all(_, mu, kap) - spgf_vec(_, mu, kap)));
      if (err > 1.0e-12)
        std::cout << "component (" << mu << ", " << kap << "): max|all - vec| = " << err
                  << ", max|all| = " << nda::max_element(nda::abs(spgf_all(_, mu, kap))) << "\n";
    }
  }

  EXPECT_LE(nda::max_element(nda::abs(spgf_all - spgf_vec)), 1.0e-12)
     << "compute_single_ptcle_gf(G, topology, f_ix_vec) disagrees with compute_single_ptcle_gf(G, topology)";

  EXPECT_LE(nda::max_element(nda::abs(spgf_all - spgf_single)), 1.0e-12)
     << "compute_single_ptcle_gf(G, topology, f_ix) disagrees with compute_single_ptcle_gf(G, topology)";
}

/**
 * @brief Check that compute_self_energy_by_pairs() agrees with compute_self_energy() in the presence of dynamical interactions
 *
 * @details The by-pairs routine evaluates both directions of the hybridization line attached to vertex 0 in one pass, and has to construct its
 * Backbone with n_int like the plain routine does.
 */
TEST(DenseDynint, self_energy_by_pairs_agrees) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}}; // crossing: the only order-2 topology that flips

  // only a crossing topology exposes the parity error
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);

  ASSERT_EQ(D.n_hyb, 1);
  ASSERT_EQ(D.n_int, 1);
  ASSERT_EQ(D.n, 2);

  // check that some backbones carry the interaction operator on the internal line
  DiagramEvaluator D_no_dynint(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad);

  int n_backbones           = D.get_num_self_energy_backbones(topology);
  int n_backbones_fermionic = D_no_dynint.get_num_self_energy_backbones(topology);
  ASSERT_GT(n_backbones, n_backbones_fermionic) << "vacuous test: no backbone carries the interaction operator";

  auto sigma_all   = D.compute_self_energy(m.G_ppsc, topology);
  auto sigma_pairs = D.compute_self_energy_by_pairs(m.G_ppsc, topology);

  // the comparison below is absolute, so require a non-negligible self-energy
  double scale = nda::max_element(nda::abs(sigma_all[0].data()));
  ASSERT_GT(scale, 0.1) << "vacuous test: the self-energy is too small for an absolute tolerance";

  double err = nda::max_element(nda::abs(sigma_all[0].data() - sigma_pairs[0].data()));
  if (err > 1.0e-12) std::cout << "max|Sigma| = " << scale << ", max|all - by_pairs| = " << err << "\n";

  EXPECT_LE(err, 1.0e-12) << "compute_self_energy_by_pairs(G, topology) disagrees with compute_self_energy(G, topology)";
}

/**
 * @brief Check that the three compute_self_energy_by_pairs() overloads agree
 *
 * @details The flat-index overloads do not filter the flat indices, so the caller has to pass only the indices with fb(0) = 0, i.e.
 * (f_ix / n_p) % 2 == 0 with n_p = o_ix_max * p_ix_max, as BlockSparseSolver.eval_pseudo_particle_self_energy_order_by_pairs does.
 */
TEST(DenseDynint, self_energy_by_pairs_flat_index_overloads_agree) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);

  // stride n_p of fb_ix in the flat index, from f_ix_max = n_p * fb_ix_max with fb_ix_max = 2^m for the Backbone
  int order    = topology.extent(0);
  int f_ix_max = D.get_num_self_energy_backbones(topology);
  int n_p      = f_ix_max / (1 << order);

  std::vector<int> seeds;
  for (int f_ix = 0; f_ix < f_ix_max; ++f_ix) {
    if ((f_ix / n_p) % 2 == 0) seeds.push_back(f_ix);
  }
  ASSERT_EQ(static_cast<int>(seeds.size()), f_ix_max / 2);

  auto sigma_all = D.compute_self_energy_by_pairs(m.G_ppsc, topology);
  ASSERT_GT(nda::max_element(nda::abs(sigma_all[0].data())), 1.0e-8) << "vacuous test: the self-energy is zero";

  nda::vector<int> f_ix_vec(seeds.size());
  for (int i = 0; i < static_cast<int>(seeds.size()); ++i) f_ix_vec(i) = seeds[i];
  auto sigma_vec = D.compute_self_energy_by_pairs(m.G_ppsc, topology, f_ix_vec);

  auto sigma_single = nda::zeros<dcomplex>(D.r, D.N, D.N);
  for (int f_ix : seeds) sigma_single += D.compute_self_energy_by_pairs(m.G_ppsc, topology, f_ix)[0].data();

  EXPECT_LE(nda::max_element(nda::abs(sigma_all[0].data() - sigma_vec[0].data())), 1.0e-14)
     << "compute_self_energy_by_pairs(G, topology, f_ix_vec) disagrees with compute_self_energy_by_pairs(G, topology)";

  EXPECT_LE(nda::max_element(nda::abs(sigma_all[0].data() - sigma_single)), 1.0e-14)
     << "compute_self_energy_by_pairs(G, topology, f_ix) disagrees with compute_self_energy_by_pairs(G, topology)";
}

/**
 * @brief Check that the dense dynamical interaction constructor rejects an atom_diag with more than one subspace
 *
 * @details The operator matrices are read from subspace 0 only, and with more than one subspace the assignment to the full Hilbert space
 * matrix reads past the end of the source in a build without bounds checks.
 */
TEST(DenseDynint, dynint_constructor_rejects_multi_subspace_atom_diag) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps, /*partition=*/true);

  ASSERT_EQ(m.ad.n_subspaces(), 2);
  ASSERT_EQ(m.ad.get_full_hilbert_space_dim(), 2);

  // match the message, std::invalid_argument is also thrown upstream of the check under test
  try {
    DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);
    (void)D;
    FAIL() << "expected std::invalid_argument for a multi-subspace atom_diag";
  } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("single subspace"), std::string::npos) << e.what(); }
}

/**
 * @brief Check that compute_one_time_correlator() rejects an atom_diag with more than one subspace
 */
TEST(DenseDynint, one_time_correlator_rejects_multi_subspace_atom_diag) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps, /*partition=*/true);

  ASSERT_EQ(m.ad.n_subspaces(), 2);

  // the constructor without dynamical interactions handles several subspaces, the correlator routine does not
  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad);

  nda::array<int, 2> topology              = {{0, 1}};
  std::vector<many_body_operator_real> ops = {n<double>("0", 0)};

  nda::vector<int> f_ix_vec(1);
  f_ix_vec(0) = 0;

  try {
    D.compute_one_time_correlator(m.G_ppsc, ops, ops, m.ad, topology, f_ix_vec);
    FAIL() << "expected std::invalid_argument for a multi-subspace atom_diag";
  } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("single subspace"), std::string::npos) << e.what(); }
}
