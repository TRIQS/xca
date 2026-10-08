#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/backbone.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/topology.hpp>

#include "test_utils/block_sparse.hpp"

using nda::dcomplex;
using nda::range;

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::many_body_operator_real;
using triqs::operators::n;

using triqs_xca::dense::DiagramEvaluator;
namespace test_utils = triqs_xca::test_utils;

/**
 * @file dense/correlator_statistics.cpp
 *
 * @brief Tests of how compute_one_time_correlator() classifies its operators as fermionic or bosonic
 *
 * @details The classification enters the diagram in one place: is_fermionic picks the external-vertex sentinel, -2 or -3, which
 * Backbone::get_parity() uses to decide whether the external pair is dropped before the permutation parity is taken, a sign on a crossing
 * topology. The rule is the fermion parity of the operators, i.e. whether their monomials have odd or even length, which differs from a
 * commutator test whenever an even-parity operator fails to commute, as for [S+, S-] = 2 Sz != 0. Since eval_correlator() takes is_fermionic
 * as an explicit argument, both classifications can be evaluated and compared to what compute_one_time_correlator() chose, which pins the
 * decision itself without an ED reference, and the two flag values giving different answers is the non-vacuity guard of each test.
 */

namespace {

  constexpr double beta = 2.0, Lambda = 40.0, eps = 1.0e-8;
  constexpr int p_poles = 2;

  /// Two spinful orbitals in a single subspace: the order-2 crossing correlator needs a vertex between the two external operators that can
  /// return the flipped spin, so chi+- vanishes on a single spinful level
  triqs::atom_diag::atom_diag<true> two_orbital_model() {
    triqs::atom_diag::fundamental_operator_set fs;
    fs.insert("up", 0);
    fs.insert("up", 1);
    fs.insert("do", 0);
    fs.insert("do", 1);
    many_body_operator_complex H = -0.35 * (n("up", 0) + n("do", 0) + n("up", 1) + n("do", 1)) + 0.4 * (n("up", 0) - n("do", 0))
       + 0.17 * (n("up", 1) - n("do", 1)) + 1.1 * n("up", 0) * n("do", 0) + 0.6 * n("up", 0) * n("up", 1)
       + 0.23 * (c_dag("up", 0) * c("up", 1) + c_dag("up", 1) * c("up", 0));
    return triqs::atom_diag::atom_diag<true>(H, fs, std::vector<many_body_operator_complex>{});
  }

  many_body_operator_real S_plus() { return c_dag<double>("up", 0) * c<double>("do", 0) + c_dag<double>("up", 1) * c<double>("do", 1); }
  many_body_operator_real S_minus() { return c_dag<double>("do", 0) * c<double>("up", 0) + c_dag<double>("do", 1) * c<double>("up", 1); }

  /// The body of compute_one_time_correlator with the statistics flag forced rather than inferred
  nda::array<dcomplex, 3> correlator_with_flag(DiagramEvaluator &D, triqs::gfs::block_gf<triqs::mesh::dlr_imtime> const &G,
                                               std::vector<many_body_operator_real> const &ops_tau, std::vector<many_body_operator_real> const &ops_0,
                                               triqs::atom_diag::atom_diag<true> const &ad, nda::array_const_view<int, 2> topology,
                                               bool is_fermionic) {
    CorrelatorBackbone backbone(topology, D.n, D.n_int);
    int N                           = D.N;
    auto U                          = ad.get_unitary_matrix(0);
    nda::array<dcomplex, 3> mu_ops  = nda::zeros<dcomplex>(ops_tau.size(), N, N);
    nda::array<dcomplex, 3> kap_ops = nda::zeros<dcomplex>(ops_0.size(), N, N);
    for (auto [i, op] : itertools::enumerate(ops_tau))
      mu_ops(i, range::all, range::all) = U * ad.get_op_mat(op).block_mat[0] * nda::conj(nda::transpose(U));
    for (auto [i, op] : itertools::enumerate(ops_0))
      kap_ops(i, range::all, range::all) = U * ad.get_op_mat(op).block_mat[0] * nda::conj(nda::transpose(U));

    nda::array<dcomplex, 3> corr = nda::zeros<dcomplex>(D.r, mu_ops.extent(0), kap_ops.extent(0));
    long nb                      = D.get_num_single_ptcle_gf_backbones(topology);
    for (long f_ix = 0; f_ix < nb; ++f_ix) corr += D.eval_correlator(G[0].data(), backbone, mu_ops, kap_ops, f_ix, is_fermionic);
    return corr;
  }

  struct Harness {
    triqs::atom_diag::atom_diag<true> ad;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G;
    DiagramEvaluator D;
  };

  Harness make_harness(triqs::atom_diag::atom_diag<true> ad) {
    auto G  = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);
    int nf  = static_cast<int>(ad.get_fops().size());
    auto hc = nda::zeros<dcomplex>(p_poles, nf, nf);
    for (int i = 0; i < nf; ++i) {
      hc(0, i, i) = 0.7 - 0.05 * i;
      hc(1, i, i) = 0.5 - 0.04 * i;
    }
    nda::vector<double> poles = {1.3, -0.8};
    return {ad, G, DiagramEvaluator(poles, hc, G[0].mesh(), ad)};
  }

  double maxabs(nda::array_const_view<dcomplex, 3> a) { return nda::max_element(nda::abs(a)); }

} // namespace

/**
 * @brief Check that S+/S- are classified bosonic, so that chi+- gets the right parity on a crossing topology
 */
TEST(DenseCorrelatorStatistics, spin_flip_operators_are_classified_bosonic) {
  auto h = make_harness(two_orbital_model());

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  std::vector<many_body_operator_real> ops_tau{S_plus()}, ops_0{S_minus()};

  auto as_boson   = correlator_with_flag(h.D, h.G, ops_tau, ops_0, h.ad, topology, false);
  auto as_fermion = correlator_with_flag(h.D, h.G, ops_tau, ops_0, h.ad, topology, true);

  // check that the topology can see the classification, at order 1 the two coincide
  ASSERT_GT(maxabs(nda::make_regular(as_boson - as_fermion)), 1.0e-3)
     << "vacuous test: the two statistics give the same correlator on this topology, so nothing here "
        "constrains the classification";

  nda::vector<long> f_ix_vec(h.D.get_num_single_ptcle_gf_backbones(topology));
  for (long i = 0; i < f_ix_vec.size(); ++i) f_ix_vec(i) = i;
  auto chosen = h.D.compute_one_time_correlator(h.G, ops_tau, ops_0, h.ad, topology, f_ix_vec);

  EXPECT_LE(maxabs(nda::make_regular(chosen - as_boson)), 1.0e-13 * std::max(1.0, maxabs(as_boson)))
     << "compute_one_time_correlator classified (S+, S-) as fermionic, but both have even fermion parity. max|chosen - bosonic| = "
     << maxabs(nda::make_regular(chosen - as_boson)) << ", max|chosen - fermionic| = " << maxabs(nda::make_regular(chosen - as_fermion));
}

/**
 * @brief Check that a density operator is classified bosonic and c / c^dag fermionic, the cases where the parity rule and a commutator test
 * agree
 */
TEST(DenseCorrelatorStatistics, densities_stay_bosonic_and_single_fermions_stay_fermionic) {
  auto h                      = make_harness(two_orbital_model());
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  nda::vector<long> f_ix_vec(h.D.get_num_single_ptcle_gf_backbones(topology));
  for (long i = 0; i < f_ix_vec.size(); ++i) f_ix_vec(i) = i;

  {
    SCOPED_TRACE("density-density, must be bosonic");
    std::vector<many_body_operator_real> ops{n<double>("up", 0)};
    auto as_boson = correlator_with_flag(h.D, h.G, ops, ops, h.ad, topology, false);
    auto chosen   = h.D.compute_one_time_correlator(h.G, ops, ops, h.ad, topology, f_ix_vec);
    ASSERT_GT(maxabs(as_boson), 1.0e-6) << "vacuous: chi_nn is zero on this model";
    EXPECT_LE(maxabs(nda::make_regular(chosen - as_boson)), 1.0e-13 * std::max(1.0, maxabs(as_boson)));
  }
  {
    SCOPED_TRACE("c / c_dag, must be fermionic");
    std::vector<many_body_operator_real> ops_tau{c<double>("up", 0)}, ops_0{c_dag<double>("up", 0)};
    auto as_fermion = correlator_with_flag(h.D, h.G, ops_tau, ops_0, h.ad, topology, true);
    auto chosen     = h.D.compute_one_time_correlator(h.G, ops_tau, ops_0, h.ad, topology, f_ix_vec);
    ASSERT_GT(maxabs(as_fermion), 1.0e-6) << "vacuous: the single-particle correlator is zero on this model";
    EXPECT_LE(maxabs(nda::make_regular(chosen - as_fermion)), 1.0e-13 * std::max(1.0, maxabs(as_fermion)));
  }
}

/**
 * @brief Check that an operator of mixed fermion parity such as c + n is rejected, while the all-odd c + c^dag c^dag c is accepted
 */
TEST(DenseCorrelatorStatistics, mixed_parity_operators_are_rejected) {
  auto h                      = make_harness(two_orbital_model());
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  nda::vector<long> f_ix_vec(h.D.get_num_single_ptcle_gf_backbones(topology));
  for (long i = 0; i < f_ix_vec.size(); ++i) f_ix_vec(i) = i;

  std::vector<many_body_operator_real> mixed{c<double>("up", 0) + n<double>("up", 0)};
  std::vector<many_body_operator_real> plain{c<double>("up", 0)};

  EXPECT_THROW(h.D.compute_one_time_correlator(h.G, mixed, plain, h.ad, topology, f_ix_vec), std::runtime_error)
     << "an operator with both even- and odd-length monomials has no fermion parity and must be rejected";

  // all-odd with several monomials must be accepted
  std::vector<many_body_operator_real> all_odd{c<double>("up", 0) + c_dag<double>("up", 1) * c_dag<double>("do", 1) * c<double>("up", 0)};
  EXPECT_NO_THROW(h.D.compute_one_time_correlator(h.G, all_odd, plain, h.ad, topology, f_ix_vec))
     << "an operator whose monomials are all odd has a well-defined fermion parity and must be accepted";
}
