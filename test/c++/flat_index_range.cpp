#include <stdexcept>
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

namespace block_sparse = triqs_xca::block_sparse;
namespace dense        = triqs_xca::dense;
namespace test_utils   = triqs_xca::test_utils;

/**
 * @file flat_index_range.cpp
 *
 * @brief Out-of-range flat indices are rejected by every flat-index overload of both diagram evaluators
 *
 * @details Both evaluators decode a flat index with Backbone::set_flat_index, which throws std::invalid_argument outside [0, number of
 * backbones). The dense self-energy accumulates in a member, so a call that throws part way through a flat-index vector must not leave a
 * partial sum behind for the next call.
 */

namespace {

  // the call throws std::invalid_argument from Backbone::set_flat_index, not from another check
  template <typename F> void expect_flat_index_rejected(F &&call) {
    try {
      call();
      ADD_FAILURE() << "no exception for an out-of-range flat index";
    } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("set_flat_index"), std::string::npos) << e.what(); }
  }

} // namespace

/// @brief Check every flat-index overload of both evaluators, and the dense self-energy after a failed call
TEST(FlatIndexRange, out_of_range_flat_indices_are_rejected) {
  double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  int p = 2;

  auto ad          = test_utils::unequal_sym_set_model(true);
  auto ad_flat     = test_utils::unequal_sym_set_model(false);
  int n_hyb        = static_cast<int>(ad.get_fops().size());
  auto labels      = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
  auto hyb_coeffs  = test_utils::sym_set_diagonal_hyb(labels, p);
  auto hyb_poles   = nda::vector<double>{1.3, -0.8};
  auto G_bs        = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);
  auto G_flat      = test_utils::ad_to_atom_prop(ad_flat, beta, Lambda, eps);
  auto ops         = std::vector<many_body_operator_real>{n("A", 0)};
  auto topology    = nda::array<int, 2>{{0, 2}, {1, 3}};
  auto D_dense     = dense::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat);
  auto D_bs        = block_sparse::DiagramEvaluator(hyb_poles, hyb_coeffs, G_bs[0].mesh(), ad);
  int n_sigma      = D_dense.get_num_self_energy_backbones(topology);
  int n_correlator = D_dense.get_num_single_ptcle_gf_backbones(topology);
  ASSERT_EQ(D_bs.get_num_self_energy_backbones(topology), n_sigma);
  ASSERT_EQ(D_bs.get_num_single_ptcle_gf_backbones(topology), n_correlator);

  // the first n_valid flat indices, and the same followed by bad
  auto first = [](int n_valid) {
    auto v = nda::vector<int>(n_valid);
    for (int i = 0; i < n_valid; ++i) v(i) = i;
    return v;
  };
  auto valid_then = [&](int n_valid, int bad) {
    auto v                 = nda::vector<int>(n_valid + 1);
    v(nda::range(n_valid)) = first(n_valid);
    v(n_valid)             = bad;
    return v;
  };
  auto all_sigma = first(n_sigma);

  for (int bad : {-1, n_sigma}) {
    SCOPED_TRACE("self-energy, f_ix = " + std::to_string(bad));
    expect_flat_index_rejected([&] { D_dense.compute_self_energy(G_flat, topology, bad); });
    expect_flat_index_rejected([&] { D_dense.compute_self_energy(G_flat, topology, valid_then(n_sigma, bad)); });
    expect_flat_index_rejected([&] { D_dense.compute_self_energy_by_pairs(G_flat, topology, bad); });
    expect_flat_index_rejected([&] { D_dense.compute_self_energy_by_pairs(G_flat, topology, valid_then(n_sigma, bad)); });
    expect_flat_index_rejected([&] { D_bs.compute_self_energy(G_bs, topology, bad); });
    expect_flat_index_rejected([&] { D_bs.compute_self_energy(G_bs, topology, valid_then(n_sigma, bad)); });
  }
  for (int bad : {-1, n_correlator}) {
    SCOPED_TRACE("correlators, f_ix = " + std::to_string(bad));
    expect_flat_index_rejected([&] { D_dense.compute_single_ptcle_gf(G_flat, topology, bad); });
    expect_flat_index_rejected([&] { D_dense.compute_single_ptcle_gf(G_flat, topology, valid_then(n_correlator, bad)); });
    expect_flat_index_rejected([&] { D_dense.compute_one_time_correlator(G_flat, ops, ops, ad_flat, topology, valid_then(n_correlator, bad)); });
    expect_flat_index_rejected([&] { D_bs.compute_single_ptcle_gf(G_bs, topology, bad); });
    expect_flat_index_rejected([&] { D_bs.compute_single_ptcle_gf(G_bs, topology, valid_then(n_correlator, bad)); });
    expect_flat_index_rejected([&] { D_bs.compute_one_time_correlator(G_bs, ops, ops, ad, topology, valid_then(n_correlator, bad)); });
  }

  // each dense self-energy overload, called after a call that failed with a partial sum in Sigma, agrees with a fresh evaluator
  auto fresh = [&] { return dense::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat); };
  auto fail  = [&] { EXPECT_THROW(D_dense.compute_self_energy(G_flat, topology, valid_then(n_sigma, n_sigma)), std::invalid_argument); };
  auto check = [&](std::string const &what, auto const &Sigma, auto const &Sigma_fresh) {
    SCOPED_TRACE(what + " after a failed call");
    ASSERT_GT(nda::max_element(nda::abs(Sigma_fresh[0].data())), 0.01);
    EXPECT_EQ(nda::max_element(nda::abs(Sigma[0].data() - Sigma_fresh[0].data())), 0.0);
  };
  int f_nonzero = 0;
  while (nda::max_element(nda::abs(fresh().compute_self_energy(G_flat, topology, f_nonzero)[0].data())) < 0.01) ++f_nonzero;

  fail();
  check("compute_self_energy(G, topology)", D_dense.compute_self_energy(G_flat, topology), fresh().compute_self_energy(G_flat, topology));
  fail();
  check("compute_self_energy(G, topology, f_ix)", D_dense.compute_self_energy(G_flat, topology, f_nonzero),
        fresh().compute_self_energy(G_flat, topology, f_nonzero));
  fail();
  check("compute_self_energy(G, topology, f_ix_vec)", D_dense.compute_self_energy(G_flat, topology, all_sigma),
        fresh().compute_self_energy(G_flat, topology, all_sigma));
  fail();
  check("compute_self_energy_by_pairs(G, topology)", D_dense.compute_self_energy_by_pairs(G_flat, topology),
        fresh().compute_self_energy_by_pairs(G_flat, topology));
  fail();
  check("compute_self_energy_by_pairs(G, topology, f_ix)", D_dense.compute_self_energy_by_pairs(G_flat, topology, f_nonzero),
        fresh().compute_self_energy_by_pairs(G_flat, topology, f_nonzero));
  fail();
  check("compute_self_energy_by_pairs(G, topology, f_ix_vec)", D_dense.compute_self_energy_by_pairs(G_flat, topology, all_sigma),
        fresh().compute_self_energy_by_pairs(G_flat, topology, all_sigma));
}
