#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense.hpp"

namespace block_sparse = triqs_xca::block_sparse;
namespace dense        = triqs_xca::dense;
namespace test_utils   = triqs_xca::test_utils;

using nda::dcomplex;
using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::many_body_operator_real;
using triqs::operators::n;

/**
 * @file dynint_spin_rotation.cpp
 *
 * @brief A spin-rotation invariant dynamical interaction in the ladder and in the Cartesian spin operators
 *
 * @details The interaction J(tau) S(tau) . S(0) is given once as the operators (S_z, S+, S-) with coefficients J diag(1, 1/2, 1/2), and once
 * as (S_x, S_y, S_z) with J diag(1, 1, 1). A dynamical-interaction line joins O_i^dag and O_j with coefficient d_ij, so both forms give the
 * kernel J (S_z S_z + (S+ S- + S- S+) / 2), and every evaluator must give the same self-energy and single-particle Green's function for
 * them. S_y has imaginary coefficients, so the Cartesian form needs complex operators, also with a real atom_diag.
 *
 * The model is a two-orbital Hubbard dimer with a spin-independent hybridization. The dense evaluator runs on the single-subspace
 * atom_diag, the block-sparse one on the particle-number partition, where all six operators are block diagonal, and the ladder form also on
 * the (N_up, N_do) partition, where S+ and S- move between subspaces.
 */

namespace {

  constexpr double beta = 2.0, Lambda = 20.0, eps = 1.0e-8;
  constexpr int norb = 2;

  std::vector<std::string> const spins{"up", "do"};

  many_body_operator_real hamiltonian(double U = 1.0, double mu = 0.4, double t = 0.3) {
    many_body_operator_real H;
    for (int i = 0; i < norb; ++i) H += U * n("up", i) * n("do", i) - mu * (n("up", i) + n("do", i));
    for (auto const &s : spins) H += t * (c_dag(s, 0) * c(s, 1) + c_dag(s, 1) * c(s, 0));
    return H;
  }

  triqs::atom_diag::fundamental_operator_set make_fops() {
    triqs::atom_diag::fundamental_operator_set fops;
    for (auto const &s : spins) {
      for (int i = 0; i < norb; ++i) fops.insert(s, i);
    }
    return fops;
  }

  many_body_operator_real number(std::string const &s) {
    many_body_operator_real N;
    for (int i = 0; i < norb; ++i) N += n(s, i);
    return N;
  }

  many_body_operator_complex S_plus() {
    many_body_operator_complex op;
    for (int i = 0; i < norb; ++i) op += c_dag("up", i) * c("do", i);
    return op;
  }
  many_body_operator_complex S_minus() { return dagger(S_plus()); }
  many_body_operator_complex S_x() { return 0.5 * (S_plus() + S_minus()); }
  many_body_operator_complex S_y() { return dcomplex(0.0, -0.5) * (S_plus() - S_minus()); }
  many_body_operator_complex S_z() { return 0.5 * many_body_operator_complex{number("up") - number("do")}; }

  // the two poles shared by the hybridization and the interaction, and the interaction weight J at each
  nda::vector<double> const poles = {1.3, -0.8};
  std::vector<double> const J     = {0.6, 0.35};

  /// Spin-diagonal hybridization with an orbital off-diagonal, the same for both spins
  nda::array<dcomplex, 3> hyb_coeffs(triqs::atom_diag::fundamental_operator_set const &fops) {
    nda::array<double, 3> M        = {{{0.5, 0.2}, {0.2, 0.3}}, {{0.4, -0.1}, {-0.1, 0.6}}};
    nda::array<dcomplex, 3> coeffs = nda::zeros<dcomplex>(poles.size(), fops.size(), fops.size());
    for (long l = 0; l < poles.size(); ++l) {
      for (auto const &s : spins) {
        for (int i = 0; i < norb; ++i) {
          for (int j = 0; j < norb; ++j) coeffs(l, fops[{s, i}], fops[{s, j}]) = M(l, i, j);
        }
      }
    }
    return coeffs;
  }

  /// The interaction operators and their coefficients J_l diag(weights)
  struct DynintForm {
    std::string name;
    std::vector<many_body_operator_complex> ops;
    nda::array<dcomplex, 3> coeffs;
  };

  DynintForm make_form(std::string name, std::vector<many_body_operator_complex> ops, std::vector<double> const &weights) {
    nda::array<dcomplex, 3> coeffs = nda::zeros<dcomplex>(poles.size(), ops.size(), ops.size());
    for (long l = 0; l < poles.size(); ++l) {
      for (long i = 0; i < ops.size(); ++i) coeffs(l, i, i) = J[l] * weights[i];
    }
    return {std::move(name), std::move(ops), coeffs};
  }

  DynintForm ladder_form() { return make_form("(S_z, S+, S-)", {S_z(), S_plus(), S_minus()}, {1.0, 0.5, 0.5}); }
  DynintForm cartesian_form(double y_weight = 1.0) { return make_form("(S_x, S_y, S_z)", {S_x(), S_y(), S_z()}, {1.0, y_weight, 1.0}); }

  /// The raw self-energy over the full Hilbert space and single-particle Green's function of one evaluator
  struct Diagrams {
    nda::array<dcomplex, 3> sigma;
    nda::array<dcomplex, 3> spgf;
  };

  /// One evaluator on one partition of the Hilbert space, with a complex or a real atom_diag
  struct EvalSetup {
    std::string name;
    bool block_sparse;
    std::vector<many_body_operator_real> qn;
    bool complex_ad;
    bool cartesian; // whether the partition admits the Cartesian form
  };

  template <bool IsComplex> triqs::atom_diag::atom_diag<IsComplex> make_ad(std::vector<many_body_operator_real> const &qn) {
    using op_t = typename triqs::atom_diag::atom_diag<IsComplex>::many_body_op_t;
    return {op_t{hamiltonian()}, make_fops(), std::vector<op_t>(qn.begin(), qn.end())};
  }

  template <bool IsComplex>
  Diagrams evaluate_with(triqs::atom_diag::atom_diag<IsComplex> const &ad, triqs::atom_diag::atom_diag<true> const &ad_c, EvalSetup const &setup,
                         DynintForm const &form, nda::array_const_view<int, 2> topology, bool with_dynint) {
    // the propagator in the Fock basis does not depend on the scalar type of the atom_diag, only on its subspaces
    auto G      = test_utils::ad_to_atom_prop(ad_c, beta, Lambda, eps);
    auto coeffs = hyb_coeffs(ad.get_fops());
    if (setup.block_sparse) {
      auto D = with_dynint ? block_sparse::DiagramEvaluator(poles, coeffs, G[0].mesh(), ad, form.ops, form.coeffs) :
                             block_sparse::DiagramEvaluator(poles, coeffs, G[0].mesh(), ad);
      return {test_utils::get_tensor_in_full_hilbert_space(D.compute_self_energy(G, topology), ad_c), D.compute_single_ptcle_gf(G, topology)};
    }
    auto D = with_dynint ? dense::DiagramEvaluator(poles, coeffs, G[0].mesh(), ad, form.ops, form.coeffs) :
                           dense::DiagramEvaluator(poles, coeffs, G[0].mesh(), ad);
    return {D.compute_self_energy(G, topology)[0].data(), D.compute_single_ptcle_gf(G, topology)};
  }

  Diagrams evaluate(EvalSetup const &setup, DynintForm const &form, nda::array_const_view<int, 2> topology, bool with_dynint = true) {
    auto ad_c = make_ad<true>(setup.qn);
    if (setup.complex_ad) return evaluate_with(ad_c, ad_c, setup, form, topology, with_dynint);

    auto ad_r = make_ad<false>(setup.qn);
    bool same = ad_r.n_subspaces() == ad_c.n_subspaces();
    for (int s = 0; same && s < ad_c.n_subspaces(); ++s) same = ad_r.get_fock_states(s) == ad_c.get_fock_states(s);
    if (!same) throw std::runtime_error("the real and complex atom_diag have different subspaces");
    return evaluate_with(ad_r, ad_c, setup, form, topology, with_dynint);
  }

  double max_abs(auto const &A) { return nda::max_element(nda::abs(A)); }

} // namespace

/**
 * @brief The ladder and the Cartesian form of J(tau) S(tau) . S(0) give the same self-energy and single-particle Green's function, for both
 * evaluators, both partitions and both scalar types of the atom_diag
 */
TEST(dynint_spin_rotation, ladder_and_cartesian_forms_agree) {
  std::vector<EvalSetup> setups;
  for (bool complex_ad : {true, false}) {
    std::string scalar = complex_ad ? ", complex atom_diag" : ", real atom_diag";
    setups.push_back({"dense" + scalar, false, {}, complex_ad, true});
    setups.push_back({"block-sparse, N" + scalar, true, {number("up") + number("do")}, complex_ad, true});
    setups.push_back({"block-sparse, (N_up, N_do)" + scalar, true, {number("up"), number("do")}, complex_ad, false});
  }

  std::vector<nda::array<int, 2>> const topologies{{{0, 1}}, {{0, 2}, {1, 3}}};

  for (auto const &topology : topologies) {
    int order = topology.extent(0);
    SCOPED_TRACE("order " + std::to_string(order));

    // the dense ladder form is the reference
    auto ref           = evaluate(setups[0], ladder_form(), topology);
    double sigma_scale = max_abs(ref.sigma), spgf_scale = max_abs(ref.spgf);
    ASSERT_GT(sigma_scale, 1e-3);
    ASSERT_GT(spgf_scale, 1e-3);

    // the interaction enters, and so does S_y; the Green's function has no internal line at first order
    auto no_dynint = evaluate(setups[0], ladder_form(), topology, false);
    auto no_S_y    = evaluate(setups[0], cartesian_form(0.0), topology);
    ASSERT_GT(max_abs(ref.sigma - no_dynint.sigma), 1e-3 * sigma_scale) << "vacuous test: the interaction does not change the self-energy";
    ASSERT_GT(max_abs(ref.sigma - no_S_y.sigma), 1e-3 * sigma_scale) << "vacuous test: S_y does not change the self-energy";
    if (order > 1) {
      ASSERT_GT(max_abs(ref.spgf - no_dynint.spgf), 1e-3 * spgf_scale) << "vacuous test: the interaction does not change the Green's function";
      ASSERT_GT(max_abs(ref.spgf - no_S_y.spgf), 1e-3 * spgf_scale) << "vacuous test: S_y does not change the Green's function";
    }

    for (auto const &setup : setups) {
      std::vector<DynintForm> forms{ladder_form()};
      if (setup.cartesian) forms.push_back(cartesian_form());
      for (auto const &form : forms) {
        SCOPED_TRACE(setup.name + ", " + form.name);
        auto d = evaluate(setup, form, topology);
        EXPECT_LE(max_abs(d.sigma - ref.sigma), 1e-12 * sigma_scale);
        EXPECT_LE(max_abs(d.spgf - ref.spgf), 1e-12 * spgf_scale);
      }
    }
  }
}
