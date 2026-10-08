#include <complex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/topology.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense.hpp"

namespace block_sparse = triqs_xca::block_sparse;
namespace dense        = triqs_xca::dense;
namespace test_utils   = triqs_xca::test_utils;

using nda::dcomplex;
using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::n;
using triqs_xca::topology::topology_parity;

/**
 * @file unitary_invariance_all_evals.cpp
 *
 * @brief Single-particle unitary invariance of the diagram evaluators, which turns a real two-fermion model into a complex one
 *
 * @details The model H = sum_ij h_ij c^dag_i c_j + U n_0 n_1 with a real hopping h and a real two-pole hybridization Delta is rotated by a
 * complex unitary V, h' = V^dag h V and Delta' = V^dag Delta V, while n_0 n_1 is invariant under any unitary of two orbitals. The rotated
 * problem is the original one written in the operators d_j = sum_b V_jb c_b. With W the Fock-space unitary with W c_j W^dag = d_j, the
 * pseudo-particle propagator and self-energy of the rotated problem are G' = W G W^dag and Sigma' = W Sigma W^dag, and its
 * single-particle Green's function is g' = V^dag g V.
 *
 * V is not a global phase times a real matrix, so V^T Delta V^* differs from V^dag Delta V and a transposed hybridization index is
 * detected. Both problems are partitioned by particle number, so the block-sparse evaluator sees the 2 x 2 sector N = 1.
 */

namespace {

  double const beta   = 2.0;
  double const Lambda = 20.0 * beta;
  double const eps    = 1.0e-12;

  /// The two-fermion model and its atomic propagator, with one block per particle-number sector and as one full Hilbert-space block
  struct Problem {
    triqs::atom_diag::atom_diag<true> ad;
    nda::array<dcomplex, 3> hyb_coeffs;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_bs;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_dense;
  };

  Problem make_problem(nda::matrix_const_view<dcomplex> h, double U, nda::array<dcomplex, 3> hyb_coeffs) {
    std::vector<std::string> const names{"0", "1"};
    many_body_operator_complex H = U * n("0", 0) * n("1", 0);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) H += h(i, j) * c_dag(names[i], 0) * c(names[j], 0);
    }
    triqs::atom_diag::fundamental_operator_set fops;
    for (auto const &name : names) fops.insert(name, 0);

    auto ad   = triqs::atom_diag::atom_diag<true>(H, fops, std::vector<many_body_operator_complex>{n("0", 0) + n("1", 0)});
    auto G_bs = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);
    std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> full{{G_bs[0].mesh(), test_utils::get_tensor_in_full_hilbert_space(G_bs, ad)}};
    return {ad, std::move(hyb_coeffs), G_bs, {full}};
  }

  /// The Fock-space unitary W with W c^dag_j1 ... c^dag_jk |0> = d^dag_j1 ... d^dag_jk |0>, d^dag_j = sum_b V^*_jb c^dag_b
  nda::matrix<dcomplex> fock_space_unitary(triqs::atom_diag::atom_diag<true> const &ad, nda::matrix_const_view<dcomplex> V) {
    int norb = V.extent(0);
    long dim = ad.get_full_hilbert_space_dim();
    std::vector<nda::matrix<dcomplex>> cdag(norb), ddag(norb);
    for (int j = 0; j < norb; ++j) cdag[j] = dense::atom_diag::get_operator(ad, j, true);
    for (int j = 0; j < norb; ++j) {
      ddag[j] = nda::zeros<dcomplex>(dim, dim);
      for (int b = 0; b < norb; ++b) ddag[j] += std::conj(V(j, b)) * cdag[b];
    }

    nda::matrix<dcomplex> W = nda::zeros<dcomplex>(dim, dim);
    for (long occ = 0; occ < (1L << norb); ++occ) {
      nda::vector<dcomplex> v = nda::zeros<dcomplex>(dim), w = nda::zeros<dcomplex>(dim);
      v(0) = w(0) = 1.0; // Fock state 0 is the vacuum
      for (int j = norb - 1; j >= 0; --j) {
        if (((occ >> j) & 1) == 0) continue;
        v = cdag[j] * v;
        w = ddag[j] * w;
      }
      for (long a = 0; a < dim; ++a) {
        for (long b = 0; b < dim; ++b) W(a, b) += w(a) * std::conj(v(b));
      }
    }
    return W;
  }

  /// A W^dag at every time
  nda::array<dcomplex, 3> conjugate_by(nda::matrix_const_view<dcomplex> W, nda::array_const_view<dcomplex, 3> A) {
    nda::array<dcomplex, 3> out(A.shape());
    for (long t = 0; t < A.extent(0); ++t)
      out(t, nda::range::all, nda::range::all) = W * nda::matrix<dcomplex>{A(t, nda::range::all, nda::range::all)} * nda::dagger(W);
    return out;
  }

  /// The pseudo-particle self-energy over the full Hilbert space and the single-particle Green's function of one evaluator, signs included
  struct Diagrams {
    nda::array<dcomplex, 3> sigma;
    nda::array<dcomplex, 3> spgf;
  };

  Diagrams eval_dense(Problem const &P, nda::vector_const_view<double> poles, nda::array_const_view<int, 2> topology) {
    dense::DiagramEvaluator D(poles, P.hyb_coeffs, P.G_bs[0].mesh(), P.ad);
    auto G = P.G_dense;
    return {nda::make_regular(topology_parity(topology) * D.compute_self_energy(G, topology)[0].data()),
            nda::make_regular(topology_parity(topology) * D.compute_single_ptcle_gf(G, topology))};
  }

  Diagrams eval_block_sparse(Problem const &P, nda::vector_const_view<double> poles, nda::array_const_view<int, 2> topology) {
    block_sparse::DiagramEvaluator D(poles, P.hyb_coeffs, P.G_bs[0].mesh(), P.ad);
    auto G = P.G_bs;
    return {nda::make_regular(topology_parity(topology) * test_utils::get_tensor_in_full_hilbert_space(D.compute_self_energy(G, topology), P.ad)),
            nda::make_regular(topology_parity(topology) * D.compute_single_ptcle_gf(G, topology))};
  }

  double max_abs(auto const &A) { return nda::max_element(nda::abs(A)); }
  double max_abs_imag(auto const &A) { return nda::max_element(nda::abs(nda::imag(A))); }

} // namespace

/**
 * @brief The self-energy and the single-particle Green's function of both evaluators are covariant under a complex rotation of the orbitals
 */
TEST(unitary_invariance, two_fermions_complex_rotation) {
  // real hopping with an orbital off-diagonal, and a real hybridization with off-diagonal coefficients at two poles
  nda::matrix<dcomplex> h            = {{-0.3, 0.25}, {0.25, 0.4}};
  double U                           = 1.0;
  nda::vector<double> poles          = {-1.2, 0.8};
  nda::array<dcomplex, 3> hyb_coeffs = {{{0.5, 0.2}, {0.2, 0.3}}, {{0.4, -0.1}, {-0.1, 0.6}}};

  nda::matrix<dcomplex> V = {{1.0, dcomplex(0.0, 1.0)}, {1.0, dcomplex(0.0, -1.0)}};
  V /= std::sqrt(2.0);
  nda::matrix<dcomplex> V_dag = nda::dagger(V);
  ASSERT_LE(nda::max_element(nda::abs(V * V_dag - nda::eye<dcomplex>(2))), 1e-15);
  // V^dag V^* is proportional to the identity exactly when V is a global phase times a real matrix, which cannot see a transposed index
  nda::matrix<dcomplex> M = V_dag * nda::conj(V);
  ASSERT_GT(nda::max_element(nda::abs(M - M(0, 0) * nda::eye<dcomplex>(2))), 0.1);

  nda::array<dcomplex, 3> hyb_coeffs_rot(hyb_coeffs.shape());
  for (long l = 0; l < hyb_coeffs.extent(0); ++l)
    hyb_coeffs_rot(l, nda::range::all, nda::range::all) = V_dag * nda::matrix<dcomplex>{hyb_coeffs(l, nda::range::all, nda::range::all)} * V;

  auto P     = make_problem(h, U, hyb_coeffs);
  auto P_rot = make_problem(V_dag * h * V, U, hyb_coeffs_rot);
  ASSERT_EQ((P.ad.get_fops()[{"0", 0}]), 0);
  ASSERT_EQ(P.ad.n_subspaces(), 3);

  // the rotated problem is the original one conjugated by W
  auto W = fock_space_unitary(P.ad, V);
  ASSERT_LE(nda::max_element(nda::abs(W * nda::dagger(W) - nda::eye<dcomplex>(W.extent(0)))), 1e-14);
  ASSERT_LE(nda::max_element(nda::abs(W * test_utils::get_full_h_atomic(P.ad) * nda::dagger(W) - test_utils::get_full_h_atomic(P_rot.ad))), 1e-13);
  auto G     = P.G_dense[0].data();
  auto G_rot = P_rot.G_dense[0].data();
  ASSERT_LE(max_abs(conjugate_by(W, G) - G_rot), 1e-13);
  ASSERT_GT(max_abs_imag(G_rot), 1e-2) << "vacuous test: the rotated propagator is real";

  struct Evaluator {
    std::string name;
    Diagrams (*eval)(Problem const &, nda::vector_const_view<double>, nda::array_const_view<int, 2>);
  };
  std::vector<Evaluator> const evaluators{{"dense", eval_dense}, {"block-sparse", eval_block_sparse}};

  std::vector<nda::array<int, 2>> const topologies{{{0, 1}}, {{0, 2}, {1, 3}}, {{0, 3}, {1, 4}, {2, 5}}};

  for (auto const &topology : topologies) {
    std::vector<Diagrams> rotated;
    for (auto const &e : evaluators) {
      SCOPED_TRACE(e.name + ", order " + std::to_string(topology.extent(0)));
      auto d     = e.eval(P, poles, topology);
      auto d_rot = e.eval(P_rot, poles, topology);
      ASSERT_GT(max_abs_imag(d_rot.sigma), 1e-3 * max_abs(d_rot.sigma)) << "vacuous test: the rotated self-energy is real";
      ASSERT_GT(max_abs_imag(d_rot.spgf), 1e-3 * max_abs(d_rot.spgf)) << "vacuous test: the rotated Green's function is real";

      EXPECT_LE(max_abs(conjugate_by(W, d.sigma) - d_rot.sigma), 1e-11 * max_abs(d.sigma));
      EXPECT_LE(max_abs(conjugate_by(V_dag, d.spgf) - d_rot.spgf), 1e-11 * max_abs(d.spgf));
      rotated.push_back(std::move(d_rot));
    }

    // the evaluators also agree with each other on the complex problem
    SCOPED_TRACE("order " + std::to_string(topology.extent(0)));
    EXPECT_LE(max_abs(rotated[0].sigma - rotated[1].sigma), 1e-11 * max_abs(rotated[0].sigma));
    EXPECT_LE(max_abs(rotated[0].spgf - rotated[1].spgf), 1e-11 * max_abs(rotated[0].spgf));
  }
}
