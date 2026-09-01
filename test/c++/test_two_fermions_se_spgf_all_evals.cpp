#include <gtest/gtest.h>

#include <nda/basic_functions.hpp>
#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/hyb.hpp>
#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/block_sparse_manual_gf.hpp>
#include <triqs_xca/topology.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::dense::DenseDiagramEvaluator;

using triqs_xca::block_sparse::DiagramEvaluator;
using triqs_xca::block_sparse::NCA_dense;
using triqs_xca::block_sparse::NCA_gf_dense;
using triqs_xca::block_sparse::OCA_dense;
using triqs_xca::block_sparse::OCA_gf_dense;

using triqs_xca::topology::topology_parity;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;
using triqs_xca::atom_diag::get_tensor_in_full_hilbert_space;

/**
 * @file test_two_fermions_se_spgf_all_evals.cpp
 *
 * @brief Tests of self-energy and single-particle Green's function diagram evaluators for models with two spinless fermions
 *
 * @details This is the two-fermion analogue of test_one_fermion_se_spgf_all_evals.cpp. Each test covers a different hybridization:
 * 
 * - a constant, diagonal hybridization, 
 * - a constant Hermitian hybridization with off-diagonal entries, and 
 * - a hybridization arising from a single pole \omega.

 * The tests compare the self-energy/single-particle Green's function computed analytically at first- (NCA), second- (OCA), and third order against 
 * several evaluation routines. For NCA and OCA the comparisons are to
 *
 * - the `compute_self_energy/single_particle_gf` routine of DenseDiagramEvaluator
 * - the `compute_self_energy_by_pairs` routine of DenseDiagramEvaluator, which has no single-particle Green's function analogue
 *   - This routine is a time optimization of the first compute_self_energy routine above and is the one actually used in Python wrappers
 * - the `compute_self_energy/single_particle_gf` routine of the DiagramEvaluator, which takes advantage of block-sparsity
 * - `N/OCA_(gf_)dense`, a routine that can only evaluate N/OCA with dense matmuls
 * - `N/OCA_(gf_)bs`, a routine that can only evaluate N/OCA taking advantage of block-sparsity
 *
 * The last two routines are not wrapped and exist just for testing purposes; at third order they have no
 * analogue, so only the two diagram evaluators are compared there.
 *
 * For third-order diagram evaluation, there are comparisons to all of the above routines except, obviously, the last two routines above, and no 
 * third-order analogues exist. 
 *
 * All three models take U = mu = 0, so the pseudo-particle propagator is G(tau) = -4^{-tau/beta} I_4 in every
 * one of them and the atom_diag object splits into occupation sectors N = 0, 1, 2 of dimensions 1, 2, 1. The
 * closed forms are derived in examples/two_fermion_analytical_solutions.ipynb. Every test differs only in its
 * hybridization and in its analytic reference values; the model setup and the evaluator-vs-evaluator
 * comparisons are shared through the TwoFermionSetup bundle and the check_* helpers below.
 */

namespace {

  // ---------------------------------------------------------------------------------------------------
  // Model construction
  // ---------------------------------------------------------------------------------------------------

  /**
   * @brief Everything the two-fermion tests need: the DLR grid, the model, and all the evaluators
   *
   * @details The dense objects sum the same backbones as the block-sparse ones but over the full Hilbert
   * space rather than block by block, so the two carry the same overall self-energy sign convention.
   *
   * Each test constructs one directly from its own beta, Lambda = 20 beta and eps, so that those values
   * stay in scope for the analytic references and tolerances in the test body.
   *
   * The Hilbert space is four-dimensional and splits into the occupation sectors N = 0, 1, 2 of dimensions
   * 1, 2, 1. Every self-energy here is diagonal in the Fock basis, with Fock state f carrying the entry of
   * sector N = popcount(f) -- so state 0 holds N = 0, states 1 and 2 hold N = 1, and state 3 holds N = 2.
   * The analytic references in the tests are written directly as dense 4x4 matrices in that layout, which is
   * what get_tensor_in_full_hilbert_space produces, so the two can be subtracted whole.
   */
  struct TwoFermionSetup {
    double beta;
    double Lambda;
    double eps;
    imtime_ops itops;
    nda::vector<double> dlr_it; // DLR imaginary time nodes, relative format
    int r;                      // DLR rank
    FermionModelData model;     // hybridization, atom_diag object, and atomic propagator

    nda::array<dcomplex, 3> Gt_dense;      // atomic propagator over the full Hilbert space
    nda::array<dcomplex, 3> Gt_dense_refl; // ... evaluated at (beta - tau)
    nda::array<dcomplex, 3> Fs_dense;
    nda::array<dcomplex, 3> F_dags_dense;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_ppsc_dense;

    DenseDiagramEvaluator D_dense;
    DiagramEvaluator D;
    BlockOpSymQuartet Fq;     // field operators of the block-sparse evaluator
    nda::vector<int> block_N; // occupation sector N of each block, which indexes the per-sector references

    /**
     * @brief Build the DLR grid, the model and every dense fixture from the physical parameters
     *
     * @details U = mu = 0 in all three models, so only the hybridization varies.
     *
     * @param[in] hyb_pole Pole omega of the single-pole hybridization; omega = 0 makes Delta tau-independent
     * @param[in] alpha Off-diagonal orbital amplitude, i.e. M = {{1, alpha}, {alpha, 1}}
     */
    TwoFermionSetup(double beta_, double Lambda_, double eps_, double hyb_pole, double alpha = 0.0)
       : beta(beta_),
         Lambda(Lambda_),
         eps(eps_),
         itops(Lambda_, build_dlr_rf(Lambda_, eps_)),
         dlr_it(itops.get_itnodes()),
         r(itops.rank()),
         model(make_model(beta_, Lambda_, eps_, hyb_pole, alpha)),
         Gt_dense(make_dense_propagator(model.ad, beta_, dlr_it)),
         Gt_dense_refl(itops.reflect(Gt_dense)),
         G_ppsc_dense(wrap_dense(model.G_ppsc[0].mesh(), Gt_dense)),
         D_dense(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         D(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         Fq(make_field_operators(model.ad, model.hyb_coeffs)),
         block_N(make_block_occupations(model.ad)) {
      auto [Fs, F_dags] = get_operators_dense(model.ad);
      Fs_dense          = std::move(Fs);
      F_dags_dense      = std::move(F_dags);
    }

    int nblocks() const { return model.ad.n_subspaces(); }

    private:
    // The U = mu = 0 model, with the identity orbital-space amplitude matrix replaced by the Hermitian
    // M = {{1, alpha}, {alpha, 1}}. The coefficients have to be overwritten before the evaluators read them,
    // hence the tweak here rather than on the setup afterwards.
    static FermionModelData make_model(double beta, double Lambda, double eps, double hyb_pole, double alpha) {
      auto model                = two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0, hyb_pole);
      model.hyb_coeffs(0, 0, 1) = alpha;
      model.hyb_coeffs(0, 1, 0) = alpha;
      return model;
    }

    // The atomic propagator over the full Hilbert space, rebuilt from the Hamiltonian rather than taken from the
    // block-sparse model, so that the dense evaluators are fed through an independent code path.
    static nda::array<dcomplex, 3> make_dense_propagator(triqs::atom_diag::atom_diag<true> const &ad, double beta,
                                                         nda::vector_const_view<double> dlr_it) {
      return Hmat_to_Gtmat(get_full_h_atomic(ad), beta, cppdlr::rel2abs(dlr_it));
    }

    // The field operators of the block-sparse evaluator. get_operators also returns the symmetry set labels,
    // which none of these tests use.
    static BlockOpSymQuartet make_field_operators(triqs::atom_diag::atom_diag<true> const &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {
      return std::get<0>(get_operators(ad, hyb_coeffs));
    }

    // The occupation sector N of each block, read off the particle number of any one of its Fock states.
    static nda::vector<int> make_block_occupations(triqs::atom_diag::atom_diag<true> const &ad) {
      nda::vector<int> block_N(ad.n_subspaces());
      for (int b = 0; b < ad.n_subspaces(); ++b) { block_N(b) = __builtin_popcountl(ad.get_fock_states(b)[0]); }
      return block_N;
    }

    // The dense propagator as a one-block block_gf, which is what the dense evaluator takes.
    static triqs::gfs::block_gf<triqs::mesh::dlr_imtime> wrap_dense(triqs::mesh::dlr_imtime const &mesh, nda::array_const_view<dcomplex, 3> Gt) {
      std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh, Gt)};
      return {blocks};
    }
  };

  // ---------------------------------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------------------------------

  // Sample an analytic reference function on the DLR nodes. The argument passed to f is t = tau / beta.
  nda::array<double, 1> tau_ref(nda::vector_const_view<double> dlr_it, auto f) {
    auto r   = dlr_it.extent(0);
    auto ref = nda::zeros<double>(r);
    for (long i = 0; i < r; ++i) { ref(i) = f(rel2abs(dlr_it(i))); }
    return ref;
  }

  // Maximally-crossing topology at the given order: hybridization line i joins vertices i and i + order,
  // i.e. {{0, 1}} at first order, {{0, 2}, {1, 3}} at second, {{0, 3}, {1, 4}, {2, 5}} at third.
  nda::array<int, 2> max_crossing_topology(int order) {
    auto topology = nda::zeros<int>(order, 2);
    for (int i = 0; i < order; ++i) {
      topology(i, 0) = i;
      topology(i, 1) = i + order;
    }
    return topology;
  }

  // The atom_diag object splits into the occupation sectors N = 0, 1, 2, of dimensions 1, 2, 1 -- so the
  // per-block checks below are not vacuously scalar, and the singly-occupied sector really is 2x2.
  void expect_block_structure(TwoFermionSetup const &s) {
    SCOPED_TRACE("block structure");
    ASSERT_EQ(s.nblocks(), 3);
    nda::vector<int> expected_dims = {1, 2, 1};
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      EXPECT_EQ(static_cast<int>(s.model.ad.get_fock_states(b).size()), expected_dims(b));
      EXPECT_EQ(s.block_N(b), b); // block order happens to coincide with occupation number here
    }
  }

} // namespace

/**
 * @brief Test evaluation of the self-energy for a two-fermion system with a constant hybridization function
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams.
 */
TEST(two_fermions, const_hyb_se) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  // Delta(tau) = K(tau, om) M, which is tau-independent at om = 0; M = I_2 at alpha = 0
  double om    = 0.0;
  double alpha = 0.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  TwoFermionSetup s{beta, Lambda, eps, om, alpha};
  double ln4 = 2 * std::numbers::ln2;

  // Before testing self-energy contributions, check atomic propagator is correct
  auto G0_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto G0_diag = tau_ref(s.dlr_it, [](double t) { return -exp(-t * 2 * std::numbers::ln2); });
  for (int f = 0; f < 4; ++f) { G0_ana(_, f, f) = G0_diag; } // U = mu = 0, so every Fock state has the same weight
  EXPECT_LE(nda::max_element(nda::abs(get_tensor_in_full_hilbert_space(s.model.G_bdof, s.model.ad) - G0_ana)), eps);
  expect_block_structure(s);

  // ----- NCA -----
  auto nca_topology = max_crossing_topology(1); // = {0, 1}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, nca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto nca_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, nca_topology);
  auto nca_dense    = nda::make_regular(topology_parity(nca_topology) * nca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto nca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, nca_topology);
  auto nca_pairs    = nda::make_regular(topology_parity(nca_topology) * nca_pairs_gf[0].data());
  // manual dense routine, and change sign
  auto nca_manual_dense = nda::make_regular(-NCA_dense(s.D.hyb.values, s.D.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense));
  // manual block-sparse routine, convert to dense format, and change sign
  auto nca_manual_bdof = NCA_bs(s.D.hyb.values, s.D.hyb.values_reflect, s.model.G_bdof, s.Fq);
  auto nca_manual_bs   = nda::make_regular(-get_tensor_in_full_hilbert_space(nca_manual_bdof, s.model.ad, s.r));

  // compute analytical reference, = -G0_ana
  auto nca_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto nca_diag = tau_ref(s.dlr_it, [&](double t) { return exp(-t * ln4); });
  for (int f = 0; f < 4; ++f) { nca_ana(_, f, f) = nca_diag; } // Delta is tau-independent: the same entry in every sector

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_pairs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  auto oca_topology = max_crossing_topology(2); // = {{0, 2}, {1, 3}}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, oca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto oca_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, oca_topology);
  auto oca_dense    = nda::make_regular(topology_parity(oca_topology) * oca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto oca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, oca_topology);
  auto oca_pairs    = nda::make_regular(topology_parity(oca_topology) * oca_pairs_gf[0].data());
  // manual dense routine
  auto oca_manual_dense = OCA_dense(s.D.hyb.values, s.D.hyb.coeffs, s.D.hyb.values_reflect, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta,
                                    s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, and convert to dense format
  auto oca_manual_bdof = OCA_bs(s.D.hyb.values, s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  auto oca_manual_bs   = get_tensor_in_full_hilbert_space(oca_manual_bdof, s.model.ad, s.r);

  // compute analytical reference
  auto oca_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto oca_diag = tau_ref(s.dlr_it, [&](double t) { return -0.25 * exp(-t * ln4) * t * t * beta * beta; });
  for (int f = 0; f < 4; ++f) { oca_ana(_, f, f) = oca_diag; } // Delta is tau-independent: the same entry in every sector

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_pairs - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_bs - oca_ana)), eps);

  // ----- third order -----
  auto third_topology = max_crossing_topology(3); // = {{0, 3}, {1, 4}, {2, 5}}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology)
                                    * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, third_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto third_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, third_topology);
  auto third_dense    = nda::make_regular(topology_parity(third_topology) * third_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto third_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, third_topology);
  auto third_pairs    = nda::make_regular(topology_parity(third_topology) * third_pairs_gf[0].data());
  // no manual third-order routine

  // compute analytical reference
  auto third_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto third_diag = tau_ref(s.dlr_it, [&](double t) { return 1.0 / 96 * exp(-t * ln4) * pow(t, 4) * pow(beta, 4); });
  for (int f = 0; f < 4; ++f) { third_ana(_, f, f) = third_diag; } // Delta is tau-independent: the same entry in every sector

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_pairs - third_ana)), eps);
}

/**
 * @brief Test evaluation of the single-particle Green's function for a two-fermion system with a constant hybridization function
 *
 * @details This tests the evaluation of the first-, second-, and third-order diagrams contributing to the single-particle Green's function.
 */
TEST(two_fermions, const_hyb_spgf) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  // Delta(tau) = K(tau, om) M, which is tau-independent at om = 0; M = I_2 at alpha = 0
  double om    = 0.0;
  double alpha = 0.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  TwoFermionSetup s{beta, Lambda, eps, om, alpha};

  // ----- NCA -----
  auto nca_topology = max_crossing_topology(1); // = {0, 1}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, nca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto nca_dense = nda::make_regular(topology_parity(nca_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, nca_topology));
  // manual dense routine
  auto nca_manual_dense = NCA_gf_dense(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto nca_manual_bs = NCA_gf_bs(s.model.G_bdof, s.model.G_bdof.reflect(s.itops), s.Fq);

  // compute analytical reference: first order has no hybridization line, so it does not see Delta at all
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  nca_ana(_, 0, 0) = 0.5;
  nca_ana(_, 1, 1) = 0.5;

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);
  EXPECT_EQ(nca_bs.extent(1), 2); // the two orbitals

  // ----- OCA -----
  auto oca_topology = max_crossing_topology(2); // = {{0, 2}, {1, 3}}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, oca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto oca_dense = nda::make_regular(topology_parity(oca_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, oca_topology));
  // manual dense routine
  auto oca_manual_dense = OCA_gf_dense(s.D.hyb.coeffs, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto oca_manual_bs = OCA_gf_bs(s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);

  // compute analytical reference: g_aa = tau (beta - tau) / 4, with no spin-flip terms in H or Delta so the
  // orbital off-diagonal entries vanish identically
  auto oca_ana  = nda::zeros<dcomplex>(s.r, 2, 2);
  auto oca_diag = tau_ref(s.dlr_it, [&](double t) { return 0.25 * beta * beta * t * (1 - t); });
  for (int o = 0; o < 2; ++o) { oca_ana(_, o, o) = oca_diag; }

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_bs - oca_ana)), eps);

  // ----- third order -----
  auto third_topology = max_crossing_topology(3); // = {{0, 3}, {1, 4}, {2, 5}}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, third_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto third_dense = nda::make_regular(topology_parity(third_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, third_topology));
  // no manual third-order routine

  // compute analytical reference: tau^2 (beta - tau)^2 / 32, the alpha = 0 case of hermitian_hyb_spgf below
  auto third_ana  = nda::zeros<dcomplex>(s.r, 2, 2);
  auto third_diag = tau_ref(s.dlr_it, [&](double t) { return pow(beta, 4) * t * t * (1 - t) * (1 - t) / 32.0; });
  for (int o = 0; o < 2; ++o) { third_ana(_, o, o) = third_diag; }

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
}

/**
 * @brief Test evaluation of the self-energy for a two-fermion system with an off-diagonal (Hermitian) hybridization
 *
 * @details Same model as two_fermions.const_hyb_se except that the orbital-space amplitude matrix is the Hermitian
 * M = {{1, alpha}, {alpha, 1}} instead of I_2. The closed forms are derived in the "Off-diagonal (Hermitian)
 * hybridization" section of examples/two_fermion_analytical_solutions.ipynb. The nonzero off-diagonal entries produce 
 * a nontrivial OCA contribution in contrast to the one-fermion tests.
 */
TEST(two_fermions, hermitian_hyb_se) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  // Delta(tau) = K(tau, om) M, with the Hermitian M = {{1, alpha}, {alpha, 1}}
  double om    = 0.0;
  double alpha = 0.4;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  TwoFermionSetup s{beta, Lambda, eps, om, alpha};
  double ln4 = 2 * std::numbers::ln2;

  // Before testing self-energy contributions, check atomic propagator is correct
  auto G0_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto G0_diag = tau_ref(s.dlr_it, [](double t) { return -exp(-t * 2 * std::numbers::ln2); });
  for (int f = 0; f < 4; ++f) { G0_ana(_, f, f) = G0_diag; } // U = mu = 0, so every Fock state has the same weight
  EXPECT_LE(nda::max_element(nda::abs(get_tensor_in_full_hilbert_space(s.model.G_bdof, s.model.ad) - G0_ana)), eps);
  expect_block_structure(s);

  // ----- NCA -----
  auto nca_topology = max_crossing_topology(1); // = {0, 1}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, nca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto nca_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, nca_topology);
  auto nca_dense    = nda::make_regular(topology_parity(nca_topology) * nca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto nca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, nca_topology);
  auto nca_pairs    = nda::make_regular(topology_parity(nca_topology) * nca_pairs_gf[0].data());
  // manual dense routine, and change sign
  auto nca_manual_dense = nda::make_regular(-NCA_dense(s.D.hyb.values, s.D.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense));
  // manual block-sparse routine, convert to dense format, and change sign
  auto nca_manual_bdof = NCA_bs(s.D.hyb.values, s.D.hyb.values_reflect, s.model.G_bdof, s.Fq);
  auto nca_manual_bs   = nda::make_regular(-get_tensor_in_full_hilbert_space(nca_manual_bdof, s.model.ad, s.r));

  // compute analytical reference, = -G0_ana and independent of alpha
  auto nca_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto nca_diag = tau_ref(s.dlr_it, [&](double t) { return exp(-t * ln4); });
  for (int f = 0; f < 4; ++f) { nca_ana(_, f, f) = nca_diag; } // Delta is tau-independent: the same entry in every sector

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_pairs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  auto oca_topology = max_crossing_topology(2); // = {{0, 2}, {1, 3}}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, oca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto oca_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, oca_topology);
  auto oca_dense    = nda::make_regular(topology_parity(oca_topology) * oca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto oca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, oca_topology);
  auto oca_pairs    = nda::make_regular(topology_parity(oca_topology) * oca_pairs_gf[0].data());
  // manual dense routine
  auto oca_manual_dense = OCA_dense(s.D.hyb.values, s.D.hyb.coeffs, s.D.hyb.values_reflect, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta,
                                    s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, and convert to dense format
  auto oca_manual_bdof = OCA_bs(s.D.hyb.values, s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  auto oca_manual_bs   = get_tensor_in_full_hilbert_space(oca_manual_bdof, s.model.ad, s.r);

  // compute analytical reference; the alpha = 0 value carries a factor -1, and the Hermitian M replaces it by alpha^2 - 1
  auto oca_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto oca_diag = tau_ref(s.dlr_it, [&](double t) {
    double tau = beta * t;
    return 0.25 * (alpha * alpha - 1) * exp(-t * ln4) * tau * tau;
  });
  for (int f = 0; f < 4; ++f) { oca_ana(_, f, f) = oca_diag; } // Delta is tau-independent: the same entry in every sector

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_pairs - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_bs - oca_ana)), eps);

  // ----- third order -----
  auto third_topology = max_crossing_topology(3); // = {{0, 3}, {1, 4}, {2, 5}}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology)
                                    * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, third_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto third_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, third_topology);
  auto third_dense    = nda::make_regular(topology_parity(third_topology) * third_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto third_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, third_topology);
  auto third_pairs    = nda::make_regular(topology_parity(third_topology) * third_pairs_gf[0].data());
  // no manual third-order routine

  // compute analytical reference
  auto third_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto third_diag = tau_ref(s.dlr_it, [&](double t) {
    double tau = beta * t;
    return (3 * alpha * alpha + 1) / 96.0 * exp(-t * ln4) * pow(tau, 4);
  });
  for (int f = 0; f < 4; ++f) { third_ana(_, f, f) = third_diag; }

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_pairs - third_ana)), eps);
}

/**
 * @brief Test evaluation of the single-particle Green's function for a two-fermion system with an off-diagonal
 * (Hermitian) hybridization
 *
 * @details Same model as two_fermions.hermitian_hyb_se: U = mu = 0 and Delta(tau) = -M/2 with the Hermitian
 * M = {{1, alpha}, {alpha, 1}}. Unlike the self-energy, the single-particle Green's function does acquire
 * off-diagonal orbital entries here, linear in alpha; the closed forms are derived in the "Off-diagonal (Hermitian)
 * hybridization" section of examples/two_fermion_analytical_solutions.ipynb. Setting alpha = 0 reproduces the
 * two_fermions.const_hyb_spgf references.
 */
TEST(two_fermions, hermitian_hyb_spgf) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  // Delta(tau) = K(tau, om) M, with the Hermitian M = {{1, alpha}, {alpha, 1}}
  double om    = 0.0;
  double alpha = 0.4;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  TwoFermionSetup s{beta, Lambda, eps, om, alpha};

  // ----- NCA -----
  auto nca_topology = max_crossing_topology(1); // = {0, 1}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, nca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto nca_dense = nda::make_regular(topology_parity(nca_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, nca_topology));
  // manual dense routine
  auto nca_manual_dense = NCA_gf_dense(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto nca_manual_bs = NCA_gf_bs(s.model.G_bdof, s.model.G_bdof.reflect(s.itops), s.Fq);

  // compute analytical reference: first order has no hybridization line, so it does not see M at all
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  nca_ana(_, 0, 0) = 0.5;
  nca_ana(_, 1, 1) = 0.5;

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);
  EXPECT_EQ(nca_bs.extent(1), 2); // the two orbitals

  // ----- OCA -----
  auto oca_topology = max_crossing_topology(2); // = {{0, 2}, {1, 3}}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, oca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto oca_dense = nda::make_regular(topology_parity(oca_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, oca_topology));
  // manual dense routine
  auto oca_manual_dense = OCA_gf_dense(s.D.hyb.coeffs, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto oca_manual_bs = OCA_gf_bs(s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);

  // compute analytical reference: g_aa = tau (beta - tau) / 4, g_ab = -alpha tau (beta - tau) / 4
  auto oca_ana  = nda::zeros<dcomplex>(s.r, 2, 2);
  auto oca_diag = tau_ref(s.dlr_it, [&](double t) { return 0.25 * beta * beta * t * (1 - t); });
  for (int o = 0; o < 2; ++o) {
    oca_ana(_, o, o)     = oca_diag;
    oca_ana(_, o, 1 - o) = -alpha * oca_diag;
  }

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_bs - oca_ana)), eps);

  // ----- third order -----
  auto third_topology = max_crossing_topology(3); // = {{0, 3}, {1, 4}, {2, 5}}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, third_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto third_dense = nda::make_regular(topology_parity(third_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, third_topology));
  // no manual third-order routine

  // compute analytical reference: g_aa = (alpha^2 + 1) tau^2 (beta - tau)^2 / 32, g_ab = alpha tau^2 (beta - tau)^2 / 16
  auto third_ana = nda::zeros<dcomplex>(s.r, 2, 2);
  auto base      = tau_ref(s.dlr_it, [&](double t) { return pow(beta, 4) * t * t * (1 - t) * (1 - t); });
  for (int o = 0; o < 2; ++o) {
    third_ana(_, o, o)     = (alpha * alpha + 1) * base / 32.0;
    third_ana(_, o, 1 - o) = alpha * base / 16.0;
  }

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
}

/**
 * @brief Test evaluation of the self-energy for a two-fermion system with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams. 
 *
 */
TEST(two_fermions, one_hyb_pole_se) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  // Delta(tau) = K(tau, om) M; M = I_2 at alpha = 0
  double om    = -1.5;
  double alpha = 0.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  TwoFermionSetup s{beta, Lambda, eps, om, alpha};
  double ln4 = 2 * std::numbers::ln2;

  // Before testing self-energy contributions, check atomic propagator is correct
  auto G0_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  auto G0_diag = tau_ref(s.dlr_it, [](double t) { return -exp(-t * 2 * std::numbers::ln2); });
  for (int f = 0; f < 4; ++f) { G0_ana(_, f, f) = G0_diag; } // U = mu = 0, so every Fock state has the same weight
  EXPECT_LE(nda::max_element(nda::abs(get_tensor_in_full_hilbert_space(s.model.G_bdof, s.model.ad) - G0_ana)), eps);
  expect_block_structure(s);

  // ----- NCA -----
  auto nca_topology = max_crossing_topology(1); // = {0, 1}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, nca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto nca_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, nca_topology);
  auto nca_dense    = nda::make_regular(topology_parity(nca_topology) * nca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto nca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, nca_topology);
  auto nca_pairs    = nda::make_regular(topology_parity(nca_topology) * nca_pairs_gf[0].data());
  // manual dense routine, and change sign
  auto nca_manual_dense = nda::make_regular(-NCA_dense(s.D.hyb.values, s.D.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense));
  // manual block-sparse routine, convert to dense format, and change sign
  auto nca_manual_bdof = NCA_bs(s.D.hyb.values, s.D.hyb.values_reflect, s.model.G_bdof, s.Fq);
  auto nca_manual_bs   = nda::make_regular(-get_tensor_in_full_hilbert_space(nca_manual_bdof, s.model.ad, s.r));

  // compute analytical reference; Fock states 0, {1, 2} and 3 hold the occupation sectors N = 0, 1 and 2
  auto nca_ana = nda::zeros<dcomplex>(s.r, 4, 4);
  for (int i = 0; i < s.r; ++i) {
    double t         = rel2abs(s.dlr_it(i)); // t = tau / beta
    double tau       = beta * t;
    double g4        = exp(-t * ln4);
    nca_ana(i, 0, 0) = 2 * g4 * exp(om * tau) / (exp(beta * om) + 1);
    nca_ana(i, 3, 3) = 2 * g4 * exp(-om * tau) / (exp(-beta * om) + 1);
    nca_ana(i, 1, 1) = 0.5 * (nca_ana(i, 0, 0) + nca_ana(i, 3, 3));
    nca_ana(i, 2, 2) = nca_ana(i, 1, 1);
  }

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_pairs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  auto oca_topology = max_crossing_topology(2); // = {{0, 2}, {1, 3}}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, oca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto oca_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, oca_topology);
  auto oca_dense    = nda::make_regular(topology_parity(oca_topology) * oca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto oca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, oca_topology);
  auto oca_pairs    = nda::make_regular(topology_parity(oca_topology) * oca_pairs_gf[0].data());
  // manual dense routine
  auto oca_manual_dense = OCA_dense(s.D.hyb.values, s.D.hyb.coeffs, s.D.hyb.values_reflect, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta,
                                    s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, and convert to dense format
  auto oca_manual_bdof = OCA_bs(s.D.hyb.values, s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  auto oca_manual_bs   = get_tensor_in_full_hilbert_space(oca_manual_bdof, s.model.ad, s.r);

  // compute analytical reference
  auto oca_ana  = nda::zeros<dcomplex>(s.r, 4, 4);
  double denom2 = om * om * (exp(beta * om) + 1) * (exp(beta * om) + 1);
  for (int i = 0; i < s.r; ++i) {
    double t         = rel2abs(s.dlr_it(i)); // t = tau / beta
    double tau       = beta * t;
    double tom       = om * tau;
    double g4        = exp(-t * ln4);
    oca_ana(i, 0, 0) = -2 * g4 * (exp(tom) - tom - 1) * exp(tom) / denom2;
    oca_ana(i, 1, 1) = -g4 * (exp(tom) - 1) * (exp(tom) - 1) * exp(om * (beta - tau)) / denom2;
    oca_ana(i, 2, 2) = oca_ana(i, 1, 1);
    oca_ana(i, 3, 3) = -2 * g4 * (tom * exp(tom) - exp(tom) + 1) * exp(2 * om * (beta - tau)) / denom2;
  }

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_pairs - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_dense - oca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_manual_bs - oca_ana)), eps);

  // ----- third order -----
  auto third_topology = max_crossing_topology(3); // = {{0, 3}, {1, 4}, {2, 5}}
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology)
                                    * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, third_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto third_dense_gf = s.D_dense.compute_self_energy(s.G_ppsc_dense, third_topology);
  auto third_dense    = nda::make_regular(topology_parity(third_topology) * third_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto third_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G_ppsc_dense, third_topology);
  auto third_pairs    = nda::make_regular(topology_parity(third_topology) * third_pairs_gf[0].data());
  // no manual third-order routine

  // compute analytical reference
  auto third_ana = nda::zeros<dcomplex>(s.r, 4, 4);
  double denom3  = om * om * om * om * (exp(beta * om) + 1) * (exp(beta * om) + 1) * (exp(beta * om) + 1);
  for (int i = 0; i < s.r; ++i) {
    double t           = rel2abs(s.dlr_it(i)); // t = tau / beta
    double tau         = beta * t;
    double tom         = om * tau;
    double g4          = exp(-t * ln4);
    third_ana(i, 0, 0) = g4 * (tom * tom * exp(tom) - 4 * tom * exp(tom) - 2 * tom + 6 * exp(tom) - 6) * exp(beta * om) / denom3;
    third_ana(i, 3, 3) = g4 * (tom * tom + 4 * tom + 2 * (tom - 3) * exp(tom) + 6) * exp(om * (2 * beta - tau)) / denom3;
    third_ana(i, 1, 1) = 0.5 * (third_ana(i, 0, 0) + third_ana(i, 3, 3));
    third_ana(i, 2, 2) = third_ana(i, 1, 1);
  }

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_pairs - third_ana)), eps);
}

/**
 * @brief Test evaluation of the single-particle Green's function for a two-fermion system with a
 * hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's
 * function diagrams. As in two_fermions.one_hyb_pole_se, U = mu = 0 so that the pseudo-particle propagator
 * is G(tau) = -4^{-tau/beta} I_4, and the hybridization is Delta(tau) = K(tau, omega) I_2 with the helper's
 * default pole omega = -1.5. The OCA and third-order closed forms are unwieldy, so those orders are compared
 * against reference values tabulated from examples/two_fermion_analytical_solutions.ipynb.
 */
TEST(two_fermions, one_hyb_pole_spgf) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  // Delta(tau) = K(tau, om) M; M = I_2 at alpha = 0
  double om    = -1.5;
  double alpha = 0.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  TwoFermionSetup s{beta, Lambda, eps, om, alpha};

  // ----- NCA -----
  auto nca_topology = max_crossing_topology(1); // = {0, 1}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, nca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto nca_dense = nda::make_regular(topology_parity(nca_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, nca_topology));
  // manual dense routine
  auto nca_manual_dense = NCA_gf_dense(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto nca_manual_bs = NCA_gf_bs(s.model.G_bdof, s.model.G_bdof.reflect(s.itops), s.Fq);

  // compute analytical reference: first order has no hybridization line, so it does not see Delta at all
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  nca_ana(_, 0, 0) = 0.5;
  nca_ana(_, 1, 1) = 0.5;

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);
  EXPECT_EQ(nca_bs.extent(1), 2); // the two orbitals

  // ----- OCA -----
  auto oca_topology = max_crossing_topology(2); // = {{0, 2}, {1, 3}}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, oca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto oca_dense = nda::make_regular(topology_parity(oca_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, oca_topology));
  // manual dense routine
  auto oca_manual_dense = OCA_gf_dense(s.D.hyb.coeffs, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto oca_manual_bs = OCA_gf_bs(s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);

  // analytical reference: the closed form is unwieldy, so it is frozen here as values of g_{up,up}(tau) at a
  // few tau points, tabulated from examples/two_fermion_analytical_solutions.ipynb
  std::vector<double> tau_pts    = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> oca_gf_ref = {0.051177221707610, 0.110236319147384, 0.127756436679493, 0.110236319147384, 0.051177221707610};

  // compare: the reference is a handful of tau points on the orbital diagonal, so the block-sparse result
  // meets it directly and the other routines are compared against that in turn
  for (int o = 0; o < 2; ++o) {
    auto coeffs = s.itops.vals2coefs(oca_bs(_, o, o));
    for (size_t k = 0; k < tau_pts.size(); ++k) { EXPECT_LE(std::abs(s.itops.coefs2eval(coeffs, tau_pts[k]) - oca_gf_ref[k]), eps); }
  }
  // no spin-flip terms in H or Delta, so the orbital off-diagonal entries vanish identically
  EXPECT_LE(max_offdiag(oca_bs), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_bs)), eps);

  // ----- third order -----
  auto third_topology = max_crossing_topology(3); // = {{0, 3}, {1, 4}, {2, 5}}
  // compute using DiagramEvaluator, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, third_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto third_dense = nda::make_regular(topology_parity(third_topology) * s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, third_topology));
  // no manual third-order routine

  // analytical reference: values of g_{up,up}(tau) at beta=2, om=-1.5, computed from the closed form derived
  // in examples/two_fermion_analytical_solutions.ipynb
  std::vector<double> third_gf_ref = {0.000393140721520, 0.003052337792819, 0.006393977070682, 0.006929916603937, 0.002036125469681};

  // compare: the reference is a handful of tau points on the orbital diagonal, so the block-sparse result
  // meets it directly and the other routines are compared against that in turn
  for (int o = 0; o < 2; ++o) {
    auto coeffs = s.itops.vals2coefs(third_bs(_, o, o));
    for (size_t k = 0; k < tau_pts.size(); ++k) { EXPECT_LE(std::abs(s.itops.coefs2eval(coeffs, tau_pts[k]) - third_gf_ref[k]), eps); }
  }
  // no spin-flip terms in H or Delta, so the orbital off-diagonal entries vanish identically
  EXPECT_LE(max_offdiag(third_bs), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_dense)), eps);
}
