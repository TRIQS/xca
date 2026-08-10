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
using triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace;

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

    TwoFermionSetup(double beta_, double Lambda_, double eps_, FermionModelData model_)
       : beta(beta_),
         Lambda(Lambda_),
         eps(eps_),
         itops(Lambda_, build_dlr_rf(Lambda_, eps_)),
         dlr_it(itops.get_itnodes()),
         r(itops.rank()),
         model(std::move(model_)),
         Gt_dense(Hmat_to_Gtmat(get_full_h_atomic(model.ad), beta_, cppdlr::rel2abs(dlr_it))),
         Gt_dense_refl(itops.reflect(Gt_dense)),
         G_ppsc_dense(wrap_dense(model.G_ppsc[0].mesh(), Gt_dense)),
         D_dense(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         D(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         Fq(std::get<0>(get_operators(model.ad, model.hyb_coeffs))),
         block_N(model.ad.n_subspaces()) {
      auto [Fs, F_dags] = get_operators_dense(model.ad);
      Fs_dense          = std::move(Fs);
      F_dags_dense      = std::move(F_dags);
      for (int b = 0; b < model.ad.n_subspaces(); ++b) { block_N(b) = __builtin_popcountl(model.ad.get_fock_states(b)[0]); }
    }

    int nblocks() const { return model.ad.n_subspaces(); }

    private:
    // The dense propagator as a one-block block_gf, which is what the dense evaluator takes.
    static triqs::gfs::block_gf<triqs::mesh::dlr_imtime> wrap_dense(triqs::mesh::dlr_imtime const &mesh, nda::array_const_view<dcomplex, 3> Gt) {
      std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh, Gt)};
      return {blocks};
    }
  };

  // All the tests use the same DLR parameters and U = mu = 0, and differ only in the hybridization. A single
  // pole at omega = 0 makes Delta(tau) = -M/2 tau-independent.
  TwoFermionSetup const_hyb_setup(double beta = 2.0, double eps = 1.0e-12) {
    double Lambda = 20.0 * beta;
    return {beta, Lambda, eps, two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0, 0.0)};
  }

  // Same model, with the identity orbital-space amplitude matrix replaced by the Hermitian
  // M = {{1, alpha}, {alpha, 1}}. The coefficients have to be overwritten before the evaluators read them,
  // hence the tweak to the model on the way in rather than to the setup afterwards.
  TwoFermionSetup hermitian_hyb_setup(double alpha, double beta = 2.0, double eps = 1.0e-12) {
    double Lambda             = 20.0 * beta;
    auto model                = two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0, 0.0);
    model.hyb_coeffs(0, 0, 1) = alpha;
    model.hyb_coeffs(0, 1, 0) = alpha;
    return {beta, Lambda, eps, std::move(model)};
  }

  // Delta(tau) = K(tau, omega) I_2 with the helper's default pole omega = -1.5.
  TwoFermionSetup one_pole_setup(double beta = 2.0, double eps = 1.0e-12) {
    double Lambda = 20.0 * beta;
    return {beta, Lambda, eps, two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0)};
  }

  // ---------------------------------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------------------------------

  // Largest absolute deviation between two arrays. Pair with EXPECT_LE, as with max_offdiag, so that a
  // failure prints the deviation that was actually reached.
  double max_dev(auto const &a, auto const &b) { return nda::max_element(nda::abs(a - b)); }

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

  // G(beta - tau) as a BDOF, built block by block since BlockDiagOpFun has no reflection routine.
  BlockDiagOpFun reflect_bdof(BlockDiagOpFun const &G, imtime_ops &itops) {
    std::vector<nda::array<dcomplex, 3>> blocks;
    nda::vector<int> zero_block_indices(G.get_num_block_cols());
    for (int b = 0; b < G.get_num_block_cols(); ++b) {
      blocks.push_back(nda::make_regular(itops.reflect(G.get_block(b))));
      zero_block_indices(b) = G.get_zero_block_index(b);
    }
    return {blocks, zero_block_indices};
  }

  // Compare a block-sparse result, block by block, against the corresponding subspaces of a dense result.
  // sign = -1 for the evaluators that use the opposite overall self-energy convention (see
  // check_nca_se_manual).
  void expect_blocks_match_dense(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> bs, nda::array_const_view<dcomplex, 3> dense,
                                 TwoFermionSetup const &s, double sign = 1.0) {
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      auto dense_block = get_tensor_in_atom_diag_subspace(dense, b, s.model.ad);
      EXPECT_LE(max_dev(bs[b].data(), sign * dense_block), s.eps);
    }
  }

  // Same, against another block-sparse result.
  void expect_blocks_match(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> bs, BlockDiagOpFun const &other, TwoFermionSetup const &s,
                           double sign = 1.0) {
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      EXPECT_LE(max_dev(bs[b].data(), sign * other.get_block(b)), s.eps);
    }
  }

  // Every diagonal entry of every block equals ref, and every off-diagonal entry vanishes. This is the
  // structure the self-energy has whenever Delta is tau-independent: diagonal in the Fock basis, with the
  // same entry in every occupation sector.
  void expect_diag_all_blocks(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> bs, nda::array_const_view<double, 1> ref, TwoFermionSetup const &s) {
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      auto block = bs[b].data();
      for (int d = 0; d < block.extent(1); ++d) { EXPECT_LE(max_dev(block(_, d, d), ref), s.eps); }
      EXPECT_LE(max_offdiag(block), s.eps);
    }
  }

  // Same, but with one reference per occupation sector (the columns of ref are N = 0, 1, 2), which is what a
  // tau-dependent hybridization produces. Blocks are matched to sectors by the particle number of their Fock
  // states rather than by block order.
  void expect_diag_by_sector(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> bs, nda::array_const_view<double, 2> ref, TwoFermionSetup const &s) {
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b) + ", occupation sector N = " + std::to_string(s.block_N(b)));
      auto block = bs[b].data();
      for (int d = 0; d < block.extent(1); ++d) { EXPECT_LE(max_dev(block(_, d, d), ref(_, s.block_N(b))), s.eps); }
      EXPECT_LE(max_offdiag(block), s.eps);
    }
  }

  // The atomic propagator is -4^{-tau/beta} in all three models, since U = mu = 0 in all of them.
  void expect_free_propagator(TwoFermionSetup const &s) {
    SCOPED_TRACE("atomic propagator");
    auto G0_ana = tau_ref(s.dlr_it, [](double t) { return -exp(-t * 2 * std::numbers::ln2); });
    for (int b = 0; b < s.nblocks(); ++b) { EXPECT_LE(max_dev(s.model.G_bdof.get_block(b)(_, 0, 0), G0_ana), s.eps); }
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

  // ---------------------------------------------------------------------------------------------------
  // Evaluator comparison
  // ---------------------------------------------------------------------------------------------------

  /**
   * @brief Cancel the fermionic topology sign that the diagram evaluators apply internally
   *
   * @details Used primarily for comparisons in tests.
   */
  double parity_of(nda::array_const_view<int, 2> topology) { return static_cast<double>(topology_parity(topology)); }

  /**
   * @brief Evaluate the self-energy for a topology with both diagram evaluators and compare them
   * @return The block-sparse self-energy with the topology sign cancelled, for the caller to compare
   *         against its analytic reference
   */
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> check_se_diagram_evaluators(TwoFermionSetup &s, nda::array_const_view<int, 2> topology) {
    SCOPED_TRACE("self-energy evaluators at order " + std::to_string(topology.extent(0)));
    // dense diagram evaluator
    auto se_dde = s.D_dense.compute_self_energy(s.G_ppsc_dense, topology);
    EXPECT_LE(max_offdiag(se_dde[0].data()), s.eps);
    // block-sparse diagram evaluator, compared with the dense one
    auto se = s.D.compute_self_energy(s.model.G_ppsc, topology);
    expect_blocks_match_dense(se, se_dde[0].data(), s);
    // both evaluators agree, so cancel the topology sign once, on the way out (see parity_of)
    auto parity = parity_of(topology);
    for (int b = 0; b < s.nblocks(); ++b) { se[b].data() *= parity; }
    return se;
  }

  /**
   * @brief Evaluate the single-particle Green's function for a topology with both diagram evaluators
   * @return The block-sparse Green's function with the topology sign cancelled, for the caller to compare
   *         against its analytic reference
   */
  nda::array<dcomplex, 3> check_gf_diagram_evaluators(TwoFermionSetup &s, nda::array_const_view<int, 2> topology) {
    SCOPED_TRACE("single-particle gf evaluators at order " + std::to_string(topology.extent(0)));
    auto gf_dde = s.D_dense.compute_single_ptcle_gf(s.G_ppsc_dense, topology);
    auto gf     = s.D.compute_single_ptcle_gf(s.model.G_ppsc, topology);
    EXPECT_EQ(gf.extent(1), 2); // the two orbitals
    EXPECT_LE(max_dev(gf, gf_dde), s.eps);
    gf *= parity_of(topology); // cancel the topology sign once, on the way out (see parity_of)
    return gf;
  }

  // Compare an NCA self-energy with the dense and block-sparse evaluators that can only do first order.
  void check_nca_se_manual(TwoFermionSetup &s, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> nca_se) {
    SCOPED_TRACE("manual NCA self-energy evaluators");
    // NCA_bs/NCA_dense and DiagramEvaluator::compute_self_energy use opposite overall sign conventions for
    // the self-energy (cf. the "-Sigma_Diagram_calc" negation in test_block_sparse_NCA_manual.cpp), hence
    // the sign = -1 below.
    auto nca_se_dense = NCA_dense(s.D.hyb.values, s.D.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
    expect_blocks_match_dense(nca_se, nca_se_dense, s, -1.0);
    EXPECT_EQ(nca_se_dense.extent(1), 4); // the four many-body states
    EXPECT_LE(max_offdiag(nca_se_dense), s.eps);
    auto nca_se_manual = NCA_bs(s.D.hyb.values, s.D.hyb.values_reflect, s.model.G_bdof, s.Fq);
    expect_blocks_match(nca_se, nca_se_manual, s, -1.0);
  }

  // Compare an OCA self-energy with the dense and block-sparse evaluators that can only do second order.
  void check_oca_se_manual(TwoFermionSetup &s, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> oca_se) {
    SCOPED_TRACE("manual OCA self-energy evaluators");
    // Unlike NCA (odd order), OCA (even order) needs no sign flip: the (-1) per hybridization line in
    // NCA_bs/OCA_bs cancels against the extra line at second order.
    auto oca_se_dense = OCA_dense(s.D.hyb.values, s.D.hyb.coeffs, s.D.hyb.values_reflect, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense,
                                  s.Fs_dense, s.F_dags_dense);
    expect_blocks_match_dense(oca_se, oca_se_dense, s);
    EXPECT_EQ(oca_se_dense.extent(1), 4);
    EXPECT_LE(max_offdiag(oca_se_dense), s.eps);
    auto oca_se_manual = OCA_bs(s.D.hyb.values, s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
    expect_blocks_match(oca_se, oca_se_manual, s);
  }

  // Compare an NCA single-particle Green's function with the dense and block-sparse first-order-only evaluators.
  void check_nca_gf_manual(TwoFermionSetup &s, nda::array_const_view<dcomplex, 3> nca_gf) {
    SCOPED_TRACE("manual NCA single-particle gf evaluators");
    auto nca_gf_dense = NCA_gf_dense(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense);
    EXPECT_LE(max_dev(nca_gf, nca_gf_dense), s.eps);
    auto nca_gf_manual = NCA_gf_bs(s.model.G_bdof, reflect_bdof(s.model.G_bdof, s.itops), s.Fq);
    EXPECT_LE(max_dev(nca_gf, nca_gf_manual), s.eps);
  }

  // Compare an OCA single-particle Green's function with the dense and block-sparse second-order-only evaluators.
  void check_oca_gf_manual(TwoFermionSetup &s, nda::array_const_view<dcomplex, 3> oca_gf) {
    SCOPED_TRACE("manual OCA single-particle gf evaluators");
    auto oca_gf_dense = OCA_gf_dense(s.D.hyb.coeffs, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
    EXPECT_LE(max_dev(oca_gf, oca_gf_dense), s.eps);
    auto oca_gf_manual = OCA_gf_bs(s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
    EXPECT_LE(max_dev(oca_gf, oca_gf_manual), s.eps);
  }

  // Evaluate the diagonal of a correlator at a few tau points and compare against tabulated reference values,
  // used where the closed form is too unwieldy to transcribe and the notebook's values are frozen here instead.
  void expect_gf_at_tau_pts(TwoFermionSetup &s, nda::array_const_view<dcomplex, 3> gf, std::vector<double> const &tau_pts,
                            std::vector<double> const &ref) {
    for (int o = 0; o < gf.extent(1); ++o) {
      SCOPED_TRACE("orbital " + std::to_string(o));
      auto coeffs = s.itops.vals2coefs(gf(_, o, o));
      for (size_t k = 0; k < tau_pts.size(); ++k) {
        SCOPED_TRACE("tau = " + std::to_string(tau_pts[k]));
        EXPECT_LE(std::abs(s.itops.coefs2eval(coeffs, tau_pts[k]) - ref[k]), s.eps);
      }
    }
    // no spin-flip terms in H or Delta, so the orbital off-diagonal entries vanish identically
    EXPECT_LE(max_offdiag(gf), s.eps);
  }

} // namespace

/**
 * @brief Test evaluation of the self-energy for a two-fermion system with a constant hybridization function
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams.
 */
TEST(two_fermions, const_hyb_se) {
  auto s      = const_hyb_setup();
  double beta = s.beta;
  double ln4  = 2 * std::numbers::ln2;
  expect_free_propagator(s);
  expect_block_structure(s);

  // ----- NCA test -----
  auto nca_se     = check_se_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_se_ana = tau_ref(s.dlr_it, [&](double t) { return exp(-t * ln4); }); // = -G0_ana
  expect_diag_all_blocks(nca_se, nca_se_ana, s);
  check_nca_se_manual(s, nca_se);

  // ----- OCA test -----
  auto oca_se     = check_se_diagram_evaluators(s, max_crossing_topology(2));
  auto oca_se_ana = tau_ref(s.dlr_it, [&](double t) { return -0.25 * exp(-t * ln4) * t * t * beta * beta; });
  expect_diag_all_blocks(oca_se, oca_se_ana, s);
  check_oca_se_manual(s, oca_se);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  auto third_order_se     = check_se_diagram_evaluators(s, max_crossing_topology(3));
  auto third_order_se_ana = tau_ref(s.dlr_it, [&](double t) { return 1.0 / 96 * exp(-t * ln4) * pow(t, 4) * pow(beta, 4); });
  expect_diag_all_blocks(third_order_se, third_order_se_ana, s);
}

/**
 * @brief Test evaluation of the single-particle Green's function for a two-fermion system with a constant hybridization function
 *
 * @details This tests the evaluation of the first-, second-, and third-order diagrams contributing to the single-particle Green's function.
 */
TEST(two_fermions, const_hyb_spgf) {
  auto s      = const_hyb_setup();
  double beta = s.beta;

  // ----- NCA test -----
  // First order has no hybridization line, so the reference is the hybridization-independent constant 1/2.
  auto nca_spgf         = check_gf_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_spgf_ana     = nda::zeros<double>(s.r, 2, 2);
  nca_spgf_ana(_, 0, 0) = 0.5;
  nca_spgf_ana(_, 1, 1) = 0.5;
  ASSERT_LE(max_dev(nca_spgf, nca_spgf_ana), s.eps);
  check_nca_gf_manual(s, nca_spgf);

  // ----- OCA test -----
  auto oca_spgf     = check_gf_diagram_evaluators(s, max_crossing_topology(2));
  auto oca_spgf_ana = tau_ref(s.dlr_it, [&](double t) { return 0.25 * beta * beta * t * (1 - t); }); // = tau (beta - tau) / 4
  ASSERT_LE(max_dev(oca_spgf(_, 0, 0), oca_spgf_ana), s.eps);
  ASSERT_LE(max_dev(oca_spgf(_, 1, 1), oca_spgf_ana), s.eps);
  // There are no spin-flip terms in H or Delta, so the spin off-diagonal blocks vanish identically
  ASSERT_LE(max_offdiag(oca_spgf), s.eps);
  check_oca_gf_manual(s, oca_spgf);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  auto third_order_spgf = check_gf_diagram_evaluators(s, max_crossing_topology(3));
  // = tau^2 (beta - tau)^2 / 32, the alpha = 0 case of the hermitian_hyb_spgf reference below
  auto third_order_spgf_ana = tau_ref(s.dlr_it, [&](double t) { return pow(beta, 4) * t * t * (1 - t) * (1 - t) / 32.0; });
  ASSERT_LE(max_dev(third_order_spgf(_, 0, 0), third_order_spgf_ana), s.eps);
  ASSERT_LE(max_dev(third_order_spgf(_, 1, 1), third_order_spgf_ana), s.eps);
  ASSERT_LE(max_offdiag(third_order_spgf), s.eps);
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
  double alpha = 0.4;
  auto s       = hermitian_hyb_setup(alpha);
  double beta  = s.beta;
  double ln4   = 2 * std::numbers::ln2;
  expect_free_propagator(s);
  expect_block_structure(s);

  // ----- NCA test -----
  auto nca_se     = check_se_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_se_ana = tau_ref(s.dlr_it, [&](double t) { return exp(-t * ln4); }); // = -G0_ana, independent of alpha
  expect_diag_all_blocks(nca_se, nca_se_ana, s);
  check_nca_se_manual(s, nca_se);

  // ----- OCA test -----
  // The alpha = 0 value carries a factor -1; the Hermitian M replaces it by alpha^2 - 1
  auto oca_se     = check_se_diagram_evaluators(s, max_crossing_topology(2));
  auto oca_se_ana = tau_ref(s.dlr_it, [&](double t) {
    double tau = beta * t;
    return 0.25 * (alpha * alpha - 1) * exp(-t * ln4) * tau * tau;
  });
  expect_diag_all_blocks(oca_se, oca_se_ana, s);
  check_oca_se_manual(s, oca_se);

  // ----- third-order test -----
  // The alpha = 0 value carries a factor 1/3 (relative to tau^4 / 32); the Hermitian M replaces it by alpha^2 + 1/3
  auto third_order_se     = check_se_diagram_evaluators(s, max_crossing_topology(3));
  auto third_order_se_ana = tau_ref(s.dlr_it, [&](double t) {
    double tau = beta * t;
    return (3 * alpha * alpha + 1) / 96.0 * exp(-t * ln4) * pow(tau, 4);
  });
  expect_diag_all_blocks(third_order_se, third_order_se_ana, s);
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
  double alpha = 0.4;
  auto s       = hermitian_hyb_setup(alpha);
  double beta  = s.beta;

  // ----- NCA test -----
  // First order has no hybridization line, so it does not see M at all
  auto nca_spgf         = check_gf_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_spgf_ana     = nda::zeros<double>(s.r, 2, 2);
  nca_spgf_ana(_, 0, 0) = 0.5;
  nca_spgf_ana(_, 1, 1) = 0.5;
  ASSERT_LE(max_dev(nca_spgf, nca_spgf_ana), s.eps);
  check_nca_gf_manual(s, nca_spgf);

  // ----- OCA test -----
  // g_aa = tau (beta - tau) / 4, g_ab = -alpha tau (beta - tau) / 4
  auto oca_spgf     = check_gf_diagram_evaluators(s, max_crossing_topology(2));
  auto oca_spgf_ana = nda::zeros<double>(s.r, 2, 2);
  auto oca_diag     = tau_ref(s.dlr_it, [&](double t) { return 0.25 * beta * beta * t * (1 - t); });
  for (int o = 0; o < 2; ++o) {
    oca_spgf_ana(_, o, o)     = oca_diag;
    oca_spgf_ana(_, o, 1 - o) = -alpha * oca_diag;
  }
  ASSERT_LE(max_dev(oca_spgf, oca_spgf_ana), s.eps);
  check_oca_gf_manual(s, oca_spgf);

  // ----- third-order test -----
  // g_aa = (alpha^2 + 1) tau^2 (beta - tau)^2 / 32, g_ab = alpha tau^2 (beta - tau)^2 / 16
  auto third_order_spgf     = check_gf_diagram_evaluators(s, max_crossing_topology(3));
  auto third_order_spgf_ana = nda::zeros<double>(s.r, 2, 2);
  auto base                 = tau_ref(s.dlr_it, [&](double t) { return pow(beta, 4) * t * t * (1 - t) * (1 - t); });
  for (int o = 0; o < 2; ++o) {
    third_order_spgf_ana(_, o, o)     = (alpha * alpha + 1) * base / 32.0;
    third_order_spgf_ana(_, o, 1 - o) = alpha * base / 16.0;
  }
  ASSERT_LE(max_dev(third_order_spgf, third_order_spgf_ana), s.eps);
}

/**
 * @brief Test evaluation of the self-energy for a two-fermion system with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams. 
 *
 */
TEST(two_fermions, one_hyb_pole_se) {
  auto s      = one_pole_setup();
  double beta = s.beta;
  double ln4  = 2 * std::numbers::ln2;
  double om   = s.model.hyb_poles(0);
  expect_free_propagator(s);
  expect_block_structure(s);

  // ----- NCA test -----
  auto nca_se     = check_se_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_se_ana = nda::zeros<double>(s.r, 3); // columns are the occupation sectors N = 0, 1, 2
  for (int i = 0; i < s.r; ++i) {
    double t         = rel2abs(s.dlr_it(i)); // t = tau / beta
    double tau       = beta * t;
    double g4        = exp(-t * ln4);
    nca_se_ana(i, 0) = 2 * g4 * exp(om * tau) / (exp(beta * om) + 1);
    nca_se_ana(i, 2) = 2 * g4 * exp(-om * tau) / (exp(-beta * om) + 1);
    nca_se_ana(i, 1) = 0.5 * (nca_se_ana(i, 0) + nca_se_ana(i, 2));
  }
  expect_diag_by_sector(nca_se, nca_se_ana, s);
  check_nca_se_manual(s, nca_se);

  // ----- OCA test -----
  auto oca_se     = check_se_diagram_evaluators(s, max_crossing_topology(2));
  auto oca_se_ana = nda::zeros<double>(s.r, 3);
  double denom2   = om * om * (exp(beta * om) + 1) * (exp(beta * om) + 1);
  for (int i = 0; i < s.r; ++i) {
    double t         = rel2abs(s.dlr_it(i)); // t = tau / beta
    double tau       = beta * t;
    double tom       = om * tau;
    double g4        = exp(-t * ln4);
    oca_se_ana(i, 0) = -2 * g4 * (exp(tom) - tom - 1) * exp(tom) / denom2;
    oca_se_ana(i, 1) = -g4 * (exp(tom) - 1) * (exp(tom) - 1) * exp(om * (beta - tau)) / denom2;
    oca_se_ana(i, 2) = -2 * g4 * (tom * exp(tom) - exp(tom) + 1) * exp(2 * om * (beta - tau)) / denom2;
  }
  expect_diag_by_sector(oca_se, oca_se_ana, s);
  check_oca_se_manual(s, oca_se);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  auto third_order_se     = check_se_diagram_evaluators(s, max_crossing_topology(3));
  auto third_order_se_ana = nda::zeros<double>(s.r, 3);
  double denom3           = om * om * om * om * (exp(beta * om) + 1) * (exp(beta * om) + 1) * (exp(beta * om) + 1);
  for (int i = 0; i < s.r; ++i) {
    double t                 = rel2abs(s.dlr_it(i)); // t = tau / beta
    double tau               = beta * t;
    double tom               = om * tau;
    double g4                = exp(-t * ln4);
    third_order_se_ana(i, 0) = g4 * (tom * tom * exp(tom) - 4 * tom * exp(tom) - 2 * tom + 6 * exp(tom) - 6) * exp(beta * om) / denom3;
    third_order_se_ana(i, 2) = g4 * (tom * tom + 4 * tom + 2 * (tom - 3) * exp(tom) + 6) * exp(om * (2 * beta - tau)) / denom3;
    third_order_se_ana(i, 1) = 0.5 * (third_order_se_ana(i, 0) + third_order_se_ana(i, 2));
  }
  expect_diag_by_sector(third_order_se, third_order_se_ana, s);
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
  auto s = one_pole_setup();

  // ----- NCA test -----
  // First order has no hybridization line at all, so the reference is the same hybridization-independent
  // constant 1/2 seen in the const-hybridization test above.
  auto nca_spgf         = check_gf_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_spgf_ana     = nda::zeros<double>(s.r, 2, 2);
  nca_spgf_ana(_, 0, 0) = 0.5;
  nca_spgf_ana(_, 1, 1) = 0.5;
  ASSERT_LE(max_dev(nca_spgf, nca_spgf_ana), s.eps);
  check_nca_gf_manual(s, nca_spgf);

  // ----- OCA test -----
  auto oca_spgf                   = check_gf_diagram_evaluators(s, max_crossing_topology(2));
  std::vector<double> oca_tau_pts = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> oca_gf_ref  = {0.051177221707610, 0.110236319147384, 0.127756436679493, 0.110236319147384, 0.051177221707610};
  expect_gf_at_tau_pts(s, oca_spgf, oca_tau_pts, oca_gf_ref);
  check_oca_gf_manual(s, oca_spgf);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  auto third_order_spgf = check_gf_diagram_evaluators(s, max_crossing_topology(3));

  // Reference values for g_{up,up}(tau) at beta=2, omega=-1.5, computed from the closed form derived
  // in examples/two_fermion_analytical_solutions.ipynb
  std::vector<double> tau_pts = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> gf_ref  = {0.000393140721520, 0.003052337792819, 0.006393977070682, 0.006929916603937, 0.002036125469681};
  expect_gf_at_tau_pts(s, third_order_spgf, tau_pts, gf_ref);
}
