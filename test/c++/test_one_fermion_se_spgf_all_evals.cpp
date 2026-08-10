#include <gtest/gtest.h>

#include <nda/algorithms.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/block_sparse_manual_gf.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/topology.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

using triqs_xca::dense::DenseDiagramEvaluator;
using triqs_xca::dense::DenseFSet;

using triqs_xca::block_sparse::DiagramEvaluator;
using triqs_xca::block_sparse::NCA_dense;
using triqs_xca::block_sparse::NCA_gf_dense;
using triqs_xca::block_sparse::OCA_dense;
using triqs_xca::block_sparse::OCA_gf_dense;

using triqs_xca::block_sparse::eval_eq;
using triqs_xca::block_sparse::OCA_gf_tpz;
using triqs_xca::block_sparse::OCA_tpz;
using triqs_xca::block_sparse::third_order_gf_tpz;
using triqs_xca::block_sparse::third_order_tpz;

using triqs_xca::topology::topology_parity;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;
using triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace;

/**
 * @file test_one_fermion_se_spgf_all_evals.cpp
 *
 * @brief Tests of self-energy and single-particle Green's function diagram evaluators for models with one spinless fermion
 *
 * @details Each test covers a different hybridization that is a linear combination of zero, one, or two K(\tau, \omega) at different values of the
 * poles \omega. All the tests compare the self-energy/single-particle Green's function computed analytically at first- (NCA), second- (OCA), and
 * third order to the results of several evaluation routines. For NCA and OCA, there are comparisons to the following evaluators:
 *
 * - the `compute_self_energy/single_particle_gf` routine of DenseDiagramEvaluator
 * - the `compute_self_energy_by_pairs` routine of DenseDiagramEvaluator, which has no single-particle Green's function analogue
 *   - This routine is a time optimization of the first compute_self_energy routine above and is the one actually used in Python wrappers
 * - the `compute_self_energy/single_particle_gf` routine of the DiagramEvaluator, which takes advantage of block-sparsity
 * - `N/OCA_(gf_)dense`, a routine that can only evaluate N/OCA with dense matmuls
 * - `N/OCA_(gf_)bs`, a routine that can only evaluate N/OCA taking advantage of block-sparsity
 *
 * The last two routines above are not wrapped and exist just for testing purposes. For third-order diagrams, there are no analogues to these last
 * two evaluators, as we are testing that the (Dense)DiagramEvaluator routines do in fact work at arbitrary order.
 *
 * For third-order diagram evaluation, there are comparisons to all of the above routines except, obviously, the last two routines above, and no 
 * third-order analogues exist. 
 *
 * For OCA and third-order diagrams, there are also comparisons to routines which compute integrals using trapezoidal quadrature. The number of 
 * quadrature points prioritizes brief test runtime over achieving accuracy competitive with the prior tests.
 *
 * Every test differs only in its hybridization and in its analytic reference values; the model setup and the evaluator-vs-evaluator comparisons are 
 * shared through the OneFermionSetup bundle and the check_* helpers below, so each test body is essentially its closed-form reference plus a list 
 * of named checks.
 */

namespace {

  // ---------------------------------------------------------------------------------------------------
  // Model construction
  // ---------------------------------------------------------------------------------------------------

  /**
   * @brief One-fermion model (trivial atomic Hamiltonian H = 0, one orbital) with the two-pole
   * hybridization Delta(tau) = K(tau, 0.6) + 2 * K(tau, -0.9)
   *
   * @details Same atom as one_fermion_model_helper (block_sparse_utils.cpp); that helper only builds
   * single-pole hybridizations, so the two-pole cases construct their coefficients here.
   */
  FermionModelData two_pole_model(double beta, double Lambda, double eps) {
    int p    = 2;
    int norb = 1;
    nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
    hyb_coeffs(0, 0, 0) = 1.0;
    hyb_coeffs(1, 0, 0) = 2.0;
    nda::vector<double> hyb_poles(p);
    hyb_poles(0) = 0.6;
    hyb_poles(1) = -0.9;

    many_body_operator_complex H;
    double mu = 0.0;
    many_body_operator_complex N;
    N = n("0", 0);
    H = -mu * N;

    triqs::atom_diag::fundamental_operator_set fop_set;
    fop_set.insert("0", 0);
    auto ad = triqs::atom_diag::atom_diag<true>(H, fop_set);

    auto G_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
    auto G_bdof = BlockDiagOpFun(G_ppsc);

    return {.hyb_coeffs = hyb_coeffs, .hyb_poles = hyb_poles, .ad = ad, .G_ppsc = G_ppsc, .G_bdof = G_bdof};
  }

  /**
   * @brief Everything the one-fermion tests need: the DLR grid, the model, and all the evaluators
   *
   * @details The dense objects sum the same backbones as the block-sparse ones but over the full Hilbert
   * space rather than block by block, so the two carry the same overall self-energy sign convention.
   */
  struct OneFermionSetup {
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
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G0_ppsc_dense;

    DenseDiagramEvaluator D_dense;
    DiagramEvaluator D;
    BlockOpSymQuartet Fq; // field operators of the block-sparse evaluator

    OneFermionSetup(double beta_, double Lambda_, double eps_, FermionModelData model_)
       : beta(beta_),
         Lambda(Lambda_),
         eps(eps_),
         itops(Lambda_, build_dlr_rf(Lambda_, eps_)),
         dlr_it(itops.get_itnodes()),
         r(itops.rank()),
         model(std::move(model_)),
         Gt_dense(Hmat_to_Gtmat(get_full_h_atomic(model.ad), beta_, cppdlr::rel2abs(dlr_it))),
         Gt_dense_refl(itops.reflect(Gt_dense)),
         G0_ppsc_dense(wrap_dense(model.G_ppsc[0].mesh(), Gt_dense)),
         D_dense(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         D(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         Fq(std::get<0>(get_operators(model.ad, model.hyb_coeffs))) {
      auto [Fs, F_dags] = get_operators_dense(model.ad);
      Fs_dense          = std::move(Fs);
      F_dags_dense      = std::move(F_dags);
    }

    int nblocks() const { return model.G_bdof.get_num_block_cols(); }

    private:
    // The dense propagator as a one-block block_gf, which is what the dense evaluator takes.
    static triqs::gfs::block_gf<triqs::mesh::dlr_imtime> wrap_dense(triqs::mesh::dlr_imtime const &mesh, nda::array_const_view<dcomplex, 3> Gt) {
      std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh, Gt)};
      return {blocks};
    }
  };

  // All six tests use the same DLR parameters, and differ only in beta and in the hybridization.
  OneFermionSetup one_pole_setup(double beta, double hyb_pole = 0.0, double eps = 1.0e-10) {
    double Lambda = 20.0 * beta;
    return {beta, Lambda, eps, one_fermion_model_helper(beta, Lambda, eps, hyb_pole)};
  }

  OneFermionSetup two_pole_setup(double beta, double eps = 1.0e-10) {
    double Lambda = 20.0 * beta;
    return {beta, Lambda, eps, two_pole_model(beta, Lambda, eps)};
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
                                 OneFermionSetup const &s, double sign = 1.0) {
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      auto dense_block = get_tensor_in_atom_diag_subspace(dense, b, s.model.ad);
      EXPECT_LE(max_dev(bs[b].data(), sign * dense_block), s.eps);
    }
  }

  // Compare two block-sparse results, block by block, with potential sign argument
  void expect_blocks_match(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> bs, BlockDiagOpFun const &other, OneFermionSetup const &s,
                           double sign = 1.0) {
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      EXPECT_LE(max_dev(bs[b].data(), sign * other.get_block(b)), s.eps);
    }
  }

  // Hybridization at the DLR imaginary-time nodes of s.itops, which is what the trapezoidal routines
  // interpolate onto their own equispaced grid. The beta/Lambda/eps overload of coefs2vals hardcodes a
  // symmetrized grid internally, so the itops-based overload is the one that matches the rest of the setup.
  nda::array<dcomplex, 3> hyb_at_dlr_nodes(OneFermionSetup &s) {
    return triqs_xca::hyb::coefs2vals(s.beta, s.itops, s.model.hyb_coeffs, s.model.hyb_poles);
  }

  // Compare a block-sparse result against a dense result that already lives on the equispaced quadrature
  // grid, re-sampling the former onto that grid. grid_range selects the points to compare, since the
  // quadrature routines leave grid endpoints they never visit at zero.
  void expect_blocks_match_eq(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> bs, nda::array_const_view<dcomplex, 3> dense_eq, OneFermionSetup &s,
                              int n_quad, nda::range grid_range, double tol) {
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      auto bs_eq       = eval_eq(s.itops, bs[b].data(), n_quad);
      auto dense_block = get_tensor_in_atom_diag_subspace(dense_eq, b, s.model.ad);
      EXPECT_LE(max_dev(bs_eq(grid_range, _, _), dense_block(grid_range, _, _)), tol);
    }
  }

  // Compare a diagonal entry of a block-sparse result against a closed-form reference.
  void expect_diag_close(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> bs, int b, nda::array_const_view<double, 1> ref,
                         OneFermionSetup const &s) {
    SCOPED_TRACE("block " + std::to_string(b));
    EXPECT_LE(max_dev(bs[b].data()(_, 0, 0), ref), s.eps);
  }

  // Check that a block-sparse result vanishes identically.
  void expect_blocks_zero(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> bs, OneFermionSetup const &s) {
    for (int b = 0; b < s.nblocks(); ++b) {
      SCOPED_TRACE("block " + std::to_string(b));
      EXPECT_LE(nda::max_element(nda::abs(bs[b].data()(_, 0, 0))), s.eps);
    }
  }

  // The atomic propagator is -exp(-tau ln2) in all three models, since H = 0 in all of them.
  void expect_atomic_propagator(OneFermionSetup const &s) {
    SCOPED_TRACE("atomic propagator");
    auto G0_ana = tau_ref(s.dlr_it, [](double t) { return -exp(-t * std::numbers::ln2); });
    for (int b = 0; b < s.nblocks(); ++b) { EXPECT_LE(max_dev(s.model.G_bdof.get_block(b)(_, 0, 0), G0_ana), s.eps); }
  }

  // ---------------------------------------------------------------------------------------------------
  // Evaluator comparison
  // ---------------------------------------------------------------------------------------------------

  /**
   * @brief Cancel the fermionic topology sign that the diagram evaluators apply internally
   *
   * @details Both diagram evaluators multiply each backbone by the fermionic permutation parity of its
   * topology (Backbone::get_parity). The analytic references below, the manual N/OCA routines and the
   * trapezoidal routines all predate that convention and carry no such factor, so the parity is divided
   * back out once here, at the point where a diagram-evaluator result leaves the evaluator-vs-evaluator
   * comparisons. Same cancellation as the "Cancel topology sign, accounted for in dense diag eval"
   * negations in test_block_sparse_backbone_eval.cpp, but written for arbitrary order rather than
   * hardcoded for OCA: the parity is +1 at first order and -1 at second and third.
   */
  double parity_of(nda::array_const_view<int, 2> topology) { return static_cast<double>(topology_parity(topology)); }

  /**
   * @brief Evaluate the self-energy for a topology with every general-order evaluator and compare them
   * @return The block-sparse self-energy with the topology sign cancelled, for the caller to compare
   *         against its analytic reference
   */
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> check_se_diagram_evaluators(OneFermionSetup &s, nda::array_const_view<int, 2> topology) {
    SCOPED_TRACE("self-energy evaluators at order " + std::to_string(topology.extent(0)));
    // dense diagram evaluator
    auto se_dde = s.D_dense.compute_self_energy(s.G0_ppsc_dense, topology);
    EXPECT_LE(max_offdiag(se_dde[0].data()), s.eps);
    // dense diagram evaluator, evaluating the backbones using the pairs optimization
    auto se_pairs = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, topology);
    EXPECT_LE(max_dev(se_dde[0].data(), se_pairs[0].data()), s.eps);
    // block-sparse diagram evaluator, compared with the dense one
    auto se = s.D.compute_self_energy(s.model.G_ppsc, topology);
    expect_blocks_match_dense(se, se_dde[0].data(), s);
    // all three evaluators agree, so cancel the topology sign once, on the way out (see parity_of)
    auto parity = parity_of(topology);
    for (int b = 0; b < s.nblocks(); ++b) { se[b].data() *= parity; }
    return se;
  }

  /**
   * @brief Evaluate the single-particle Green's function for a topology with both block-sparse and dense diagram evaluators
   * @return The block-sparse Green's function with the topology sign cancelled, for the caller to compare
   *         against its analytic reference
   */
  nda::array<dcomplex, 3> check_gf_diagram_evaluators(OneFermionSetup &s, nda::array_const_view<int, 2> topology) {
    SCOPED_TRACE("single-particle gf evaluators at order " + std::to_string(topology.extent(0)));
    auto gf_dde = s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, topology);
    auto gf     = s.D.compute_single_ptcle_gf(s.model.G_ppsc, topology);
    EXPECT_LE(max_dev(gf, gf_dde), s.eps);
    gf *= parity_of(topology); // cancel the topology sign once, on the way out (see parity_of)
    return gf;
  }

  // Compare an NCA self-energy with the dense and block-sparse evaluators that can only do first order.
  void check_nca_se_manual(OneFermionSetup &s, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> nca_se) {
    SCOPED_TRACE("manual NCA self-energy evaluators");
    // NCA_bs/NCA_dense and DiagramEvaluator::compute_self_energy use opposite overall sign conventions for
    // the self-energy (cf. the "-Sigma_Diagram_calc" negation in test_block_sparse_NCA_manual.cpp), hence
    // the sign = -1 below.
    auto nca_se_dense = NCA_dense(s.D.hyb.values, s.D.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
    expect_blocks_match_dense(nca_se, nca_se_dense, s, -1.0);
    EXPECT_LE(max_offdiag(nca_se_dense), s.eps);
    auto nca_se_manual = NCA_bs(s.D.hyb.values, s.D.hyb.values_reflect, s.model.G_bdof, s.Fq);
    expect_blocks_match(nca_se, nca_se_manual, s, -1.0);
  }

  // Compare an OCA self-energy with the dense and block-sparse evaluators that can only do second order.
  void check_oca_se_manual(OneFermionSetup &s, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> oca_se) {
    SCOPED_TRACE("manual OCA self-energy evaluators");
    // Unlike NCA (odd order), OCA (even order) needs no sign flip: the (-1) per hybridization line in
    // NCA_bs/OCA_bs cancels against the extra line at second order.
    auto oca_se_dense = OCA_dense(s.D.hyb.values, s.D.hyb.coeffs, s.D.hyb.values_reflect, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense,
                                  s.Fs_dense, s.F_dags_dense);
    expect_blocks_match_dense(oca_se, oca_se_dense, s);
    EXPECT_LE(max_offdiag(oca_se_dense), s.eps);
    auto oca_se_manual = OCA_bs(s.D.hyb.values, s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
    expect_blocks_match(oca_se, oca_se_manual, s);
  }

  /**
   * @brief Compare an OCA self-energy with direct trapezoidal quadrature of the same diagram
   *
   * @details OCA_tpz shares none of the backbone enumeration or analytic DLR edge integration of the
   * diagram evaluators, so this is an independent check of the machinery rather than of one evaluator
   * against another. The tolerance is set by the quadrature error at n_quad, not by eps.
   */
  void check_oca_se_tpz(OneFermionSetup &s, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> oca_se, int n_quad = 20, double tol = 2.0e-3) {
    SCOPED_TRACE("OCA self-energy vs trapezoidal quadrature");
    auto oca_se_tpz = OCA_tpz(hyb_at_dlr_nodes(s), s.itops, s.beta, s.Gt_dense, s.Fs_dense, n_quad);
    // OCA_tpz's outer loop starts at i = 1, leaving grid point 0 untouched, so skip it.
    expect_blocks_match_eq(oca_se, oca_se_tpz, s, n_quad, nda::range(1, n_quad + 1), tol);
    EXPECT_LE(max_offdiag(oca_se_tpz), tol);
  }

  // Compare an NCA single-particle Green's function with the dense and block-sparse first-order-only evaluators.
  void check_nca_gf_manual(OneFermionSetup &s, nda::array_const_view<dcomplex, 3> nca_gf) {
    SCOPED_TRACE("manual NCA single-particle gf evaluators");
    auto nca_gf_dense = NCA_gf_dense(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense);
    EXPECT_LE(max_dev(nca_gf, nca_gf_dense), s.eps);
    auto nca_gf_manual = NCA_gf_bs(s.model.G_bdof, reflect_bdof(s.model.G_bdof, s.itops), s.Fq);
    EXPECT_LE(max_dev(nca_gf, nca_gf_manual), s.eps);
  }

  // Compare an OCA single-particle Green's function with the dense and block-sparse second-order-only evaluators.
  void check_oca_gf_manual(OneFermionSetup &s, nda::array_const_view<dcomplex, 3> oca_gf) {
    SCOPED_TRACE("manual OCA single-particle gf evaluators");
    auto oca_gf_dense = OCA_gf_dense(s.D.hyb.coeffs, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
    EXPECT_LE(max_dev(oca_gf, oca_gf_dense), s.eps);
    auto oca_gf_manual = OCA_gf_bs(s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
    EXPECT_LE(max_dev(oca_gf, oca_gf_manual), s.eps);
  }

  // The three DLR-coefficient arrays that the *_gf_tpz routines take: unlike OCA_tpz and third_order_tpz they
  // want coefficients rather than values, and they do not build the reflected hybridization themselves, so the
  // -reflect(hyb) convention that OCA_tpz applies internally is reproduced here.
  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> gf_tpz_coeffs(OneFermionSetup &s) {
    auto hyb = hyb_at_dlr_nodes(s);
    return {s.itops.vals2coefs(hyb), s.itops.vals2coefs(nda::make_regular(-s.itops.reflect(hyb))), s.itops.vals2coefs(s.Gt_dense)};
  }

  // Compare a correlator against a quadrature result on the equispaced grid. Both *_gf_tpz routines run their
  // outer loop over 1 <= i <= n_quad - 1, leaving the two grid endpoints untouched, so those are skipped.
  void expect_matches_tpz(nda::array_const_view<dcomplex, 3> gf, nda::array_const_view<dcomplex, 3> tpz, OneFermionSetup &s, int n_quad, double tol) {
    auto gf_eq    = eval_eq(s.itops, gf, n_quad);
    auto interior = nda::range(1, n_quad);
    EXPECT_LE(max_dev(gf_eq(interior, _, _), tpz(interior, _, _)), tol);
  }

  /**
   * @brief Compare an OCA single-particle Green's function with direct trapezoidal quadrature of the diagram
   *
   * @details The single-particle analogue of check_oca_se_tpz.
   *
   * @note OCA vanishes identically for a single fermion, so in these tests both sides are zero. The
   * agreement was checked separately on the two-fermion model of two_fermion_model_helper, where OCA does
   * not vanish: the two agree to ~2e-4 at n_quad = 20, i.e. to quadrature error.
   */
  void check_oca_gf_tpz(OneFermionSetup &s, nda::array_const_view<dcomplex, 3> oca_gf, int n_quad = 20, double tol = 2.0e-3) {
    SCOPED_TRACE("OCA single-particle gf vs trapezoidal quadrature");
    auto [hyb_coeffs, hyb_refl_coeffs, Gt_coeffs] = gf_tpz_coeffs(s);
    auto oca_gf_tpz                               = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n_quad);
    expect_matches_tpz(oca_gf, oca_gf_tpz, s, n_quad, tol);
  }

  /**
   * @brief Compare a third-order single-particle Green's function with direct trapezoidal quadrature
   *
   * @details Same idea as check_oca_gf_tpz one order up, and the only third-order check here that goes
   * through neither the diagram evaluators nor the tabulated notebook values. Unlike the OCA case this
   * diagram does not vanish for a single fermion, so the comparison has real content: the deviations from
   * the evaluator at n_quad = 20 are 5.7e-4 (constant hybridization), 3.9e-5 (one pole) and 3.2e-4 (two
   * poles), and they fall off as O(dt^2) with n_quad, so the default tolerance is set by the quadrature
   * error rather than by eps.
   */
  void check_third_order_gf_tpz(OneFermionSetup &s, nda::array_const_view<dcomplex, 3> third_order_gf, int n_quad = 20, double tol = 2.0e-3) {
    SCOPED_TRACE("third-order single-particle gf vs trapezoidal quadrature");
    auto [hyb_coeffs, hyb_refl_coeffs, Gt_coeffs] = gf_tpz_coeffs(s);
    auto third_order_gf_tpz_result                = third_order_gf_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n_quad);
    expect_matches_tpz(third_order_gf, third_order_gf_tpz_result, s, n_quad, tol);
  }

} // namespace

/**
 * @brief Test computation of several diagrams for a spinless fermion with a constant hybridization
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams by comparing analytical calculations carried out
 * in examples/one_fermion_analytical_solutions.ipynb to the results of calls to the DiagramEvaluator compute_self_energy routine.
 */
TEST(one_fermion, const_hyb_se) {
  auto s = one_pole_setup(2.0);
  expect_atomic_propagator(s);

  // ----- NCA test -----
  auto nca_se     = check_se_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_se_ana = tau_ref(s.dlr_it, [](double t) { return exp(-t * std::numbers::ln2) / 2; }); // = -G0_ana / 2
  for (int b = 0; b < s.nblocks(); ++b) { expect_diag_close(nca_se, b, nca_se_ana, s); }
  check_nca_se_manual(s, nca_se);

  // ----- OCA test -----
  auto oca_se = check_se_diagram_evaluators(s, max_crossing_topology(2));
  expect_blocks_zero(oca_se, s); // OCA contribution should be identically zero
  check_oca_se_manual(s, oca_se);
  check_oca_se_tpz(s, oca_se);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  auto third_order_se     = check_se_diagram_evaluators(s, max_crossing_topology(3));
  auto third_order_se_ana = tau_ref(s.dlr_it, [beta = s.beta](double t) {
    double bt4 = beta * t;
    bt4        = bt4 * bt4;
    bt4        = bt4 * bt4;
    return bt4 * exp(-t * std::numbers::ln2) / 192.0;
  });
  for (int b = 0; b < s.nblocks(); ++b) { expect_diag_close(third_order_se, b, third_order_se_ana, s); }
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams.
 */
TEST(one_fermion, one_hyb_pole_se) {
  auto s      = one_pole_setup(1.0, 0.8);
  double beta = s.beta;
  double om   = s.model.hyb_poles(0);
  expect_atomic_propagator(s);

  // ----- NCA test -----
  auto nca_se      = check_se_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_se0_ana = tau_ref(s.dlr_it, [&](double t) { return exp(-t * std::numbers::ln2) * exp(t * om) / (exp(beta * om) + 1); });
  auto nca_se1_ana = tau_ref(s.dlr_it, [&](double t) { return exp(-t * std::numbers::ln2) * exp(-t * om) / (exp(-beta * om) + 1); });
  expect_diag_close(nca_se, 0, nca_se0_ana, s);
  expect_diag_close(nca_se, 1, nca_se1_ana, s);
  check_nca_se_manual(s, nca_se);

  // ----- OCA test -----
  auto oca_se = check_se_diagram_evaluators(s, max_crossing_topology(2));
  expect_blocks_zero(oca_se, s);
  check_oca_se_manual(s, oca_se);
  check_oca_se_tpz(s, oca_se);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  auto third_order_se      = check_se_diagram_evaluators(s, max_crossing_topology(3));
  double denom             = om * (exp(beta * om) + 1);
  denom                    = denom * denom * denom;
  denom                    = 2 * om * denom;
  auto third_order_se0_ana = tau_ref(s.dlr_it, [&](double t) {
    double tom = t * om;
    return (tom * tom * exp(tom) - 4 * tom * exp(tom) - 2 * tom + 6 * exp(tom) - 6) * exp(beta * om) * exp(-t * std::numbers::ln2) / denom;
  });
  auto third_order_se1_ana = tau_ref(s.dlr_it, [&](double t) {
    double tom = t * om;
    return (tom * tom + 4 * tom + 2 * (tom - 3) * exp(tom) + 6) * exp(om * (2 * beta - t)) * exp(-t * std::numbers::ln2) / denom;
  });
  expect_diag_close(third_order_se, 0, third_order_se0_ana, s);
  expect_diag_close(third_order_se, 1, third_order_se1_ana, s);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from two poles
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams for a
 * two-pole hybridization Delta(tau) = K(tau, omega_1) + 2*K(tau, omega_2).
 */
TEST(one_fermion, two_hyb_poles_se) {
  auto s      = two_pole_setup(1.0);
  double beta = s.beta;
  double om1  = s.model.hyb_poles(0);
  double om2  = s.model.hyb_poles(1);
  double c1   = s.model.hyb_coeffs(0, 0, 0).real();
  double c2   = s.model.hyb_coeffs(1, 0, 0).real();
  expect_atomic_propagator(s);

  // ----- NCA test -----
  // First order is linear in Delta, so the analytic reference is just the coefficient-weighted sum of
  // the single-pole NCA formula over the two poles.
  auto nca_se      = check_se_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_se0_ana = tau_ref(s.dlr_it, [&](double t) {
    return c1 * exp(-t * std::numbers::ln2) * exp(t * om1) / (exp(beta * om1) + 1)
       + c2 * exp(-t * std::numbers::ln2) * exp(t * om2) / (exp(beta * om2) + 1);
  });
  auto nca_se1_ana = tau_ref(s.dlr_it, [&](double t) {
    return c1 * exp(-t * std::numbers::ln2) * exp(-t * om1) / (exp(-beta * om1) + 1)
       + c2 * exp(-t * std::numbers::ln2) * exp(-t * om2) / (exp(-beta * om2) + 1);
  });
  expect_diag_close(nca_se, 0, nca_se0_ana, s);
  expect_diag_close(nca_se, 1, nca_se1_ana, s);
  check_nca_se_manual(s, nca_se);

  // ----- OCA test -----
  // Still identically zero: the combinatorial argument (creation/annihilation operators must alternate
  // for a single fermion level) doesn't depend on how many poles Delta has.
  auto oca_se = check_se_diagram_evaluators(s, max_crossing_topology(2));
  expect_blocks_zero(oca_se, s);
  check_oca_se_manual(s, oca_se);
  check_oca_se_tpz(s, oca_se);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  auto third_order_se = check_se_diagram_evaluators(s, max_crossing_topology(3));
  auto se00_coeffs    = s.itops.vals2coefs(third_order_se[0].data()(_, 0, 0));
  auto se11_coeffs    = s.itops.vals2coefs(third_order_se[1].data()(_, 0, 0));

  // Reference values for Sigma_00(tau), Sigma_11(tau) at beta=1, omega_1=0.6, omega_2=-0.9, computed from
  // the closed form in one_fermion_two_poles_analytical_solutions.ipynb and cross-validated there against
  // independent brute-force nested quadrature of the undecomposed diagram integral.
  std::vector<double> tau_pts  = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> se00_ref = {0.000014094454748, 0.000890679147681, 0.005485107140040, 0.017223807607182, 0.039435093044707};
  std::vector<double> se11_ref = {0.000010136868438, 0.000699571262738, 0.004714452501594, 0.016214271857205, 0.040647750033025};

  for (size_t k = 0; k < tau_pts.size(); ++k) {
    dcomplex se00_val = s.itops.coefs2eval(se00_coeffs, tau_pts[k]);
    dcomplex se11_val = s.itops.coefs2eval(se11_coeffs, tau_pts[k]);
    ASSERT_LE(std::abs(se00_val - se00_ref[k]), s.eps);
    ASSERT_LE(std::abs(se11_val - se11_ref[k]), s.eps);
  }

  // ----- trapezoidal third-order comparison -----
  // Independent verification of third_order_se, computed by direct trapezoidal quadrature
  // (third_order_tpz, c++/triqs_xca/block_sparse_manual.hpp) of the same topology {{0,3},{1,4},{2,5}}.
  //
  int n_quad              = 20;
  auto third_order_se_tpz = third_order_tpz(hyb_at_dlr_nodes(s), s.itops, beta, s.Gt_dense, s.Fs_dense, n_quad);
  auto third_order_se0_eq = eval_eq(s.itops, third_order_se[0].data(), n_quad);
  auto third_order_se1_eq = eval_eq(s.itops, third_order_se[1].data(), n_quad);

  double tpz_tol = 2.0e-3; // ~2x the empirically observed n_quad=20 error (~9.2e-4)
  ASSERT_LE(max_dev(third_order_se0_eq(_, 0, 0), third_order_se_tpz(_, 0, 0)), tpz_tol);
  ASSERT_LE(max_dev(third_order_se1_eq(_, 0, 0), third_order_se_tpz(_, 1, 1)), tpz_tol);
  ASSERT_LE(max_offdiag(third_order_se_tpz), tpz_tol);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a constant hybridization
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's functions diagrams.
 */
TEST(one_fermion, const_hyb_spgf) {
  auto s = one_pole_setup(2.0);

  // ----- NCA test -----
  auto nca_gf     = check_gf_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_gf_ana = nda::make_regular(nda::ones<dcomplex>(s.r) / 2);
  ASSERT_LE(max_dev(nca_gf(_, 0, 0), nca_gf_ana), s.eps);
  check_nca_gf_manual(s, nca_gf);

  // ----- OCA test -----
  auto oca_gf = check_gf_diagram_evaluators(s, max_crossing_topology(2));
  ASSERT_LE(nda::max_element(nda::abs(oca_gf)), s.eps); // OCA contribution should be identically zero
  check_oca_gf_manual(s, oca_gf);
  check_oca_gf_tpz(s, oca_gf);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the two diagram evaluators are compared here.
  auto third_order_gf     = check_gf_diagram_evaluators(s, max_crossing_topology(3));
  auto third_order_gf_ana = tau_ref(s.dlr_it, [beta = s.beta](double t) {
    double halfbeta   = beta / 2.0;
    double halfbetasq = halfbeta * halfbeta;
    double halfbeta4  = halfbetasq * halfbetasq;
    return halfbeta4 * (1.0 - t) * (1.0 - t) * t * t / 2.0;
  });
  ASSERT_LE(max_dev(third_order_gf(_, 0, 0), third_order_gf_ana), s.eps);
  check_third_order_gf_tpz(s, third_order_gf);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's functions diagrams.
 */
TEST(one_fermion, one_hyb_pole_spgf) {
  auto s = one_pole_setup(1.0, 0.8);

  // ----- NCA test -----
  auto nca_gf     = check_gf_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_gf_ana = nda::make_regular(nda::ones<dcomplex>(s.r) / 2);
  ASSERT_LE(max_dev(nca_gf(_, 0, 0), nca_gf_ana), s.eps);
  check_nca_gf_manual(s, nca_gf);

  // ----- OCA test -----
  auto oca_gf = check_gf_diagram_evaluators(s, max_crossing_topology(2));
  ASSERT_LE(nda::max_element(nda::abs(oca_gf)), s.eps); // OCA contribution should be identically zero
  check_oca_gf_manual(s, oca_gf);
  check_oca_gf_tpz(s, oca_gf);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the two diagram evaluators are compared here.
  auto third_order_gf     = check_gf_diagram_evaluators(s, max_crossing_topology(3));
  double om               = s.model.hyb_poles(0);
  auto third_order_gf_ana = tau_ref(s.dlr_it, [beta = s.beta, om](double t) {
    return (t + (exp(-om * t) - 1.0) / om) * (t - beta + (exp(om * (beta - t)) - 1.0) / om)
       / (2 * om * om * (1 + exp(-beta * om)) * (exp(beta * om) + 1));
  });
  ASSERT_LE(max_dev(third_order_gf(_, 0, 0), third_order_gf_ana), s.eps);
  check_third_order_gf_tpz(s, third_order_gf);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from two poles
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's function
 * diagrams for a two-pole hybridization Delta(tau) = K(tau, omega_1) + 2*K(tau, omega_2), mirroring the
 * two-pole self-energy test `one_fermion.two_hyb_poles_se` above.
 */
TEST(one_fermion, two_hyb_poles_spgf) {
  auto s = two_pole_setup(1.0);

  // ----- NCA test -----
  // NCA is independent of the hybridization (no Delta lines at this order), so the reference is the same
  // constant 1/2 seen in the const-hybridization and one-pole tests above.
  auto nca_gf     = check_gf_diagram_evaluators(s, max_crossing_topology(1));
  auto nca_gf_ana = nda::make_regular(nda::ones<dcomplex>(s.r) / 2);
  ASSERT_LE(max_dev(nca_gf(_, 0, 0), nca_gf_ana), s.eps);
  check_nca_gf_manual(s, nca_gf);

  // ----- OCA test -----
  // Still identically zero: the combinatorial argument (creation/annihilation operators must alternate
  // for a single fermion level) doesn't depend on how many poles Delta has.
  auto oca_gf = check_gf_diagram_evaluators(s, max_crossing_topology(2));
  ASSERT_LE(nda::max_element(nda::abs(oca_gf)), s.eps);
  check_oca_gf_manual(s, oca_gf);
  check_oca_gf_tpz(s, oca_gf);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the two diagram evaluators are compared here.
  auto third_order_gf        = check_gf_diagram_evaluators(s, max_crossing_topology(3));
  auto third_order_gf_coeffs = s.itops.vals2coefs(third_order_gf(_, 0, 0));

  // Reference values for g_3(tau) at beta=1, omega_1=0.6, omega_2=-0.9, computed from the closed form
  // in one_fermion_two_poles_analytical_solutions.ipynb and cross-validated there against independent
  // brute-force nested quadrature of the undecomposed diagram integral.
  std::vector<double> tau_pts = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> gf_ref  = {0.001818250732044, 0.010263216835087, 0.015229427191986, 0.011360586534472, 0.002226904649606};

  for (size_t k = 0; k < tau_pts.size(); ++k) {
    dcomplex gf_val = s.itops.coefs2eval(third_order_gf_coeffs, tau_pts[k]);
    ASSERT_LE(std::abs(gf_val - gf_ref[k]), s.eps);
  }
  check_third_order_gf_tpz(s, third_order_gf);
}
