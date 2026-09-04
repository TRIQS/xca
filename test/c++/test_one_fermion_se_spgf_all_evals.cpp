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
using triqs_xca::atom_diag::get_tensor_in_full_hilbert_space;

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
 * When comparisons are carried out, block-sparse results are converted to dense formats so that all outputs are the same type and shape.
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
 * Every test uses the maximally-crossing topology at each order, in which hybridization line i joins vertices i and i + order.
 *
 * Both diagram evaluators multiply each backbone by the fermionic permutation parity of its topology (Backbone::get_parity), while the analytic
 * references, the manual N/OCA routines and the trapezoidal routines carry no such factor. Every evaluator result is therefore multiplied by
 * topology_parity to divide that sign back out, which is what the "ensure sign is correct" comments in the compute sections refer to. This is the
 * same cancellation as the "Cancel topology sign, accounted for in dense diag eval" negations in test_block_sparse_backbone_eval.cpp, but for
 * arbitrary order rather than hardcoded for OCA: the parity is +1 at first order and -1 at second and third.
 */

namespace {

  // ---------------------------------------------------------------------------------------------------
  // Model construction
  // ---------------------------------------------------------------------------------------------------

  /**
   * @brief One-fermion model (trivial atomic Hamiltonian H = 0, one orbital) with the hybridization
   * Delta(tau) = sum_k coeffs[k] K(tau, poles[k])
   *
   * @details Same atom as the one-fermion helper in block_sparse_utils.cpp; that helper only builds
   * single-pole hybridizations, so the general sum over poles is assembled here.
   *
   * @param[in] poles Hybridization poles omega_k
   * @param[in] coeffs Coefficients c_k, one per pole
   */
  FermionModelData one_fermion_model(double beta, double Lambda, double eps, std::vector<double> const &poles, std::vector<dcomplex> const &coeffs) {
    int p    = poles.size();
    int norb = 1;
    nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
    nda::vector<double> hyb_poles(p);
    for (int k = 0; k < p; ++k) {
      hyb_coeffs(k, 0, 0) = coeffs[k];
      hyb_poles(k)        = poles[k];
    }

    // H = 0: a single spinless level at mu = 0
    double mu = 0.0;
    many_body_operator_complex H;
    H = -mu * n("0", 0);

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
   *
   * Each test constructs one directly from its own beta, Lambda = 20 beta and eps, so that those values
   * stay in scope for the analytic references and tolerances in the test body.
   *
   * The Hilbert space is two-dimensional and every atom_diag subspace is one-dimensional -- subspace 0 holds
   * the empty level, subspace 1 the occupied one -- so every self-energy here is a diagonal 2x2 matrix in the
   * Fock basis, and the analytic references in the tests are written directly in that layout.
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

    /**
     * @brief Build the DLR grid, the model and every dense fixture from the physical parameters
     * @param[in] poles Hybridization poles omega_k
     * @param[in] coeffs Coefficients c_k in Delta(tau) = sum_k c_k K(tau, omega_k)
     */
    OneFermionSetup(double beta_, double Lambda_, double eps_, std::vector<double> const &poles, std::vector<dcomplex> const &coeffs)
       : beta(beta_),
         Lambda(Lambda_),
         eps(eps_),
         itops(Lambda_, build_dlr_rf(Lambda_, eps_)),
         dlr_it(itops.get_itnodes()),
         r(itops.rank()),
         model(one_fermion_model(beta_, Lambda_, eps_, poles, coeffs)),
         Gt_dense(make_dense_propagator(model.ad, beta_, dlr_it)),
         Gt_dense_refl(itops.reflect(Gt_dense)),
         G0_ppsc_dense(wrap_dense(model.G_ppsc[0].mesh(), Gt_dense)),
         D_dense(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         D(model.hyb_poles, model.hyb_coeffs, model.G_ppsc[0].mesh(), model.ad),
         Fq(make_field_operators(model.ad, model.hyb_coeffs)) {
      auto [Fs, F_dags] = get_operators_dense(model.ad);
      Fs_dense          = std::move(Fs);
      F_dags_dense      = std::move(F_dags);
    }

    int nblocks() const { return model.G_bdof.get_num_block_cols(); }

    private:
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

} // namespace

/**
 * @brief Test computation of several diagrams for a spinless fermion with a constant hybridization
 * * Every test differs only in its hybridization and in its analytic reference values. The model setup is shared through the OneFermionSetup bundle,
 * but each test body then runs every evaluator itself and holds all of its own comparisons, so that the list of routines being cross-checked at each
 * order is visible in the test rather than behind a helper. Each order reads the same way: compute with every method, compute the analytic
 * reference, then compare.
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams by comparing analytical calculations carried out
 * in examples/one_fermion_analytical_solutions.ipynb to the results of calls to the DiagramEvaluator compute_self_energy routine.
 */
TEST(one_fermion, const_hyb_se) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  // Delta(tau) = K(tau, om), which is tau-independent at om = 0
  double om = 0.0;
  double c  = 1.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  OneFermionSetup s{beta, Lambda, eps, /*poles=*/{om}, /*coeffs=*/{c}};

  // Before testing self-energy contributions, check atomic propagator is correct
  auto G0_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  G0_ana(_, 0, 0) = tau_ref(s.dlr_it, [](double t) { return -exp(-t * std::numbers::ln2); }); // the atom has H = 0
  G0_ana(_, 1, 1) = G0_ana(_, 0, 0);                                                          // same in both sectors
  EXPECT_LE(nda::max_element(nda::abs(get_tensor_in_full_hilbert_space(s.model.G_bdof, s.model.ad) - G0_ana)), eps);

  // parameters for trapezoidal routines
  int n_quad     = 20;
  double tpz_tol = 2.0e-3; // ~2x the empirically observed n_quad=20 error (~9.2e-4)
  // the itops overload of coefs2vals; the beta/Lambda/eps one hardcodes a symmetrized grid internally
  auto hyb = triqs_xca::hyb::coefs2vals(s.beta, s.itops, s.model.hyb_coeffs, s.model.hyb_poles);

  // ----- NCA -----
  nda::array<int, 2> nca_topology = {{0, 1}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, nca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto nca_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, nca_topology);
  auto nca_dense    = nda::make_regular(topology_parity(nca_topology) * nca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto nca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, nca_topology);
  auto nca_pairs    = nda::make_regular(topology_parity(nca_topology) * nca_pairs_gf[0].data());
  // manual dense routine
  auto nca_manual_dense = NCA_dense(s.D.hyb.values, s.D.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, convert to dense format
  auto nca_manual_bdof = NCA_bs(s.D.hyb.values, s.D.hyb.values_reflect, s.model.G_bdof, s.Fq);
  auto nca_manual_bs   = get_tensor_in_full_hilbert_space(nca_manual_bdof, s.model.ad, s.r);

  // compute analytical reference
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  nca_ana(_, 0, 0) = tau_ref(s.dlr_it, [](double t) { return exp(-t * std::numbers::ln2) / 2; }); // = -G0_ana / 2
  nca_ana(_, 1, 1) = nca_ana(_, 0, 0);                                                            // same in both sectors

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_pairs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  nda::array<int, 2> oca_topology = {{0, 2}, {1, 3}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, oca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto oca_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, oca_topology);
  auto oca_dense    = nda::make_regular(topology_parity(oca_topology) * oca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto oca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, oca_topology);
  auto oca_pairs    = nda::make_regular(topology_parity(oca_topology) * oca_pairs_gf[0].data());
  // manual dense routine
  auto oca_manual_dense = OCA_dense(s.D.hyb.values, s.D.hyb.coeffs, s.D.hyb.values_reflect, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta,
                                    s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, and convert to dense format
  auto oca_manual_bdof = OCA_bs(s.D.hyb.values, s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  auto oca_manual_bs   = get_tensor_in_full_hilbert_space(oca_manual_bdof, s.model.ad, s.r);
  // compute using trapezoidal quadrature
  auto oca_tpz   = OCA_tpz(hyb, s.itops, s.beta, s.Gt_dense, s.Fs_dense, n_quad);
  auto oca_bs_eq = eval_eq(s.itops, oca_bs, n_quad);

  // no analytical reference here because the expected result is zero

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_dense - oca_pairs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_bs)), eps);
  EXPECT_LE(max_offdiag(oca_dense), eps);
  // oca_bs is exactly diagonal, so comparing whole tensors also bounds the quadrature result's off-diagonals
  EXPECT_LE(nda::max_element(nda::abs(oca_bs_eq - oca_tpz)), tpz_tol);

  // ----- third order -----
  nda::array<int, 2> third_topology = {{0, 3}, {1, 4}, {2, 5}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology)
                                    * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, third_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto third_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, third_topology);
  auto third_dense    = nda::make_regular(topology_parity(third_topology) * third_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto third_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, third_topology);
  auto third_pairs    = nda::make_regular(topology_parity(third_topology) * third_pairs_gf[0].data());
  // no manual third-order routine

  // compute analytical reference
  auto third_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  third_ana(_, 0, 0) = tau_ref(s.dlr_it, [beta](double t) {
    double bt4 = beta * t;
    bt4        = bt4 * bt4;
    bt4        = bt4 * bt4;
    return bt4 * exp(-t * std::numbers::ln2) / 192.0;
  });
  third_ana(_, 1, 1) = third_ana(_, 0, 0); // same in both sectors

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_pairs - third_ana)), eps);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams.
 */
TEST(one_fermion, one_hyb_pole_se) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  // Delta(tau) = c K(tau, om)
  double om = 0.8;
  double c  = 1.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  OneFermionSetup s{beta, Lambda, eps, /*poles=*/{om}, /*coeffs=*/{c}};

  // Before testing self-energy contributions, check atomic propagator is correct
  auto G0_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  G0_ana(_, 0, 0) = tau_ref(s.dlr_it, [](double t) { return -exp(-t * std::numbers::ln2); }); // the atom has H = 0
  G0_ana(_, 1, 1) = G0_ana(_, 0, 0);                                                          // same in both sectors
  EXPECT_LE(nda::max_element(nda::abs(get_tensor_in_full_hilbert_space(s.model.G_bdof, s.model.ad) - G0_ana)), eps);

  // parameters for trapezoidal routines
  int n_quad     = 20;
  double tpz_tol = 2.0e-3; // ~2x the empirically observed n_quad=20 error (~9.2e-4)
  // the itops overload of coefs2vals; the beta/Lambda/eps one hardcodes a symmetrized grid internally
  auto hyb = triqs_xca::hyb::coefs2vals(s.beta, s.itops, s.model.hyb_coeffs, s.model.hyb_poles);

  // ----- NCA -----
  nda::array<int, 2> nca_topology = {{0, 1}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, nca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto nca_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, nca_topology);
  auto nca_dense    = nda::make_regular(topology_parity(nca_topology) * nca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto nca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, nca_topology);
  auto nca_pairs    = nda::make_regular(topology_parity(nca_topology) * nca_pairs_gf[0].data());
  // manual dense routine
  auto nca_manual_dense = NCA_dense(s.D.hyb.values, s.D.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, convert to dense format
  auto nca_manual_bdof = NCA_bs(s.D.hyb.values, s.D.hyb.values_reflect, s.model.G_bdof, s.Fq);
  auto nca_manual_bs   = get_tensor_in_full_hilbert_space(nca_manual_bdof, s.model.ad, s.r);

  // compute analytical reference
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  nca_ana(_, 0, 0) = tau_ref(s.dlr_it, [&](double t) { return exp(-t * std::numbers::ln2) * exp(t * om) / (exp(beta * om) + 1); });
  nca_ana(_, 1, 1) = tau_ref(s.dlr_it, [&](double t) { return exp(-t * std::numbers::ln2) * exp(-t * om) / (exp(-beta * om) + 1); });

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_pairs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  nda::array<int, 2> oca_topology = {{0, 2}, {1, 3}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, oca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto oca_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, oca_topology);
  auto oca_dense    = nda::make_regular(topology_parity(oca_topology) * oca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto oca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, oca_topology);
  auto oca_pairs    = nda::make_regular(topology_parity(oca_topology) * oca_pairs_gf[0].data());
  // manual dense routine
  auto oca_manual_dense = OCA_dense(s.D.hyb.values, s.D.hyb.coeffs, s.D.hyb.values_reflect, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta,
                                    s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, and convert to dense format
  auto oca_manual_bdof = OCA_bs(s.D.hyb.values, s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  auto oca_manual_bs   = get_tensor_in_full_hilbert_space(oca_manual_bdof, s.model.ad, s.r);
  // compute using trapezoidal quadrature
  auto oca_tpz   = OCA_tpz(hyb, s.itops, s.beta, s.Gt_dense, s.Fs_dense, n_quad);
  auto oca_bs_eq = eval_eq(s.itops, oca_bs, n_quad);

  // no analytical reference here because the expected result is zero

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_dense - oca_pairs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_bs)), eps);
  EXPECT_LE(max_offdiag(oca_dense), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs_eq - oca_tpz)), tpz_tol);

  // ----- third order -----
  nda::array<int, 2> third_topology = {{0, 3}, {1, 4}, {2, 5}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology)
                                    * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, third_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto third_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, third_topology);
  auto third_dense    = nda::make_regular(topology_parity(third_topology) * third_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto third_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, third_topology);
  auto third_pairs    = nda::make_regular(topology_parity(third_topology) * third_pairs_gf[0].data());
  // no manual third-order routine

  // compute analytical reference
  double denom       = om * (exp(beta * om) + 1);
  denom              = denom * denom * denom;
  denom              = 2 * om * denom;
  auto third_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  third_ana(_, 0, 0) = tau_ref(s.dlr_it, [&](double t) {
    double tom = t * om;
    return (tom * tom * exp(tom) - 4 * tom * exp(tom) - 2 * tom + 6 * exp(tom) - 6) * exp(beta * om) * exp(-t * std::numbers::ln2) / denom;
  });
  third_ana(_, 1, 1) = tau_ref(s.dlr_it, [&](double t) {
    double tom = t * om;
    return (tom * tom + 4 * tom + 2 * (tom - 3) * exp(tom) + 6) * exp(om * (2 * beta - t)) * exp(-t * std::numbers::ln2) / denom;
  });

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_pairs - third_ana)), eps);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from two poles
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams for a
 * two-pole hybridization Delta(tau) = K(tau, omega_1) + 2*K(tau, omega_2).
 */
TEST(one_fermion, two_hyb_poles_se) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  // Delta(tau) = c1 K(tau, om1) + c2 K(tau, om2)
  double om1 = 0.6;
  double om2 = -0.9;
  double c1  = 1.0;
  double c2  = 2.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  OneFermionSetup s{beta, Lambda, eps, /*poles=*/{om1, om2}, /*coeffs=*/{c1, c2}};

  // Before testing self-energy contributions, check atomic propagator is correct
  auto G0_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  G0_ana(_, 0, 0) = tau_ref(s.dlr_it, [](double t) { return -exp(-t * std::numbers::ln2); }); // the atom has H = 0
  G0_ana(_, 1, 1) = G0_ana(_, 0, 0);                                                          // same in both sectors
  EXPECT_LE(nda::max_element(nda::abs(get_tensor_in_full_hilbert_space(s.model.G_bdof, s.model.ad) - G0_ana)), eps);

  // parameters for trapezoidal routines
  int n_quad     = 20;
  double tpz_tol = 2.0e-3; // ~2x the empirically observed n_quad=20 error (~9.2e-4)
  // the itops overload of coefs2vals; the beta/Lambda/eps one hardcodes a symmetrized grid internally
  auto hyb = triqs_xca::hyb::coefs2vals(s.beta, s.itops, s.model.hyb_coeffs, s.model.hyb_poles);

  // ----- NCA -----
  nda::array<int, 2> nca_topology = {{0, 1}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, nca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto nca_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, nca_topology);
  auto nca_dense    = nda::make_regular(topology_parity(nca_topology) * nca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto nca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, nca_topology);
  auto nca_pairs    = nda::make_regular(topology_parity(nca_topology) * nca_pairs_gf[0].data());
  // manual dense routine
  auto nca_manual_dense = NCA_dense(s.D.hyb.values, s.D.hyb.values_reflect, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, convert to dense format
  auto nca_manual_bdof = NCA_bs(s.D.hyb.values, s.D.hyb.values_reflect, s.model.G_bdof, s.Fq);
  auto nca_manual_bs   = get_tensor_in_full_hilbert_space(nca_manual_bdof, s.model.ad, s.r);

  // compute analytical reference
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 2, 2);
  nca_ana(_, 0, 0) = tau_ref(s.dlr_it, [&](double t) {
    return c1 * exp(-t * std::numbers::ln2) * exp(t * om1) / (exp(beta * om1) + 1)
       + c2 * exp(-t * std::numbers::ln2) * exp(t * om2) / (exp(beta * om2) + 1);
  });
  nca_ana(_, 1, 1) = tau_ref(s.dlr_it, [&](double t) {
    return c1 * exp(-t * std::numbers::ln2) * exp(-t * om1) / (exp(-beta * om1) + 1)
       + c2 * exp(-t * std::numbers::ln2) * exp(-t * om2) / (exp(-beta * om2) + 1);
  });

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_pairs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  nda::array<int, 2> oca_topology = {{0, 2}, {1, 3}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology)
                                  * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, oca_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto oca_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, oca_topology);
  auto oca_dense    = nda::make_regular(topology_parity(oca_topology) * oca_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto oca_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, oca_topology);
  auto oca_pairs    = nda::make_regular(topology_parity(oca_topology) * oca_pairs_gf[0].data());
  // manual dense routine
  auto oca_manual_dense = OCA_dense(s.D.hyb.values, s.D.hyb.coeffs, s.D.hyb.values_reflect, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta,
                                    s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine, and convert to dense format
  auto oca_manual_bdof = OCA_bs(s.D.hyb.values, s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  auto oca_manual_bs   = get_tensor_in_full_hilbert_space(oca_manual_bdof, s.model.ad, s.r);
  // compute using trapezoidal quadrature
  auto oca_tpz   = OCA_tpz(hyb, s.itops, s.beta, s.Gt_dense, s.Fs_dense, n_quad);
  auto oca_bs_eq = eval_eq(s.itops, oca_bs, n_quad);

  // no analytical reference here because the expected result is zero

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_dense - oca_pairs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_bs)), eps);
  EXPECT_LE(max_offdiag(oca_dense), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs_eq - oca_tpz)), tpz_tol);

  // ----- third order -----
  nda::array<int, 2> third_topology = {{0, 3}, {1, 4}, {2, 5}};
  // compute using DiagramEvaluator, convert block-sparse result to dense format, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology)
                                    * get_tensor_in_full_hilbert_space(s.D.compute_self_energy(s.model.G_ppsc, third_topology), s.model.ad));
  // compute using DenseDiagramEvaluator, and ensure sign is correct
  auto third_dense_gf = s.D_dense.compute_self_energy(s.G0_ppsc_dense, third_topology);
  auto third_dense    = nda::make_regular(topology_parity(third_topology) * third_dense_gf[0].data());
  // DenseDiagramEvaluator with pairs optimization, which is what the Python wrappers actually use, and ensure sign is correct
  auto third_pairs_gf = s.D_dense.compute_self_energy_by_pairs(s.G0_ppsc_dense, third_topology);
  auto third_pairs    = nda::make_regular(topology_parity(third_topology) * third_pairs_gf[0].data());
  // no manual third-order routine
  // compute using trapezoidal rule
  auto third_tpz = third_order_tpz(hyb, s.itops, s.beta, s.Gt_dense, s.Fs_dense, n_quad);
  // evaluate block-sparse results on equispaced grid to compare against the output of the trapezoidal routine above
  auto third_bs_eq = eval_eq(s.itops, third_bs, n_quad);

  // analytical reference: computed from the closed form in one_fermion_two_poles_analytical_solutions.ipynb and cross-validated there against
  // brute-force nested quadrature of the undecomposed diagram integral
  std::vector<double> tau_pts  = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> se00_ref = {0.000014094454748, 0.000890679147681, 0.005485107140040, 0.017223807607182, 0.039435093044707};
  std::vector<double> se11_ref = {0.000010136868438, 0.000699571262738, 0.004714452501594, 0.016214271857205, 0.040647750033025};

  // compare
  auto se00_coeffs = s.itops.vals2coefs(third_bs(_, 0, 0));
  auto se11_coeffs = s.itops.vals2coefs(third_bs(_, 1, 1));
  for (size_t k = 0; k < tau_pts.size(); ++k) {
    ASSERT_LE(std::abs(s.itops.coefs2eval(se00_coeffs, tau_pts[k]) - se00_ref[k]), eps);
    ASSERT_LE(std::abs(s.itops.coefs2eval(se11_coeffs, tau_pts[k]) - se11_ref[k]), eps);
  }
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_pairs)), eps);
  EXPECT_LE(max_offdiag(third_dense), eps);
  // third_bs is exactly diagonal, so comparing whole tensors also bounds the quadrature result's off-diagonals
  EXPECT_LE(nda::max_element(nda::abs(third_bs_eq - third_tpz)), tpz_tol);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a constant hybridization
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's functions diagrams.
 */
TEST(one_fermion, const_hyb_spgf) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  // Delta(tau) = K(tau, om), which is tau-independent at om = 0
  double om = 0.0;
  double c  = 1.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  OneFermionSetup s{beta, Lambda, eps, /*poles=*/{om}, /*coeffs=*/{c}};

  // parameters for trapezoidal routines
  int n_quad     = 20;
  double tpz_tol = 2.0e-3; // ~2x the empirically observed n_quad=20 error
  // the itops overload of coefs2vals; the beta/Lambda/eps one hardcodes a symmetrized grid internally
  auto hyb = triqs_xca::hyb::coefs2vals(s.beta, s.itops, s.model.hyb_coeffs, s.model.hyb_poles);
  // unlike OCA_tpz and third_order_tpz, the *_gf_tpz routines want DLR coefficients rather than values, and
  // they do not build the reflected hybridization themselves, so the -reflect(hyb) convention is applied here
  auto hyb_coeffs      = s.itops.vals2coefs(hyb);
  auto hyb_refl_coeffs = s.itops.vals2coefs(nda::make_regular(-s.itops.reflect(hyb)));
  auto Gt_coeffs       = s.itops.vals2coefs(s.Gt_dense);

  // ----- NCA -----
  nda::array<int, 2> nca_topology = {{0, 1}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, nca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto nca_dense = nda::make_regular(topology_parity(nca_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, nca_topology));
  // manual dense routine
  auto nca_manual_dense = NCA_gf_dense(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto nca_manual_bs = NCA_gf_bs(s.model.G_bdof, s.model.G_bdof.reflect(s.itops), s.Fq);

  // compute analytical reference: first order has no hybridization line, so it is the constant 1/2
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 1, 1);
  nca_ana(_, 0, 0) = 0.5;

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  nda::array<int, 2> oca_topology = {{0, 2}, {1, 3}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, oca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto oca_dense = nda::make_regular(topology_parity(oca_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, oca_topology));
  // manual dense routine
  auto oca_manual_dense = OCA_gf_dense(s.D.hyb.coeffs, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto oca_manual_bs = OCA_gf_bs(s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  // compute using trapezoidal quadrature
  auto oca_tpz   = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n_quad);
  auto oca_bs_eq = eval_eq(s.itops, oca_bs, n_quad);

  // no analytical reference here because the expected result is zero

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_bs)), eps);
  // both *_gf_tpz routines run their outer loop over 1 <= i <= n_quad - 1, so both grid endpoints are skipped
  EXPECT_LE(nda::max_element(nda::abs(oca_bs_eq - oca_tpz)), tpz_tol);

  // ----- third order -----
  nda::array<int, 2> third_topology = {{0, 3}, {1, 4}, {2, 5}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, third_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto third_dense = nda::make_regular(topology_parity(third_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, third_topology));
  // no manual third-order routine
  // compute using trapezoidal quadrature
  auto third_tpz   = third_order_gf_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n_quad);
  auto third_bs_eq = eval_eq(s.itops, third_bs, n_quad);

  // compute analytical reference
  auto third_ana     = nda::zeros<dcomplex>(s.r, 1, 1);
  third_ana(_, 0, 0) = tau_ref(s.dlr_it, [beta](double t) {
    double halfbeta   = beta / 2.0;
    double halfbetasq = halfbeta * halfbeta;
    double halfbeta4  = halfbetasq * halfbetasq;
    return halfbeta4 * (1.0 - t) * (1.0 - t) * t * t / 2.0;
  });

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_bs_eq - third_tpz)), tpz_tol);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's functions diagrams.
 */
TEST(one_fermion, one_hyb_pole_spgf) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  // Delta(tau) = c K(tau, om)
  double om = 0.8;
  double c  = 1.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  OneFermionSetup s{beta, Lambda, eps, /*poles=*/{om}, /*coeffs=*/{c}};

  // parameters for trapezoidal routines
  int n_quad     = 20;
  double tpz_tol = 2.0e-3; // ~2x the empirically observed n_quad=20 error
  // the itops overload of coefs2vals; the beta/Lambda/eps one hardcodes a symmetrized grid internally
  auto hyb = triqs_xca::hyb::coefs2vals(s.beta, s.itops, s.model.hyb_coeffs, s.model.hyb_poles);
  // unlike OCA_tpz and third_order_tpz, the *_gf_tpz routines want DLR coefficients rather than values, and
  // they do not build the reflected hybridization themselves, so the -reflect(hyb) convention is applied here
  auto hyb_coeffs      = s.itops.vals2coefs(hyb);
  auto hyb_refl_coeffs = s.itops.vals2coefs(nda::make_regular(-s.itops.reflect(hyb)));
  auto Gt_coeffs       = s.itops.vals2coefs(s.Gt_dense);

  // ----- NCA -----
  nda::array<int, 2> nca_topology = {{0, 1}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, nca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto nca_dense = nda::make_regular(topology_parity(nca_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, nca_topology));
  // manual dense routine
  auto nca_manual_dense = NCA_gf_dense(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto nca_manual_bs = NCA_gf_bs(s.model.G_bdof, s.model.G_bdof.reflect(s.itops), s.Fq);

  // compute analytical reference: first order has no hybridization line, so it is the constant 1/2
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 1, 1);
  nca_ana(_, 0, 0) = 0.5;

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  nda::array<int, 2> oca_topology = {{0, 2}, {1, 3}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, oca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto oca_dense = nda::make_regular(topology_parity(oca_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, oca_topology));
  // manual dense routine
  auto oca_manual_dense = OCA_gf_dense(s.D.hyb.coeffs, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto oca_manual_bs = OCA_gf_bs(s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  // compute using trapezoidal quadrature
  auto oca_tpz   = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n_quad);
  auto oca_bs_eq = eval_eq(s.itops, oca_bs, n_quad);

  // no analytical reference here because the expected result is zero

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_bs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs_eq - oca_tpz)), tpz_tol);

  // ----- third order -----
  nda::array<int, 2> third_topology = {{0, 3}, {1, 4}, {2, 5}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, third_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto third_dense = nda::make_regular(topology_parity(third_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, third_topology));
  // no manual third-order routine
  // compute using trapezoidal quadrature
  auto third_tpz   = third_order_gf_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n_quad);
  auto third_bs_eq = eval_eq(s.itops, third_bs, n_quad);

  // compute analytical reference
  auto third_ana     = nda::zeros<dcomplex>(s.r, 1, 1);
  third_ana(_, 0, 0) = tau_ref(s.dlr_it, [beta, om](double t) {
    return (t + (exp(-om * t) - 1.0) / om) * (t - beta + (exp(om * (beta - t)) - 1.0) / om)
       / (2 * om * om * (1 + exp(-beta * om)) * (exp(beta * om) + 1));
  });

  // compare
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_dense - third_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_bs_eq - third_tpz)), tpz_tol);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from two poles
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's function
 * diagrams for a two-pole hybridization Delta(tau) = K(tau, omega_1) + 2*K(tau, omega_2), mirroring the
 * two-pole self-energy test `one_fermion.two_hyb_poles_se` above.
 */
TEST(one_fermion, two_hyb_poles_spgf) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  // Delta(tau) = c1 K(tau, om1) + c2 K(tau, om2)
  double om1 = 0.6;
  double om2 = -0.9;
  double c1  = 1.0;
  double c2  = 2.0;
  // Create a struct for generating propagators, operators, diagram evaluators, atom_diag object used in tests
  OneFermionSetup s{beta, Lambda, eps, /*poles=*/{om1, om2}, /*coeffs=*/{c1, c2}};

  // parameters for trapezoidal routines
  int n_quad     = 20;
  double tpz_tol = 2.0e-3; // ~2x the empirically observed n_quad=20 error
  // the itops overload of coefs2vals; the beta/Lambda/eps one hardcodes a symmetrized grid internally
  auto hyb = triqs_xca::hyb::coefs2vals(s.beta, s.itops, s.model.hyb_coeffs, s.model.hyb_poles);
  // unlike OCA_tpz and third_order_tpz, the *_gf_tpz routines want DLR coefficients rather than values, and
  // they do not build the reflected hybridization themselves, so the -reflect(hyb) convention is applied here
  auto hyb_coeffs      = s.itops.vals2coefs(hyb);
  auto hyb_refl_coeffs = s.itops.vals2coefs(nda::make_regular(-s.itops.reflect(hyb)));
  auto Gt_coeffs       = s.itops.vals2coefs(s.Gt_dense);

  // ----- NCA -----
  nda::array<int, 2> nca_topology = {{0, 1}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto nca_bs = nda::make_regular(topology_parity(nca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, nca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto nca_dense = nda::make_regular(topology_parity(nca_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, nca_topology));
  // manual dense routine
  auto nca_manual_dense = NCA_gf_dense(s.Gt_dense, s.Gt_dense_refl, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto nca_manual_bs = NCA_gf_bs(s.model.G_bdof, s.model.G_bdof.reflect(s.itops), s.Fq);

  // compute analytical reference: first order has no hybridization line, so it does not see Delta at all and
  // is the same constant 1/2 as in the const-hybridization and one-pole tests above
  auto nca_ana     = nda::zeros<dcomplex>(s.r, 1, 1);
  nca_ana(_, 0, 0) = 0.5;

  // compare
  EXPECT_LE(nda::max_element(nda::abs(nca_bs - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_dense - nca_ana)), eps);
  EXPECT_LE(nda::max_element(nda::abs(nca_manual_bs - nca_ana)), eps);

  // ----- OCA -----
  nda::array<int, 2> oca_topology = {{0, 2}, {1, 3}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto oca_bs = nda::make_regular(topology_parity(oca_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, oca_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto oca_dense = nda::make_regular(topology_parity(oca_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, oca_topology));
  // manual dense routine
  auto oca_manual_dense = OCA_gf_dense(s.D.hyb.coeffs, s.D.hyb.coeffs, s.D.hyb.poles, s.itops, s.beta, s.Gt_dense, s.Fs_dense, s.F_dags_dense);
  // manual block-sparse routine
  auto oca_manual_bs = OCA_gf_bs(s.D.hyb.poles, s.itops, s.beta, s.model.G_bdof, s.Fq);
  // compute using trapezoidal quadrature
  auto oca_tpz   = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n_quad);
  auto oca_bs_eq = eval_eq(s.itops, oca_bs, n_quad);

  // no analytical reference here because the expected result is zero

  // compare
  EXPECT_LE(nda::max_element(nda::abs(oca_bs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs - oca_manual_bs)), eps);
  EXPECT_LE(nda::max_element(nda::abs(oca_bs_eq - oca_tpz)), tpz_tol);

  // ----- third order -----
  nda::array<int, 2> third_topology = {{0, 3}, {1, 4}, {2, 5}};
  // compute using DiagramEvaluator, and ensure sign is correct
  auto third_bs = nda::make_regular(topology_parity(third_topology) * s.D.compute_single_ptcle_gf(s.model.G_ppsc, third_topology));
  // compute using DenseDiagramEvaluator, and ensure sign is correct; there is no by-pairs analogue for the single-particle Green's function
  auto third_dense = nda::make_regular(topology_parity(third_topology) * s.D_dense.compute_single_ptcle_gf(s.G0_ppsc_dense, third_topology));
  // no manual third-order routine
  // compute using trapezoidal quadrature
  auto third_tpz   = third_order_gf_tpz(hyb_coeffs, hyb_refl_coeffs, s.itops, s.beta, Gt_coeffs, s.Fs_dense, n_quad);
  auto third_bs_eq = eval_eq(s.itops, third_bs, n_quad);

  // analytical reference: computed from the closed form in one_fermion_two_poles_analytical_solutions.ipynb and
  // cross-validated there against brute-force nested quadrature of the undecomposed diagram integral
  std::vector<double> tau_pts = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> gf_ref  = {0.001818250732044, 0.010263216835087, 0.015229427191986, 0.011360586534472, 0.002226904649606};

  // compare
  auto third_bs_coeffs = s.itops.vals2coefs(third_bs(_, 0, 0));
  for (size_t k = 0; k < tau_pts.size(); ++k) { ASSERT_LE(std::abs(s.itops.coefs2eval(third_bs_coeffs, tau_pts[k]) - gf_ref[k]), eps); }
  EXPECT_LE(nda::max_element(nda::abs(third_bs - third_dense)), eps);
  EXPECT_LE(nda::max_element(nda::abs(third_bs_eq - third_tpz)), tpz_tol);
}
