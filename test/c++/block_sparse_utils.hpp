#pragma once

#include <algorithm>

#include <nda/algorithms.hpp>
#include <triqs_xca/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse.hpp>

using nda::dcomplex;

using triqs_xca::block_sparse::BlockDiagOpFun;
using triqs_xca::block_sparse::BlockOp;
using triqs_xca::block_sparse::BlockOpSymQuartet;

struct FermionModelData {
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::vector<double> hyb_poles;
    triqs::atom_diag::atom_diag<true> ad;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_ppsc;
    BlockDiagOpFun G_bdof;
};

struct DenseFermionModelData {
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::vector<double> hyb_poles;
    triqs::atom_diag::atom_diag<true> ad;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_ppsc_dense;
    triqs_xca::dense::DenseFSet Fset_dense;
};

/**
 * @brief Helper function for setting up the one-fermion test model with trivial atomic Hamiltonian H = 0
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] hyb_pole Pole value for the single-pole hybridization decomposition
 * @return FermionModelData containing hybridization coefficients/poles, atom_diag object, and non-interacting propagators
 */
FermionModelData one_fermion_model_helper(double beta, double Lambda, double eps, double hyb_pole = 0.0);

/**
 * @brief Helper function for setting up a two-fermion model with interaction U * n0 * n1 and one-pole hybridization
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] U Interaction strength
 * @param[in] mu Chemical potential
 * @param[in] hyb_pole Pole value for the single-pole hybridization decomposition
 * @return FermionModelData containing hybridization coefficients/poles, atom_diag object, and non-interacting propagator
 */
FermionModelData two_fermion_model_helper(double beta, double Lambda, double eps, double U = 3.0, double mu = 0.0,
                                          double hyb_pole = -1.5);

/**
 * @brief Helper function for setting up the one-fermion test model with trivial atomic Hamiltonian H = 0, using dense operator storage
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] hyb_pole Pole value for the single-pole hybridization decomposition
 * @return DenseFermionModelData containing hybridization coefficients/poles, atom_diag object, and non-interacting propagators in dense storage
 */
DenseFermionModelData one_fermion_model_dense_helper(double beta, double Lambda, double eps, double hyb_pole = 0.0);

/**
 * @brief Convert a Hamiltonian matrix to a non-interacting Green's function matrix in dense storage
 * @param[in] Hmat Hamiltonian matrix
 * @param[in] beta Inverse temperature
 * @param[in] dlr_it_abs DLR imaginary time nodes in absolute format
 * @return Non-interacting Green's function matrix in dense storage
 */
nda::array<dcomplex, 3> Hmat_to_Gtmat(nda::array<dcomplex, 2> Hmat, double beta, nda::array<double, 1> dlr_it_abs);

/**
 * @brief Helper function for setting up the discrete bath hybridization function used in two-band tests
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @return Tuple of hybridization function and its reflection
 */
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> discrete_bath_helper(double beta, double Lambda, double eps);

/**
 * @brief Helper function for setting up the discrete bath hybridization function used in spin-flip-fermion tests
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] n 2 * number of orbitals
 * @return Tuple of hybridization function and its reflection
 */
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> discrete_bath_spin_flip_helper(double beta, double Lambda, double eps, int n);

/**
 * @brief Helper function for setting up the two-band model's atom_diag object
 * @return triqs::atom_diag::atom_diag<true> object representing the two-band model's atomic Hamiltonian
 */
triqs::atom_diag::atom_diag<true> two_band_atom_diag_helper();

/**
 * @brief Helper function for setting up the spin-flip fermion model's atom_diag object
 *
 * @details The Hamiltonian is H = sum_i [U n_up,i n_do,i + mu (n_up,i + n_do,i) + V (c^dag_up,i c_do,i + c^dag_do,i c_up,i)], whose spin-flip term
 * couples the two spin species on each orbital.
 *
 * @param[in] norb Number of orbitals
 * @param[in] use_particle_number_sym If true, the atom_diag subspaces are labeled by the particle number N, so that all field operators share a
 * single symmetry set; if false, the subspaces come from autopartitioning alone and the field operators are spread over several symmetry sets
 * @param[in] mu Chemical potential
 * @param[in] U Interaction strength
 * @param[in] V Spin-flip amplitude
 * @return triqs::atom_diag::atom_diag<true> object representing the spin-flip model's atomic Hamiltonian
 */
triqs::atom_diag::atom_diag<true> spin_flip_atom_diag_helper(int norb, bool use_particle_number_sym, double mu = 0.25, double U = 1.0,
                                                             double V = 0.1);

/**
 * @brief Helper function for setting up the two-band model in dense storage
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @return Tuple of non-interacting Green's function in dense storage, annihilation operators in dense storage, and creation operators in dense storage
 */
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> two_band_dense_helper(double beta, double Lambda, double eps);

/**
 * @brief Helper function for setting up the two-band model
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] hyb_coeffs Hybridization function coefficients
 * @return Tuple of non-interacting Green's function as a BDOF, field operators as a BlockOpSymQuartet, and vector of block symmetry labels
 */
std::tuple<BlockDiagOpFun, BlockOpSymQuartet, nda::vector<int>> two_band_helper(double beta, double Lambda, double eps,
                                                                                nda::array_const_view<dcomplex, 3> hyb_coeffs);

/**
 * @brief Helper function for making dense objects compatible with a block-sparse evaluation that discards the block structure
 *
 * @details The trivial sparsity pattern is a single block spanning the whole Hilbert space and a single symmetry set holding every flavor, so a
 * block-sparse evaluator built from the result sums the same backbones as a dense one.
 *
 * @param[in] Gt_dense Atomic propagator over the full Hilbert space
 * @param[in] Fs_dense Annihilation operators in dense storage
 * @param[in] F_dags_dense Creation operators in dense storage
 * @param[in] hyb_coeffs Hybridization function coefficients
 * @param[in] nflav Number of flavors
 * @return Pair of the propagator as a one-block BDOF and the field operators as a one-set BlockOpSymQuartet
 */
std::pair<BlockDiagOpFun, BlockOpSymQuartet> trivial_sparsity_helper(nda::array<dcomplex, 3> Gt_dense, nda::array<dcomplex, 3> Fs_dense,
                                                                     nda::array<dcomplex, 3> F_dags_dense,
                                                                     nda::array_const_view<dcomplex, 3> hyb_coeffs, int nflav);

/**
 * @brief Split a BlockOpSymQuartet into the per-flavor BlockOp lists that eval_correlator takes
 *
 * @details eval_correlator predates the symmetry-set storage and still wants one BlockOp per flavor, so each flavor has to be picked out of whichever
 * symmetry set holds it: Fq.sym_set_labels(oidx) says which set, and Fq.sym_set_inds(oidx) says where within that set. Block-columns in which the
 * operator has no block are filled with a 1x1 zero, which is the placeholder the block-sparse routines expect.
 *
 * @param[in] Fq Field operators in symmetry-set storage
 * @param[in] nflav Number of flavors
 * @return Pair of the annihilation and creation operators, one BlockOp per flavor
 */
std::pair<std::vector<BlockOp>, std::vector<BlockOp>> make_correlator_ops(BlockOpSymQuartet &Fq, int nflav);

/**
 * @brief Largest absolute value taken by any off-diagonal entry of a (time, n, n) tensor
 *
 * @details Used by the analytic self-energy / single-particle Green's function tests to check that the
 * matrix structure really is diagonal, and not just diagonal in the entries that are compared against
 * closed forms. Returns 0 for a 1x1 matrix, which has no off-diagonal entries -- call sites that rely on
 * this being a non-vacuous check should assert the matrix dimension separately.
 *
 * @param[in] A Tensor whose second and third indices are the matrix indices
 * @return max_{i != j} max_t |A(t, i, j)|, or 0 if A has no off-diagonal entries
 */
inline double max_offdiag(nda::array_const_view<dcomplex, 3> A) {
  double max_abs = 0.0;
  for (int i = 0; i < A.extent(1); ++i) {
    for (int j = 0; j < A.extent(2); ++j) {
      if (i != j) { max_abs = std::max(max_abs, nda::max_element(nda::abs(A(nda::range::all, i, j)))); }
    }
  }
  return max_abs;
}
