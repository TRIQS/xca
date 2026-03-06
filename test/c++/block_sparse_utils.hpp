#pragma once

#include <triqs_xca/atom_diag_utils.hpp>
#include <triqs_xca/block_sparse.hpp>

using nda::dcomplex;

using triqs_xca::block_sparse::BlockDiagOpFun;
using triqs_xca::block_sparse::BlockOpSymQuartet;

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
 * @return triqs::atom_diag::atom_diag<false> object representing the two-band model's atomic Hamiltonian
 */
triqs::atom_diag::atom_diag<false> two_band_atom_diag_helper();

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
