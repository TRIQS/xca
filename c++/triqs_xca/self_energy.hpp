#pragma once

#include <nda/nda.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <vector>

namespace triqs::atom_diag {
  template <bool C> class atom_diag;
}

/**
 * @brief Solve for the self-energy up to a given order
 * 
 * @param beta Inverse temperature
 * @param Lambda DLR Lambda parameter
 * @param eps DLR epsilon parameter
 * @param hyb Hybridization function
 * @param hyb_poles Hybridization poles
 * @param hyb_coeffs Hybridization coefficients
 * @param ad Atom diagonalization object
 * @param order Perturbation order (1, 2, or 3)
 * @return std::vector<nda::array<dcomplex, 3>> Self-energy blocks
 */
std::vector<nda::array<dcomplex, 3>> compute_self_energy(double beta, double Lambda, double eps, nda::array<dcomplex, 3> hyb,
                                                         nda::vector<double> hyb_poles, nda::array<dcomplex, 3> hyb_coeffs,
                                                         triqs::atom_diag::atom_diag<false> ad, int order);
