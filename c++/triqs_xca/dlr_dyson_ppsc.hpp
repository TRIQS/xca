/*******************************************************************************
 *
 * triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2023, Simons Foundation
 * Copyright (C) 2023, Hugo U.R. Strand
 * Authors: Jason Kaye, Hugo U. R. Strand
 *
 * triqs_xca is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * triqs_xca is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * triqs_xca. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once

#include "nda/layout_transforms.hpp"
#include <nda/nda.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>

#include <nda/linalg/eigenelements.hpp>
#include <type_traits>

namespace cppdlr {

  /**
  * @class dyson_it_ppsc
  * @brief Class for solving the pseudo-particle Dyson equation in imaginary time
  * @tparam Ht Type of Hamiltonian
  */

  // Type of Hamiltonian, and scalar type of Hamiltonian
  template <typename Ht, nda::Scalar Sh = std::conditional_t<std::floating_point<Ht>, Ht, nda::get_value_t<Ht>>>
    requires(std::floating_point<Ht> || nda::MemoryMatrix<Ht>)
  class dyson_it_ppsc {

    public:
    /**
    * @brief Constructor for dyson_it
    * @param[in] beta Inverse temperature
    * @param[in] itops DLR imaginary time object
    * @param[in] h Hamiltonian
    *
    * \note Hamiltonian must either be a symmetric matrix, a Hermitian matrix,
    * or a real scalar.
    */
    dyson_it_ppsc(double beta, imtime_ops itops, Ht const &h)
       : beta(beta),
         itops_ptr(std::make_shared<imtime_ops>(itops)),
         r(itops.rank()),
         norb(std::floating_point<Ht> ? 1 : h.shape(0)),
         time_order(true),
         g0(r, norb, norb),
         g0c(r, norb, norb),
         sysmat(r * norb, r * norb) {

      // dyson_it_ppsc object contains a shared pointer to the imtime_ops object
      // itops. This is done to avoid making a copy of itops, which is meant to
      // handle all imaginary time operations on the given DLR imaginary time
      // grid.

      g0()  = free_gf_ppsc(beta, itops, h); // Free Green's function (right hand side of Dyson equation
      g0c() = itops_ptr->vals2coefs(g0);    // DLR coefficients of free Green's function

      // Get right hand side of Dyson equation
      if constexpr (std::floating_point<Ht>) { // If h is real scalar, rhs is a vector
        rhs.resize(r);
        rhs() = g0;
      } else { // Otherwise, rhs is given by g0 w/ some indices transposed (for compatibility w/ LAPACK)
        rhs.resize(norb, r, norb);
        rhs() = permuted_indices_view<nda::encode<3>({1, 2, 0})>(g0);
      }
    }

    /**
    * @brief Solve pseudo-particle Dyson equation for given self-energy
    *
    * @tparam Tsig Type of self-energy
    * @param[in] sig Self-energy at DLR imaginary time nodes
    * @param[in] eta Chemical potential
    *
    * @return Green's function at DLR imaginary time nodes
    *
    * \note Free Green's function (right hand side of Dyson equation) specified
    * at construction of dyson_it object
    */
    // Tsig is type of sig. Tg is type of Green's
    // function, which is of type Tsig, with scalar type replaced by the common
    // type of the Hamiltonian's scalar type, and the self-energy's scalar type
    // (real if both are real, complex otherwise).
    template <nda::MemoryArray Tsig, nda::MemoryArray Tg = make_common_t<Tsig, Sh, nda::get_value_t<Tsig>>> Tg solve(Tsig const &sig, double eta) {

      int r     = itops_ptr->rank();          // DLR rank
      auto sigc = itops_ptr->vals2coefs(sig); // DLR coefficients of self-energy

      // Obtain Dyson equation system matrix I - G0 * eta - G0 * Sig by only
      // building a single matrix of convolution for (G0 * eta - G0 * Sig)
      // the product G0 * Sig is computed using matrix free convolution
      // avoiding matrix matrix operations (2x speedup and 1/2 memory foot print)

      auto ggs = itops_ptr->convolve(beta, Fermion, g0c, sigc, time_order);
      ggs += eta * g0;
      ggs *= -1.0;

      itops_ptr->convmat_inplace(nda::matrix_view<Sh, nda::C_layout>(sysmat), beta, Fermion, itops_ptr->vals2coefs(ggs), time_order);

      sysmat += nda::eye<double>(r * norb);

      // Factorize system matrix
      auto ipiv = nda::vector<int>(r * norb);
      nda::lapack::getrf(sysmat, ipiv);

      // Solve Dyson equation
      auto g    = Tg(sig.shape());                                                         // Declare Green's function
      g         = rhs;                                                                     // Get right hand side of Dyson equation
      auto g_rs = nda::matrix_view<nda::get_value_t<Tg>>(nda::reshape(g, norb, r * norb)); // Reshape g to be compatible w/ LAPACK
      nda::lapack::getrs(sysmat, g_rs, ipiv);                                              // Back solve

      if constexpr (std::floating_point<Ht>) { // If h is scalar, g is scalar-valued
        return g;
      } else { // Otherwise, g is matrix-valued, and need to transpose some indices after solve to undo LAPACK formatting
        return permuted_indices_view<nda::encode<3>({2, 0, 1})>(g);
      }
    }

    /**
    * @brief Solve pseudo-particle Dyson equation for given self-energy, chemical potential, and operator.
    *
    * @tparam Tsig Type of self-energy
    * @param[in] sig Self-energy at DLR imaginary time nodes
    * @param[in] eta Chemical potential
    * @param[in] op Static operator (used to vary the chemical potential)
    *
    * @return Green's function at DLR imaginary time nodes
    *
    * \note Free Green's function (right hand side of Dyson equation) specified
    * at construction of dyson_it object
    */
    // Tsig is type of sig. Tg is type of Green's
    // function, which is of type Tsig, with scalar type replaced by the common
    // type of the Hamiltonian's scalar type, and the self-energy's scalar type
    // (real if both are real, complex otherwise).
    template <nda::MemoryArray Tsig, nda::MemoryArray Tg = make_common_t<Tsig, Sh, nda::get_value_t<Tsig>>>
    Tg solve_with_op(Tsig const &sig, double eta, nda::matrix_view<dcomplex> op) {

      int r     = itops_ptr->rank();          // DLR rank
      auto sigc = itops_ptr->vals2coefs(sig); // DLR coefficients of self-energy

      auto ggs = itops_ptr->convolve(beta, Fermion, g0c, sigc, time_order);
      ggs += eta * g0;
      for (int k = 0; k < r; ++k) ggs(k, _, _) += nda::matmul(g0(k, _, _), op);
      ggs *= -1.0;

      itops_ptr->convmat_inplace(nda::matrix_view<Sh, nda::C_layout>(sysmat), beta, Fermion, itops_ptr->vals2coefs(ggs), time_order);

      sysmat += nda::eye<double>(r * norb);

      // Factorize system matrix
      auto ipiv = nda::vector<int>(r * norb);
      nda::lapack::getrf(sysmat, ipiv);

      // Solve Dyson equation
      auto g    = Tg(sig.shape());                                                         // Declare Green's function
      g         = rhs;                                                                     // Get right hand side of Dyson equation
      auto g_rs = nda::matrix_view<nda::get_value_t<Tg>>(nda::reshape(g, norb, r * norb)); // Reshape g to be compatible w/ LAPACK
      nda::lapack::getrs(sysmat, g_rs, ipiv);                                              // Back solve

      if constexpr (std::floating_point<Ht>) { // If h is scalar, g is scalar-valued
        return g;
      } else { // Otherwise, g is matrix-valued, and need to transpose some indices after solve to undo LAPACK formatting
        return permuted_indices_view<nda::encode<3>({2, 0, 1})>(g);
      }
    }

    private:
    double beta;                           ///< Inverse temperature
    std::shared_ptr<imtime_ops> itops_ptr; ///< shared pointer to imtime_ops object
    int r;                                 ///< DLR rank
    int norb;                              ///< Number of orbital indices
    bool time_order;                       ///< Flag for ordinary (false) or time-ordered (true) Dyson equation

    typename std::conditional_t<std::floating_point<Ht>, nda::array<Sh, 1>, nda::array<Sh, 3>>
       rhs; ///< Right hand side of Dyson equation (in format compatible w/ LAPACK); vector if Hamiltonian is scalar, rank-3 array otherwise
    nda::array<Sh, 3> g0;                  ///< free Green's function
    nda::array<Sh, 3> g0c;                 ///< free Green's function DLR coefficients
    nda::matrix<Sh, nda::C_layout> sysmat; ///< Dyson equation matrix representation
  };

  /**
  * @brief Compute pseudo-particle imaginary time Green's function for a given
  * Hamiltonian
  *
  * The Green's function is computed by diagonalizing the Hamiltonian, and is
  * returned by its values at the DLR imaginary time nodes.
  *
  * @param[in] beta Inverse temperature
  * @param[in] it imtime_ops object
  * @param[in] h Hamiltonian
  *
  * @return Green's function at DLR imaginary time nodes
  *
  * \note Hamiltonian must either be a symmetric matrix, a Hermitian matrix,
  * or a real scalar.
  */
  template <typename Ht>
    requires(std::floating_point<Ht> || nda::MemoryMatrix<Ht>)
  // If h is scalar, return scalar-valued Green's function; if h is matrix,
  // return matrix-valued Green's function
  auto free_gf_ppsc(double beta, imtime_ops const &itops, Ht const &h) {

    int r = itops.rank();

    if constexpr (std::floating_point<Ht>) { // If h is scalar, return scalar-valued Green's function
      auto g    = nda::array<Ht, 1>(r);
      double mu = h;
      // Free Green's function for time-ordered Dyson equation
      for (int i = 0; i < r; i++) { g(i) = -exp(-beta * itops.get_itnodes(i) * (h - mu)); }
      return g;
    } else { // Otherwise, return matrix-valued Green's function

      int norb = h.shape(0);

      // Diagonalize Hamiltonian
      auto [eval, evec] = nda::linalg::eigenelements(h);

      auto E0 = nda::min_element(eval);
      eval -= E0;
      auto Z   = nda::sum(nda::exp(-beta * eval));
      auto eta = log(Z) / beta;
      eval += eta;

      // Get free Green's function
      auto g = nda::array<nda::get_value_t<Ht>, 3>(r, norb, norb);
      g      = 0;
      for (int i = 0; i < r; i++) {
        for (int j = 0; j < norb; j++) { g(i, j, j) = -exp(-beta * rel2abs(itops.get_itnodes(i)) * eval(j)); }
        g(i, _, _) = matmul(evec, matmul(g(i, _, _), transpose(conj(evec))));
      }

      return g;
    }
  }

  using dyson_it_ppsc_complex_matrix = dyson_it_ppsc<nda::array<dcomplex, 2>, dcomplex>;

} // namespace cppdlr
