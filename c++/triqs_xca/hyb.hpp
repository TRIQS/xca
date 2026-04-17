#pragma once

#include <nda/nda.hpp>

#include <triqs/gfs.hpp>

#include "triqs_xca/backbone.hpp"

namespace triqs_xca::hyb {

  using nda::dcomplex;

  class Hybridization {

    public:
    triqs::mesh::dlr_imtime tau_mesh; // imaginary time mesh

    nda::vector<double> poles;      // hybridization poles
    nda::array<dcomplex, 3> coeffs; // hybridization (matrix) coefficients

    nda::array<dcomplex, 3> values;         // hybridization function at imaginary time nodes
    nda::array<dcomplex, 3> values_reflect; // hybridization function at imaginary time nodes (reversed)

    nda::array<double, 2> k_it_p; // fermionic kernel evaluated at imaginary time nodes and poles
    nda::array<double, 2> k_it_m; // fermionic kernel evaluated at imaginary time nodes and (-1 * poles)
    nda::vector<double> k_0_p; // fermionic kernel evaluated at tau = 0 and poles
    nda::vector<double> k_0_m; // fermionic kernel evaluated at tau = 0 and (-1 * poles)

    /**
     * @brief Constructor for DenseFSet
     * @param[in] tau_mesh TRIQS imaginary time DLR mesh
     * @param[in] hyb_poles poles of the hybridization
     * @param[in] hyb_coeffs pole coefficients of the hybridization
     */
    Hybridization(const triqs::mesh::dlr_imtime &tau_mesh, nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                  double refl_sign = 1.0);

    void multiply_kernel_on_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int v_ix, int l_ix, double sign = 1.0);
    void multiply_kernels_on_edge(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int e_ix);
    void multiply_kernels_prefactor(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone);
  };

  /**
 * @brief Convert hybridization coefficients to values at DLR imaginary time nodes
 * @param[in] beta inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] coefs DLR coefficients
 * @param[in] poles DLR poles
 */
  nda::array<dcomplex, 3> coefs2vals(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> coefs,
                                     nda::vector_const_view<double> poles);

  /**
 * @brief Convert hybridization coefficients to values at DLR imaginary time nodes
 * @param[in] itos cppdlr imaginary time object
 * @param[in] coefs DLR coefficients
 * @param[in] poles DLR poles
 */
  nda::array<dcomplex, 3> coefs2vals(double beta, const cppdlr::imtime_ops &itops, nda::array_const_view<dcomplex, 3> coefs,
                                     nda::vector_const_view<double> poles);
} // namespace triqs_xca::hyb