#include "triqs_xca/hyb.hpp"

using nda::linalg::matmul;

namespace triqs_xca::hyb {

    /**
     * @brief Constructor for DenseFSet
     * @param[in] tau_mesh TRIQS imaginary time DLR mesh
     * @param[in] hyb_poles poles of the hybridization
     * @param[in] hyb_coeffs pole coefficients of the hybridization
     */
    Hybridization::Hybridization(
        const triqs::mesh::dlr_imtime &tau_mesh, 
        nda::vector_const_view<double> hyb_poles, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs) 
     : 
     tau_mesh(tau_mesh), 
     poles(hyb_poles * tau_mesh.beta()), 
     coeffs(hyb_coeffs),
     values(coefs2vals(tau_mesh.beta(), tau_mesh.dlr_it(), hyb_coeffs, hyb_poles)),
     values_reflect(-tau_mesh.dlr_it().reflect(values)) // Follow sign convention of block_sparse_backbone for reflected hybridization function.
     {};    

nda::array<dcomplex, 3> coefs2vals(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> coefs,
                                       nda::vector_const_view<double> poles) {
  auto dlr_rf = cppdlr::build_dlr_rf(Lambda, eps);
  auto itops  = cppdlr::imtime_ops(Lambda, dlr_rf);
  return coefs2vals(beta, itops, coefs, poles);
}

nda::array<dcomplex, 3> coefs2vals(double beta, const cppdlr::imtime_ops &itops, nda::array_const_view<dcomplex, 3> coefs, nda::vector_const_view<double> poles) {
  long n1     = coefs.extent(1);
  long n2     = coefs.extent(2);
  int r       = itops.rank();
  auto dlr_it = itops.get_itnodes();
  int p       = static_cast<int>(poles.size());
  auto kmat   = cppdlr::build_k_it(dlr_it, nda::make_regular(beta * poles));
  auto cf_r   = nda::reshape(coefs, p, coefs.size() / p);
  nda::array<dcomplex, 3> vals(r, n1, n2);
  reshape(vals, r, n1 * n2) = matmul(kmat, cf_r);
  return vals;
}

nda::array<dcomplex, 3> reflect(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> coefs,
                                    nda::vector_const_view<double> poles) {
  auto vals   = coefs2vals(beta, Lambda, eps, coefs, poles);
  auto dlr_rf = cppdlr::build_dlr_rf(Lambda, eps);
  auto itops  = cppdlr::imtime_ops(Lambda, dlr_rf);
  auto refl   = itops.reflect(vals);
  // TODO
  return coefs; // placeholder
}

} // namespace triqs_xca::hyb