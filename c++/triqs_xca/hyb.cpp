#include "triqs_xca/hyb.hpp"

using nda::linalg::matmul;

using cppdlr::_;

namespace triqs_xca::hyb {

  /**
     * @brief Constructor for DenseFSet
     * @param[in] tau_mesh TRIQS imaginary time DLR mesh
     * @param[in] hyb_poles poles of the hybridization
     * @param[in] hyb_coeffs pole coefficients of the hybridization
     */
  Hybridization::Hybridization(const triqs::mesh::dlr_imtime &tau_mesh, nda::vector_const_view<double> hyb_poles,
                               nda::array_const_view<dcomplex, 3> hyb_coeffs, double refl_sign)
     : tau_mesh(tau_mesh),
       poles(hyb_poles * tau_mesh.beta()),
       coeffs(hyb_coeffs),
       values(coefs2vals(tau_mesh.beta(), tau_mesh.dlr_it(), hyb_coeffs, hyb_poles)),
       values_reflect(refl_sign
                      * tau_mesh.dlr_it().reflect(values)) // Follow sign convention of block_sparse_backbone for reflected hybridization function.
  {};

  void Hybridization::multiply_kernel_on_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int v_ix, int l_ix, double sign) {
    // Multiply the kernel associated with vertex v_ix and pole index l_ix into T_buf

    int r       = tau_mesh.dlr_it().rank();
    int Ksign   = backbone.get_vertex_Ksign(v_ix);
    auto dlr_it = tau_mesh.dlr_it().get_itnodes();

    if (Ksign != 0) {
      double pole = sign * poles(l_ix);
      for (int t = 0; t < r; t++) T_buf(t, _, _) *= cppdlr::k_it(dlr_it(t), Ksign * pole);
    }
  }

  void Hybridization::multiply_kernels_on_edge(nda::array_view<dcomplex, 3> Gt, Backbone &backbone, int e_ix) {
    // Multiply the kernels associated with edge e_ix into T_buf

    int m       = backbone.m;
    int r       = tau_mesh.dlr_it().rank();
    auto dlr_it = tau_mesh.dlr_it().get_itnodes();

    for (int x = 0; x < m - 1; x++) {
      int Ksign = backbone.get_edge(e_ix, x);
      if (Ksign != 0) {
        double pole = poles(backbone.get_pole_ind(x));
        for (int t = 0; t < r; t++) Gt(t, _, _) *= cppdlr::k_it(dlr_it(t), Ksign * pole);
      }
    }
  }

  void Hybridization::multiply_kernels_prefactor(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone) {
    // Multiply the kernels associated with the prefactor into T_buf

    int m            = backbone.m;
    double prefactor = 1.;

    for (int m_ix = 0; m_ix < m - 1; m_ix++) {     // loop over hybridization indices
      int exp = backbone.get_prefactor_Kexp(m_ix); // exponent on K for this hybridization index
      if (exp != 0) {
        int Ksign   = backbone.get_prefactor_Ksign(m_ix); // sign on K for this hybridization index
        double pole = poles(backbone.get_pole_ind(m_ix)); // DLR frequency for this value of this hybridization index
        prefactor *= std::pow(cppdlr::k_it(0, Ksign * pole), -exp);
      }
    }

    T_buf *= prefactor;
  }

  nda::array<dcomplex, 3> coefs2vals(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> coefs,
                                     nda::vector_const_view<double> poles) {
    auto dlr_rf = cppdlr::build_dlr_rf(Lambda, eps);
    auto itops  = cppdlr::imtime_ops(Lambda, dlr_rf);
    return coefs2vals(beta, itops, coefs, poles);
  }

  nda::array<dcomplex, 3> coefs2vals(double beta, const cppdlr::imtime_ops &itops, nda::array_const_view<dcomplex, 3> coefs,
                                     nda::vector_const_view<double> poles) {
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