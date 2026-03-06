#include "triqs_xca/dense.hpp"

namespace triqs_xca::dense {

    DenseFSet::DenseFSet(nda::array_const_view<dcomplex, 3> Fs,
                         nda::array_const_view<dcomplex, 3> F_dags,
                         nda::array_const_view<dcomplex, 3> hyb_coeffs)
      :
      N(Fs.extent(1)), n(Fs.extent(0)), p(hyb_coeffs.extent(0)),
      Fs(Fs), F_dags(F_dags),
      F_dag_bars(nda::zeros<dcomplex>(n, p, N, N)),
      F_bars_refl(nda::zeros<dcomplex>(n, p, N, N))
      {

      update_hybridization(hyb_coeffs);
    }

    std::size_t DenseFSet::get_num_orb_inds() const { return n; }

    void DenseFSet::update_hybridization(nda::array_const_view<dcomplex, 3> hyb_coeffs) {

      if (hyb_coeffs.extent(1) != n || hyb_coeffs.extent(2) != n) {
        throw std::invalid_argument("hyb_coeffs must have shape (p, n, n)");
      }

      if (hyb_coeffs.extent(0) != p) {
        p = hyb_coeffs.extent(0);
        F_dag_bars = nda::zeros<dcomplex>(n, p, N, N);
        F_bars_refl = nda::zeros<dcomplex>(n, p, N, N);
      }

      for (int l = 0; l < p; l++) {
        for (int nu = 0; nu < n; nu++) {
          for (int lam = 0; lam < n; lam++) {
            F_bars_refl(nu, l, _, _) -= hyb_coeffs(l, nu, lam) * Fs(lam, _, _);
            F_dag_bars (nu, l, _, _) += hyb_coeffs(l, lam, nu) * F_dags(lam, _, _);
          }
        }
      }
    }


} // namespace triqs_xca::dense