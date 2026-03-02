#include "triqs_xca/dense.hpp"

DenseFSet::DenseFSet(nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags,
                     nda::array_const_view<dcomplex, 3> hyb_coeffs)
   : Fs(Fs), F_dags(F_dags) {

  int n       = Fs.extent(0);
  int N       = Fs.extent(1);
  int p       = hyb_coeffs.extent(0);
  F_dag_bars  = nda::array<dcomplex, 4>(n, p, N, N);
  F_bars_refl = nda::array<dcomplex, 4>(n, p, N, N);

  static constexpr auto _ = nda::range::all;

  for (int lam = 0; lam < n; lam++) {
    for (int l = 0; l < p; l++) {
      for (int nu = 0; nu < n; nu++) {
        F_dag_bars(lam, l, _, _) += hyb_coeffs(l, nu, lam) * F_dags(nu, _, _);
        F_bars_refl(nu, l, _, _) -= hyb_coeffs(l, nu, lam) * Fs(lam, _, _);
      }
    }
  }
}

std::size_t DenseFSet::get_num_orb_inds() const { return Fs.extent(0); }
