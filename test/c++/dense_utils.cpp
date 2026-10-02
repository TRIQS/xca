#include "dense_utils.hpp"

using cppdlr::_;

nda::array<dcomplex, 3> eval_eq(imtime_ops &itops, nda::array_const_view<dcomplex, 3> f, int n_quad) {
  auto fc    = itops.vals2coefs(f);
  auto it_eq = cppdlr::eqptsrel(n_quad + 1);
  auto f_eq  = nda::array<dcomplex, 3>(n_quad + 1, f.extent(1), f.extent(2));
  for (int i = 0; i <= n_quad; i++) { f_eq(i, _, _) = itops.coefs2eval(fc, it_eq(i)); }
  return f_eq;
}
