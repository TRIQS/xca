// clair-c2py --gen-default-config _cppdlr.cpp  --> _cppdlr.toml (template)

#include <c2py/c2py.hpp>

#include <cppdlr/cppdlr.hpp>
#include <cppdlr/dyson_it_ppsc.hpp>

template nda::array<nda::dcomplex, 3> cppdlr::imtime_ops::vals2coefs<nda::array<nda::dcomplex, 3>>(nda::array<nda::dcomplex, 3> const &g,
                                                                                                   bool transpose = false) const;

template nda::array<nda::dcomplex, 3> cppdlr::imtime_ops::coefs2vals<nda::array<nda::dcomplex, 3>>(nda::array<nda::dcomplex, 3> const &g) const;

template nda::array<nda::dcomplex, 3> cppdlr::imtime_ops::reflect<nda::array<nda::dcomplex, 3>>(nda::array<nda::dcomplex, 3> const &g) const;

template auto cppdlr::imtime_ops::coefs2eval<nda::array<nda::dcomplex, 3>>(nda::array<nda::dcomplex, 3> const &g, double t) const;

template nda::array<nda::dcomplex, 3> cppdlr::imtime_ops::convolve<nda::array<nda::dcomplex, 3>>(double beta, nda::array<nda::dcomplex, 3> const &fc,
                                                                                                 nda::array<nda::dcomplex, 3> const &gc,
                                                                                                 bool time_order = false) const;

template cppdlr::dyson_it_ppsc<nda::array<nda::dcomplex, 2>, nda::dcomplex>::dyson_it_ppsc(double beta, cppdlr::imtime_ops itops,
                                                                                            nda::array_view<nda::dcomplex, 3> const &g0);

template nda::array<nda::dcomplex, 3> cppdlr::dyson_it_ppsc<nda::array<nda::dcomplex, 2>, nda::dcomplex>::solve<nda::array_view<nda::dcomplex, 3>>(
   nda::array_view<nda::dcomplex, 3> const &sig, double eta);

template nda::array<nda::dcomplex, 3> cppdlr::dyson_it_ppsc<nda::array<nda::dcomplex, 2>, nda::dcomplex>::solve_with_op<
   nda::array_view<nda::dcomplex, 3>>(nda::array_view<nda::dcomplex, 3> const &sig, double eta, nda::matrix_view<nda::dcomplex> op);

namespace c2py_module {
  using ImTimeOps    = cppdlr::imtime_ops;
  using DysonItPPSC = cppdlr::dyson_it_ppsc<nda::array<nda::dcomplex, 2>, nda::dcomplex>;
}

#include "_cppdlr.wrap.cxx"
