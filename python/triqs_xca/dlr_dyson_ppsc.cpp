// clair-c2py --gen-default-config dlr_dyson_ppsc.cpp  --> dlr_dyson_ppsc.toml (template)

#include <c2py/c2py.hpp>

#include "triqs_xca/dlr_dyson_ppsc.hpp"

#include "pycppdlr.wrap.hxx"

namespace c2py_module {
  using DysonItPPSC = cppdlr::dyson_it_ppsc<nda::array<nda::dcomplex, 2>, nda::dcomplex>;
}

template c2py_module::DysonItPPSC::dyson_it_ppsc(
  double beta, imtime_ops itops, nda::array_view<nda::dcomplex, 3> const &g0);

template nda::array<nda::dcomplex, 3> c2py_module::DysonItPPSC::solve<nda::array_view<nda::dcomplex, 3>>(
  nda::array_view<nda::dcomplex, 3> const &sig, double eta);

template nda::array<nda::dcomplex, 3> c2py_module::DysonItPPSC::solve_with_op<nda::array_view<nda::dcomplex, 3>>(
  nda::array_view<nda::dcomplex, 3> const &sig, double eta, nda::matrix_view<dcomplex> op);

#include "dlr_dyson_ppsc.wrap.cxx"
