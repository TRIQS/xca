// clair-c2py --gen-default-config pycppdlr.cpp  --> pycppdlr.toml (template)

#include <c2py/c2py.hpp>

#include <cppdlr/cppdlr.hpp>

template nda::array<nda::dcomplex, 3> cppdlr::imtime_ops::vals2coefs<nda::array<nda::dcomplex, 3>>(nda::array<nda::dcomplex, 3> const &g,
                                                                                                   bool transpose = false) const;

template nda::array<nda::dcomplex, 3> cppdlr::imtime_ops::coefs2vals<nda::array<nda::dcomplex, 3>>(nda::array<nda::dcomplex, 3> const &g) const;

template nda::array<nda::dcomplex, 3> cppdlr::imtime_ops::reflect<nda::array<nda::dcomplex, 3>>(nda::array<nda::dcomplex, 3> const &g) const;

template auto cppdlr::imtime_ops::coefs2eval<nda::array<nda::dcomplex, 3>>(nda::array<nda::dcomplex, 3> const &g, double t) const;

template nda::array<nda::dcomplex, 3> cppdlr::imtime_ops::convolve<nda::array<nda::dcomplex, 3>>(double beta, nda::array<nda::dcomplex, 3> const &fc,
                                                                                                 nda::array<nda::dcomplex, 3> const &gc,
                                                                                                 bool time_order = false) const;

namespace c2py_module {
  using ImTimeOps = cppdlr::imtime_ops;
}

#include "pycppdlr.wrap.cxx"
