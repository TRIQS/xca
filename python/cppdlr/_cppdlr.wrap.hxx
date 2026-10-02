#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION__cppdlr_GUARDS
#define C2PY_HXX_DECLARATION__cppdlr_GUARDS
template <> constexpr bool c2py::is_wrapped<cppdlr::imtime_ops>     = true;
template <> inline constexpr auto c2py::tp_name<cppdlr::imtime_ops> = "cppdlr._cppdlr.ImTimeOps";
template <>
constexpr bool c2py::is_wrapped<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>> = true;
template <>
inline constexpr auto c2py::tp_name<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>> = "cppdlr._cppdlr.DysonItPPSC";
#endif