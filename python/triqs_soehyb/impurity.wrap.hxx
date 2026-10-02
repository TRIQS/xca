#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION_impurity_GUARDS
#define C2PY_HXX_DECLARATION_impurity_GUARDS
template <> constexpr bool c2py::is_wrapped<fastdiagram>     = true;
template <> inline constexpr auto c2py::tp_name<fastdiagram> = "triqs_soehyb.impurity.Fastdiagram";
#endif