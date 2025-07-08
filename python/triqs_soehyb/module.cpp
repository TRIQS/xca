#include <c2py/c2py.hpp>

#include "triqs_soehyb/triqs_soehyb.hpp"

using namespace std::string_literals;
using triqs_soehyb::toto;
template <> struct c2py::arithmetic<toto, c2py::OpName::Add> : std::tuple<triplet<toto, toto, toto>> {};

#include "module.wrap.cxx"
