// clair-c2py --gen-default-config impurity.cpp  --> impurity.toml (template)

#include <c2py/c2py.hpp>

#include "triqs_soehyb/impurity.hpp"

#include <cppdlr/dyson_it_ppsc.hpp>

#include "cppdlr/_cppdlr.wrap.hxx"

#include "impurity.wrap.cxx"
