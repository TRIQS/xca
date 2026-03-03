// clair-c2py --gen-default-config dense.cpp  --> dense.toml (template)

#include <c2py/c2py.hpp>

#include <triqs/atom_diag.hpp>

#include "triqs_xca/dense_backbone.hpp"

#include "triqs_xca/block_sparse_manual.hpp"

#include "pycppdlr.wrap.hxx"

#include "dense.wrap.cxx"
