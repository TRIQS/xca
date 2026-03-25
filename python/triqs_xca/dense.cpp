// clair-c2py --gen-default-config dense.cpp  --> dense.toml (template)

#include <c2py/c2py.hpp>

#include <triqs/atom_diag.hpp>

#include "triqs_xca/dense_backbone.hpp"

#include "triqs_xca/block_sparse_manual.hpp"

#include "pycppdlr.wrap.hxx"

extern template
triqs_xca::dense::DenseDiagramEvaluator::DenseDiagramEvaluator(
    nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
    triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<true> const &);

extern template
triqs_xca::dense::DenseDiagramEvaluator::DenseDiagramEvaluator(
    nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
    triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<false> const &);

#include "dense.wrap.cxx"