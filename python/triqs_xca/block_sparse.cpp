// clair-c2py --gen-default-config block_sparse.cpp  --> block_sparse.toml (template)

#include <c2py/c2py.hpp>
#include <triqs/atom_diag.hpp>
#include <triqs/operators.hpp>

#include "triqs_xca/block_sparse_backbone.hpp"

extern template triqs_xca::block_sparse::DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
  triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<true> const &);

extern template triqs_xca::block_sparse::DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
  triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<false> const &);

#include "block_sparse.wrap.cxx"