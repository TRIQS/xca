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

extern template
nda::array<dcomplex, 3> triqs_xca::block_sparse::DiagramEvaluator::compute_one_time_correlator<true>(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>,
    std::vector<triqs::operators::many_body_operator_real> const &,
    std::vector<triqs::operators::many_body_operator_real> const &,
    triqs::atom_diag::atom_diag<true> const &, nda::array_const_view<int, 2>,
    nda::array_const_view<int, 1>);

extern template
nda::array<dcomplex, 3> triqs_xca::block_sparse::DiagramEvaluator::compute_one_time_correlator<false>(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>,
    std::vector<triqs::operators::many_body_operator_real> const &,
    std::vector<triqs::operators::many_body_operator_real> const &,
    triqs::atom_diag::atom_diag<false> const &, nda::array_const_view<int, 2>,
    nda::array_const_view<int, 1>);

#include "block_sparse.wrap.cxx"