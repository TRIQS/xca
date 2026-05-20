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

extern template
nda::array<dcomplex, 3> triqs_xca::dense::DenseDiagramEvaluator::compute_one_time_correlator<true>(
    gf_vt, std::vector<triqs::operators::many_body_operator_real> const &,
    std::vector<triqs::operators::many_body_operator_real> const &,
    triqs::atom_diag::atom_diag<true> const &, nda::array_const_view<int, 2>,
    nda::array_const_view<int, 1>);

extern template
nda::array<dcomplex, 3> triqs_xca::dense::DenseDiagramEvaluator::compute_one_time_correlator<false>(
    gf_vt, std::vector<triqs::operators::many_body_operator_real> const &,
    std::vector<triqs::operators::many_body_operator_real> const &,
    triqs::atom_diag::atom_diag<false> const &, nda::array_const_view<int, 2>,
    nda::array_const_view<int, 1>);

#include "dense.wrap.cxx"