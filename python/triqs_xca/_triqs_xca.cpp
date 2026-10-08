// clair-c2py --gen-default-config _triqs_xca.cpp  --> _triqs_xca.toml (template)

#include <c2py/c2py.hpp>

#include <triqs/atom_diag.hpp>
#include <triqs/operators.hpp>

#include "triqs_xca/dense/diagram_evaluator.hpp"
#include "triqs_xca/dense/manual/sigma.hpp"
#include "triqs_xca/block_sparse/diagram_evaluator.hpp"

#include <cppdlr/dyson_it_ppsc.hpp>
#include "cppdlr/_cppdlr.wrap.hxx"

// Both evaluators are named DiagramEvaluator in C++; these aliases give them distinct Python names.
// A class aliased here must not also be matched by match_names in _triqs_xca.toml, or the plain name wins.
namespace c2py_module {
  using DenseDiagramEvaluator       = triqs_xca::dense::DiagramEvaluator;
  using BlockSparseDiagramEvaluator = triqs_xca::block_sparse::DiagramEvaluator;
} // namespace c2py_module

// clair-c2py binds the constructor and method template overloads declared as extern templates in this file, not those in the headers

// ==================== dense::DiagramEvaluator ====================

extern template
triqs_xca::dense::DiagramEvaluator::DiagramEvaluator(
    nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
    triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<true> const &);

extern template
triqs_xca::dense::DiagramEvaluator::DiagramEvaluator(
    nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
    triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<false> const &);

// -- Dynamic interaction constructor

extern template
triqs_xca::dense::DiagramEvaluator::DiagramEvaluator(
    nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
    triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<true> const &,
    std::vector<triqs::operators::many_body_operator_complex> const &,
    nda::array_const_view<dcomplex, 3>);

extern template
triqs_xca::dense::DiagramEvaluator::DiagramEvaluator(
    nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
    triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<false> const &,
    std::vector<triqs::operators::many_body_operator_complex> const &,
    nda::array_const_view<dcomplex, 3>);

// -- Correlator evaluator

extern template
nda::array<dcomplex, 3> triqs_xca::dense::DiagramEvaluator::compute_one_time_correlator<true>(
    gf_vt, std::vector<triqs::operators::many_body_operator_complex> const &,
    std::vector<triqs::operators::many_body_operator_complex> const &,
    triqs::atom_diag::atom_diag<true> const &, nda::array_const_view<int, 2>,
    nda::array_const_view<long, 1>);

extern template
nda::array<dcomplex, 3> triqs_xca::dense::DiagramEvaluator::compute_one_time_correlator<false>(
    gf_vt, std::vector<triqs::operators::many_body_operator_complex> const &,
    std::vector<triqs::operators::many_body_operator_complex> const &,
    triqs::atom_diag::atom_diag<false> const &, nda::array_const_view<int, 2>,
    nda::array_const_view<long, 1>);

// ==================== block_sparse::DiagramEvaluator ====================

extern template triqs_xca::block_sparse::DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
  triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<true> const &);

extern template triqs_xca::block_sparse::DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
  triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<false> const &);

// -- Dynamic interaction constructor

extern template triqs_xca::block_sparse::DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
  triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<true> const &,
  std::vector<triqs::operators::many_body_operator_complex> const &,
  nda::array_const_view<dcomplex, 3>);

extern template triqs_xca::block_sparse::DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
  triqs::mesh::dlr_imtime, triqs::atom_diag::atom_diag<false> const &,
  std::vector<triqs::operators::many_body_operator_complex> const &,
  nda::array_const_view<dcomplex, 3>);

// -- Correlator evaluator

extern template
nda::array<dcomplex, 3> triqs_xca::block_sparse::DiagramEvaluator::compute_one_time_correlator<true>(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>,
    std::vector<triqs::operators::many_body_operator_complex> const &,
    std::vector<triqs::operators::many_body_operator_complex> const &,
    triqs::atom_diag::atom_diag<true> const &, nda::array_const_view<int, 2>,
    nda::array_const_view<long, 1>);

extern template
nda::array<dcomplex, 3> triqs_xca::block_sparse::DiagramEvaluator::compute_one_time_correlator<false>(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>,
    std::vector<triqs::operators::many_body_operator_complex> const &,
    std::vector<triqs::operators::many_body_operator_complex> const &,
    triqs::atom_diag::atom_diag<false> const &, nda::array_const_view<int, 2>,
    nda::array_const_view<long, 1>);

// -- Free functions

extern template
dcomplex triqs_xca::block_sparse::expectation_value<false>(
  triqs::operators::many_body_operator_complex const &op,
  triqs::atom_diag::atom_diag<false> const &ad,
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc);

extern template
dcomplex triqs_xca::block_sparse::expectation_value<true>(
  triqs::operators::many_body_operator_complex const &op,
  triqs::atom_diag::atom_diag<true> const &ad,
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc);

#include "_triqs_xca.wrap.cxx"
