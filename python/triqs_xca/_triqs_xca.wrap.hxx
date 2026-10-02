#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION__triqs_xca_GUARDS
#define C2PY_HXX_DECLARATION__triqs_xca_GUARDS
template <> constexpr bool c2py::is_wrapped<triqs_xca::dense::DiagramEvaluator>            = true;
template <> inline constexpr auto c2py::tp_name<triqs_xca::dense::DiagramEvaluator>        = "triqs_xca._triqs_xca.DenseDiagramEvaluator";
template <> constexpr bool c2py::is_wrapped<triqs_xca::block_sparse::DiagramEvaluator>     = true;
template <> inline constexpr auto c2py::tp_name<triqs_xca::block_sparse::DiagramEvaluator> = "triqs_xca._triqs_xca.BlockSparseDiagramEvaluator";
#endif