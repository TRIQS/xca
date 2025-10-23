from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "dense", doc = r"", app_name = "dense")

# Imports

# Add here all includes
#module.add_include("triqs_xca/block_sparse.hpp")
module.add_include("cppdlr/cppdlr.hpp")
module.add_include("triqs_xca/block_sparse_manual.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <cpp2py/converters/complex.hpp>
#include <cpp2py/converters/string.hpp>
#include <nda_py/cpp2py_converters.hpp>

""")

module.add_function ("nda::array<dcomplex, 3> NCA_dense(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags)", doc = r"""""")

module.add_function ("nda::array<dcomplex, 3> OCA_dense(nda::array_const_view<dcomplex, 3> hyb, imtime_ops itops, double beta, nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags)", doc = r"""""")

module.generate_code()

