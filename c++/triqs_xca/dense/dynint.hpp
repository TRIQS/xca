#pragma once

#include <nda/nda.hpp>

#include <triqs/gfs.hpp>

#include "triqs_xca/atom_diag.hpp"
#include "triqs_xca/dense/atom_diag.hpp"
#include "triqs_xca/dense/fset.hpp"


namespace triqs_xca::dense::dynint {

    using triqs_xca::atom_diag::triqs_atom_diag_t;

    FSet get_operators_and_interactions(
        const triqs_atom_diag_t<true> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops);

    FSet get_operators_and_interactions(
        const triqs_atom_diag_t<false> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops);

}
