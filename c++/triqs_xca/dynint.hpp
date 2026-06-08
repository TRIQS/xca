#pragma once


#include <nda/nda.hpp>

#include <triqs/gfs.hpp>


#include "triqs_xca/atom_diag_utils.hpp"
#include "triqs_xca/dense.hpp"


namespace triqs_xca::dynint {

    using triqs_xca::dense::DenseFSet;
    using triqs_xca::atom_diag::triqs_atom_diag_t;

    DenseFSet get_operators_and_interactions_dense(
        const triqs_atom_diag_t<true> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops);

    DenseFSet get_operators_and_interactions_dense(
        const triqs_atom_diag_t<false> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops);

}
