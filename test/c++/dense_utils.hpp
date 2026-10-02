#pragma once

#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>

using nda::dcomplex;

using cppdlr::imtime_ops;

/**
 * @brief Evaluate a function on the DLR imaginary time grid at n_quad + 1 equidistant points
 * @param[in] itops imaginary time DLR operations
 * @param[in] f function values on the DLR imaginary time nodes
 * @param[in] n_quad number of equidistant intervals
 * @return function values on the equidistant grid
 */
nda::array<dcomplex, 3> eval_eq(imtime_ops &itops, nda::array_const_view<dcomplex, 3> f, int n_quad);
