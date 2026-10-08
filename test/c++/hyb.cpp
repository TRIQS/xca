#include <gtest/gtest.h>

#include <nda/nda.hpp>
#include <triqs/gfs.hpp>

#include <triqs_xca/hyb.hpp>

/**
 * @file hyb.cpp
 *
 * @brief Tests of hyb::Hybridization, which both diagram evaluators build from the hybridization poles and coefficients
 *
 * @details The coefficients reach the library as a generic nda view, so a view with any strides has to give the same hybridization as a
 * contiguous copy of the same values.
 */

using nda::dcomplex;
using nda::range;

/// @brief Views of the coefficients with a skipped column and with a reversed pole index give the same hybridization as contiguous copies
TEST(Hybridization, strided_coefficients) {
  double beta = 2.0, Lambda = 40.0, eps = 1e-10;
  auto tau_mesh             = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false);
  nda::vector<double> poles = {1.3, -0.8};
  long p = poles.size(), n = 3;

  // every other column of a wider array, with filler values in the skipped columns
  nda::array<dcomplex, 3> wide(p, n, 2 * n);
  wide = dcomplex(99.0, -7.0);
  for (long l = 0; l < p; ++l)
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) wide(l, i, 2 * j) = dcomplex(1.0 + l + i, 0.5 * j - i);
  auto strided = wide(range::all, range::all, range(0, 2 * n, 2));
  ASSERT_FALSE(strided.indexmap().is_contiguous());

  // a negative stride, which nda reports as contiguous
  nda::array<dcomplex, 3> base = strided;
  auto reversed                = base(range(p - 1, -1, -1), range::all, range::all);
  ASSERT_TRUE(reversed.indexmap().is_contiguous());

  for (auto const &view : {nda::array_const_view<dcomplex, 3>(strided), nda::array_const_view<dcomplex, 3>(reversed)}) {
    nda::array<dcomplex, 3> contiguous = view;
    auto H_view                        = triqs_xca::hyb::Hybridization(tau_mesh, poles, view);
    auto H_contiguous                  = triqs_xca::hyb::Hybridization(tau_mesh, poles, contiguous);
    ASSERT_GT(nda::max_element(nda::abs(H_contiguous.values)), 0.1);
    EXPECT_EQ(nda::max_element(nda::abs(H_view.coeffs - H_contiguous.coeffs)), 0.0);
    EXPECT_EQ(nda::max_element(nda::abs(H_view.values - H_contiguous.values)), 0.0);
    EXPECT_EQ(nda::max_element(nda::abs(H_view.values_reflect - H_contiguous.values_reflect)), 0.0);
  }
}
