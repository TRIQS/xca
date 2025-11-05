/*******************************************************************************
 *
 * triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2025, Z. Huang
 *
 * triqs_xca is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * triqs_xca is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * triqs_xca. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include "nda/nda.hpp"
#include <nda/linalg/matmul.hpp>
#include <gtest/gtest.h>

using namespace nda;
using nda::linalg::matmul;

double oca_equi_error(const int Num) {
  double beta = 1;
  double h_t  = beta / Num;
  auto tgrid  = nda::vector<double>(Num + 1);

  for (int i = 0; i <= Num; ++i) { tgrid(i) = i * h_t; }
  // Is there an nda analog of np.arange?

  const int dim = 2;
  auto Gtau     = nda::array<dcomplex, 3>(dim, dim, Num + 1);

  double alpha_1 = 0.1;
  double alpha_2 = 0.2;

  auto G_01                     = exp(-alpha_1 * tgrid);
  Gtau(0, 0, range(0, Num + 1)) = 0;
  Gtau(1, 1, range(0, Num + 1)) = 0;
  Gtau(0, 1, range(0, Num + 1)) = G_01;
  Gtau(1, 0, range(0, Num + 1)) = G_01;

  auto Delta = exp(-alpha_2 * tgrid);

  auto Sigma      = nda::array<dcomplex, 3>(dim, dim, Num + 1);
  auto Sigma_true = nda::array<dcomplex, 3>(dim, dim, Num + 1);

  Sigma_true(0, 0, range(0, Num + 1)) = 0;
  Sigma_true(1, 1, range(0, Num + 1)) = 0;
  for (int n = 0; n <= Num; ++n) {
    Sigma_true(0, 1, n) = exp(-(alpha_1 + alpha_2) * tgrid(n)) * (tgrid(n) + (exp(-alpha_2 * tgrid(n)) - 1) / alpha_2) / alpha_2;
    Sigma_true(1, 0, n) = exp(-(alpha_1 + alpha_2) * tgrid(n)) * (tgrid(n) + (exp(-alpha_2 * tgrid(n)) - 1) / alpha_2) / alpha_2;
  }

  auto tmp_matr     = matrix<dcomplex>(dim, dim);
  auto tmp_tmp_matr = matrix<dcomplex>(dim, dim);

  auto ttmp_m = nda::array<dcomplex, 4>(Num + 1, Num + 1, dim, dim);
  // res = matmul(Gtau( range(0,dim), range(0,dim),0) ,Gtau(range(0,dim), range(0,dim),1));
  // res = Gtau( range(0,dim), range(0,dim),0);
  // res = matmul(res,res);

  auto Gtmp_k       = matrix<dcomplex>(dim, dim);
  auto Gtmp_jminusk = matrix<dcomplex>(dim, dim);
  auto Gtmp_nminusj = matrix<dcomplex>(dim, dim);
  for (int n = 0; n <= Num; ++n) {
    if (n == 0) {
      ttmp_m(n, range(0, Num + 1), range(0, dim), range(0, dim)) = 0;
      continue;
    }
    for (int j = 0; j <= n; ++j) {
      tmp_matr = 0;
      if (j == 0) {
        ttmp_m(n, j, range(0, dim), range(0, dim)) = 0;
        continue;
      }
      for (int k = 0; k <= j; ++k) {
        tmp_tmp_matr = 0;
        Gtmp_k       = Gtau(range(dim), range(dim), k);
        Gtmp_jminusk = Gtau(range(dim), range(dim), j - k);
        tmp_tmp_matr = matmul(Gtmp_jminusk, Gtmp_k);
        tmp_tmp_matr = tmp_tmp_matr * Delta[n - k];
        if (k == 0 || k == j) { tmp_tmp_matr *= 0.5; }
        tmp_matr = tmp_matr + tmp_tmp_matr;
      }
      Gtmp_nminusj                               = Gtau(range(dim), range(dim), n - j);
      ttmp_m(n, j, range(0, dim), range(0, dim)) = matmul(Gtmp_nminusj, tmp_matr) * Delta[j];
    }
  }
  ttmp_m = ttmp_m * h_t;
  for (int n = 0; n <= Num; ++n) {
    if (n == 0) {
      Sigma(range(0, dim), range(0, dim), n) = 0;
      continue;
    }
    tmp_matr = 0;
    for (int j = 0; j <= n; ++j) {
      if (j == 0 || j == n) {
        tmp_matr = tmp_matr + ttmp_m(n, j, range(0, dim), range(0, dim)) * 0.5;
      } else {
        tmp_matr = tmp_matr + ttmp_m(n, j, range(0, dim), range(0, dim));
      }
    }
    tmp_matr = tmp_matr * h_t;
    //std::cout<<Sigma_true(0,1,n)<<" ";
    //std::cout<<tmp_matr(0,1)<<" ";
    Sigma(range(0, dim), range(0, dim), n) = tmp_matr;
  }
  //max(abs(Sigma-Sigma_real))
  std::cout << Sigma_true(0, 1, range(Num + 1));
  std::cout << Sigma_true(range(dim), range(dim), 10);

  return max_element(abs(Sigma - Sigma_true));
}
int main() {
  //Num_list = [4,8,16,32,64,128];
  std::cout << "Grid-size| " << "|Error| " << "|log(Error)/log(2)\n";
  int Num;
  double error_Num;
  for (int i = 128; i < 200; i *= 2) {
    Num       = i;
    error_Num = oca_equi_error(Num);
    std::cout << Num << "   ";
    std::cout << error_Num << "   ";
    std::cout << std::log(error_Num) / std::log(2) << " \n";
  }
}
