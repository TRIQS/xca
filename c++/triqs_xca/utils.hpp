/*******************************************************************************
 *
 * triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2025, H. U.R. Strand
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

#include <cstdint>

// https://en.wikipedia.org/wiki/Exponentiation_by_squaring

namespace triqs_xca::utils {

static constexpr inline int64_t pown(int64_t x, unsigned p) {
  int64_t result = 1;
  while (p) {
    if (p & 0x1) result *= x;
    x *= x;
    p >>= 1;
  }
  return result;
}

} // namespace triqs_xca::utils