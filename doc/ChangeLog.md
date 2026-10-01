# Changelog

## Version 0.0.1

`triqs_xca` Version 0.0.1 provides a 
sum-of-exponentials based bold hybridization 
expansion solver for quantum impurity problems.

This is the initial release for this project, 
with a diagram evaluation implemented using dense linear algebra 
in the local Hilbert space by Z. Huang and J. Kaye.

## Version 0.1.0

`triqs_xca` Version 0.1.0 is a major reimplementation of the diagram evaluation 
exploiting local atomic symmetries by F. Rilloraza, H. UR. Strand, and J. Kaye.

* The new `BlockSparseSolver` evaluates the pseudo-particle diagrams in block-sparse storage over the
  symmetry sectors of the local Hamiltonian. The sectors are found by automatic partitioning or from
  user-supplied conserved operators, and without symmetries the solver falls back to a dense evaluator.
* The hybridization function is compressed to a small number of poles with adapol, with the DLR
  expansion as fallback.
* Self-consistent (`solve`) and bare (`solve_bare`) hybridization expansions to a given order.
* Single-particle Green's functions, one-time correlators of general operators, expectation values,
  and the partition function.
* TRIQS 4.0.x and adapol 0.2.x are required.

**Note:** The solver API has changed since v0.0.1, see the API documentation.
