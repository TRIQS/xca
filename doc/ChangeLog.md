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

**Note:** The solver API has changed since v0.0.1, see the API documentation.

Fixes since the 0.1.0 release candidates:

* The single-particle Green's function of the block-sparse evaluator is now indexed by orbital; its components were permuted when the symmetry sets interleave with the orbital order, e.g. with `conserved_operators='automatic'`.
* Correlators no longer drop the $[\beta, \tau]$ side of a diagram when vertex zero pairs with the last vertex.
* The backward hybridization line connected to vertex zero now transposes its orbital index, which restores unitary invariance under complex single-particle basis rotations.
* Hybridization functions that couple different symmetry sets are rejected instead of being silently truncated.
* The evaluator work arrays are sized by the largest block over all symmetry sets.
* The adapol dependency is pinned to its 0.2.0 tag.
