.. _ppsc:

A word on the algorithm
=======================

The hybridization expansion approach to solving generalized Anderson impurity models has been combined with a plethora of numerical methods. One of these methods is the bold hybridization expansion [#ppscintro]_ in terms of the local atomic propagator :math:`G(\tau)`, often referred to as the pseudo-particle approach, since it can be derived by introducing a pseudo particle for each many-body state in the impurity local Hilbert space.

In the bold formulation :math:`G(\tau)` is self-consistently computed using the Dyson equation

.. math::

   (1 - G_0 \ast \Sigma \ast) G = G_0 \\
   
where :math:`G_0(\tau)` is the atomic many-body propagator and :math:`\Sigma(\tau)` is the pseudo-particle self-energy truncated at a finite expansion order :math:`n` in the hybridization function :math:`\Delta(\tau)`.

.. math::

   \Sigma[G] = \Sigma_1[G] + \Sigma_2[G] + \dots + \Sigma_n[G]

Once convergence is reached physical response functions like the single particle Green's function :math:`g(\tau) = \langle \mathcal{T} c(\tau) c^\dagger(0) \rangle` can be evaluted by a separate diagrammatic series (similar to :math:`\Sigma`).

The ``triqs_xca`` solver implements the bold hybridization expansion using the Discrete Lehmann Representation (DLR) [#dlr]_ [#cppdlr]_ for compact representation of propagators in imaginary time :math:`\tau` and a separate hybridization function compression approach [#xca]_ (based on the famous AAA algorithm) to evaluate the diagram series for :math:`\Sigma` with lower computational complexity than standard quadrature integration. [#dlrhyb]_


.. [#ppscintro] `M. Eckstein, P. Werner, Phys. Rev. B 82, 115115 (2010) <https://doi.org/10.1103/PhysRevB.82.115115>`_
.. [#dlrhyb] `J. Kaye, Z. Huang, H. U.R. Strand, D. Golež, Phys. Rev. X 14, 031034 (2024) <https://doi.org/10.1103/PhysRevX.14.031034>`_
.. [#xca] `Z. Huang, D. Golež, H. U.R. Strand, J. Kaye, arXiv:2503.19727 (2025) <https://doi.org/10.48550/arXiv.2503.19727>`_
.. [#cppdlr] `J. Kaye, H. U.R. Strand, N. Wentzell, J. Open Source Softw., 9(100), 6297, (2024) <https://doi.org/10.21105/joss.06297>`_
.. [#dlr] `J. Kaye, K. Chen, O. Parcollet, Phys. Rev. B 105, 235115 (2021) <https://doi.org/10.1103/PhysRevB.105.235115>`_   
