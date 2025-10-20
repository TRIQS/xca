.. _welcome:

The TRIQS Sum-Of-Exponentials bold HYBridization expansion impurity solver (triqs_soehyb)
*********

.. sidebar:: triqs_soehyb |PROJECT_VERSION|

   This is the homepage of triqs_soehyb |PROJECT_VERSION|.
   For changes see the :doc:`changelog page <ChangeLog>`.
      
      .. image:: _static/logo_github.png
         :width: 75%
         :align: center
         :target: https://github.com/triqs/soehyb


The TRIQS Sum-Of-Exponentials bold HYBridization expansion impurity solver (triqs_soehyb) can solve generic multiband Anderson impurity problems approximately by perturbative expansion in the coupling to the environment (a.k.a. the hybridization function :math:`\Delta(\tau)`).

This implementation is leveraging the sum-of-exponentials (SOE) trick to acheive parametrically better computational complexity compared to direct integration and uses the Discrete Lehmann Representation (DLR) for representing response functions.

Learn how to use triqs_soehyb in the :ref:`documentation`.


Citation
========

We kindly ask you to cite the following method papers when using triqs_soehyb in your work

.. [#dlrhyb] `J. Kaye, Z. Huang, H. U.R. Strand, D. Golež, Phys. Rev. X 14, 031034 (2024) <https://doi.org/10.1103/PhysRevX.14.031034>`_
.. [#soehyb] `Z. Huang, D. Golež, H. U.R. Strand, J. Kaye, arXiv:2503.19727 (2025) <https://doi.org/10.48550/arXiv.2503.19727>`_
.. [#cppdlr] `J. Kaye, H. U.R. Strand, N. Wentzell, J. Open Source Softw., 9(100), 6297, (2024) <https://doi.org/10.21105/joss.06297>`_
.. [#dlr] `J. Kaye, K. Chen, O. Parcollet, Phys. Rev. B 105, 235115 (2021) <https://doi.org/10.1103/PhysRevB.105.235115>`_

Note that this application uses the TRIQS library please also see its `citation page. <https://triqs.github.io/triqs/latest/about.html#citation>`_

    
.. toctree::
   :maxdepth: 2
   :hidden:

   install
   documentation
   issues
   ChangeLog
   about
