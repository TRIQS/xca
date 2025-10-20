.. _python_api:

Python API Reference
====================

This section contains the complete Python API reference for the triqs_soehyb package,
including all modules, classes, and compiled extensions.

Main Module
-----------

The main triqs_soehyb module provides access to all submodules and the high-level Solver interface.

High-Level Solver Interface
----------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary_class_template.rst

   triqs_soehyb.triqs_solver.TriqsSolver
   triqs_soehyb.solver

Compiled Extension Modules
--------------------------

These are the C++ modules compiled with Python bindings:

Impurity Module
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary_module_template.rst

   triqs_soehyb.impurity
   triqs_soehyb.impurity.Fastdiagram

DLR Dyson Module
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary_module_template.rst

   triqs_soehyb.dlr_dyson_ppsc
   triqs_soehyb.dlr_dyson_ppsc.DysonItPPSC

PyCppDLR Module
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary_module_template.rst

   triqs_soehyb.pycppdlr
   triqs_soehyb.pycppdlr.ImTimeOps
   triqs_soehyb.pycppdlr.build_dlr_rf

Additional Modules
------------------

AAA Module
~~~~~~~~~~

.. autosummary::
   :toctree: generated

   triqs_soehyb.aaa.aaa
   triqs_soehyb.aaa.aaa_matrix

Utilities
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary_module_template.rst

   triqs_soehyb.diag
   triqs_soehyb.dimer
   triqs_soehyb.dimer_new
   triqs_soehyb.ac_pes

