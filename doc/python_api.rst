.. _python_api:

Python API Reference
====================

This section contains the complete Python API reference for the triqs_xca package,
including all modules, classes, and compiled extensions.

Main Module
-----------

The main triqs_xca module provides access to all submodules and the high-level Solver interface.

High-Level Solver Interface
----------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary_class_template.rst

   triqs_xca.triqs_solver.TriqsSolver
   triqs_xca.solver

Compiled Extension Modules
--------------------------

These are the C++ modules compiled with Python bindings:

Impurity Module
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary_module_template.rst

   triqs_xca.impurity
   triqs_xca.impurity.Fastdiagram

DLR Dyson Module
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary_module_template.rst

   triqs_xca.dlr_dyson_ppsc
   triqs_xca.dlr_dyson_ppsc.DysonItPPSC

PyCppDLR Module
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary_module_template.rst

   triqs_xca.pycppdlr
   triqs_xca.pycppdlr.ImTimeOps
   triqs_xca.pycppdlr.build_dlr_rf

Additional Modules
------------------

Utilities
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary_module_template.rst

   triqs_xca.diag
   triqs_xca.dimer
   triqs_xca.dimer_new

