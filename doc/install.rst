.. highlight:: bash

.. _install:

Install xca
*******************

Compiling xca from source
===============================

.. note:: To guarantee reproducibility in scientific calculations we strongly recommend the use of a stable `release <https://github.com/TRIQS/triqs/releases>`_ of both TRIQS and its applications.

Prerequisites
-------------

#. The :ref:`TRIQS <triqslibs:welcome>` library, see :ref:`TRIQS installation instruction <triqslibs:triqs_install>`.
   In the following, we assume that TRIQS is installed in the directory ``path_to_triqs``.
   
   We are currently depending on the `unstable` development version of TRIQS (that will be released as version 4.0 in the end of 2025). However, for a smoother installation experience we provide tested branches with the naming pattern `triqs/DEV_PPSC_*` (as of writing `triqs/DEV_PPSC_20250912`) that trail the `triqs/unstable` branch with fixed versions of all subpackages (e.g. `itertools`, `mpi`, `h5`, `nda`, etc.). Long story short. **Please install TRIQS on the latest `triqs/DEV_PPSC_*` branch.**

#. The `adapol <https://github.com/flatironinstitute/adapol>`_ package (Adaptive Pole Fitting for Quantum Many-Body Physics) has to be installed on the `main` branch. This can be achieved using a local `pip` install, e.g.::

    $ git clone git@github.com:flatironinstitute/adapol.git --branch main
    $ python -m pip install -e ./adapol

#. The `pyed <https://github.com/HugoStrand/pyed>`_ package (PYED: Exact diagonalization for finite quantum systems) has to be available in the python environment. This can be tested by running::

    $ python -c "import pyed.TriqsExactDiagonalization"

.. note:: The `pyed` python module has to be installed manually. Trying to install it using `pip` will not work since there is another python module with the same name (doing something completely different) registered in the PIP package index.

Installation steps
------------------

#. Download the source code of the latest stable version by cloning the ``TRIQS/xca`` repository from GitHub::

     $ git clone https://github.com/TRIQS/xca xca.src

#. Create and move to a new directory where you will compile the code::

     $ mkdir xca.build && cd xca.build

#. Ensure that your shell contains the TRIQS environment variables by sourcing the ``triqsvars.sh`` file from your TRIQS installation::

     $ source path_to_triqs/share/triqs/triqsvars.sh

#. In the build directory call cmake, including any additional custom CMake options, see below::

     $ cmake ../xca.src

#. Compile the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

Version compatibility
---------------------

Keep in mind that the version of ``xca`` must be compatible with your TRIQS library version,
see :ref:`TRIQS website <triqslibs:versions>`.
In particular the Major and Minor Version numbers have to be the same.
To use a particular version, go into the directory with the sources, and look at all available versions::

     $ cd xca.src && git tag

Checkout the version of the code that you want::

     $ git checkout 2.1.0

and follow steps 2 to 4 above to compile the code.

Custom CMake options
--------------------

The compilation of ``xca`` can be configured using CMake-options::

    cmake ../xca.src -DOPTION1=value1 -DOPTION2=value2 ...

+-----------------------------------------------------------------+-----------------------------------------------+
| Options                                                         | Syntax                                        |
+=================================================================+===============================================+
| Specify an installation path other than path_to_triqs           | -DCMAKE_INSTALL_PREFIX=path_to_xca         |
+-----------------------------------------------------------------+-----------------------------------------------+
| Build in Debugging Mode                                         | -DCMAKE_BUILD_TYPE=Debug                      |
+-----------------------------------------------------------------+-----------------------------------------------+
| Disable testing (not recommended)                               | -DBuild_Tests=OFF                             |
+-----------------------------------------------------------------+-----------------------------------------------+
| Build the documentation                                         | -DBuild_Documentation=ON                      |
+-----------------------------------------------------------------+-----------------------------------------------+
