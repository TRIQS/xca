.. highlight:: bash

.. _install:

Install xca
***********

Compiling xca from source
=========================

Prerequisites
-------------

#. The :ref:`TRIQS <triqslibs:welcome>` library, see :ref:`TRIQS installation instructions. <triqslibs:triqs_install>`.
   We are currently depending on the ``unstable`` development branch of TRIQS that has to be compiled from source. (*Note that the packaged versions of TRIQS are not yet supported.*)

   In the following, we assume that TRIQS is installed in the directory ``path_to_triqs``.

Installation steps
------------------

#. Download the source code by cloning the ``TRIQS/xca`` repository from GitHub::

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

Custom CMake options
--------------------

The compilation of ``xca`` can be configured using CMake-options::

    cmake ../xca.src -DOPTION1=value1 -DOPTION2=value2 ...

+-----------------------------------------------------------------+-----------------------------------------------+
| Options                                                         | Syntax                                        |
+=================================================================+===============================================+
| Specify an installation path other than path_to_triqs           | -DCMAKE_INSTALL_PREFIX=path_to_xca            |
+-----------------------------------------------------------------+-----------------------------------------------+
| Build in Debugging Mode                                         | -DCMAKE_BUILD_TYPE=Debug                      |
+-----------------------------------------------------------------+-----------------------------------------------+
| Disable testing (not recommended)                               | -DBuild_Tests=OFF                             |
+-----------------------------------------------------------------+-----------------------------------------------+
| Build the documentation                                         | -DBuild_Documentation=ON                      |
+-----------------------------------------------------------------+-----------------------------------------------+
| Update the Python bindings (developers only)                    | -DUpdate_Python_Bindings=ON                   |
+-----------------------------------------------------------------+-----------------------------------------------+
