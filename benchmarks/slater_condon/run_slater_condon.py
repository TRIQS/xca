"""Driver script for the Slater-Condon benchmarks.

Copy this file (e.g. ``run_slater_condon_d_shell.py``) and edit the CONFIGURATION
block below for each cluster job, so that ``slater_condon_interaction.py`` itself
stays untouched and shared between jobs.

Usage:

    mpirun -n <nranks> python run_slater_condon.py

If the copy lives outside this directory, point it at the module with

    export SLATER_CONDON_DIR=/path/to/xca/benchmarks/slater_condon
"""

import os
import sys
from pathlib import Path

# Make slater_condon_interaction.py importable even when this driver is copied
# to a job/scratch directory.
_module_dir = Path(os.environ.get('SLATER_CONDON_DIR', Path(__file__).resolve().parent))
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from slater_condon_interaction import (
    _N_tot_operator,
    solve_slater_condon_bethe_half_filling,
    one_se_iter_slater_condon_bethe_half_filling,
    one_spgf_iter_slater_condon_bethe_half_filling,
)

# ----------------------------------------------------------------------------
# SYMMETRY MODES
# ----------------------------------------------------------------------------

# Each entry maps a mode name to a function of the shell `l` returning the
# `conserved_operators` argument handed to BlockSparseSolver:
#
#   'automatic'         -> all symmetries, found by the atom_diag autopartition
#   []                  -> no symmetries, i.e. the dense diagram evaluator
#   [op1, op2, ...]     -> only the listed quantum numbers
#
# Add a new mode by adding an entry here and listing its name in the
# `*_symmetry_modes` lists below.
SYMMETRY_MODES = {
    'all_sym': lambda l: 'automatic',
    'no_sym': lambda l: [],
    'N_tot_sym': lambda l: [_N_tot_operator(l)],
}


# ----------------------------------------------------------------------------
# CONFIGURATION -- edit per job
# ----------------------------------------------------------------------------

# Which benchmark(s) to run, and with which symmetry modes. An empty list (or
# the corresponding `run_*` flag set to False) skips the benchmark.
run_full_solves = False
full_solve_symmetry_modes = ['all_sym']

run_one_se_iters = True
one_se_symmetry_modes = ['all_sym', 'no_sym', 'N_tot_sym']

run_one_spgf_iters = False
one_spgf_symmetry_modes = ['no_sym']

# Write the one-self-energy-iteration output of each symmetry mode into its own
# subdirectory (named after the mode), so that modes do not overwrite each
# other's .h5 files. Set to None to write everything into the current directory.
output_root = '.'

# Perturbation orders to sweep (full solves use `full_solve_order`)
orders = [1, 2, 3]

# Solver frontends to sweep: False -> BlockSparseSolver, True -> dense TriqsSolver
# Note: the dense frontend ignores `conserved_operators`, so it produces the same
# run for every symmetry mode.
denses = [False]

# Shells to sweep, as (l, Fs) pairs
shells = [
    (0, [3.0]),
    (1, [3.0, 0.5]),
    (2, [3.0, 0.5, 0.3]),
]

# DLR accuracies to sweep
eps_values = [1e-6]

# Shared solver parameters
beta = 1.0
ppsc_tol = 1e-4

# Full-solve-only parameters
full_solve_order = 1
ppsc_maxiter = 10


def conserved_operators_for(l, mode):
    """Conserved operators handed to BlockSparseSolver for shell `l` in `mode`."""

    try:
        return SYMMETRY_MODES[mode](l)
    except KeyError:
        raise KeyError(
            f"Unknown symmetry mode '{mode}', "
            f"known modes are {sorted(SYMMETRY_MODES)}") from None


def output_dir_for(mode):
    """Directory the outputs of symmetry mode `mode` are written to."""

    if output_root is None:
        return None
    path = Path(output_root) / mode
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# ----------------------------------------------------------------------------
# DRIVER
# ----------------------------------------------------------------------------

def main():

    # Fail on a mistyped mode name before doing any work
    for modes in (full_solve_symmetry_modes, one_se_symmetry_modes, one_spgf_symmetry_modes):
        for mode in modes:
            conserved_operators_for(shells[0][0], mode)

    if run_full_solves:
        for mode in full_solve_symmetry_modes:
            for eps in eps_values:
                opts = dict(
                    beta=beta,
                    eps=eps,
                    ppsc_tol=ppsc_tol,
                    ppsc_maxiter=ppsc_maxiter,
                    order=full_solve_order,
                )

                for l, Fs in shells:
                    solve_slater_condon_bethe_half_filling(
                        conserved_operators=conserved_operators_for(l, mode),
                        l=l, Fs=Fs, **opts)

    if run_one_se_iters:
        for mode in one_se_symmetry_modes:
            output_dir = output_dir_for(mode)

            for eps in eps_values:
                for order in orders:
                    for dense in denses:
                        opts = dict(
                            dense=dense,
                            beta=beta,
                            eps=eps,
                            ppsc_tol=ppsc_tol,
                            order=order,
                            output_dir=output_dir,
                        )

                        for l, Fs in shells:
                            one_se_iter_slater_condon_bethe_half_filling(
                                conserved_operators=conserved_operators_for(l, mode),
                                l=l, Fs=Fs, **opts)

    if run_one_spgf_iters:
        for mode in one_spgf_symmetry_modes:
            for eps in eps_values:
                for order in orders:
                    for dense in denses:
                        opts = dict(
                            dense=dense,
                            beta=beta,
                            eps=eps,
                            ppsc_tol=ppsc_tol,
                            order=order,
                        )

                        for l, Fs in shells:
                            one_spgf_iter_slater_condon_bethe_half_filling(
                                conserved_operators=conserved_operators_for(l, mode),
                                l=l, Fs=Fs, **opts)


if __name__ == '__main__':
    main()
