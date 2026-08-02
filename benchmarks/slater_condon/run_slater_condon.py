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
# CONFIGURATION -- edit per job
# ----------------------------------------------------------------------------

# Which benchmark(s) to run
run_full_solves = False
run_one_se_no_sym_iters = False
run_one_se_all_sym_iters = True
run_one_spgf_no_sym_iters = False

# Perturbation orders to sweep (full solves use `full_solve_order`)
orders = [1, 2, 3]

# Solver frontends to sweep: False -> BlockSparseSolver, True -> dense TriqsSolver
denses = [False, True]

# Shells to sweep, as (l, Fs) pairs
shells = [
    (0, [3.0]),
    (1, [3.0, 0.5]),
    # (2, [3.0, 0.5, 0.3]),
]

# DLR accuracies to sweep
eps_values = [1e-9]

# Shared solver parameters
beta = 1.0
ppsc_tol = 1e-4

# Full-solve-only parameters
full_solve_order = 1
ppsc_maxiter = 10


def conserved_operators_for(l, use_symmetries):
    """Conserved operators handed to BlockSparseSolver for shell `l`.

    Edit this to change the symmetry content of a job, e.g. return
    'automatic' to let the solver detect the symmetries itself.
    """

    if not use_symmetries:
        return []
    return [_N_tot_operator(l)]


# ----------------------------------------------------------------------------
# DRIVER
# ----------------------------------------------------------------------------

def main():

    if run_full_solves:
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
                    conserved_operators=conserved_operators_for(l, use_symmetries=True),
                    l=l, Fs=Fs, **opts)

    for use_symmetries, enabled in ((False, run_one_se_no_sym_iters),
                                    (True, run_one_se_all_sym_iters)):
        if not enabled:
            continue

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
                        one_se_iter_slater_condon_bethe_half_filling(
                            conserved_operators=conserved_operators_for(l, use_symmetries),
                            l=l, Fs=Fs, **opts)

    if run_one_spgf_no_sym_iters:
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
                            conserved_operators=conserved_operators_for(l, use_symmetries=False),
                            l=l, Fs=Fs, **opts)


if __name__ == '__main__':
    main()
