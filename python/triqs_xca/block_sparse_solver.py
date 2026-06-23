import copy
import numpy as np

from collections import defaultdict

import triqs.utility.mpi as mpi

from triqs.gfs import Gf, MeshDLRImTime, BlockGf, make_gf_dlr, make_gf_dlr_imfreq
from triqs.atom_diag import AtomDiag, AtomDiagReal, AtomDiagComplex

from .diag import all_pairings, all_connected_pairings

from .dlr_dyson_ppsc import DysonItPPSC

from .block_sparse import DiagramEvaluator
from .dense import DenseDiagramEvaluator

from .block_sparse import trace
from .block_sparse import convolve_ppsc as conv
from .block_sparse import expectation_value

from .ase.utils.timing import Timer, timer


def is_root():
    return mpi.is_master_node()


def scatter_array_over_ranks(arr):
    size = mpi.size
    rank = mpi.rank
    arr_rank = np.array_split(np.array(arr), size, axis=0)[rank]
    return arr_rank


class BlockSparseSolver(object):
    
    """ Solver class for triqs_xca using the block sparse algorithm. 
    
    Parameters
    ----------
    H_loc : triqs.operators.Operator
        Local Hamiltonian, including both quadratic and quartic terms (:math:`H_{\\textrm{loc}} = \\sum c^\\dagger c + \\sum c^\\dagger c^\\dagger c c`).
    beta : float
        Inverse temperature :math:`\\beta`.
    w_max : float
        Imaginary time DLR discretization frequency cutoff :math:`\\omega_{\\text{max}}`.
    eps : float
        Imaginary time DLR accuracy tolerance :math:`\\epsilon`.
    gf_struct : dict
        Triqs style Green's function structure, e.g. ``[('up', 1), ('down', 1)]``, matching the operator content of ``H_loc``.
    conserved_operators : list, optional
        List of conserved operators (``triqs.operators.Operator`` instances). 
        Default is ``'automatic'`` using the autopartition algorithm of ``triqs.atom_diag``.
        To disable symmetries and use the dense diagram evaluator (``DenseDiagramEvaluator``), pass an empty list ``[]``.
    timer : Timer, optional
        Timer for performance measurements. Default: ``None`` (will be setup automatically).
    atom_diag : AtomDiag, optional
        Triqs AtomDiag object ``triqs.atom_diag.AtomDiag``. Default: ``None``. 
        If not provided, it will be initialized automatically using the provided local Hamiltonian and conserved operators.
    verbose : bool/int, optional
        Verbosity flag controlling level of printouts. Default: ``True``.

    Notes
    -----

    The hybridization function is available as the ``Delta_tau`` attribute, which is a block Green's function in imaginary time. 
    It has to be set by the user before calling ``solve()``. If not set by the user, it will be initialized to zero.

    The main method of the class is ``solve()``, which runs the pseudo particle self-consistent perturbation theory until convergence.

    The single particle Green's function is available as the ``G_tau`` attribute after calling ``solve()``. 
    It is a block Green's function in imaginary time.

    """

    def __init__(self, H_loc, beta, w_max, eps, gf_struct, 
                 conserved_operators='automatic', dlr_symmetrize=False,
                 timer=None, atom_diag=None, verbose=True):

        self.dlr_symmetrize = dlr_symmetrize
        self.H_loc = H_loc
        self.gf_struct = gf_struct
        self.beta = beta
        self.w_max = w_max
        self.eps = eps
        self.conserved_operators = conserved_operators
        self.verbose = verbose
        self.has_dynamic_interactions = False        

        self.fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)
        self.use_dense_solver = (self.conserved_operators == []) # Without symmetries, use the dense solver

        self.timer = timer if timer is not None else Timer()

        if verbose and is_root():
            print(logo())
            print()
            if self.use_dense_solver: 
                print(f'use_dense_solver = {self.use_dense_solver}')
            if self.conserved_operators != 'automatic': 
                print(f'conserved_operators = {self.conserved_operators}')
        
        self.eta = 0. # Pseudo particle chemical potential (shift of the pseudo particle energies).

        self.mesh_tau = MeshDLRImTime(beta=self.beta, statistic='Fermion', w_max=self.w_max, eps=self.eps, symmetrize=self.dlr_symmetrize)

        self.Delta_tau = BlockGf(mesh=self.mesh_tau, gf_struct=self.gf_struct, name='Delta_tau')

        self.timer.start('AtomDiag Init')
        if atom_diag is not None:
            self.atom_diag = atom_diag
        elif self.conserved_operators == 'automatic':
            # Uses autopartition algorithm for symmetry discovery, when no conserved operators are provided. 
            # This is the default, and recommended, usage.
            self.atom_diag = AtomDiag(self.H_loc, self.fundamental_operators) 
        else:
            self.atom_diag = AtomDiag(self.H_loc, self.fundamental_operators, self.conserved_operators)
        self.timer.stop()

        if verbose and is_root(): 
            #print(f'mesh: {self.mesh_tau}')
            print(f'beta = {beta}, w_max = {w_max}, eps = {eps}, N_DLR = {len(self.mesh_tau)}')
            print_atom_diag_info(self.atom_diag)

        self.timer.start('Dyson')
        self.timer.start('Setup')
        self.G0, self.eta0 = atomic_pseudo_particle_greens_function(self.atom_diag, self.beta, self.mesh_tau)
        self.G = self.G0.copy()
        self.Sigma = self.get_zero_pseudo_particle_propagator()

        # FIXME! Get ito from mesh_tau
        from .pycppdlr import build_dlr_rf
        from .pycppdlr import ImTimeOps
        ito = ImTimeOps(w_max * beta, build_dlr_rf(w_max * beta, eps, self.dlr_symmetrize), symmetrize=self.dlr_symmetrize)

        # Compare DLR meshes
        #np.testing.assert_array_almost_equal(np.array([ float(t) for t in self.mesh_tau]), ito.get_itnodes() * beta)
        np.testing.assert_array_almost_equal(self.mesh_tau.dlr_freq, ito.get_rfnodes())

        self.dysons = [DysonItPPSC(self.beta, ito, G0_block.data) for _, G0_block in self.G0]
    
        self.timer.stop()
        self.timer.stop()


    @timer('Adapol hybridization fit')
    def fit_hybridization(self, tol=None, compression=False, verbose=True):

        Delta_tau_dense = self.__from_blockgf_to_dense(self.Delta_tau)

        if compression:

            assert( tol is not None and tol > 0 ), 'Error: tol must be provided and positive when compression is enabled.'

            from adapol.triqs import TriqsDLRCompression
        
            try:
                tdc = TriqsDLRCompression(Delta_tau_dense, tol=tol, verbose=verbose and is_root())
                poles, pole_weights, fit_error = tdc.poles, tdc.residues, tdc.error
            except ValueError as e:
                fit_error = None
                if is_root():
                    print(f'Adapol: WARNING! TriqsDLRCompression failed with error: {e}. Using direct DLR coefficients instead.')

            if verbose and is_root():
                if fit_error >= tol:
                    print(f'Adapol: WARNING! Fit error = {fit_error:2.2E} >= tol = {tol:2.2E}, N_poles = {len(poles)}')
                else:
                    print(f'Adapol: Fit error = {fit_error:2.2E} < tol = {tol:2.2E}, N_poles = {len(poles)}')

        if not compression or fit_error is None:

            Delta_dlr_dense = make_gf_dlr(Delta_tau_dense)
            poles = np.array([ float(x) for x in Delta_dlr_dense.mesh ]) / self.beta
            pole_weights = Delta_dlr_dense.data.copy()
            fit_error = Delta_dlr_dense.mesh.eps

            if verbose and is_root():
                print(f'Hybridization: using DLR expansion with N_poles = {len(poles)}')

        self.set_hybridization_poles_and_coefficients(poles, pole_weights)

        self.hyb.tol = tol
        self.hyb.compression = compression
        self.hyb.fit_error = fit_error


    def __from_blockgf_to_dense(self, G):

        for b, g in G:
            assert( len(g.target_shape) == 2)
            assert( g.target_shape[0] == g.target_shape[1] )

        norb = sum([ g.target_shape[0] for b, g in G ])
        
        G_dense = Gf(mesh=G.mesh, target_shape=[norb]*2)
        
        sidx = 0
        for b, g in G:
            size = g.target_shape[0]
            G_dense.data[:, sidx:sidx+size, sidx:sidx+size] = g.data
            sidx += size

        return G_dense


    def __from_dense_to_blockgf(self, G_dense, gf_struct):

        G = BlockGf(mesh=G_dense.mesh, gf_struct=gf_struct)

        sidx = 0
        for b, g in G:
            size = g.target_shape[0]
            g.data[:] = G_dense.data[:, sidx:sidx+size, sidx:sidx+size]
            sidx += size

        return G


    def set_hybridization_poles_and_coefficients(self, poles, coefficients):

        self.hyb = PoleRepresentation(poles, coefficients)

        #if is_root():
            #print(f'hyb poles = {len(self.hyb.poles)}')
            #print(f'hyb.poles = {self.hyb.poles}')
            #print(f'hyb.coefficients =\n{self.hyb.coefficients}')


    def set_dynamic_interactions(self, dynint_ops, dynint_coeffs):
        self.has_dynamic_interactions = True
        self.dynint_ops = dynint_ops
        self.dynint_coeffs = dynint_coeffs


    @timer('DiagramEvaluator Init')
    def init_diagram_evaluator(self):
        #if is_root(): print(f'Initializing diagram evaluator with use_dense_solver = {self.use_dense_solver}')
        if self.use_dense_solver:
            if self.has_dynamic_interactions:
                self.d = DenseDiagramEvaluator(self.hyb.poles, self.hyb.coefficients, self.mesh_tau, self.atom_diag, self.dynint_ops, self.dynint_coeffs)
            else:
                self.d = DenseDiagramEvaluator(self.hyb.poles, self.hyb.coefficients, self.mesh_tau, self.atom_diag)
        else:
            if self.has_dynamic_interactions:
                raise NotImplementedError('Dynamic interactions are not yet implemented for the block sparse solver (DiagramEvaluator).')
            self.d = DiagramEvaluator(self.hyb.poles, self.hyb.coefficients, self.mesh_tau, self.atom_diag)
        #if is_root(): print(f'done.')


    def solve(self, max_order, tol=1e-4, maxiter=10, mix=1., hyb_tol=None, hyb_comp=True, normalization='classic', spgf_max_order=None, verbose=True):
        """ Solve the impurity problem using pseudo particle self-consistent perturbation theory.
        
        Parameters
        ----------
        max_order : int
            Maximum order of the perturbation theory.
        tol : float, optional
            Tolerance for the convergence criterion.
        maxiter : int, optional
            Maximum number of iterations.
        mix : float, optional
            Mixing parameter for the self-consistent iteration.
        hyb_tol : float, optional
            Tolerance for the hybridization function compression. Defaults to ``0.1 * tol`` if not provided.
        hyb_comp : bool, optional
            Whether to compress the hybridization function. Default: ``True``.
            When set to ``False``, the hybridization function is represented using the full DLR basis.
        spgf_max_order : int, optional
            Maximum order for the single particle Green's function evaluation. If not provided, it defaults to ``max_order``.
        normalization : str, optional
            Normalization method for the pseudo particle Green's function. Default: ``'classic'``.

        Returns
        -------
        None

        Notes
        -----

        Available normalization methods:

        * ``'classic'``: Classical normalization (Default)
        * ``'root'``: Root normalization
        * ``'ode+classic'``: ODE with classic normalization
        * ``'ode+root'``: ODE with root normalization
        * ``'odeG+classic'``: ODE on G with classic normalization
        * ``'odeG+root'``: ODE on G with root normalization

        """

        self.normalization = normalization
        self.max_order = max_order
        self.spgf_max_order = spgf_max_order if spgf_max_order is not None else max_order

        self.hyb_tol = hyb_tol if hyb_tol is not None else 0.1 * tol

        self.hyb_comp = hyb_comp if max_order > 1 else False # Skip hybridization compression for max_order = 1 (NCA) since the pole representation is not used.
        
        #if verbose and is_root(): print(f'hyb_tol = {self.hyb_tol:2.2E}')

        self.fit_hybridization(tol=self.hyb_tol, compression=self.hyb_comp, verbose=verbose)
        self.init_diagram_evaluator() # FIXME! Evaluator takes hyb poles and coeffs in constructor

        iter = 0
        for iter in range(1, maxiter+1):

            #if is_root(): print(f'Sigma max_order = {self.max_order}')
            self.Sigma = self.eval_pseudo_particle_self_energy(self.G, self.max_order, verbose=verbose > 1)

            self.timer.start('Dyson')

            if normalization == 'classic':
                G_new = self.solve_dyson(self.Sigma, self.eta)
                G_new = self.normalize_pseudo_particle_gf(G_new)
                #G_new, Sigma_new = self.normalize_pseudo_particle_gf_and_sigma(G_new, self.Sigma)

            elif normalization == 'root':
                self.solve_ppsc_chempot_newton(xtol=10*self.eps)
                G_new = self.solve_dyson(self.Sigma, self.eta)

            elif normalization == 'ode+root':
                self.solve_ppsc_chempot_adiabatic_ode(tol=100*self.eps)               
                self.solve_ppsc_chempot_newton(xtol=10*self.eps)
                G_new = self.solve_dyson(self.Sigma, self.eta)

            elif normalization == 'ode+classic':
                self.solve_ppsc_chempot_adiabatic_ode(tol=100*self.eps)                                
                G_new = self.solve_dyson(self.Sigma, self.eta)
                G_new = self.normalize_pseudo_particle_gf(G_new)

            elif normalization == 'odeG+root':
                self.solve_ppsc_chempot_adiabatic_ode_G(tol=100*self.eps)               
                self.solve_ppsc_chempot_newton(xtol=10*self.eps)
                G_new = self.solve_dyson(self.Sigma, self.eta)

            elif normalization == 'odeG+classic':
                self.solve_ppsc_chempot_adiabatic_ode_G(tol=100*self.eps)                                
                G_new = self.solve_dyson(self.Sigma, self.eta)
                G_new = self.normalize_pseudo_particle_gf(G_new)

            else:
                raise NotImplementedError(f'Normalization method {normalization} not implemented.')

            Z = self.partition_function_from_ppgf(G_new)

            self.timer.stop()

            self.diff_G = max_abs_diff_BlockGf(G_new, self.G)

            if verbose and is_root(): print(f'iter = {iter}, diff_G = {self.diff_G:2.2E}, Z-1 = {Z-1:+2.2E}, eta = {self.eta:2.2E}')

            if self.diff_G < tol:
                if verbose and is_root(): print(f'Converged after {iter} iterations with diff_G = {self.diff_G:2.2E} < tol = {tol:2.2E}')
                break

            self.G = mix * G_new + (1 - mix) * self.G

        G_tau = self.eval_single_particle_greens_function(self.G, max_order=self.spgf_max_order)
        self.G_tau = self.__from_dense_to_blockgf(G_tau, self.gf_struct)

        self.n_ppsc_iter = iter

        if verbose and is_root():
            print(); self.timer.write()


    def solve_bare(self, max_order, hyb_tol=None, hyb_comp=True, use_dyson=False, spgf_max_order=None, verbose=True):
        """ Solve the impurity problem using the bare pseudo particle self-consistent perturbation theory, 
        i.e. with the pseudo particle self-energy evaluated using the atomic pseudo particle Green's function.
         
        .. math::
            \\Sigma = \\Sigma[G_0]

        The pseudo-particle Green's function :math:`G` is then obtained by

        .. math::
            G = G_0 + G_0 \\ast \\Sigma \\ast G_0

        or when ``use_dyson=True`` by solving the Dyson equation

        .. math::
            (1 - G_0 \\ast \\Sigma \\ast \\, ) G = G_0 
             
        This is useful for benchmarking and testing purposes.
        
        Parameters
        ----------
        max_order : int
            Maximum order of the perturbation theory.
        hyb_tol : float, optional
            Tolerance for the hybridization function compression. Defaults to ``10 * self.eps`` if not provided.
        hyb_comp : bool, optional
            Whether to compress the hybridization function. Default: ``True``.
            When set to ``False``, the hybridization function is represented using the full DLR basis.
        use_dyson : bool, optional
            Whether to solve the Dyson equation to obtain the pseudo particle Green's function. Default: ``False``.
        spgf_max_order : int, optional
            Maximum order for the single particle Green's function evaluation. If not provided, it defaults to ``max_order``.
        verbose : bool, optional
            Verbosity flag controlling level of printouts. Default: ``True``.
        """

        self.max_order = max_order
        self.spgf_max_order = spgf_max_order if spgf_max_order is not None else max_order

        self.hyb_tol = hyb_tol if hyb_tol is not None else 10 * self.eps
        self.hyb_comp = hyb_comp if max_order > 1 else False # Skip hybridization compression for max_order = 1 (NCA) since the pole representation is not used.
        
        self.fit_hybridization(tol=self.hyb_tol, compression=self.hyb_comp, verbose=verbose)
        self.init_diagram_evaluator() # FIXME! Evaluator takes hyb poles and coeffs in constructor

        self.Sigma = self.eval_pseudo_particle_self_energy(self.G0, self.max_order, connected=False, verbose=verbose > 1)

        if use_dyson:   
            self.timer.start('Dyson')
            G = self.solve_dyson(self.Sigma, self.eta)
        else:
            self.timer.start('G = G0 + G0 Sigma G0')
            G = self.G0 + conv(self.G0, conv(self.Sigma, self.G0)) # Add self.eta here? (eta * G0 * G0)

        G = self.normalize_pseudo_particle_gf(G)
        Z = self.partition_function_from_ppgf(G)

        self.timer.stop()

        if verbose and is_root(): print(f'Bare: Z-1 = {Z-1:+2.2E}, eta = {self.eta:2.2E}')

        self.G = G

        G_tau = self.eval_single_particle_greens_function(self.G, max_order=self.spgf_max_order)
        self.G_tau = self.__from_dense_to_blockgf(G_tau, self.gf_struct)

        if verbose and is_root():
            print(); self.timer.write()
        

    def normalize_pseudo_particle_gf(self, G):
        """ Normalize the pseudo particle Green's function by updating the pseudo particle chemical potential eta, such that the partition function Z is equal to 1. """

        Z = self.partition_function_from_ppgf(G)

        deta = np.log(np.abs(Z)) / self.beta

        def energy_shift_ppgf(G, deta):
            G_new = G.copy()
            tau = np.array([ float(t) for t in G.mesh ])
            for bidx, g in G_new:
                g.data[:] *= np.exp(-tau * deta)[:, None, None]
            return G_new

        G_new = energy_shift_ppgf(G, deta)

        Z_new = self.partition_function_from_ppgf(G_new)

        #if is_root(): print(f'  Update eta = {self.eta:2.2E}, Z = {Z:2.2E}, Z_new-1 = {Z_new-1:+2.2E}')

        self.eta += deta

        return G_new
    

    def normalize_pseudo_particle_gf_and_sigma(self, G, Sigma):
        """ Normalize the pseudo particle Green's function by updating the pseudo particle chemical potential eta, such that the partition function Z is equal to 1. """

        Z = self.partition_function_from_ppgf(G)

        deta = np.log(np.abs(Z)) / self.beta

        def energy_shift_ppgf(G, deta):
            G_new = G.copy()
            tau = np.array([ float(t) for t in G.mesh ])
            for bidx, g in G_new:
                g.data[:] *= np.exp(-tau * deta)[:, None, None]
            return G_new

        G_new = energy_shift_ppgf(G, deta)
        Sigma_new = energy_shift_ppgf(Sigma, deta)

        Z_new = self.partition_function_from_ppgf(G_new)

        #if is_root(): print(f'  Update eta = {self.eta:2.2E}, Z = {Z:2.2E}, Z_new-1 = {Z_new-1:+2.2E}')

        self.eta += deta

        return G_new, Sigma_new


    def pseudo_particle_greens_function(self):
        return self.G


    def partition_function_from_ppgf(self, G):
        Z = -trace(G)
        #print(f'partition_function_from_ppgf: Z = {Z}, eps = {self.eps}')
        #assert(Z.imag < 1e-12)
        #assert(Z.imag < 10 * self.eps)
        return Z.real


    def partition_function(self):
        """ Partition function :math:`Z` of the impurity model, computed from the pseudo particle Green's function as
        
        .. math::
            Z = -\\mathrm{Tr}[ G(\\beta) ] \\cdot e^{-\\beta \\eta}
          
        Returns
        -------
        float
            Partition function :math:`Z` of the impurity model.
        """
        return self.partition_function_from_ppgf(self.G) * np.exp(-self.beta * self.eta)


    def pseudo_particle_chemical_potential(self):
        """ Pseudo particle chemical potential :math:`\\eta`.

        The pseudo particle chemical potential :math:`\\eta` is a shift of the pseudo particle energies used to enforce
        the normalization condition :math:`\\textrm{Tr}[G(\\beta)] = -1` on the pseudo particle Green's function :math:`G`.

        Returns
        -------
        float
            Pseudo particle chemical potential :math:`\\eta`.
        """
        return self.eta


    def expectation_value(self, operator):
        """ Expectation value :math:`\\langle \\hat{O} \\rangle` of an operator :math:`\\hat{O}`, computed from the pseudo particle Green's function as
        
        .. math::
            \\langle \\hat{O} \\rangle = -\\mathrm{Tr}[ \\hat{O} G(\\beta) ]
          
        Parameters
        ----------
        operator : triqs.operators.Operator
            Operator :math:`\\hat{O}` for which the expectation value is computed. 
            It has to be compatible with the operator content of the local Hamiltonian and the Green's function structure.

        Returns
        -------
        float
            Expectation value :math:`\\langle \\hat{O} \\rangle` of the operator :math:`\\hat{O}`.
        """
        return expectation_value(operator, self.atom_diag, self.G)
    

    def many_body_density_matrix(self, G):
        """ Get the block sparse many body density matrix :math:`\\rho` from the pseudo particle Green's function :math:`G`.
        
        Returns
        -------
        
        rho : list of tuples(str, np.ndarray)
            List of tuples ``(bidx, rho_b)``, where ``bidx`` is the block index 
            and ``rho_b`` is the many body density matrix for block ``bidx``
        """
        rho = []
        for bidx, g_b in G:
            g_b_dlr = make_gf_dlr(g_b)
            rho_b = -g_b_dlr(g_b.mesh.beta)
            rho.append((bidx, rho_b))
        return rho
    

    def herm_err(self, G):
        """ Compute maximum hermiticity error of a block Green's function :math:`G` """
        err = 0.
        for bidx, g_b in G:
            err_b = np.max(np.abs(g_b.data - np.conj(g_b.data.transpose(0,2,1))))
            err = max(err, err_b)
        return err


    def make_hermitian(self, G):
        """ Enforce hermiticity of a block Green's function :math:`G` by symmetrizing it with its conjugate transpose. """
        for bidx, g_b in G:
            g_b.data[:] = 0.5 * (g_b.data + g_b.data.transpose(0,2,1).conjugate())
        return G


    @timer('Solve')
    def solve_dyson(self, Sigma, eta):

        assert type(Sigma) is BlockGf, 'Sigma must be a BlockGf'

        G = self.get_zero_pseudo_particle_propagator()

        for dyson, (bidx, sigma_b) in zip(self.dysons, Sigma):
            #if is_root(): print(f'solve_dyson: block {bidx}, sigma_b.data.shape = {sigma_b.data.shape}')
            #self.timer.start(f'Dyson block {bidx}')
            G[bidx].data[:] = dyson.solve(sigma_b.data, eta)

            # The Dyson solver does not enforce hermiticity of the solution, which can lead to numerical issues in the self-consistent iteration.
            # Enforce hermiticity explicitly here, which is important for convergence of the self-consistent iteration.
            G[bidx].data[:] = 0.5*(G[bidx].data + G[bidx].data.conjugate().transpose(0,2,1))

            #self.timer.stop()

        return G


    def pseudo_particle_self_energy(self):
        return self.Sigma


    @timer('Sigma')
    def eval_pseudo_particle_self_energy(self, G, max_order, connected=True, verbose=False):

        self.Sigma = self.get_zero_pseudo_particle_propagator()

        for order in range(1, max_order+1):
            with self.timer(f'Order {order}'):
                self.Sigma += self.__eval_pseudo_particle_self_energy_order(G, order, connected, verbose=verbose)

        return self.Sigma


    def __eval_pseudo_particle_self_energy_order(self, G, order, connected, verbose=False):

        Sigma = self.get_zero_pseudo_particle_propagator()
        
        pairings = all_connected_pairings if connected else all_pairings

        for sign, topology in pairings(order):
            if verbose and is_root(): print(f'SIGMA: O{order} topo {topology} sign {sign:+d}')
            topology = np.array(topology, dtype=np.int32)
            Sigma -= self.__eval_pseudo_particle_self_energy_topology_loop(G, topology, verbose=verbose)
            
        return Sigma
    
    
    def eval_pseudo_particle_self_energy_topology(self, G, topology):
        order = len(topology)
        return self.d.compute_self_energy(G, topology)


    def __eval_pseudo_particle_self_energy_topology_loop(self, G, topology, verbose=False):

        order = len(topology)
        n_max = self.d.get_num_self_energy_backbones(topology)
        n_vec = scatter_array_over_ranks(np.arange(n_max, dtype=np.int32))

        Sigma = self.get_zero_pseudo_particle_propagator()
        Sigma = self.d.compute_self_energy(G, topology, n_vec)
        for bidx, sigma_b in Sigma:
            sigma_b.data[:] = mpi.all_reduce(sigma_b.data)

        return Sigma
    

    def get_zero_pseudo_particle_propagator(self):
        return zero_pseudo_particle_propagator(self.atom_diag, self.mesh_tau)


    def get_zero_single_particle_greens_function(self):
        spgf = Gf(mesh=self.mesh_tau, target_shape=[len(self.fundamental_operators)]*2)
        spgf.data[:] = 0.
        return spgf


    def single_particle_greens_function(self, max_order):
        return self.eval_single_particle_greens_function(self.G, max_order=max_order)


    @timer('Single-particle Gf')
    def eval_single_particle_greens_function(self, G, max_order):
        self.spgf = self.get_zero_single_particle_greens_function()

        for order in range(1, max_order+1):
            with self.timer(f'Order {order}'):
                self.spgf += self.__eval_single_particle_greens_function_order(G, order)
        
        return self.spgf


    def __eval_single_particle_greens_function_order(self, G, order):

        spgf = self.get_zero_single_particle_greens_function()

        for sign, topology in all_connected_pairings(order):
            topology = np.array(topology, dtype=np.int32)
            spgf += pow(-1, order) * \
                self.__eval_single_particle_greens_function_topology_loop(G, topology)

        return spgf


    def eval_single_particle_greens_function_topology(self, G, topology):

        spgf = self.get_zero_single_particle_greens_function()

        spgf.data[:] = self.d.compute_single_ptcle_gf(G, topology) # FIXME! return triqs::gfs::gf

        return spgf


    def __eval_single_particle_greens_function_topology_loop(self, G, topology):

        n_max = self.d.get_num_single_ptcle_gf_backbones(topology)
        n_vec = scatter_array_over_ranks(np.arange(n_max, dtype=np.int32))

        spgf = self.get_zero_single_particle_greens_function()
        if self.has_dynamic_interactions:
            n_orb = spgf.target_shape[0]
            # FIXME! With the dense solver we compute all operator combinations not only single particle ones
            # fix this by adjusting the dense spgf calculator
            spgf.data[:] = self.d.compute_single_ptcle_gf(G, topology, n_vec)[:, :n_orb, :n_orb]  
        else:
            spgf.data[:] = self.d.compute_single_ptcle_gf(G, topology, n_vec)
        spgf.data[:] = mpi.all_reduce(spgf.data)

        return spgf


    @timer('One-time correlator')
    def eval_one_time_correlator(self, G, max_order, ops_tau, ops_0):
        """ Evaluate a one time correlator of the form

        .. math::
            C(\\tau) = \\langle T_\\tau O(\\tau) O(0) \\rangle
        
        where :math:`O(\\tau)` and :math:`O(0)` are operators in imaginary time, 
        and :math:`T_\\tau` is the imaginary time ordering operator. 
        
        The operators are specified by the lists of fundamental operators 
        ``ops_tau`` and ``ops_0``, which contain the operators at imaginary time 
        :math:`\\tau` and :math:`0`, respectively.
        """
        corr = Gf(mesh=self.mesh_tau, target_shape=[len(ops_tau), len(ops_0)])
        corr.data[:] = 0.

        for order in range(1, max_order+1):
            with self.timer(f'Order {order}'):
                self.__inplace_eval_one_time_correlator_order(corr, G, order, ops_tau, ops_0)

        return corr


    def __inplace_eval_one_time_correlator_order(self, corr, G, order, ops_tau, ops_0):

        prefactor = pow(-1, order)
        for sign, topology in all_connected_pairings(order):
            topology = np.array(topology, dtype=np.int32)
            self.__inplace_eval_one_time_correlator_topology_loop(
                corr, G, topology, ops_tau, ops_0, prefactor)
    

    def __inplace_eval_one_time_correlator_topology_loop(self, corr, G, topology, ops_tau, ops_0, prefactor):

        n_max = self.d.get_num_single_ptcle_gf_backbones(topology)
        n_vec = scatter_array_over_ranks(np.arange(n_max, dtype=np.int32))

        corr_arr = prefactor * self.d.compute_one_time_correlator(
            G, ops_tau, ops_0, self.atom_diag, topology, n_vec)
        corr.data[:] += mpi.all_reduce(corr_arr)


    def Z_alpha(self, alpha, eta):
        r""" Partition function with Sigma scaled by alpha 
        Solving
        
        .. math::
            (1 - G_0\ast*(\alpha \cdot \Sigma - \eta)\ast*)G = G_0

            Z_\alpha = - \textrm{Tr} \left[ G_\alpha (\beta) \right]
        
        """

        #if is_root(): print(f'Z_alpha: alpha = {alpha}, eta = {eta}')

        if type(eta) == np.ndarray: eta = eta.item()
        
        Sigma = self.pseudo_particle_self_energy()
        #Ga = self.solve_dyson(alpha**2 * Sigma, eta)
        Ga = self.solve_dyson(alpha * Sigma, eta)
        Za = -trace(Ga)
        #Za = self.partition_function_from_ppgf(Ga).real
        return Za    


    def dZ_alpha_deta(self, alpha, eta):
        r""" Derivative of partition function Z_\alpha with respect to
        the pseudo-particle chemical potential eta. 
        
        .. math::
            \frac{d Z_\alpha}{d \eta} = - \textr{Tr} \left[G_\alpha G_\alpha \right]
        """

        if is_root(): print(f'dZ_alpha_deta: alpha = {alpha}, eta = {eta}')

        if type(eta) == np.ndarray: eta = eta.item()

        Sigma = self.pseudo_particle_self_energy()
        #G = self.solve_dyson(alpha**2 * Sigma, eta)
        G = self.solve_dyson(alpha * Sigma, eta)
        dZ_deta = -trace(conv(G, G)).real
        return dZ_deta


    def Z_alpha_minus_one_and_derivative(self, alpha, eta):
        r""" Compute Z_alpha - 1 and its derivative with respect to eta, for root finding in the Newton solver. """

        if is_root(): print(f'Z_alpha_minus_one_and_derivative: alpha = {alpha}, eta = {eta}')

        if type(eta) == np.ndarray: eta = eta.item()

        Sigma = self.pseudo_particle_self_energy()
        #G = self.solve_dyson(alpha**2 * Sigma, eta)
        G = self.solve_dyson(alpha * Sigma, eta)
        Za = self.partition_function_from_ppgf(G).real
        dZ_deta = -trace(conv(G, G)).real
        return Za - 1, dZ_deta
    

    def Z_alpha_minus_one_and_derivative_and_2nd_derivative(self, alpha, eta):
        r""" Compute Z_alpha - 1 and its derivatives with respect to eta, for root finding in the Newton solver. """

        if is_root(): print(f'Z_alpha_minus_one_and_derivative_and_2nd_derivative: alpha = {alpha}, eta = {eta}')

        if type(eta) == np.ndarray: eta = eta.item()

        Sigma = self.pseudo_particle_self_energy()
        #G = self.solve_dyson(alpha**2 * Sigma, eta)
        G = self.solve_dyson(alpha * Sigma, eta)
        #Za = self.partition_function_from_ppgf(G).real
        Za = -trace(G)
        print(f'Za = {Za}')
        Za = Za.real
        G2 = conv(G, G)
        dZ_deta = -trace(G2).real
        d2Z_deta2 = -2 * trace(conv(G2, G)).real
        return Za - 1, dZ_deta, d2Z_deta2


    def deta_dalpha(self, alpha, eta):
        r""" Analytic derivative of :math:`\eta(\alpha)`

        .. math::
            \frac{d \eta}{d \alpha} = 
                - \textr{Tr} \left[ G_\alpha \ast \Sigma \ast G_\alpha \right]
                / \textrm{Tr} \left[ G_\alpha \ast G_\alpha \right]

        """
        if is_root(): print(f'deta_dalpha: alpha = {alpha}, eta = {eta}')
        Sigma = self.pseudo_particle_self_energy()
        #Ga = self.solve_dyson(alpha**2 * Sigma, eta)
        Ga = self.solve_dyson(alpha * Sigma, eta)
        TrGaGa = -trace(conv(Ga, Ga))
        TrGaSigmaGa = -trace(conv(Ga, conv(Sigma, Ga)))
        #deta_dalpha = - 2 * alpha * TrGaSigmaGa / TrGaGa
        deta_dalpha = - TrGaSigmaGa / TrGaGa
        if is_root(): print(f'deta_dalpha: Za = {-trace(Ga)}, deta/dalpha = {deta_dalpha}')

        return deta_dalpha


    def derivative_components(self, alpha, eta):
        if is_root(): print(f'derivative_components: alpha = {alpha}, eta = {eta}')
        Sigma = self.pseudo_particle_self_energy()
        Ga = self.solve_dyson(alpha * Sigma, eta)
        TrGaGa = -trace(conv(Ga, Ga))
        TrGaSigmaGa = -trace(conv(Ga, conv(Sigma, Ga)))
        return TrGaGa, TrGaSigmaGa


    def d2eta_dalpha_deta(self, alpha, eta):
        r""" Higher order derivative (used for stiff ODE solver)
        
        .. math:: 
            \frac{d}{d\eta} \left (\frac{d \eta}{d \alpha})
            =
            - \textr{Tr} \left[ G^2_\alpha \ast \Sigma \ast G_\alpha \right]
            / \textrm{Tr} \left[ G_\alpha \ast G_\alpha \right]
            +
            \textr{Tr} \left[ G_\alpha \ast \Sigma \ast G^2_\alpha \right]
            / \textrm{Tr} \left[ G_\alpha \ast G_\alpha \right]
            - 
            2 \textrm{Tr} \left[ G_\alpha \ast G_\alpha \ast G_\alpha \right]
            / \textrm{Tr} \left[ G_\alpha \ast G_\alpha \right]^2

        """
    
        #raise NotImplementedError('d2eta_dalpha_deta is not implemented yet, and may contain errors in the formula.')
    
        print('--> d2eta_dalpha_deta')

        Sigma = self.pseudo_particle_self_energy()

        #Ga = self.solve_dyson(alpha**2 * Sigma, eta) # fix to alpha * Sigma
        Ga = self.solve_dyson(alpha * Sigma, eta)
        
        Ga2 = conv(Ga, Ga)
        Ga3 = conv(Ga, Ga2)
        
        TrGa2 = -trace(Ga2)
        TrGa3 = -trace(Ga3)

        TrGaSigmaGa  = -trace(conv(Ga,  conv(Sigma, Ga )))
        TrGa2SigmaGa = -trace(conv(Ga2, conv(Sigma, Ga )))
        TrGaSigmaGa2 = -trace(conv(Ga,  conv(Sigma, Ga2)))

        # Fix for alpha * Sigma
        #d2eta_dalpha_deta = -2*alpha * ((TrGa2SigmaGa + TrGaSigmaGa2)/TrGa2 - 2*TrGaSigmaGa*TrGa3/TrGa2**2)
        d2eta_dalpha_deta = -(TrGa2SigmaGa + TrGaSigmaGa2)/TrGa2 + 2*TrGaSigmaGa*TrGa3/TrGa2**2
        
        return d2eta_dalpha_deta.real
    
    
    @timer('ODE normalization')
    def solve_ppsc_chempot_adiabatic_ode(
        self, tol=1e-9, 
        method='RK23',
        #method='DOP853',
        #method='RK45',
        #method='LSODA',
        ):

        func = lambda t, y : self.deta_dalpha(t, y[0]).real
        jac = lambda t, y : np.array([self.d2eta_dalpha_deta(t, y[0])]).reshape(1, 1).real

        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            func, (0., 1.), 
            np.array([0.]), 
            #jac=jac,
            atol=tol, 
            method=method,
            dense_output=True,
            )
        
        self.eta = sol.y[0, -1]
        G = self.solve_dyson(self.Sigma, self.eta)
        Z = self.partition_function_from_ppgf(G)
        
        if is_root(): print(f'PPSC: Chempot eta from adiabatic ODE: Z-1={Z-1:+2.2E}')
        return sol


    @timer('ODE normalization G')
    def solve_ppsc_chempot_adiabatic_ode_G(
        self, tol=1e-9, 
        method='RK23',
        ):

        def from_array(y):
            G = self.get_zero_pseudo_particle_propagator()
            idx = 0
            for bidx, g in G:
                size = np.prod(g.data.shape)
                g.data[:] = y[idx:idx+size].reshape(g.data.shape)
                idx += size
            return G
        
        def to_array(G):
            arr = []
            for bidx, g in G:
                arr.append(g.data.flatten())
            return np.concatenate(arr)
        
        def get_eta(G0, G, S):
            
            TrG0G = trace(conv(G0, G))
            TrG0SG  = trace(G - G0 - conv(G0, conv(S, G)))

            eta = TrG0SG / TrG0G
            return eta

        def func(t, y):

            G = from_array(y)

            S = self.pseudo_particle_self_energy()
        
            GG = conv(G, G)
            GSG = conv(G, conv(S, G))
            
            TrGG = trace(GG)
            TrGSG  = trace(GSG)

            dG = GSG - GG * TrGSG / TrGG

            dy = to_array(dG)
            
            print(f'func: t = {t}, trace(G) = {trace(G)}, max(abs(dG)) = {np.max(np.abs(dy))}')

            return dy

        from scipy.integrate import solve_ivp

        y0 = to_array(self.G0)

        sol = solve_ivp(
            func, (0., 1.), y0,
            atol=tol, 
            method=method,
            dense_output=False,
            )
        

        G = from_array(sol.y[:, -1])
        self.eta = get_eta(self.G0, G, self.Sigma).real


        G = self.solve_dyson(self.Sigma, self.eta)
        Z = self.partition_function_from_ppgf(G)

        if is_root(): print(f'PPSC: Chempot eta from adiabatic ODE: Z-1={Z-1:+2.2E}, eta = {self.eta:2.2E}')

        return sol


    @timer('Root normalization')
    def solve_ppsc_chempot_newton(self, **kwargs):

        if is_root(): print(f'PPSC: Starting root solver for eta with kwargs = {kwargs}')
        
        #f = lambda eta : self.Z_alpha(1., eta) - 1
        #fprime = lambda eta : self.dZ_alpha_deta(1., eta)
        #f = lambda eta : self.Z_alpha_minus_one_and_derivative(1., eta)
        f = lambda eta : self.Z_alpha_minus_one_and_derivative_and_2nd_derivative(1., eta)
        
        from scipy.optimize import root_scalar

        sol = root_scalar(
            f, x0=self.eta, 
            #fprime=fprime,
            fprime=True,
            fprime2=True,
            #method='newton', # Newton's method, using the first derivative
            method='halley', # Halley's cubic method, using the first and second derivatives
            #xtol=tol,
            **kwargs,
            )
        
        self.eta = sol.root

        G = self.solve_dyson(self.Sigma, self.eta)
        Z = self.partition_function_from_ppgf(G)

        if is_root(): print(f'PPSC: Root solver done, eta = {self.eta:+2.2E} Z-1 = {Z-1:+2.2E}')

        return sol
    

    @timer('Bisection normalization')
    def solve_ppsc_chempot_bisection(self, **kwargs):

        if is_root(): print(f'PPSC: Starting bisection solver for eta with kwargs = {kwargs}')
        
        def func(eta):
            Sigma = self.pseudo_particle_self_energy()
            G = self.solve_dyson(Sigma, eta)

            # Try to detect extreme limits by oscillations in Ga

            def num_sign_changes_blockgf(G):
                max_num_changes = 0
                for bidx, g in G:
                    g_diag = np.diagonal(g.data.real, axis1=1, axis2=2)
                    num_changes = np.sum(np.abs(np.diff(np.sign(g_diag), axis=0)))
                    max_num_changes = max(max_num_changes, num_changes)
                return max_num_changes

            def max_blockgf(G):
                max_val = float('-inf')
                for bidx, g in G:
                    max_val = max(max_val, np.max(g.data.real))
                return max_val

            def min_blockgf(G):
                min_val = float('inf')
                for bidx, g in G:
                    min_val = min(min_val, np.min(g.data.real))
                return min_val

            def max_tau0_blockgf(G):
                max_val = float('-inf')
                for bidx, g in G:
                    max_val = max(max_val, np.max(np.diag(g.data[0].real)))
                return max_val

            def min_tau0_blockgf(G):
                min_val = float('inf')
                for bidx, g in G:
                    min_val = min(min_val, np.min(np.diag(g.data[0].real)))
                return min_val

            Z = -trace(G)
            val = Z.real - 1

            # Are we in a trouble some regime?
            #if np.abs(val) > 1e2 or np.abs(val) < 1e-2:
            if max_blockgf(G) > 1. or min_blockgf(G) < -1.:
                print(f'Warning: max(G) = {max_blockgf(G)}, min(G) = {min_blockgf(G)}')
                print(f'Warning: max(G(tau0)) = {max_tau0_blockgf(G)}, min(G(tau0)) = {min_tau0_blockgf(G)}')
                print(f'Warning: num_sign_changes = {num_sign_changes_blockgf(G)}')

                #if max_tau0_blockgf(G) > -1. and min_tau0_blockgf(G) > -1.:
                #    val = +1.
                
                #if max_tau0_blockgf(G) < -1. and min_tau0_blockgf(G) < -1.:
                #    val = -1.

                if max_blockgf(G) > 1.:
                    val = -1.
                
                if max_blockgf(G) < 1.:
                    val = +1.

            print(f'func: eta = {eta}, Z = {Z}, max(G) = {max_blockgf(G)}, min(G) = {min_blockgf(G)}, val = {val}')

            from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
            oplot(G)
            plt.show()


            return val
        
        print(f'Pole energies = {np.max(np.abs(self.hyb.poles))}, max AD energy = {np.max(np.abs(self.atom_diag.energies))}')

        #E_max = 0.5 * np.max([np.max(self.hyb.poles), np.max(self.atom_diag.energies)])
        E_max = np.max(np.abs(self.atom_diag.energies)) * 10
        #E_min = np.min([np.min(self.hyb.poles), np.min(self.atom_diag.energies)])
        E_min = 0.

        print(f'E_max = {E_max}, E_min = {E_min}')

        from scipy.optimize import root_scalar

        sol = root_scalar(
            func, bracket=(E_min, E_max),
            method='bisect',
            **kwargs,
            )
        
        self.eta = sol.root

        G = self.solve_dyson(self.Sigma, self.eta)
        Z = self.partition_function_from_ppgf(G)

        if is_root(): print(f'PPSC: Bisection solver done, eta = {self.eta:+2.2E} Z-1 = {Z-1:+2.2E}')

        return sol    


    def __eq__(self, obj):

        for key in self.__dict__.keys():
            if key not in obj.__dict__.keys():
                if key != 'd': # -- Skip the diagram evaluator
                    return False
        
        for key in self.__dict__.keys():
            if key not in self.__skip_keys():
                a = getattr(self, key)
                b = getattr(obj, key)                
                if type(a) == np.ndarray:
                    if not np.array_equal(a, b):
                        print(f'BlockSparseSolver: __eq__: attribute {key} differ')
                        return False
                elif type(a) is Gf:
                    if not np.array_equal(a.data, b.data):
                        print(f'BlockSparseSolver: __eq__: attribute {key} differ')
                        return False
                elif type(a) is BlockGf:
                    for bidx, g in a:
                        g_b = b[bidx]
                        if not np.array_equal(g.data, g_b.data):
                            print(f'BlockSparseSolver: __eq__: attribute {key} differ in block {bidx}')
                            return False
                elif type(a) in (AtomDiagReal, AtomDiagComplex):
                    if not np.array_equal(a.energies, b.energies):
                        print(f'BlockSparseSolver: __eq__: attribute {key}.energies differ')
                        return False
                    for sidx in range(a.n_subspaces):
                        if not np.array_equal(a.unitary_matrices[sidx], b.unitary_matrices[sidx]):
                            print(f'BlockSparseSolver: __eq__: attribute {key}.unitary_matrices differ in subspace {sidx}')
                            return False
                else:
                    if not a == b:
                        print(f'BlockSparseSolver: __eq__: attribute {key} differ')
                        return False

        return True

    
    def __skip_keys(self):
        return ['d', 'dysons', 'timer']


    def __reduce_to_dict__(self):
        d = self.__dict__.copy()
        keys = set(d.keys()).intersection(self.__skip_keys())
        for key in keys: del d[key]
        return d


    def __copy__(self):
        return self.__factory_from_dict__(self.__class__.__name__, self.__reduce_to_dict__())

    
    def __deepcopy__(self, memo):
        d = copy.deepcopy(self.__reduce_to_dict__())
        return self.__factory_from_dict__(self.__class__.__name__, d)


    @classmethod
    def __factory_from_dict__(cls, name, d):
        arg_keys = ['H_loc', 'beta', 'w_max', 'eps', 'gf_struct', 'conserved_operators']
        argv_keys = ['atom_diag', 'verbose']
        verbose = d['verbose']
        d['verbose'] = False # -- Suppress printouts on reconstruction from dict
        ret = cls(*[ d[key] for key in arg_keys ],
                  **{ key : d[key] for key in argv_keys })
        ret.__dict__.update(d)
        ret.verbose = verbose
        return ret


def logo():
    """ https://patorjk.com/software/taag/#p=display&f=Red+Phoenix&t=XCA """
    return r"""____  ____________     _____
\   \/  /\_   ___ \   /  _  \
 \     / /    \  \/  /  /_\  \
 /     \ \     \____/    |    \
/___/\  \ \______  /\____|__  /
      \_/        \/         \/  [github.com/TRIQS/xca]"""


class PoleRepresentation(object):
    def __init__(self, poles, coefficients):
        self.poles = poles
        self.coefficients = np.asarray(coefficients, order='C')


    def __eq__(self, obj):
        return np.array_equal(self.poles, obj.poles) and \
            np.array_equal(self.coefficients, obj.coefficients)


    def __reduce_to_dict__(self):
        d = self.__dict__.copy()
        return d


    @classmethod
    def __factory_from_dict__(cls, name, d):
        arg_keys = ['poles', 'coefficients']
        argv_keys = []
        ret = cls(*[ d[key] for key in arg_keys ],
                  **{ key : d[key] for key in argv_keys })
        ret.__dict__.update(d)
        return ret


def hamiltonian_matrix(ad):

    H_mat = np.zeros([ad.full_hilbert_space_dim]*2, dtype=float)

    for sidx in range(ad.n_subspaces):
        H_block = hamiltonian_matrix_block(ad, sidx)
        fidx = ad.fock_states[sidx]
        bidx = np.ix_(fidx, fidx)
        H_mat[bidx] = H_block

    return H_mat


def hamiltonian_matrix_blocks(ad):
  
    H_blocks = []

    for sidx in range(ad.n_subspaces):
        H_block = hamiltonian_matrix_block(ad, sidx)
        H_blocks.append(H_block)

    return H_blocks


def hamiltonian_matrix_block(ad, sidx):

    U = ad.unitary_matrices[sidx]
    E = ad.energies[sidx]
    H_block_diag = np.diag(E + ad.gs_energy)
    H_block = U @ H_block_diag @ U.T.conj()

    return H_block


def zero_pseudo_particle_propagator_dense(ad, mesh_tau):

    G_tau = Gf(mesh=mesh_tau, target_shape=[ad.full_hilbert_space_dim]*2)
    G_tau.data[:] = 0.
    
    return G_tau


def zero_pseudo_particle_propagator(ad, mesh_tau):

    G_tau_blocks = []
    
    for sidx in range(ad.n_subspaces):
        G_tau = Gf(mesh=mesh_tau, target_shape=[ad.get_subspace_dims()[sidx]]*2)
        G_tau.data[:] = 0.
        G_tau_blocks.append(G_tau)
        
    G_tau = BlockGf(block_list=G_tau_blocks)

    return G_tau
        

def atomic_pseudo_particle_greens_function_dense(ad, beta, mesh_tau):
    G_b_tau = atomic_pseudo_particle_greens_function(ad, beta, mesh_tau)
    G_tau = pseudo_particle_block_gf_to_dense(G_b_tau, ad)
    return G_tau


def atomic_pseudo_particle_greens_function(ad, beta, mesh_tau):

    G_tau_blocks = []

    Z0 = np.sum([ np.sum(np.exp(-beta * E)) for E in ad.energies ])
    eta0 = -np.log(Z0) / beta

    tau = np.array([ float(t) for t in mesh_tau ])
    
    for sidx in range(ad.n_subspaces):

        G_tau = Gf(mesh=mesh_tau, target_shape=[ad.get_subspace_dims()[sidx]]*2)
        
        U = ad.unitary_matrices[sidx]
        E = ad.energies[sidx]

        G_diag_tau = -np.exp(-tau[:, None]*(E - eta0)[None, :])
        G_tau.data[:] = np.matmul(U, G_diag_tau[:, :, None] * U.T.conj()[None, :, :])

        G_tau_blocks.append(G_tau)
        
    G_tau = BlockGf(block_list=G_tau_blocks)

    return G_tau, eta0


def print_atom_diag_info(ad, verbose=False):

    if False:
        print(r"""   _____    __                   ________   .___    _____     ________
    /  _  \ _/  |_  ____    _____  \______ \  |   |  /  _  \   /  _____/
    /  /_\  \\   __\/  _ \  /     \  |    |  \ |   | /  /_\  \ /   \  ___
    /    |    \|  | (  <_> )|  Y Y  \ |    `   \|   |/    |    \\    \_\  \
    \____|__  /|__|  \____/ |__|_|  //_______  /|___|\____|__  / \______  /
            \/                    \/         \/              \/         \/  """)


    hist = defaultdict(int)
    for dim in ad.get_subspace_dims():
        hist[dim] += 1

    subspace_dims = np.array(list(hist.keys()))
    num_subspaces_per_dim = np.array(list(hist.values()))

    sidx = np.argsort(subspace_dims)
    subspace_dims = subspace_dims[sidx]
    num_subspaces_per_dim = num_subspaces_per_dim[sidx]
    emin = np.min([np.min(x) for x in ad.energies])
    emax = np.max([np.max(x) for x in ad.energies])

    print(f'{type(ad).__name__}: dim {ad.full_hilbert_space_dim} with ' + \
        f'{ad.n_subspaces} subspaces dims {subspace_dims} freq {num_subspaces_per_dim} ' + \
        f'E min/max {emin:+2.2E}/{emax:+2.2E}')
    
    if verbose:
        print(f'full_hilbert_space_dim = {ad.full_hilbert_space_dim}')
        print(f'n_subspaces = {ad.n_subspaces}')
        #print(f'subspace_dims = {ad.get_subspace_dims()}')

        print(f'subspace_dims = {subspace_dims}')
        print(f'num_subspaces_per_dim = {num_subspaces_per_dim}')

        print(f'gs_energy = {ad.gs_energy}')
        print(f'energies min/max = {emin:+2.2E} / {emax:+2.2E}')

        print(f'fops = {ad.fops}')
        print(f'quantum_numbers = {ad.quantum_numbers}')
        print(f'fock_states = {ad.fock_states}')
        print(f'unitary_matrics = {ad.unitary_matrices}')
        print(f'energies = {ad.energies}')
        print(f'energies + gs_energy = {[ e + ad.gs_energy for e in ad.energies]}')
        print(ad)


def max_abs_diff_BlockGf(G1, G2):
    return np.max([ np.max(np.abs(g1.data - g2.data)) for (_, g1), (_, g2) in zip(G1, G2) ])


def pseudo_particle_block_gf_to_dense(block_gf, ad):

    mesh = block_gf[0].mesh
    dense_gf = Gf(mesh=mesh, target_shape=[ad.full_hilbert_space_dim]*2)

    for sidx in range(ad.n_subspaces):
        fidx = ad.fock_states[sidx]
        bidx = np.ix_(range(len(mesh)), fidx, fidx)
        dense_gf.data[bidx] = block_gf[sidx].data

    return dense_gf    


def get_max_abs_trapz_block_gf(G):

    max_abs = -float('inf')
    for b, g in G:
        g_dlr = make_gf_dlr(g)
        beta = g.mesh.beta

        max_abs_block = np.max(np.abs(np.diag(g_dlr(0.) + g_dlr(beta)))) / 2

        if max_abs_block > max_abs:
            max_abs = max_abs_block

    return max_abs


def get_max_abs_block_gf(G):

    max_abs = -float('inf')
    for b, g in G:
        max_abs_block = np.max(np.abs(g.data))
        if max_abs_block > max_abs:
            max_abs = max_abs_block

    return max_abs


def trace_dlr_imtime_BlockGf(G_dlr_imtime_BlockGf):

    G_dlr_coeff_BlockGf = make_gf_dlr(G_dlr_imtime_BlockGf)

    def block_trace(g_dlr_coeff): 
        return np.trace(g_dlr_coeff(g_dlr_coeff.mesh.beta))

    return np.sum([block_trace(g) for _, g in G_dlr_coeff_BlockGf])


def fundamental_operators_from_gf_struct(gf_struct):
    fundamental_operators = []
    for s, n in gf_struct:
        fundamental_operators += [ (s, i) for i in range(n) ]
    return fundamental_operators


# -- Register Solver in Triqs formats

from h5.formats import register_class
register_class(BlockSparseSolver)
register_class(PoleRepresentation)
