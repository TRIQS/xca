################################################################################
#
# triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2025 by H. U.R. Strand
#
# triqs_xca is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_xca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_xca. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import time

import numpy as np
from scipy.optimize import root_scalar

from mpi4py import MPI as mpi

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from adapol.fit_utils import polefitting

from .pycppdlr import build_dlr_rf
from .pycppdlr import ImTimeOps

from .impurity import Fastdiagram
from .dlr_dyson_ppsc import DysonItPPSC
from .diag import all_connected_pairings

from .ase.utils.timing import Timer, timer


def logo():
    """ https://patorjk.com/software/taag/#p=display&f=Red+Phoenix&t=XCA """
    return r"""____  ____________     _____
\   \/  /\_   ___ \   /  _  \
 \     / /    \  \/  /  /_\  \
 /     \ \     \____/    |    \
/___/\  \ \______  /\____|__  /
      \_/        \/         \/  [github.com/TRIQS/xca]"""


def is_root():
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    return rank == 0


def scatter_array_over_ranks(arr):
    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    arr_rank = np.array_split(np.array(arr), size, axis=0)[rank]
    return arr_rank


def Sigma_calc_loop(fd, G_iaa, max_order, verbose=True):

    assert( max_order >= 1 )
    assert( type(fd) == Fastdiagram )
    
    if verbose:
        start_time = time.time()    

    Sigma_t = np.zeros((G_iaa.shape[0], G_iaa.shape[1], G_iaa.shape[2]), dtype=complex)
    
    for ord in range(1, max_order+1):
        n_diags = fd.number_of_diagrams(ord)
        
        if is_root() and verbose:
            print(f"PPSC: Sigma order = {ord}, n_diags = {n_diags}", flush=True)

        for sign, diag in all_connected_pairings(ord):

            #if is_root() and verbose: print(sign, diag)
            
            diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
            diag_idx_vec = np.arange(n_diags, dtype=np.int32)
            diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
            Sigma_t += pow(-1,ord)* sign * fd.Sigma_calc_group(G_iaa, diag_vec, diag_idx_vec)

    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, Sigma_t)

    if is_root() and verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"PPSC: Sigma time {elapsed_time:2.2E}s.")

    return Sigma_t


def G_calc_loop(fd, G_iaa, max_order, n_g, verbose=True):

    if verbose:
        start_time = time.time()
    
    g_iaa = np.zeros((G_iaa.shape[0], n_g, n_g), dtype=complex)

    for ord in range(1, max_order+1):
        n_diags = fd.number_of_diagrams(ord)

        if is_root() and verbose:
            print(f"PPSC: SPGF order = {ord}, n_diags = {n_diags}", flush=True)
            
        for sign, diag in all_connected_pairings(ord):

            #if is_root(): print(sign, diag)
                
            diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
            diag_idx_vec = np.arange(n_diags, dtype=np.int32)
            diag_idx_vec = scatter_array_over_ranks(diag_idx_vec)
            g_iaa += pow(-1,ord)* sign * fd.G_calc_group(G_iaa, diag_vec, diag_idx_vec)

    mpi.COMM_WORLD.Allreduce(mpi.IN_PLACE, g_iaa) 

    if is_root() and verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"PPSC: SPGF time {elapsed_time:2.2E}s.")
        
    return g_iaa


def eval_dlr_freq(G_xaa, z, beta, dlr_rf):
    w_x = dlr_rf / beta
    kernel_zx = 1./(z[:, None] - w_x[None, :])
    G_zaa = np.einsum('zx,xab->zab',kernel_zx, G_xaa)
    return G_zaa


def make_hermitian(A_iaa):
    return 0.5*(A_iaa +  np.swapaxes(A_iaa.conj(), 1, 2))


class Solver(object):

    def __init__(self, beta, lamb, eps,
                 H_loc, fundamental_operators,
                 ntau=100, timer=None, G_iaa=None, eta=None, verbose=True):

        self.timer = timer if timer is not None else Timer()

        self.__setup_dlr_basis(beta, lamb, eps)
        self.__setup_ed_solver(beta, H_loc, fundamental_operators)
        self.__setup_ppsc_solver()
        self.__setup_initial_guess(G_iaa=G_iaa, eta=eta)
    
        # -- AAA pole fitting setups
        self.ntau = max(ntau, int(10*lamb)) # This is a hacky solution to be fixed later 
        self.tau_f = np.linspace(0, self.beta, num=self.ntau)
        
        def interp(g_xaa):
            eval = lambda t : self.ito.coefs2eval(g_xaa, t/self.beta)
            return np.vectorize(eval, signature='()->(m,m)')(self.tau_f)

        self.interp = interp

        self.verbose = verbose
        if verbose: self._print_info()


    def __setup_dlr_basis(self, beta, lamb, eps):
        self.beta, self.lamb, self.eps = beta, lamb, eps        
        self.dlr_rf = build_dlr_rf(lamb, eps)
        self.ito = ImTimeOps(lamb, self.dlr_rf)
        

    def __setup_ed_solver(self, beta, H_loc, fundamental_operators):
        self.H_loc, self.fundamental_operators = H_loc, fundamental_operators

        self.ed = TriqsExactDiagonalization(H_loc, fundamental_operators, beta)

        self.F_dag = np.array([
            np.array(self.ed.rep.sparse_operators.c_dag[idx].todense())
            for idx in range(len(fundamental_operators))])

        self.F = np.array([np.array(
            self.ed.rep.sparse_operators.c_dag[idx].T.conj().todense())
            for idx in range(len(fundamental_operators)) ])

        self.N_op = 0 * self.ed.rep.sparse_operators.I
        for c_dag in self.ed.rep.sparse_operators.c_dag:
            self.N_op += c_dag * c_dag.T.conj()
        self.N_op = np.array(self.N_op.todense())
        
        self.H_mat = np.array(self.ed.ed.H.todense())
        

    def __setup_ppsc_solver(self):
        self.fd = Fastdiagram(self.beta, self.lamb, self.ito, self.F, self.F_dag)
        self.tau_i = self.fd.get_it_actual().real
        
        self.G0_iaa = self.fd.free_greens_ppsc(self.beta, self.H_mat)
        self.dyson = DysonItPPSC(self.beta, self.ito, self.H_mat)

        
    def __setup_initial_guess(self, G_iaa=None, eta=None):
        self.G_iaa = self.G0_iaa.copy() if G_iaa is None else G_iaa
        self.eta = 0.0 if eta is None else eta
        self.dmu = 0.0


    def set_H_loc(self, H_loc):
        self.H_loc = H_loc
        self.__setup_ed_solver(self.beta, self.H_loc, self.fundamental_operators)
        self.__setup_ppsc_solver()

        
    def _print_info(self):

        if is_root():
            print(logo())
            print()
            print(f'beta = {self.beta}, lamb = {self.lamb:2.2E}, eps = {self.eps:2.2E}, N_DLR = {self.ito.rank()}')
            print(f'fundamental_operators = {self.fundamental_operators}')
            print(f'H_mat.shape = {self.H_mat.shape}')
            print(f'H_loc = {self.H_loc}')
            print()

        
    @timer('Hybridization compression (AAA)')
    def set_hybridization(self, delta_iaa,
                          compress=False, delta_diff=1.0, fittingeps=2e-6,
                          Hermitian=False, verbose=True, svd_trunc=True):
        
        """Set the hybridization function of the expansion.

        Parameters
        ----------

        delta_iaa : (n, m, m) ndarray
            Hybridization function on imaginary time DLR nodes.

        compress : bool, optional
            The hybridization is by default represented using the DLR poles. When enabling
            compression the representation is reduced to an (if possible) even smaller
            number of customized poles, using the AAA algorithm.
            Default `False`

        delta_diff : float, optional
            Current maximal difference between DMFT self-consistent steps in the  hybridization function. 
            The pole fitting tolerance `fittingeps` will be constrained to `fittingeps < delta_diff / 1000`
            to ensure the pole fitting error will not affect the self-consistent iterations.
            In one-shot impurity solvers, one can neglect this argument. 
            Default 1.0

        fittingeps : float, optional
            The pole fitting error tolerance
            Default 2e-6

        Hermitian : bool, optional
            Enforce the hybridization representation to be Hermitian. 
            Default `False`

        verbose : bool, optional
            Enable (more) verbose printouts of the method.
            Defailt `True`
        
        """
        if svd_trunc ==False:
            eps_svd = 0.0
        else:
            eps_svd = fittingeps/delta_iaa.shape[1]

        if compress == False:        
            self.fd.hyb_init(delta_iaa, poledlrflag=True)
            self.fd.hyb_decomposition(poledlrflag=True, eps=eps_svd)

            if is_root() and verbose:
                print(f"AdaPol: Hybridization using all {self.ito.rank()} DLR poles.")
            
        else:
            # decomposition and reflection of Delta(t) using aaa poles
            delta_xaa = self.ito.vals2coefs(delta_iaa) 
            self.fd.hyb_init(delta_iaa, poledlrflag=False)

            # Requested accuracy of pole fit
            # epstol = min(fittingeps, delta_diff/100)
            # epstol = min(1e-6, max(fittingeps, delta_diff/100)) # This defaults to 1e-6 most of the time. FIXME?
            epstol = fittingeps # TEST: Disable delta_diff feature

            dlr_if_dense = self.fd.dlr_if_dense

            Deltat = self.interp(delta_xaa)
            Deltaiw_dense = eval_dlr_freq(delta_xaa, 1j*dlr_if_dense, self.beta, self.dlr_rf)
            Npmax = Deltaiw_dense.shape[0] -1
            weights, pol, error = polefitting(
                Deltaiw_dense, 1.j*dlr_if_dense, delta_iaa, self.tau_i, Deltat, self.tau_f,self.beta,
                eps=epstol, Np_max=Npmax, Hermitian=Hermitian)

            if error < epstol and len(pol)<=len(self.tau_i):
                if is_root() and verbose:
                    print(f"AdaPol: Hybridization fit accuracy {error:2.2E}, using {len(pol)} poles.")
                
                self.fd.copy_aaa_result(pol, weights)
                self.fd.hyb_decomposition(poledlrflag=False, eps=eps_svd)
            else:
                if is_root() and verbose:
                    print(f"AdaPol: Hybridization using all {self.ito.rank()} DLR poles.")
            
                self.fd.hyb_init(delta_iaa, poledlrflag=True)
                self.fd.hyb_decomposition(poledlrflag=True, eps=eps_svd)


    @timer('Eta search (bisection)')
    def energyshift_bisection(self, Sigma_iaa, tol=1e-10, verbose=True):
        
        def target_function(eta):
            G_iaa_new = self.solve_dyson(Sigma_iaa, eta, tol, dmu=self.dmu)
            Z = self.fd.partition_function(G_iaa_new)
            Omega = np.log(np.abs(Z)) / self.beta

            if is_root() and verbose:
                print(f'PPSC: Eta bisection: Z-1 = {Z-1:+2.2E}, Omega = {Omega:+2.2E}')

            return Omega
        
        Omega = target_function(self.eta)

        if np.abs(Omega) > 0:
            
            E_max = self.eta.real if Omega < 0. else 0.5*self.lamb/self.beta
            E_min = self.eta.real if Omega > 0. else 0.
            
            bracket = [E_min, E_max]
            
            sol = root_scalar(target_function, method='brenth',
                              fprime=False, bracket=bracket, rtol=tol, options={'disp': True})
            
            if not sol.converged and is_root():
                print("PPSC: Warning! Energy shift failed.")
                print(sol)

            return sol.root


    #@timer('Eta search (Newton)')
    def energyshift_newton(self, Sigma_iaa, tol=1e-10, verbose=True):

        def target_function(eta):
        
            #G_iaa_new = self.dyson.solve(Sigma_iaa, eta)
            G_iaa_new = self.solve_dyson(Sigma_iaa, eta, tol, dmu=self.dmu)
            
            Z = self.fd.partition_function(G_iaa_new)
            Omega = np.log(np.abs(Z)) / self.beta

            if verbose and is_root():
                print(f'PPSC: Eta Newton: Z-1 = {Z-1:+2.2E}, Omega = {Omega:+2.2E}')

            G_xaa = self.ito.vals2coefs(G_iaa_new)
            GG_iaa = self.ito.convolve(self.beta, G_xaa, G_xaa, True)
            TrGGb = self.fd.partition_function(GG_iaa)
            dOmega = TrGGb / self.beta / Z

            return Omega, dOmega

        sol = root_scalar(
            target_function, x0=self.eta, method='newton', fprime=True, rtol=tol)

        if not sol.converged and is_root():
            print('PPSC: Warning! Energy shift Newton search failed.')
            print(sol)

        if not sol.converged:
            return self.energyshift_bisection(Sigma_iaa, verbose=verbose)
        
        return sol.root


    @timer('Eta and mu search (Newton)')
    def energyshift_density_newton(self, Sigma_iaa, N_fix, tol=1e-10, verbose=True, single_step=False):
        
        def target_function(x, verbose=True):
            eta, dmu = x

            G_iaa_new = self.solve_dyson(Sigma_iaa, eta, tol, dmu=dmu)

            G_xaa = self.ito.vals2coefs(G_iaa_new)

            GG_iaa = self.ito.convolve(self.beta, G_xaa, G_xaa, True)
            GNG_iaa = self.ito.convolve(self.beta, G_xaa, self.N_op @ G_xaa, True)

            TrGb = -self.fd.partition_function(G_iaa_new)
            TrNGb = -self.fd.partition_function(self.N_op @ G_iaa_new)

            TrGGb = -self.fd.partition_function(GG_iaa)
            TrNGGb = -self.fd.partition_function(self.N_op @ GG_iaa)

            TrGNGb = -self.fd.partition_function(GNG_iaa)
            TrNGNGb = -self.fd.partition_function(self.N_op @ GNG_iaa)

            Z = - TrGb
            N = - TrNGb / Z

            Omega = - np.log(np.abs(Z)) / self.beta

            F = np.array([Omega, N_fix - N]) # Root function

            dOmega_deta = TrGGb / self.beta / Z
            dOmega_dmu = TrGNGb / self.beta / Z

            dN_deta = TrGGb * TrNGb / Z**2 + TrNGGb / Z
            dN_dmu = TrGNGb * TrNGb / Z**2 + TrNGNGb / Z

            # Jacobian

            J = np.array([
                [dOmega_deta, dOmega_dmu],
                [dN_deta,     dN_dmu],
                ])

            if verbose and is_root():
                print(f'PPSC: Z-1 = {Z-1:+2.2E}, Omega = {Omega:+2.2E}, N-Nfix = {N-N_fix:+2.2E} (Nfix={N_fix}, tol={tol:2.2E})')
                print(f'PPSC: eta = {eta:+6.6E}, dmu = {dmu:+6.6E} (pre)')

            #print(f'F = {F}')
            #print(f'J = \n{J}')

            return F, J


        x0 = np.array([self.eta, self.dmu])

        if False:
            # -- Numerical check of gradient
            from scipy.optimize import check_grad, approx_fprime
            func = lambda x: target_function(x)[0]
            grad = lambda x: target_function(x)[1]
            grad_err = check_grad(func, grad, x0, epsilon=1e-8)
            print('='*72)
            print(f'grad_err = {grad_err:+2.2E}')

            J = grad(x0)
            print(f'J =\n{J}')

            J_approx = approx_fprime(x0, func)
            print(f'J_approx =\n{J_approx}')
            print('='*72)

            exit()

        if single_step:
            #if is_root(): print(f'PPSC: N_fix = {N_fix}, tol = {tol:2.2E}')
            df, H = target_function(x0)
            s = np.linalg.solve(H, -df)
            #if is_root(): print(f'norm(s) = {np.linalg.norm(s)}')
            x = x0 + s
            eta, dmu = x
            if is_root(): print(f'PPSC: eta = {eta:+6.6E}, dmu = {dmu:+6.6E} (post Newton step)')
            return x

        else:
            if is_root(): print(f'--> energyshift_density_newton: N_fix = {N_fix}, tol = {tol:2.2E}')
            
            from scipy.optimize import root
            sol = root(
                target_function, x0=x0, method='hybr', jac=True, tol=tol,
                options=dict(
                    xtol=tol,
                    ))

            #F, J = target_function(sol.x, verbose=True)
            #print(f'F = {F}')

            if not sol.success and is_root():
                print('PPSC: Warning! Energy and density shift Newton search failed.')
                print(sol)

            return sol.x


    def solve_fix_N(self, max_order, N_fix, tol=1e-9, maxiter=10, mix=1.0,
                    G0_iaa=None, single_step=False, verbose=True):

        if is_root(): print('--> solve_fix_N')
        
        if G0_iaa is not None:
            assert( type(G0_iaa) == np.ndarray )
            assert( G0_iaa.shape == self.G_iaa.shape )
            self.G_iaa = G0_iaa

        self.G0_iaa = self.fd.free_greens_ppsc(self.beta, self.H_mat)
        self.G0_xaa = self.ito.vals2coefs(self.G0_iaa)

        for iter in range(maxiter):

            self.Sigma_iaa = self.calc_Sigma(max_order, verbose=verbose)

            eta_old, dmu_old = self.eta, self.dmu

            self.eta, self.dmu = self.energyshift_density_newton(
                self.Sigma_iaa, N_fix, tol=tol, single_step=single_step, verbose=verbose)

            if is_root():
            #    print(f'PPSC: eta_old = {eta_old}, dmu_old = {dmu_old}')
            #    print(f'PPSC: eta = {self.eta}, dmu = {self.dmu}')
            #    print(f'PPSc: deta = {self.eta - eta_old}')
                print(f'PPSC: Fix N mix = {mix}')
              
            G_iaa_new = self.solve_dyson(self.Sigma_iaa, self.eta, tol, dmu=self.dmu)

            diff = np.max(np.abs(self.G_iaa - G_iaa_new))

            self.G_iaa = mix*G_iaa_new + (1-mix)*self.G_iaa
            #self.G_iaa = make_hermitian(self.G_iaa)

            self.eta = mix*self.eta + (1-mix)*eta_old # Try to stabilize!?
            self.dmu = mix*self.dmu + (1-mix)*dmu_old # Try to stabilize!?
            
            if is_root(): print(f'PPSC: iter = {iter:3d} diff = {diff:2.2E} (tol = {tol:2.2E} maxiter = {maxiter})')
            if diff < tol: break

        if is_root():
            print(); self.timer.write()

        return diff


    def solve(self, max_order, tol=1e-9, maxiter=100, update_eta_exact=True, mix=1.0, verbose=True, G0_iaa=None):

        self.timer = Timer() # Reset timer for each solve call

        if verbose == False:
            verbose = 0
            
        if verbose == True:
            verbose = 1

        if G0_iaa is not None:
            assert( type(G0_iaa) == np.ndarray )
            assert( G0_iaa.shape == self.G_iaa.shape )
            self.G_iaa = G0_iaa

        diff = 1.0

        self.G0_iaa = self.fd.free_greens_ppsc(self.beta, self.H_mat)        
        self.G0_xaa = self.ito.vals2coefs(self.G0_iaa)

        if is_root() and verbose > 0:
            print()
            print(" iter |   conv   |    Z-1    ")
            print("------+----------+-----------")
        
        for iter in range(1, maxiter+1):
            
            #Sigma_iaa = Sigma_calc_loop(self.fd, self.G_iaa, max_order, verbose=verbose)
            Sigma_iaa = self.calc_Sigma(max_order, verbose=verbose > 1)

            if is_root() and verbose:
                dyson_start_time = time.time()

            if update_eta_exact:
                #self.eta = self.energyshift_newton(Sigma_iaa, tol=0.1*diff, verbose=verbose)
                #self.eta = self.energyshift_bisection(Sigma_iaa, tol=tol, verbose=verbose)
                self.eta = self.energyshift_newton(Sigma_iaa, tol=tol, verbose=verbose > 1)
                G_iaa_new = self.solve_dyson(Sigma_iaa, self.eta, tol, dmu=self.dmu)
                
            else:
                G_iaa_new = self.solve_dyson(Sigma_iaa, self.eta, tol, dmu=self.dmu)
                Z = self.partition_function(G_iaa_new)
                deta = np.log(np.abs(Z)) / self.beta
                G_iaa_new[:] *= np.exp(-self.tau_i * deta)[:, None, None]
                if is_root() and verbose > 1: print(f'deta = {deta}, eta = {self.eta}')
                self.eta += deta

            if is_root() and verbose > 1:
                dyson_end_time = time.time()
                dyson_elapsed_time = dyson_end_time - dyson_start_time
                print(f"PPSC: Dyson time {dyson_elapsed_time:2.2E}s.")
                
            Z = self.partition_function(G_iaa_new)

            diff = np.max(np.abs(self.G_iaa - G_iaa_new))
            
            self.G_iaa = mix*G_iaa_new + (1-mix)*self.G_iaa
            #self.G_iaa = make_hermitian(self.G_iaa)
            self.Sigma_iaa = Sigma_iaa

            if is_root() and verbose > 0:
                #print(f"PPSC: Z-1 = {Z-1:+2.2E}")
                print(f' {iter:4d} | {diff:2.2E} | {Z-1:+2.2E}')
            if diff < tol: break

        if is_root() and verbose > 0:
            print(); self.timer.write()

        return diff
            

    @timer('Dyson equation')
    def solve_dyson(self, Sigma_iaa, eta, tol, iterative=False, dmu=0.0):
        """ Dyson solver frontend

        For direct solver or iterative solver (with given tolerance).
        """
        
        if iterative:
            G_iaa = self.solve_dyson_iterative(Sigma_iaa, eta, G_iaa_guess=self.G_iaa, tol=tol, dmu=dmu)
        else:
            #G_iaa = self.dyson.solve(Sigma_iaa, eta)
            G_iaa = self.dyson.solve_with_op(Sigma_iaa, eta, np.array(dmu * self.N_op, dtype=complex))
            
        return G_iaa

    
    @timer('Dyson Iterative')
    def solve_dyson_iterative(self, Sigma_iaa, eta, G_iaa_guess=None, tol=1e-9, dmu=0.0):
        """ Solve ( 1 - G0*(eta + Sigma)) * G = G0 iteratively. """

        shape_iaa = Sigma_iaa.shape
        Sigma_xaa = self.ito.vals2coefs(Sigma_iaa)
        G0Sigma_iaa = self.ito.convolve(self.beta, self.G0_xaa, Sigma_xaa, True)
        
        K_iaa = self.G0_iaa * eta + G0Sigma_iaa
        K_iaa += dmu * self.G0_iaa @ self.N_op
        K_xaa = self.ito.vals2coefs(K_iaa)

        def matvec(x):
            """ Apply the matrix ( 1 - K ) to G.  """
            G_iaa = x.reshape(shape_iaa)
            G_xaa = self.ito.vals2coefs(G_iaa)            
            KG_iaa = self.ito.convolve(self.beta, K_xaa, G_xaa, True)
            LHS_iaa = G_iaa - KG_iaa
            return LHS_iaa.flatten()

        from scipy.sparse.linalg import LinearOperator, bicgstab

        x0 = self.G0_iaa.flatten() if G_iaa_guess is None else G_iaa_guess.flatten()
        A = LinearOperator(shape=[x0.shape[0]]*2, matvec=matvec)
        b = self.G0_iaa.flatten()

        x, info = bicgstab(A, b, x0=x0, tol=tol)
        if info != 0:
            print(f'WARNING: scipy.sparse.linalg.bicgstab, info = {info}')

        G_iaa = x.reshape(shape_iaa)
        return G_iaa
    

    #@timer('Partition function')
    def partition_function(self, G_iaa):
        Z = self.fd.partition_function(G_iaa)
        return Z
    

    @timer('Pseudo-particle self-energy')
    def calc_Sigma(self, max_order, verbose=True):

        Sigma_iaa = Sigma_calc_loop(self.fd, self.G_iaa, max_order, verbose=verbose)
        return Sigma_iaa

    
    @timer("Single particle Green's function")
    def calc_spgf(self, max_order, verbose=True):
        
        n_g = self.F.shape[0]
        g_iaa = G_calc_loop(self.fd, self.G_iaa, max_order, n_g, verbose=verbose)

        self.g_iaa = g_iaa
        
        return g_iaa
        

    #@timer('mb dens mat')
    def get_many_body_density_matrix(self):

        assert( hasattr(self, 'G_iaa') )
        G_xaa = self.ito.vals2coefs(self.G_iaa)
        rho_GG = -self.ito.coefs2eval(G_xaa, 1.0)
        
        return rho_GG

    #@timer('mb exp val')
    def get_expectation_value(self, triqs_operator):
        op_mat = self.ed.rep.sparse_matrix(triqs_operator)
        exp_val = np.sum(np.diag( op_mat @ self.get_many_body_density_matrix() ))
        return exp_val
    
    
    #@timer('sp dens mat')
    def get_single_particle_density_matrix(self):

        assert( hasattr(self, 'g_iaa') )

        g_xaa = self.ito.vals2coefs(self.g_iaa)
        rho_aa = -self.ito.coefs2eval(g_xaa, 1.0)
        
        return rho_aa

    
    #@timer('tot dens')
    def get_density(self, tol=None):

        if tol is None:
            tol = 10 * self.eps

        N = np.sum(np.diag(self.get_single_particle_density_matrix()))

        if np.abs(np.imag(N)) > tol:
            print(f'PPSC: WARNING Total density is complex: {N}')
        
        return N.real


    def interpolate_dlr_tau_to_tau(self, g_iaa, tau_j):
        g_xaa = self.ito.vals2coefs(g_iaa)
        eval = lambda t : self.ito.coefs2eval(g_xaa, t/self.beta)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)
    
    
    def __skip_keys(self):
        return ['timer', 'ito', 'fd', 'ed', 'interp', 'dyson']

    
    def __eq__(self, obj):

        if obj.__dict__.keys() != self.__dict__.keys():
            return False
        
        for key in self.__dict__.keys():
            if key not in self.__skip_keys():
                a = getattr(self, key)
                b = getattr(obj, key)                
                if not np.equal(a, b).all():
                    return False

        return True
    
        
    def __reduce_to_dict__(self):
        d = self.__dict__.copy()
        keys = set(d.keys()).intersection(self.__skip_keys())
        for key in keys: del d[key]
        return d

    
    @classmethod
    def __factory_from_dict__(cls, name, d):
        arg_keys = ['beta', 'lamb', 'eps', 'H_loc', 'fundamental_operators']
        argv_keys = ['ntau', 'G_iaa', 'eta', 'verbose']
        verbose = d['verbose']
        d['verbose'] = False # -- Suppress printouts on reconstruction from dict
        ret = cls(*[ d[key] for key in arg_keys ],
                  **{ key : d[key] for key in argv_keys })
        ret.__dict__.update(d)
        ret.verbose = verbose
        return ret
    
    
# -- Register Solver in Triqs formats

from h5.formats import register_class 
register_class(Solver)    
