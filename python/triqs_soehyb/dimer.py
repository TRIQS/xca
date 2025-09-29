import numpy as np
import time
from .impurity import Fastdiagram
from .ac_pes import *
def two_band_init(U, v, mu):
    F = np.zeros((2,4,4), dtype=np.complex128)
    F_dag = np.zeros((2,4,4), dtype=np.complex128)
    c0_S_dag = np.zeros((4,4), dtype=np.complex128)
    c1_S_dag = np.zeros((4,4), dtype=np.complex128)
    c0_S_dag[2,0] = 1
    c0_S_dag[3,1] = 1
    c1_S_dag[1,0] = 1
    c1_S_dag[3,2] = -1
    c0_S = np.conjugate(np.transpose(c0_S_dag))
    c1_S = np.conjugate(np.transpose(c1_S_dag))
    F[0,:,:] = c0_S
    F[1,:,:] = c1_S
    F_dag[0,:,:] = c0_S_dag
    F_dag[1,:,:] = c1_S_dag

    H_S = U*np.matmul(np.matmul(c0_S_dag,c0_S) ,np.matmul(c1_S_dag,c1_S))
    H_S = H_S- v*(np.matmul(c0_S_dag,c1_S)+np.matmul(c1_S_dag,c0_S)); 
    H_S = H_S - mu*(np.matmul(c0_S_dag,c0_S)+np.matmul(c1_S_dag,c1_S))

    return H_S, F, F_dag

def calc_eta0(H_S):
    E_S = np.linalg.eigvals(H_S)
    E0_S = np.min(np.real(E_S))
    E_S-=E0_S
    Z_S = np.sum(np.exp(-beta*E_S))
    eta_0 = E0_S - np.log(np.real(Z_S))/beta
    return eta_0


if __name__ == '__main__':
    beta = 8.0
    t, U, v, t1, ek, mu= 1.0, 4.0, 1.5, 1.5, 0.0, 0.0

    #create Hamiltonian, F and F_dag matrices. Should be replaced with triqs in other projects
    H_S, F, F_dag = two_band_init(U, v, mu)

    #calculate eta0, this is for the shift in dyspn solver later
    eta_0 = calc_eta0(H_S)

    # dlr parameters
    lamb, eps = 640.0, 1.0e-12

    #initialize diagram evaluator
    diagramsolver = Fastdiagram(beta,lamb,eps,F,F_dag)

    # construct hybridization function Delta(t)
    Hbath=np.array([[ek,-t1],[-t1,ek]])
    Deltat = 2* diagramsolver.free_greens(beta,Hbath)* (t**2)

    ##decomposition and reflection of Delta(t) using aaa poles
    poledlrflag = False
    diagramsolver.hyb_init(Deltat,poledlrflag)
    weights, pol, error = polefitting(diagramsolver.Deltaiw, 1j*diagramsolver.dlr_if,eps= 1e-8)
    weights_reflect, pol_reflect, error_reflect = polefitting(diagramsolver.Deltaiw_reflect, 1j*diagramsolver.dlr_if,eps= 1e-8)
    diagramsolver.copy_aaa_result(pol, weights,pol_reflect,weights_reflect)
    diagramsolver.hyb_decomposition(poledlrflag)
    breakpoint()
    ##decomposition and reflection of Delta(t) using dlr poles
    #diagramsolver.hyb_init(Deltat)
    # diagramsolver.hyb_decomposition()

    #construct initial value of G(t) by time-ordered free Green's function
    G_S = diagramsolver.free_greens(beta, H_S,eta_0,True)

    G_S_old = np.zeros_like(G_S)

    #specify diagram order
    order = "OCA"

    #obtain actual imaginary time nodes on [0,beta]
    tau_actual = diagramsolver.get_it_actual()
    r = tau_actual.size

    #start iteration
    for ppsc_iter in range(10):
        start_time = time.time()
        if np.max(np.abs(G_S_old-G_S))<1e-10:
            break
        G_S_old = G_S.copy()

        #calculate pseudo-particle self energy diagrams
        Sigma_t = diagramsolver.Sigma_calc(G_S,order)
        #calculate pseudo-particle Green's function through time-ordered Dyson's equation
        G_new = diagramsolver.time_ordered_dyson(beta,H_S,eta_0,Sigma_t)

        # G(t) = G(t)*exp(-eta*t)
        Z_S = diagramsolver.partition_function(G_new)
        eta = np.log(Z_S)/beta
        H_S += eta * np.eye(H_S.shape[0])
        for k in range(r):
            G_new[k,:,:] =  G_new[k,:,:] * np.exp(-tau_actual[k]*eta)
        
        #linear damping
        G_S = 1.0*G_new+0.0*G_S_old

        print("iter ",ppsc_iter)
        print("diff is ",np.max(np.abs(G_S_old-G_S)))
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("Time spent is ",elapsed_time)
    
    #calculate impurity Green's function diagrams
    g_S = diagramsolver.G_calc(G_S,order)
    breakpoint()