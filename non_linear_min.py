from typing import Callable, Tuple
import numpy as np
from grad import grad_c
from grad import grad_p
from Wolf import wolfe

def non_linear_min(f : Callable[[np.ndarray], float], \
    x0 : np.ndarray, \
    method : str, \
    restart : bool, \
    D_k : np.ndarray) \
    -> Tuple[np.ndarray, int, int, float, np.ndarray]:
    #if (method == "DFP") :
    return DFP(f,x0,restart,D_k)
    #elif (method == "BFGS") :
        #return BFGS(f,x0,restart,D_k)
    
def DFP(f : Callable[[np.ndarray], float], \
    x0 : np.ndarray, \
    restart : bool, \
    D_k : np.ndarray) \
    -> Tuple[np.ndarray, int, int, float,np.ndarray]:
    dim = x0.shape[0]
    i = 0
    x = x0
    func_eval = 0
    if restart :
        dim = x0.shape[0]
        D_k = np.identity(dim)
        print(D_k)
    dk = -D_k @ grad_c(f,x)
    lam = wolfe(f,1,2,0.1,0.1,x,dk)
    p_k = lam*dk
    q_k = grad_c(f,p_k + x) - grad_c(f,x)
    print("p shape")
    print(p_k.shape)
    print("q shape")
    print(q_k.shape)
    D_k = D_k + (p_k@p_k.T / (p_k.T@p_k)) - (D_k @ np.outer(q_k, q_k) @ D_k) / (q_k.T @D_k @q_k)
    x_old = x
    x = x + lam*dk
    i += 1
    return x , func_eval, i,grad_c(f,x), D_k

def BFGS(f : Callable[[np.ndarray], float], \
    x0 : np.ndarray, \
    tol : float, \
    restart : bool, \
    D_k : np.ndarray) \
    -> Tuple[np.ndarray, int, int, float]:
    return "hej"