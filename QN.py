from typing import Callable, Tuple
import numpy as np
from grad import grad_c
from grad import grad_p
from Wolf import wolfe
    
lamb = 1
alpha = 2
epsilon = 0.1
sigma = 0.1

def DFP(f : Callable[[np.ndarray], float], \
    x0 : np.ndarray, \
    D_k : np.ndarray) \
    -> Tuple[np.ndarray,int,int,float,np.ndarray]:
    x = x0.astype(float)
    func_eval = 0
    dk = -D_k @ grad_c(f,x)
    lam,func_eval = wolfe(f,lamb,alpha,epsilon,sigma,x,dk,func_eval)
    p_k = lam*dk
    q_k = grad_c(f,p_k + x) - grad_c(f,x)
    D_k = D_k + (p_k@p_k.T / (p_k.T@p_k)) - (D_k @ np.outer(q_k, q_k) @ D_k) / (q_k.T @D_k @q_k)
    x = x + lam*dk
    return x , func_eval, lam, np.linalg.norm(grad_c(f,x)), D_k

def BFGS(f : Callable[[np.ndarray], float], \
    x0 : np.ndarray, \
    D_k : np.ndarray) \
    -> Tuple[np.ndarray, int, int, float]:
    x = x0.astype(float)
    func_eval = 0
    dk = -D_k @ grad_c(f,x)
    lam,func_eval = wolfe(f,lamb,alpha,epsilon,sigma,x,dk,func_eval)
    p_k = lam*dk
    print("p_k = ",p_k)
    print("lam = ",lam)
    q_k = grad_c(f,p_k + x) - grad_c(f,x)
    try:
        D_k = D_k + (1 / (p_k.T @ q_k))*((1 + (q_k.T @ D_k @ q_k)/(p_k.T @ q_k))*p_k@p_k.T - D_k @ np.outer(q_k, p_k) - np.outer(p_k, q_k) @ D_k)
    except ZeroDivisionError:
        raise ValueError("division by zero")
    x = x + lam*dk
    return x, func_eval, lam, np.linalg.norm(grad_c(f,x)), D_k
    
