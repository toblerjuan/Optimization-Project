from typing import Callable, Tuple
import numpy as np
from grad import grad_c
from grad import grad_p
from Wolf import wolfe
    
lamb = 1
alpha = 2
epsilon = 0.1
sigma = 0.1

def QuasiN(f : Callable[[np.ndarray], float], \
    x0 : np.ndarray, \
    D_k : np.ndarray, \
    method : str) \
    -> Tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    x = x0.astype(float)
    func_eval = 0
    grad_x0 = grad_c(f,x)
    dk = -D_k @ grad_x0
    # print("Gradient in x: ", grad_x0)
    # print("D_K: ", D_k)
    decendingDir = dk.T @ grad_x0
    if decendingDir > 0 :
        D_k = np.eye(D_k.shape[0])
        dk = -D_k @ grad_x0
    # print("dk: ", dk)
    # print("x: ", x)
    # print("DecendingDir: ", decendingDir)
    lam,func_eval = wolfe(f,lamb,alpha,epsilon,sigma,x,dk,func_eval)
    p_k = lam*dk
    q_k = grad_c(f,x + p_k) - grad_x0
    # print("p_k: ", p_k)
    # print("q_k: ", q_k)
    if method == "DFP" :
        D_k = D_k + (p_k@p_k.T / (p_k.T@p_k)) - (D_k @ np.outer(q_k, q_k) @ D_k) / (q_k.T @D_k @q_k)
    elif method == "BFGS" :
        D_k = D_k + (1 / (p_k.T @ q_k))*((1 + (q_k.T @ D_k @ q_k)/(p_k.T @ q_k))*p_k@p_k.T - D_k @ np.outer(q_k, p_k) - np.outer(p_k, q_k) @ D_k)
    
    x = x + lam*dk
    return x, func_eval, lam, grad_c(f,x), D_k

    