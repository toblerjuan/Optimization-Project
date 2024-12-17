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
    # print("D_k", D_k)
    # print("grad_x0", grad_x0)
    # print("Gradient in x: ", grad_x0)
    # print("Norm of grad: ", np.linalg.norm(grad_x0))
    # print("D_K: ", D_k)
    decendingDir = dk.T @ grad_x0
    # if decendingDir > 0 :
    #     D_k = np.eye(D_k.shape[0])
    #     dk = -D_k @ grad_x0
    # print("dk: ", dk)
    # print("x: ", x)
    # print("DecendingDir: ", decendingDir)
    lam,func_eval = wolfe(f,lamb,alpha,epsilon,sigma,x,dk,func_eval)
    # print("lambda: ",lam)
    # print("dk: ", dk)
    p_k = lam*dk
    # if (p_k == 0).any() :
    #     print("Zerro")
    q_k = grad_c(f,x + p_k) - grad_x0
    # print("p_k: ", p_k)
    # print("q_k: ", q_k)
    qTDq = np.inner(q_k, D_k @ q_k)
    pTq = np.inner(p_k,q_k)
    ppT = np.outer(p_k,p_k)

    if (pTq == 0).any() :
        raise Exception("Stepsize to small, divide by zero.")
    
    if method == "DFP" :
        D_k += (ppT / pTq) - ((D_k @ np.outer(q_k, q_k) @ D_k) / qTDq)
    elif method == "BFGS" :
        qpT = np.outer(q_k,p_k)
        pqT = np.transpose(qpT)
        D_k += (1 / pTq)*((1 + (qTDq / pTq))*ppT - (D_k @ qpT) - (pqT @ D_k))
    
    x = x + lam*dk
    
    # print("")
    return x, func_eval, lam, np.linalg.norm(grad_c(f,x)), D_k

    
