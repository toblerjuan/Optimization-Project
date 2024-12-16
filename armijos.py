from typing import Callable, Tuple
import numpy as np
from grad import grad_c
from grad import grad_p


def armijo(f: Callable[[np.ndarray], float], 
           lambda0: float, 
           epsilon: float, 
           alfa: float, 
           x0: np.ndarray, 
           dk: np.ndarray,
           func_eval: float) -> Tuple[float,float]:
    F = lambda lamb : f(x0 + lamb * dk)
    start = F(0) 
    func_eval += 1
    grad = grad_p(F, 0)
    #print("amijos grad",grad)
    func_eval += 2
    T = lambda lam : start + (epsilon*grad*lam)
    iteration = 0
    while True :
        F_val = F(alfa*lambda0) 
        T_val = T(alfa*lambda0)
        func_eval += 1
        #print("F_val = ",F_val)
        #print("T_val = ", T_val)
        if F_val >= T_val:
            break 
        lambda0 *= alfa
        iteration += 1
        #print("here")
        #print("lambda0 = ",lambda0)
        #print("iteration = ",iteration)
        #print("F_val = ",F_val)
        #print("T_val = ",T_val)
        if lambda0 > 1e8 :
               raise ValueError("Step size lambda0 became too big")
        if iteration > 600:
                raise ValueError("Too many iterations")
    while True :
        F_val = F(lambda0) 
        T_val = T(lambda0)
        #print("F_val = ",F_val)
        #print("T_val = ", T_val)
        func_eval += 1
        if F_val <= T_val:
            break
        lambda0 /= alfa  # Reduce step size
        iteration += 1
        #if lambda0 < 1e-10 :
        #        raise ValueError("Step size lambda0 became too small")
        if iteration > 1000:
                raise ValueError("Too many iterations")
    return lambda0, func_eval

