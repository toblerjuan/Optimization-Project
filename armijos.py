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
    grad_0 = grad_p(F, 0)
    func_eval += 2
    # print("armoij gradient with 0: ", grad_0)
    
    # if grad_0 > 0 : # check that the direction is decending, if not, change direction
    #     lambda0 = -lambda0
    #     print("SWITCHED DIRECTIONS")
    start = F(0)
    func_eval += 1
    T = lambda lam : start + (epsilon * grad_0 * lam)
    iteration = 0
    while True :
        F_val = F(alfa*lambda0) 
        T_val = T(alfa*lambda0)
        func_eval += 1
        if F_val >= T_val:
            break 
        lambda0 *= alfa
        iteration += 1
        if iteration > 100:
                raise ValueError("Too many iterations")
    while True :
        F_val = F(lambda0) 
        T_val = T(lambda0)
        func_eval += 1
        if F_val <= T_val:
            break
        lambda0 /= alfa  # Reduce step size
        iteration += 1
        if iteration > 100:
                raise ValueError("Too many iterations")
    return lambda0, func_eval

