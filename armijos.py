from typing import Callable
import numpy as np
from grad import grad_c
from grad import grad_p
#from grad import grad_c

# Define the Armijo condition function


def armijo(f: Callable[[np.ndarray], float], 
           lambda0: float, 
           epsilon: float, 
           alfa: float, 
           x0: np.ndarray, 
           dk: np.ndarray) -> float:
    F = lambda lamb : f(x0 + lamb * dk)
    grad = grad_p(F, 0)
    T = lambda lam : F(0) + (epsilon*grad*lam)
    iteration = 0
    while F(alfa*lambda0) < T(alfa*lambda0) :
        lambda0 *= alfa
        iteration += 1
        if lambda0 > 1e10 :
                raise ValueError("Step size lambda0 became too big")
        if iteration > 100:
                raise ValueError("Too many iterations")
    while F(lambda0) > T(lambda0):
        lambda0 /= alfa  # Reduce step size
        iteration += 1
        if lambda0 < 1e-100 :
                raise ValueError("Step size lambda0 became too small")
        if iteration > 100:
                raise ValueError("Too many iterations")
    return lambda0