from typing import Callable, Tuple
import numpy as np
from grad import grad_p

def armijo(f: Callable[[np.ndarray], float], 
           lambda0: float, 
           epsilon: float, 
           alfa: float, 
           x0: np.ndarray, 
           dk: np.ndarray,
           func_eval: float) -> Tuple[float, float]:
    """
    Perform the Armijo line search to find an appropriate step size.
    
    Parameters:
        f: Callable, the objective function.
        lambda0: float, the initial step size.
        epsilon: float, the Armijo condition constant.
        alfa: float, the step size scaling factor (< 1).
        x0: np.ndarray, the current point.
        dk: np.ndarray, the descent direction.
        func_eval: float, the current function evaluation count.
    
    Returns:
        lambda0: float, the adjusted step size.
        func_eval: float, the updated function evaluation count.
    """
    # Define F(lambda) and its gradient at lambda = 0
    F = lambda lamb: f(x0 + lamb * dk)
    F_0 = F(0)
    func_eval += 1
    grad = grad_p(F, 0)
    func_eval += 2

    # Define the Armijo condition threshold
    T = lambda lam: F_0 + epsilon * grad * lam

    # Iteratively decrease lambda0 until Armijo condition is satisfied
    max_iterations = 1000
    iteration = 0

    while F(lambda0) > T(lambda0):
        lambda0 *= alfa  # Reduce step size
        func_eval += 1
        iteration += 1
        if iteration > max_iterations:
            raise ValueError("Too many iterations in Armijo line search (scaling down).")

    return lambda0, func_eval
