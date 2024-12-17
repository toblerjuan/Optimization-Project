from typing import Callable
import numpy as np
from QN import QuasiN
from rosenbrock import rosenbrock

# i want 4 decimals when i print the results 
np.set_printoptions(precision=6, suppress=True)

def Program(f : Callable[[np.ndarray], float], \
    tol : float, \
    init_guess : np.ndarray, \
    method : str, \
    restart : bool, \
    printOption : bool) \
    -> None:
    max_iter = 10000
    dim = init_guess.shape[0]
    total_func_eval = 0
    Next_x, func_eval, lam, normgrad, D_k = QuasiN(f,init_guess,np.eye(dim),method)
    total_func_eval += func_eval
    i = 1
    while i < max_iter and normgrad > tol:
        if (i % 3 == 0) and printOption :
            print(f"Iteration: {i:3d}, x: {Next_x}, f(x): {f(Next_x):.6f}, Gradient norm: {normgrad:.6f}, Function evaluations: {func_eval}, Lambda: {lam:.6f}")
        if restart and i % 20 == 0 :
            Next_x, func_eval, lam, normgrad, D_k = QuasiN(f,Next_x,np.eye(dim),method)
            if printOption :
                print("D_K restarted")
        else :
            Next_x, func_eval, lam, normgrad, D_k = QuasiN(f,Next_x,D_k,method)
        i += 1
        total_func_eval += func_eval
    mes = "Stopping criteria meet: "
    if (i == max_iter):
        mes += "Max iterations reached"
    if normgrad < tol:
        mes += "Gradient norm less than tol"
    print(mes)
    return Next_x, f(Next_x), total_func_eval