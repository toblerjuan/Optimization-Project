from typing import Callable, Tuple
import numpy as np
from grad import grad_c
from grad import grad_p
from Wolf import wolfe
from non_linear_min import non_linear_min
k = 2
def f(x : np.ndarray) -> float:
    return x[0]**2 + x[1]**2

tol = 1e-10
Current_x = np.array([1,1])
def Program(f) :
    Next_x, func_eval, i, normgrad, D_k = non_linear_min(f,Current_x, "DFP",True,0)
    while abs(f(Current_x)  - f(Next_x)) > tol :
        print(func_eval)
        print(i)
        print(normgrad)
        Next_x, func_eval, i, normgrad,D_k = non_linear_min(Next_x, "DFP",False,D_k)

print(Program(f))