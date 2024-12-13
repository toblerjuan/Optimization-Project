from typing import Callable, Tuple
import numpy as np
from QN import DFP , BFGS
from rosenbrock import rosenbrock
k = 2
def f(x : np.ndarray) -> float:
    return x[0]**2 + x[1]**2

def g (x : np.ndarray) -> np.ndarray:
    return (4*x[0]-2*x[1]-3)**2 + (2*x[0]-x[1]-4)**2 + (x[0])**4
def h (x : np.ndarray) -> np.ndarray:
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def j (x : np.ndarray) -> np.ndarray: # with 4 variables: x[0],x[1],x[2],x[3]
    return (x[0] + 10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4
def Program(f : Callable[[np.ndarray], float], \
    tol : float, \
    init_guess : np.ndarray, \
    method : str, \
    restart : bool) \
    -> None:
    max_iter = 1000000
    dim = init_guess.shape[0]
    dfp = True
    total_func_eval = 0
    if method == "DFP" :
        dfp = True
    elif method == "BFGS" :
        dfp = False
    Next_x, func_eval, lam, normgrad, D_k = DFP(f,init_guess,np.eye(dim)) if dfp else BFGS(f,init_guess,np.eye(dim))
    total_func_eval += func_eval
    x_old = init_guess + 1
    i = 1
    while i < max_iter and np.linalg.norm(normgrad) > tol and abs(f(x_old) - f(Next_x)) > tol: #abs(f(Current_x)  - f(Next_x)) > tol
        if (i % 10 == 0):
            print("iteration: ",i)
            print("Current_x: ",Next_x)
            print("f(x)",f(Next_x))
            print("normgrad: ",normgrad)
            print("func_eval: ",func_eval)
            print("lam: ",lam) 
            print("")
        x_old = Next_x
        if restart and i % 20 == 0 :
            Next_x, func_eval, lam, normgrad, D_k = DFP(f,Next_x,np.eye(dim)) if dfp else BFGS(f,Next_x,np.eye(dim))
            print("D_K restarted")
        else :
            Next_x, func_eval, lam, normgrad,D_k = DFP(f,Next_x,D_k) if dfp else BFGS(f,Next_x,D_k)
        i += 1
        total_func_eval += func_eval
    print("Stopping criteria meet")
    if (i == max_iter):
        print("Max iterations reached")
    if (np.linalg.norm(normgrad) < tol):
        print("Gradient norm less than tol")
    if (abs(f(x_old) - f(Next_x)) < tol):
        print("Function value difference less than tol")
    return Next_x, f(Next_x), total_func_eval
tol = 1e-100
start_x = np.array([1,1,1,-1])
start_x1 = np.array([10,10])
x,f,total = Program(rosenbrock,tol,start_x1,"DFPz",True)
print("Solution to problem is x = ",x)
print("f(x) = ",f)
print("Total function evaluations: ",total)
