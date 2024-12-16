from typing import Callable, Tuple
import numpy as np
from QN import DFP , BFGS
from rosenbrock import rosenbrock
k = 2
def f(x : np.ndarray) -> float:
    return x[0]**2 + x[1]**2

m1 = 1
m2 = 1
m3 = 1

def h (x : np.ndarray) -> np.ndarray:
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def j (x : np.ndarray) -> np.ndarray: # with 4 variables: x[0],x[1],x[2],x[3]
    return (x[0] + 10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4
def e (x : np.ndarray) -> np.ndarray:
    return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])+ m1*max(0,x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2) + m2*max(0,x[1]*x[2] - 5*x[3]*x[4]) + m3*max(0,x[0]**3+x[2]**3+1)

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
    i = 1
    while i < max_iter and normgrad > tol:
        print("normgrad is",normgrad)
        if (i % 10 == 0):
            print("iteration: ",i)
            print("Current_x: ",Next_x)
            print("f(x)",f(Next_x))
            print("grad: ",normgrad)
            print("func_eval: ",func_eval)
            print("lam: ",lam) 
            print("")
        if restart and i % 20 == 0 :
            Next_x, func_eval, lam, normgrad, D_k = DFP(f,Next_x,np.eye(dim)) if dfp else BFGS(f,Next_x,np.eye(dim))
            print("D_K restarted")
        else :
            Next_x, func_eval, lam, normgrad, D_k = DFP(f,Next_x,D_k) if dfp else BFGS(f,Next_x,D_k)
        i += 1  
        total_func_eval += func_eval
    print("Stopping criteria meet")
    if (i == max_iter):
        print("Max iterations reached")
    if normgrad < tol:
        print("Gradient norm less than tol")
    return Next_x, f(Next_x), total_func_eval
tol = 1e-6
start_x = np.array([1,1,1,-1])
start_x1 = np.array([5,5])
start_x3 = np.array([-2,2,2,-1,-1])
x,res,total = Program(f,tol,start_x1,"BFGS",True)
print("Solution to problem is x = ",x)
print("f(x) = ",res)
print("Total function evaluations: ",total)
