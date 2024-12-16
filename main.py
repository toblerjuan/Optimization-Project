from typing import Callable, Tuple
import numpy as np
from QN import QuasiN
from rosenbrock import rosenbrock
np.set_printoptions(precision=6, suppress=True)


m = lambda lam : 10**(lam - 5)

def f(x : np.ndarray) -> float:
    return x[0]**2 + x[1]**2
def p (x : np.ndarray) -> np.ndarray:
    return (x[0]-2)**2 + (x[1]-3)**2 + m[i]*(x[0]+x[1]-4)**2+ m[i]*(x[0]-x[1]-1)**2
def h (x : np.ndarray) -> np.ndarray:
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def j (x : np.ndarray) -> np.ndarray: # with 4 variables: x[0],x[1],x[2],x[3]
    return (x[0] + 10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4
def e (x : np.ndarray) -> np.ndarray:
    return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])+ m*(i)*((x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10)**2 + m(i)*(x[1]*x[2] - (5*x[3]*x[4]))**2 + m(i)*abs((x[0]**3+x[2]**3)+1)**2

def checkboundaries(x : np.ndarray) -> np.ndarray:
    print("Checking boundaries")
    print((x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10)
    print(x[1]*x[2] - 5*x[3]*x[4])
    print((x[0]**3+x[2]**3)+1)

def Program(f : Callable[[np.ndarray], float], \
    tol : float, \
    init_guess : np.ndarray, \
    method : str, \
    restart : bool) \
    -> None:
    max_iter = 10000
    dim = init_guess.shape[0]
    total_func_eval = 0
    Next_x, func_eval, lam, normgrad, D_k = QuasiN(f,init_guess,np.eye(dim),method)
    total_func_eval += func_eval
    i = 0
    while i < max_iter and normgrad > tol:
        #if (i % 3 == 0):
            #print(f"Iteration: {i:3d}, x: {Next_x}, f(x): {f(Next_x):.6f}, Gradient norm: {normgrad:.6f}, Function evaluations: {func_eval}, Lambda: {lam:.6f}")
        if restart and i % 20 == 0 :
            Next_x, func_eval, lam, normgrad, D_k = QuasiN(f,Next_x,np.eye(dim),method)
            #print("D_K restarted")
        else :
            Next_x, func_eval, lam, normgrad, D_k = QuasiN(f,Next_x,D_k,method)
        i += 1
        total_func_eval += func_eval
    print("Stopping criteria meet")
    if (i == max_iter):
        print("Max iterations reached")
    if normgrad < tol:
        print("Gradient norm less than tol")
    return Next_x, f(Next_x), total_func_eval
tol = 1e-4
start_x = np.array([1,1,1,-1])
start_x1 = np.array([1.6,2.3])
start_x2 = np.array([1.5,2.5])
start_x3 = np.array([-2,2,2,-1,-1])
new = np.array([-1.526286 ,1.404612 ,1.367206 ,-1.383988 ,-1.383988])
x,res,total = Program(j,tol,start_x,"BFGS",True)
np.set_printoptions(precision=16)  # Set precision to 16 decimal places (or as required)
print(f"Solution to problem is x = {x}")
print("f(x) = ",res)
print("Total function evaluations: ",total)

def pen() : # Run this to try the penalty problem
    i = 0
    while (i < 15) :
        x_new = start_x3
        def e (x : np.ndarray) -> np.ndarray:
            return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])+ m(i)*((x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10)**2 + m(i)*(x[1]*x[2] - (5*x[3]*x[4]))**2 + m(i)*abs((x[0]**3+x[2]**3)+1)**2
        x_new,res,total = Program(e,tol,x_new,"DFP",False)
        print(f"Solution to problem is x = {x_new}")
        print("f(x) = ",res)
        print("Total function evaluations: ",total)
        print("i = ",i)
        checkboundaries(x_new)
        i += 1
