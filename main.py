import numpy as np
from rosenbrock import rosenbrock
from program import Program

# i want 4 decimals when i print the results 
np.set_printoptions(precision=6, suppress=True)

def f(x : np.ndarray) -> float:
    return x[0]**2 + x[1]**2
def h (x : np.ndarray) -> np.ndarray:
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def j (x : np.ndarray) -> np.ndarray: # with 4 variables: x[0],x[1],x[2],x[3]
    return (x[0] + 10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4

m = 1
def e (x : np.ndarray) -> np.ndarray:
    return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])+ m*max(0,x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2-10) + m*max(0,x[1]*x[2] - 5*x[3]*x[4]) + m*max(0,x[0]**3+x[2]**3+1)


tol = 1e-4
start_x = np.array([1,1,1,-1])
start_x1 = np.array([-2,3])
start_x3 = np.array([1,1,1,1,1])
x,res,total = Program(rosenbrock,tol,start_x1,"BFGS",True,True)
np.set_printoptions(precision=16)  # Set precision to 16 decimal places (or as required)
print(f"Solution to problem is x = {x}")
print(f"f(x) = {res}")
print(f"Total function evaluations: {total}")