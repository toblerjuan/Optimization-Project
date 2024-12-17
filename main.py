import numpy as np
from rosenbrock import rosenbrock
from program import Program

# i want 4 decimals when i print the results 
np.set_printoptions(precision=6, suppress=True)


m = lambda lam : 10**(lam - 5)
i = 1

def f (x : np.ndarray) -> np.ndarray: # with 2 variables: x[0],x[1]
    return x[0]**2 + x[1]**2
def p (x : np.ndarray) -> np.ndarray: # with 2 variables: x[0],x[1]
    return (x[0]-2)**2 + (x[1]-3)**2 + m(i)*(x[0]+x[1]-4)**2+ m(i)*(x[0]-x[1]-1)**2
def h (x : np.ndarray) -> np.ndarray: # with 2 variables: x[0],x[1]
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def j (x : np.ndarray) -> np.ndarray: # with 4 variables: x[0],x[1],x[2],x[3]
    return (x[0] + 10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4
# def e (x : np.ndarray) -> np.ndarray:
#     return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])+ m*(i)*((x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10)**2 + m(i)*(x[1]*x[2] - (5*x[3]*x[4]))**2 + m(i)*abs((x[0]**3+x[2]**3)+1)**2

def checkboundaries(x : np.ndarray) -> np.ndarray:
    print("Checking boundaries")
    print((x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10)
    print(x[1]*x[2] - 5*x[3]*x[4])
    print((x[0]**3+x[2]**3)+1)

tol = 1e-4
start_x_2d = np.array([-100,-3231])
start_x = np.array([1,1,1,-1])
start_x1 = np.array([1.6,2.3])
start_x2 = np.array([1.5,2.5])
start_x3 = np.array([-2,2,2,-1,-1])
new = np.array([-1.526286 ,1.404612 ,1.367206 ,-1.383988 ,-1.383988])
if True :
    x,res,total = Program(h,tol,start_x,"BFGS",False,True)
    np.set_printoptions(precision=16)  # Set precision to 16 decimal places (or as required)
    print(f"Solution to problem is x = {x}")
    print("f(x) = ",res)
    print("Total function evaluations: ",total)

# print("Testing matrix multiplication")

# A = np.array([1, 2])
# B = np.array([3, 4])
# C = A.T @ B
# D = A @ B.T
# E = np.outer(B,A)
# F = np.outer(A,B)
# print(A)
# print(B)
# print(C)
# print(D)
# print(E)
# print(F)
# G = np.transpose(F)
# print(G)

def pen() : # Run this to try the penalty problem
    i = 0
    while (i < 15) :
        x_new = start_x3
        def e (x : np.ndarray) -> np.ndarray:
            return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])+ m(i)*((x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10)**2 + m(i)*(x[1]*x[2] - (5*x[3]*x[4]))**2 + m(i)*abs((x[0]**3+x[2]**3)+1)**2
        x_new,res,total = Program(e,tol,x_new,"DFP",False,True)
        print(f"Solution to problem is x = {x_new}")
        print("f(x) = ",res)
        print("Total function evaluations: ",total)
        print("i = ",i)
        checkboundaries(x_new)
        i += 1

