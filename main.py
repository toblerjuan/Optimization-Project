import numpy as np
from rosenbrock import rosenbrock
from program import Program
from grad import grad_c

# i want 4 decimals when i print the results 
np.set_printoptions(precision=6, suppress=False)

def f (x : np.ndarray) -> np.ndarray: # with 2 variables: x[0],x[1]
    return x[0]**2 + x[1]**2
def h (x : np.ndarray) -> np.ndarray: # with 2 variables: x[0],x[1]
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def j (x : np.ndarray) -> np.ndarray: # with 4 variables: x[0],x[1],x[2],x[3]
    return (x[0] + 10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4
# def e (x : np.ndarray) -> np.ndarray:
#     return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])+ m*(i)*((x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10)**2 + m(i)*(x[1]*x[2] - (5*x[3]*x[4]))**2 + m(i)*abs((x[0]**3+x[2]**3)+1)**2

def checkboundaries(x : np.ndarray) -> np.ndarray:
    return np.array([(x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10 ,(x[1]*x[2] - 5*x[3]*x[4]),(x[0]**3+x[2]**3)+1])

start_x_2d = np.array([14,132])
start_x_4d = np.array([1,1,1,-1])
start_x_5d = np.array([-2,2,2,-1,-1])
new = np.array([-1.526286 ,1.404612 ,1.367206 ,-1.383988 ,-1.383988])
if False :
    tol = 1e-9                  #<- Change here
    startpoint =  start_x_2d    #<- Change here
    restart = False              #<- Change here 
    restart_freq = 20           #<- Change here
    method = "DFP"             #<- Change here
    function = f                #<- Change here
    x,res,total = Program(function,tol,startpoint,method,restart,True,restart_freq)

    np.set_printoptions(precision=16)  # Set precision to 16 decimal places (or as required)
    print("startingpoint = ",startpoint)
    print("method = ",method)
    print(f"Solution to problem is x = {x}")
    print("tol = ",tol)
    print("gradient = ",grad_c(function, x))
    print("f(x) = ",res)
    print("Total function evaluations: ",total)
    print("restart = ",restart)
    if restart :
        print("restart_freq = ",restart_freq)
    else :
        print("restart_freq = None")
elif False :

    function = rosenbrock
    x_range = np.array([-10,1000])
    cycles = 50
    tol = 1e-10
    restart_freq = 20
    x_sol = np.array([1,1])
    

    x_start = np.random.rand(cycles, x_sol.shape[0])*(x_range[1]-x_range[0])+x_range[0]
    x_ans_D, x_ans_B = np.empty((cycles,x_sol.shape[0])), np.empty((cycles,x_sol.shape[0]))
    eval_D, conv_D, eval_B, conv_B= np.empty(cycles), np.empty(cycles), np.empty(cycles), np.empty(cycles)
    
    i = 0
    while i < cycles :
        x_ans_D[i,:],res,eval_D[i] = Program(function,tol,x_start[i,:],"DFP",True,False,restart_freq)
        if np.linalg.norm(x_ans_D[i,:] - x_sol) < 1e-6 :
            conv_D[i] = True
        else :
            conv_D[i] = False
        x_ans_B[i,:],res,eval_B[i] = Program(function,tol,x_start[i,:],"BFGS",True,False,restart_freq)
        if np.linalg.norm(x_ans_B[i,:] - x_sol) < 1e-6 :
            conv_B[i] = True
        else :
            conv_B[i] = False
        i += 1
    i = 0
    while i < 50 :
        print(f"Start: {x_start[i,:]}")
        print(f"Method: DFP , Convredge: {conv_D[i]}, Func.Eval: {eval_D[i]:4.1f}")
        print(f"Method: BFGS, Convredge: {conv_B[i]}, Func.Eval: {eval_B[i]:4.1f}")
        print("")
        i += 1
else :
    print("New code here")

def pen() : # Run this to try the penalty problem
    outer_tol = 1e-6
    m = lambda lam : 10**(lam - 5)
    i = 0
    tol = 1e-4                #<- Change here   
    restart = True              #<- Change here 
    restart_freq = 20           #<- Change here
    method = "DFP"             #<- Change here
    x_new = start_x_5d
    print("method = ",method)
    while (i < 20) :
        def e (x : np.ndarray) -> np.ndarray:
            return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])+ m(i)*((x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2)-10)**2 + m(i)*(x[1]*x[2] - (5*x[3]*x[4]))**2 + m(i)*((x[0]**3+x[2]**3)+1)**2
        print("Startingpoint = ",x_new)
        x_new,res,total = Program(e,tol,x_new,method,restart,False,restart_freq)
        print("mu = ",m(i))
        print("Endpoint = ",x_new)
        print("f(x) = ",res)
        print("Iteration = ",i)
        i += 1
        bound = checkboundaries(x_new)
        print("Boundaries = ",bound)
        if np.linalg.norm(bound) < outer_tol :
            print("Penalty problem done")
            break
        elif total == -1 :
            print("penalty problem not solved")
            break
pen()

