from typing import Callable,Tuple
import numpy as np
from grad import grad_c,grad_p
from armijos import armijo

def wolfe(f : Callable[[np.ndarray], float], \
    lambda0 : float, \
    alfa : float, \
    epsilon : float, \
    sigma: float,
    x0 : np.ndarray,\
    dk : np.ndarray, \
    func_eval : float
    ) -> Tuple[float,float]:
    try :
        lambda1,func_eval = armijo(f,lambda0,epsilon,alfa,x0,dk,func_eval)
    except ValueError as e:
        print (f"Armijo failed to converge due to {e}")
    a = 0
    F = lambda lamb : f(x0+ lamb * dk)
    def F_prime(lam,func_eval) :
        func_eval += 2
        return grad_p(F,lam),func_eval
    # F_prime = lambda lam : grad_p(F,lam)
    F_prime_0,func_eval = F_prime(0,func_eval)
    # func_eval += 2
    if F_prime_0 > 0: # if the derivative is positive
        return lambda1,func_eval
    F_prime_lambda1,func_eval = F_prime(lambda1,func_eval)
    if abs(F_prime_lambda1) > -sigma*F_prime_0 :
        while F_prime_lambda1 < 0 :
            a = lambda1 
            lambda1 *= alfa # increase lambda
            F_prime_lambda1,func_eval = F_prime(lambda1,func_eval) # calculate the derivative at the new lambda and increase the function evaluation count
            if (F_prime_lambda1) <= -sigma*F_prime_0 :
                return lambda1,func_eval
    b = lambda1
    lambda1 = (a + b) / 2
    F_prime_lambda1,func_eval = F_prime(lambda1,func_eval)
    while abs(F_prime_lambda1) > -sigma*F_prime_0 :
        if F_prime_lambda1 < 0 :
            a = lambda1
        else:
            b = lambda1
        lambda1 = (a + b) / 2
        F_prime_lambda1,func_eval = F_prime(lambda1,func_eval)
    return lambda1,func_eval


