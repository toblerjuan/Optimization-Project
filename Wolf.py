from typing import Callable
import numpy as np
from grad import grad_c,grad_p
from armijos import armijo

def wolfe(f : Callable[[np.ndarray], float], \
    lambda0 : float, \
    alfa : float, \
    epsilon : float, \
    sigma: float,
    x0 : np.ndarray,\
    dk : np.ndarray
    ) -> float:
    lambda1 = armijo(f,lambda0,epsilon,alfa,x0,dk)
    a = 0
    F = lambda lamb : f(x0+ lamb * dk)
    F_prime = lambda lam : grad_p(F,lam)
    F_prime_0 = grad_p(F,0)
    if abs(F_prime(lambda1)) > -sigma*F_prime_0 :
        while F_prime(lambda1) < 0 :
            a = lambda1
            lambda1 *= alfa
            if (F_prime(lambda1)) <= -sigma*F_prime_0 :
                return lambda1
    b = lambda1
    lambda1 = (a + b) / 2
    while abs(F_prime(lambda1)) > -sigma*F_prime_0 :
        i = i+ 1
        if F_prime(lambda1) < 0 :
            a = lambda1
        else:
            b = lambda1
        lambda1 = (a + b) / 2
    return lambda1

# k = 2
# def f(x : np.ndarray) -> float:
#     return (1-((10**k)* x[0]))**2
# def g(x: np.ndarray) -> float:
#     return 2*np.exp(-x[0])+x[0]
# def h(x: np.ndarray) -> float:
#     return (x-4)**2 + 6*np.cos(x[0])

# # Initial values
# x0 = np.ndarray(shape=(1,), dtype=float)
# x0[0] = 0 # Starting point (arbitrary)
# sol = 3.3574
# lambda0 = 1
# param = [2,0.1,0.1]
# # param (alfa,epsilon,sigma)
# dk = np.ndarray(shape=(1,), dtype=float)
# dk[0] = 1
# result = wolfe(h, lambda0, param[0], param[1], param[2], x0, dk)
# stepsize = result*1
# print(f"Optimal (lambda) found: {result}")
# print(f"Optimal stepsize found: {stepsize}")
# print(f"distance from solution{abs(sol-(x0[0]+stepsize))}")


