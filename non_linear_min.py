from typing import Callable, Tuple
import numpy as np

def non_linear_min(f : Callable[[np.ndarray], float], \
    x0 : np.ndarray, \
    method : str, \
    tol : float, \
    restart : bool, \
    printout : bool) \
    -> Tuple[np.ndarray, int, int, float]:
    return None