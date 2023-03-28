from math import log
import cmath
import numpy as np
from numba import jit
import csv

@jit(nopython=True, fastmath=True)
def barrier(x,t):
    if x<= 0 : return 1E9
    if x>t: return 0
    return log(x/t)**2

@jit(nopython=True, fastmath=True)
def barrier_prime(x, t):
    if x<0 : return 1E9
    if x>t : return 0
    return 2*log(x/t)/x

@jit(nopython=True, fastmath=True)
def barrier_sym(x,t):
    if abs(x)>= t : return 1E9
    return log(t*t)-log( (t-x)*(x+t))

@jit(nopython=True, fastmath=True)
def barrier_sym_prime(x, t):
    if abs(x)>t : return 1E9
    return 2*x/(t-x)/(x+t)

@jit(nopython=True)
def complex_to_mat(c : complex) -> np.ndarray:
    return np.array([[c.real, c.imag], [ -c.imag, c.real]], dtype=np.float64)

@jit(nopython=True, fastmath=True)
def L1reg(x, l):
    if abs(x)>l: return x - l/2
    return x*x/(2*l)

@jit(nopython=True, fastmath=True)
def L1reg_prime(x,l):
    if abs(x)>l: return 1
    return x/l

def export_dict_as_csv(data : dict, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)