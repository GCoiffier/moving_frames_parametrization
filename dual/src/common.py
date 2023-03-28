import cmath
from cmath import polar, rect, phase
from math import log, pi, sqrt, cos, sin, atan, atan2
import numpy as np
from numba import jit, prange
import csv

import mouette as M
from mouette import geometry as geom


"""
Miscalenous utility functions
"""

def fourth_roots(c : complex, normalize=True):
    r,t = polar(c)
    r = 1 if normalize else r**(1/4)
    return (rect(r, (t + 2*k*pi)/4) for k in range(4))

@jit(nopython=True, parallel=True)
def normalize(x):
    for i in prange(x.size):
        x[i] /= abs(x[i])
    return x

@jit(nopython=True)
def custom_round(x, n=2):
    n = 10**n
    return round(x*n)/n

def replace_in_list(l, x, y):
    return [y if e==x else e for e in l]

def c2vec(c):
    # complex to vec2
    return M.Vec(c.real, c.imag)

def c3vec(c):
    # complex to vec3 (with z=0)
    return M.Vec(c.real, c.imag, 0.)

def vec2c(*args):
    # vec2 to complex
    if len(args)==1:
        x,y = args[0][0], args[0][1]
    else:
        x,y = args[0], args[1]
    return complex(x,y)

@jit(nopython=True, fastmath=True)
def sign(x):
    if x>0: return 1
    if x<0 : return -1
    return 0

@jit(nopython=True, fastmath=True)
def pows4(x):
    x2 = x*x
    x3 = x2*x
    x4 = x2*x2
    return x2,x3,x4

# @jit(nopython=True, fastmath=True)
# def barrier_neg(x,t):
#     if x>t: return 0
#     return (x-t)**2

# @jit(nopython=True, fastmath=True)
# def barrier_neg_prime(x, t):
#     if x>t : return 0
#     return 2*(x-t)

@jit(nopython=True, fastmath=True)
def barrier_neg(x,t):
    if x<= 0 : return 1000000000
    if x>t: return 0
    return log(x/t)**2

@jit(nopython=True, fastmath=True)
def barrier_neg_prime(x, t):
    if x<0 : return 1000000000
    if x>t : return 0
    return 2*log(x/t)/x

@jit(nopython=True, fastmath=True)
def barrier_pos(x,t):
    if x<t : return 0
    if x>=2*t: return 1000000000
    return 2/(x-2*t) * log(2 - x/t)

@jit(nopython=True, fastmath=True)
def barrier_pos_prime(x, t):
    if x<t : return 0
    if x>=2*t: return 1000000000
    return log(2 - x/t)**2

@jit(nopython=True, fastmath=True)
def barrier_zero(x,t):
    if x>=t : return (x-t)**2
    if x<=-t : return (x+t)**2
    return 0

@jit(nopython=True, fastmath=True)
def barrier_zero_prime(x, t):
    if x>=t : return 2*(x-t)
    if x<=-t : return  2*(x+t)
    return 0.

def crot(w):
    # exp(iw)
    return cmath.rect(1., w)

def roots(c : complex, pow: int, normalize=True):
    """Given a complex number c, compute and returns the four ^(1/4) roots 
    Parameters:
        c (complex): input complex number
        pow (int): the power of the root. Returns power-th roots of c
        normalize (bool, optional): If True, roots will have module 1. Defaults to True

    Returns:
        list of complex roots
    """
    r,t = cmath.polar(c)
    r = 1 if normalize else r**(1/pow)
    return [cmath.rect(r, (t + 2*k*pi)/pow) for k in range(pow)]

@jit(nopython=True)
def angle_diff(a,b):
    return (a - b + pi) % (2*pi) - pi

def principal_angle(a):
    """
    From an arbitrary angle value, returns the equivalent angle which values lays in [-pi, pi[
    """
    b = a%(2*pi)
    if b>pi:
        b-=2*pi
    return b

def solve_quadratic(A : float ,B : float, C : float):
    """Solves Ax² + Bx + C = 0 for real-valued roots

    Parameters:
        A (float): Coefficient
        B (float): Coefficient
        C (float): Coefficient
    """
    if A==0: # linear case
        if B == 0:
            return []
        return [-C/B]
    delta = B*B-4*A*C
    if delta<0:
        return []
    if abs(delta)<1e-14: # delta = 0
        return [-B/(2*A)]
    else:
        delta = sqrt(delta)
        return [ (-B + delta)/(2*A), (-B - delta)/(2*A)]

@jit(nopython=True)
def other_end(xa,ya,w):
    cw,sw = cos(w), sin(w)
    xb = cw*xa - sw*ya
    yb = sw*xa + cw*ya
    return xb,yb

def align_edges(pA,pB, qA, qB, qC):
    """Apply a similarity (translation/rotation/scale) on points (qA,qB,qC) so that edge (qA,qB) corresponds to edge (pA,pB)

    Returns:
        qA,qB,qC
    """

    target = pB - pA

    # 1) Scaling to match edge length
    # q = qB - qA
    # scale = sqrt(np.dot(target,target) / np.dot(q,q))
    # qB = qA + (qB-qA)*scale
    # qC = qA + (qC-qA)*scale

    # 2) translation to align point A
    translation = pA - qA
    qA += translation
    qB += translation
    qC += translation

    # 3) rotation around point A to align the point B
    q = qB - qA
    rot = atan2(target.y, target.x) - atan2(q.y,q.x)
    rc, rs = cos(rot), sin(rot)

    q = qB - qA
    qB.x, qB.y = qA.x + rc*q.x - rs*q.y , qA.y + rs*q.x + rc*q.y
    q = qC-qA
    qC.x, qC.y = qA.x + rc*q.x - rs*q.y , qA.y + rs*q.x + rc*q.y

    return qA,qB,qC

def export_dict_as_csv(data : dict, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)