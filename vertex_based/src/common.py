import cmath
from cmath import polar, rect
from math import log, pi, cos, sin, atan2
import numpy as np
from numba import jit, prange
import csv
from enum import Enum
from dataclasses import dataclass
import numpy as np
import mouette as M

########## Distortion ##########

class Distortion(Enum):
    """Description of the different energies used a a distortion in the optimization.
    """
    NONE = 0
    ARAP = 1
    LSCM = 2
    AREA = 3

    @staticmethod
    def from_string(s :str):
        if "lscm" in s.lower():
            return Distortion.LSCM
        if "arap" in s.lower():
            return Distortion.ARAP
        if "area" in s.lower():
            return Distortion.AREA
        return Distortion.NONE

##### Init Mode #####

class InitMode(Enum):
    """
    Initialization mode
    """
    ZERO = 0
    SMOOTH = 1
    CURVATURE = 2
    RANDOM = 3

    @staticmethod
    def from_string(s :str):
        if "zero" in s.lower():
            return InitMode.ZERO
        if "smooth" in s.lower():
            return InitMode.SMOOTH
        if "curv" in s.lower():
            return InitMode.CURVATURE
        if "random" in s.lower():
            return InitMode.RANDOM
        raise Exception(f"InitMode {s} not recognized")

########## Default running options ##########

@dataclass
class VerboseOptions:
    output_dir : str = ""
    logger_verbose : bool = True
    qp_solver_verbose : bool = False
    optim_verbose : bool = True
    log_freq : int = 0
    tqdm : bool = True

@dataclass
class Options:
    distortion : Distortion = Distortion.NONE
    features : bool = False
    initMode : InitMode = InitMode.ZERO
    optimFixedFF : bool = False
    n_iter_max : int = 1000
    free_boundary : bool = False
    dist_schedule : list = None

    def set_schedule(self,sch : list = None):
        if sch is None:
            if self.distortion == Distortion.NONE:
                self.dist_schedule = []
            elif self.distortion == Distortion.LSCM:
                self.dist_schedule = [x for x in np.logspace(2, -2, 5)]
            elif self.distortion in [Distortion.ARAP, Distortion.AREA]:
                self.dist_schedule = [x for x in np.logspace(2, -4, 7)]
        else:
            self.dist_schedule = sch


########## Miscalenous utility functions ##########

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