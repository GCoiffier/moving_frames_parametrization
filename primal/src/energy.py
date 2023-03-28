from .common import *
import numpy as np
import math
from math import pi, log, atan
from numba import jit, prange

##### Energy functions #####

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_fixedFF_noJ(X : np.ndarray, I, E, PT, sepRot:int):
    """Edges should match up to the rotation between adjacent triangles. Version where angles and the frame field are fixed"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        e, T1, T2 = I[i, :]
        w = 2*math.atan(X[sepRot+e])
        a = PT[i]
        cw,sw = math.cos(a - w), math.sin(a - w)
        e1,e2 = E[i,0,:], E[i,1,:]
        a1,b1,c1,d1 = X[4*T1: 4*(T1 + 1)] # jac of T1
        a2,b2,c2,d2 = X[4*T2: 4*(T2 + 1)] # jac of T2
        val[2*i] = e1[0] * a1 + e1[1] * c1 - cw*e2[0]*a2 - cw*e2[1]*c2 - sw*e2[0]*b2 - sw*e2[1]*d2
        val[2*i+1] = e1[0] * b1 + e1[1] * d1 + sw*e2[0]*a2 + sw*e2[1]*c2 - cw*e2[0]*b2 - cw*e2[1]*d2
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_fixedFF(X : np.ndarray, I, E, PT, sepRot:int):
    """Edges should match up to the rotation between adjacent triangles. Version where angles and the frame field are fixed"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(12*n, dtype=np.float64)
    rowJ = np.zeros(12*n, dtype=np.int32)
    colJ = np.zeros(12*n, dtype=np.int32)

    for i in prange(n):
        e, T1, T2 = I[i, :]
        w = 2*math.atan(X[sepRot+e])
        a = PT[i]
        cw,sw = math.cos(a - w), math.sin(a - w)

        e1,e2 = E[i,0,:], E[i,1,:]
        a1,b1,c1,d1 = X[4*T1: 4*(T1 + 1)] # jac of T1
        a2,b2,c2,d2 = X[4*T2: 4*(T2 + 1)] # jac of T2
        # a c
        # b d

        val[2*i] = e1[0] * a1 + e1[1] * c1 - cw*e2[0]*a2 - cw*e2[1]*c2 - sw*e2[0]*b2 - sw*e2[1]*d2
        rowJ[12*i],   colJ[12*i],   valJ[12*i]   = 2*i, 4*T1,    e1[0] # a1
        rowJ[12*i+1], colJ[12*i+1], valJ[12*i+1] = 2*i, 4*T1+2,  e1[1] # c1
        rowJ[12*i+2], colJ[12*i+2], valJ[12*i+2] = 2*i, 4*T2,   -cw*e2[0]# a2
        rowJ[12*i+3], colJ[12*i+3], valJ[12*i+3] = 2*i, 4*T2+1, -sw*e2[0] # b2
        rowJ[12*i+4], colJ[12*i+4], valJ[12*i+4] = 2*i, 4*T2+2, -cw*e2[1] # c2
        rowJ[12*i+5], colJ[12*i+5], valJ[12*i+5] = 2*i, 4*T2+3, -sw*e2[1] # d2

        val[2*i+1] = e1[0] * b1 + e1[1] * d1 + sw*e2[0]*a2 + sw*e2[1]*c2 - cw*e2[0]*b2 - cw*e2[1]*d2
        rowJ[12*i+6],  colJ[12*i+6],  valJ[12*i+6]  = 2*i+1, 4*T1+1,  e1[0] # b1
        rowJ[12*i+7],  colJ[12*i+7],  valJ[12*i+7]  = 2*i+1, 4*T1+3,  e1[1] # d1
        rowJ[12*i+8],  colJ[12*i+8],  valJ[12*i+8]  = 2*i+1, 4*T2,    sw*e2[0]# a2
        rowJ[12*i+9],  colJ[12*i+9],  valJ[12*i+9]  = 2*i+1, 4*T2+1, -cw*e2[0] # b2
        rowJ[12*i+10], colJ[12*i+10], valJ[12*i+10] = 2*i+1, 4*T2+2,  sw*e2[1] # c2
        rowJ[12*i+11], colJ[12*i+11], valJ[12*i+11] = 2*i+1, 4*T2+3, -cw*e2[1] # d2
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_noJ(X, I, E, PT, sepRot):
    """Edges should match up to the rotation they carry. Version where rotations are considered as variables"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        e, T1, T2 = I[i, :]
        iw = sepRot + e
        w = X[iw]
        a = PT[i]
        pt = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) # parallel transport rotation matrix

        a1,b1,c1,d1 = X[4*T1 : 4*(T1 + 1)]
        a2,b2,c2,d2 = X[4*T2 : 4*(T2 + 1)]

        J1 = np.array([[a1,c1], [b1,d1]])
        J2 = np.array([[a2,c2], [b2,d2]])
        e1, e2 = E[i,0,:], E[i,1,:]

        wm = np.array([[1, w], [-w, 1]])
        wp = np.array([[1, -w], [w, 1]])

        en = wm @ J1 @ e1 - wp @ pt @ J2 @ e2
        val[2*i] = en[0]
        val[2*i+1] = en[1]
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge(X, I, E, PT, sepRot):
    """Edges should match up to the rotation they carry. Version where rotations are considered as variables"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(18*n, dtype=np.float64)
    rowJ = np.zeros(18*n, dtype=np.int32)
    colJ = np.zeros(18*n, dtype=np.int32)

    dJa = np.array([[1.,0.], [0.,0.]], dtype=np.float64)
    dJb = np.array([[0.,0.], [1.,0.]], dtype=np.float64)
    dJc = np.array([[0.,1.], [0.,0.]], dtype=np.float64)
    dJd = np.array([[0.,0.], [0.,1.]], dtype=np.float64)
    dw = np.array([[0.,1.], [-1.,0.]], dtype=np.float64)

    for i in prange(n):
        e, T1, T2 = I[i, :]
        iw = sepRot + e
        w = X[iw]
        a = PT[i]
        pt = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) # parallel transport rotation matrix

        a1,b1,c1,d1 = X[4*T1 : 4*(T1 + 1)]
        a2,b2,c2,d2 = X[4*T2 : 4*(T2 + 1)]

        J1 = np.array([[a1,c1], [b1,d1]])
        J2 = np.array([[a2,c2], [b2,d2]])
        e1, e2 = E[i,0,:], E[i,1,:]

        wm = np.array([[1, w], [-w, 1]])
        wp = np.array([[1, -w], [w, 1]])

        en = wm @ J1 @ e1 - wp @ pt @ J2 @ e2
        val[2*i] = en[0]
        val[2*i+1] = en[1]

        dev_w = dw @ ( J1 @ e1 + pt @ J2 @ e2)
        rowJ[18*i], colJ[18*i], valJ[18*i] = 2*i, iw, dev_w[0]
        rowJ[18*i+1], colJ[18*i+1], valJ[18*i+1] = 2*i+1, iw, dev_w[1]

        dev_A1 = wm @ dJa @ e1
        rowJ[18*i+2], colJ[18*i+2], valJ[18*i+2] = 2*i, 4*T1, dev_A1[0]
        rowJ[18*i+3], colJ[18*i+3], valJ[18*i+3] = 2*i+1, 4*T1, dev_A1[1]

        dev_A2 = - wp @ pt @ dJa @ e2
        rowJ[18*i+4], colJ[18*i+4], valJ[18*i+4] = 2*i, 4*T2, dev_A2[0]
        rowJ[18*i+5], colJ[18*i+5], valJ[18*i+5] = 2*i+1, 4*T2, dev_A2[1]
        
        dev_B1 = wm @ dJb @ e1
        rowJ[18*i+6], colJ[18*i+6], valJ[18*i+6] = 2*i, 4*T1+1, dev_B1[0]
        rowJ[18*i+7], colJ[18*i+7], valJ[18*i+7] = 2*i+1, 4*T1+1, dev_B1[1]

        dev_B2 = -wp @ pt @ dJb @ e2 
        rowJ[18*i+8], colJ[18*i+8], valJ[18*i+8] = 2*i, 4*T2+1, dev_B2[0]
        rowJ[18*i+9], colJ[18*i+9], valJ[18*i+9] = 2*i+1, 4*T2+1, dev_B2[1]
        
        dev_C1 = wm @ dJc @ e1
        rowJ[18*i+10], colJ[18*i+10], valJ[18*i+10] = 2*i, 4*T1+2, dev_C1[0]
        rowJ[18*i+11], colJ[18*i+11], valJ[18*i+11] = 2*i+1, 4*T1+2, dev_C1[1]

        dev_C2 = - wp @ pt @ dJc @ e2
        rowJ[18*i+12], colJ[18*i+12], valJ[18*i+12] = 2*i, 4*T2+2, dev_C2[0]
        rowJ[18*i+13], colJ[18*i+13], valJ[18*i+13] = 2*i+1, 4*T2+2, dev_C2[1]
        
        dev_D1 = wm @ dJd @ e1
        rowJ[18*i+14], colJ[18*i+14], valJ[18*i+14] = 2*i, 4*T1+3, dev_D1[0]
        rowJ[18*i+15], colJ[18*i+15], valJ[18*i+15] = 2*i+1, 4*T1+3, dev_D1[1]

        dev_D2 = - wp @ pt @ dJd @ e2
        rowJ[18*i+16], colJ[18*i+16], valJ[18*i+16] = 2*i, 4*T2+3, dev_D2[0]
        rowJ[18*i+17], colJ[18*i+17], valJ[18*i+17] = 2*i+1, 4*T2+3, dev_D2[1]

    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def constraint_rotations_follows_ff_noJ(X,I,PT, sepRot):
    """Rotations along edges should match the rotation between the two frames (up to a symmetry of the square)"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        e, iz1, iz2 = I[i,:]
        iw = sepRot+e
        w = X[iw]
        z1, z2 = complex(X[iz1], X[iz1+1]), complex(X[iz2], X[iz2+1]) # the ff representation
        a = PT[i]
        a = cmath.rect(1, 4*a)
        eiwp = complex(1, w) # positive
        eiwn = complex(1,-w) # negative
        val_e = (eiwp**4)*z2 - (eiwn**4)*a*z1
        val[2*i] = val_e.real
        val[2*i+1] = val_e.imag
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_rotations_follows_ff(X,I,PT, sepRot):
    """Rotations along edges should match the rotation between the two frames (up to a symmetry of the square)"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(10*n, dtype=np.float64)
    rowJ = np.zeros(10*n, dtype=np.int32)
    colJ = np.zeros(10*n, dtype=np.int32)

    for i in prange(n):
        e, iz1, iz2 = I[i,:]
        iw = sepRot+e
        w = X[iw]
        z1, z2 = complex(X[iz1], X[iz1+1]), complex(X[iz2], X[iz2+1]) # the ff representation
        a = PT[i]
        a = cmath.rect(1, 4*a)

        #  Cayley transform
        eiwp = complex(1, w) # positive
        eiwn = complex(1,-w) # negative
        val_e = (eiwp**4)*z2 - (eiwn**4)*a*z1
        
        val[2*i] = val_e.real
        val[2*i+1] = val_e.imag

        dev_w =  4j * ( (eiwp**3) * z2  + (eiwn**3) * a * z1)
        rowJ[10*i], colJ[10*i], valJ[10*i] = 2*i, iw, dev_w.real
        rowJ[10*i+1], colJ[10*i+1], valJ[10*i+1] = 2*i+1, iw, dev_w.imag

        dev_a = -(eiwn**4) * a
        rowJ[10*i+2], colJ[10*i+2], valJ[10*i+2] = 2*i, iz1, dev_a.real
        rowJ[10*i+3], colJ[10*i+3], valJ[10*i+3] = 2*i+1, iz1, dev_a.imag
        rowJ[10*i+4], colJ[10*i+4], valJ[10*i+4] = 2*i, iz1+1, -dev_a.imag
        rowJ[10*i+5], colJ[10*i+5], valJ[10*i+5] = 2*i+1, iz1+1, dev_a.real

        dev_b = eiwp**4
        rowJ[10*i+6], colJ[10*i+6], valJ[10*i+6] = 2*i, iz2, dev_b.real
        rowJ[10*i+7], colJ[10*i+7], valJ[10*i+7] = 2*i+1, iz2, dev_b.imag
        rowJ[10*i+8], colJ[10*i+8], valJ[10*i+8] = 2*i, iz2+1, -dev_b.imag
        rowJ[10*i+9], colJ[10*i+9], valJ[10*i+9] = 2*i+1, iz2+1, dev_b.real

    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def ff_norm_noJ(X, n, sep):
    val  = np.zeros(n , dtype=np.float64)
    for T in prange(n):
        zx, zy = X[sep+T], X[sep+T+1]
        val[T] = zx*zx+zy*zy-1
    return val

@jit(nopython=True, parallel=True, cache=True)
def ff_norm(X, n, sep):
    val  = np.zeros(n , dtype=np.float64)
    valJ = np.zeros(2*n, dtype=np.float64)
    rowJ = np.zeros(2*n, dtype=np.int32)
    colJ = np.zeros(2*n, dtype=np.int32)
    for T in prange(n):
        zx, zy = X[sep+T], X[sep+T+1]
        val[T] = zx*zx+zy*zy-1
        rowJ[2*T],   colJ[2*T],   valJ[2*T]   = T, sep+T,   2*zx
        rowJ[2*T+1], colJ[2*T+1], valJ[2*T+1] = T, sep+T+1, 2*zy
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def rot_norm_noJ(X, n, sep):
    val  = np.zeros(n , dtype=np.float64)
    for e in prange(n):
        val[e] = X[sep + e] * X[sep + e]
    return val

@jit(nopython=True, parallel=True, cache=True)
def rot_norm(X, n, sep):
    val  = np.zeros(n , dtype=np.float64)
    valJ = np.zeros(n, dtype=np.float64)
    rowJ = np.arange(0,n).astype(np.int32)
    colJ = np.arange(0,n).astype(np.int32)
    for e in prange(n):
        val[e] = X[sep + e] * X[sep + e]
        valJ[e] = 2*X[sep + e]
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def rot_norm(X, n, sep):
    val  = np.zeros(n , dtype=np.float64)
    valJ = np.zeros(n, dtype=np.float64)
    rowJ = np.arange(0,n).astype(np.int32)
    colJ = np.arange(0,n).astype(np.int32) + sep
    for e in prange(n):
        val[e] = X[sep + e] * X[sep + e]
        valJ[e] = 2*X[sep + e]
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def barrier_det_noJ(X, n, t):
    # n is number of faces
    val = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        a,b,c,d = X[4*i:4*(i+1)]
        det = a*d-b*c
        v = barrier(det, t)
        val[i] = v
    return val

@jit(nopython=True, parallel=True, cache=True)
def barrier_det(X, n, t):
    """Barrier term to prevent triangles from flipping"""
    # n is number of faces
    val = np.zeros(n, dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)

    for i in prange(n):
        # a c
        # b d
        a,b,c,d = X[4*i:4*(i+1)]
        det = a*d-b*c
        v = barrier(det, t)
        dv = barrier_prime(det, t)
        val[i] = v
        rowJ[4*i],   colJ[4*i],   valJ[4*i]   = i, 4*i,    d*dv
        rowJ[4*i+1], colJ[4*i+1], valJ[4*i+1] = i, 4*i+1, -c*dv
        rowJ[4*i+2], colJ[4*i+2], valJ[4*i+2] = i, 4*i+2, -b*dv
        rowJ[4*i+3], colJ[4*i+3], valJ[4*i+3] = i, 4*i+3,  a*dv
    return val, valJ, rowJ, colJ

##### Distortions #####

@jit(nopython=True, parallel=True, cache=True)
def distortion_lscm_noJ(X, n):
    """UV and XYZ of triangles should minimize the Least-Square Conformal Map energy"""
    val = np.zeros(2*n, dtype=np.float64)
    for T in prange(n):
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[2*T] = a-d
        val[2*T+1] = b+c
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_lscm(X, n):
    """UV and XYZ of triangles should minimize the Least-Square Conformal Map energy"""
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)
    for T in prange(n):
        ia,ib,ic,id = 4*T, 4*T+1, 4*T+2, 4*T+3
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[2*T] = a-d
        val[2*T+1] = b+c
        rowJ[4*T],   colJ[4*T],   valJ[4*T]   = 2*T, ia, 1
        rowJ[4*T+1], colJ[4*T+1], valJ[4*T+1] = 2*T, id, -1
        rowJ[4*T+2], colJ[4*T+2], valJ[4*T+2] = 2*T+1, ib, 1
        rowJ[4*T+3], colJ[4*T+3], valJ[4*T+3] = 2*T+1, ic, 1
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_id_noJ(X, n):
    """J = Id"""
    val = np.copy(X[:4*n])
    for T in prange(n):
        val[4*T] -= 1.
        val[4*T+3] -= 1.
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_id(X, n):
    """J = Id"""
    val  = np.copy(X[:4*n])
    valJ = np.ones(4*n, dtype=np.float64)
    rowJ = np.arange(0,4*n).astype(np.int32)
    colJ = np.arange(0,4*n).astype(np.int32)
    for T in prange(n):
        val[4*T] -= 1.
        val[4*T+3] -= 1.
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_isometric_noJ(X, n):
    """JtJ = Id"""
    val = np.zeros(3*n, dtype=np.float64)
    for T in prange(n):
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[3*T] = a*a+b*b-1
        val[3*T+1] = c*c+d*d-1
        val[3*T+2] = a*c+b*d
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_isometric(X, n):
    """JtJ = Id"""
    val = np.zeros(3*n, dtype=np.float64)
    valJ = np.zeros(8*n, dtype=np.float64)
    rowJ = np.zeros(8*n, dtype=np.int32)
    colJ = np.zeros(8*n, dtype=np.int32)
    for T in prange(n):
        ia,ib,ic,id = 4*T, 4*T+1, 4*T+2, 4*T+3
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[3*T] = a*a+b*b-1
        rowJ[8*T], colJ[8*T], valJ[8*T] = 3*T, ia, 2*a
        rowJ[8*T+1], colJ[8*T+1], valJ[8*T+1] = 3*T, ib, 2*b
        
        val[3*T+1] = c*c+d*d-1
        rowJ[8*T+2], colJ[8*T+2], valJ[8*T+2] = 3*T+1, ic, 2*c
        rowJ[8*T+3], colJ[8*T+3], valJ[8*T+3] = 3*T+1, id, 2*d
        
        val[3*T+2] = 2*(a*c+b*d)
        rowJ[8*T+4], colJ[8*T+4], valJ[8*T+4] = 3*T+2, ia, 2*c
        rowJ[8*T+5], colJ[8*T+5], valJ[8*T+5] = 3*T+2, ib, 2*d
        rowJ[8*T+6], colJ[8*T+6], valJ[8*T+6] = 3*T+2, ic, 2*a
        rowJ[8*T+7], colJ[8*T+7], valJ[8*T+7] = 3*T+2, id, 2*b
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_shear_noJ(X, n):
    """Forces only the third term of isometric energy"""
    val = np.zeros(n, dtype=np.float64)
    for T in prange(n):
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[T] = a*c+b*d
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_shear(X, n):
    """Forces only the third term of isometric energy"""
    val = np.zeros(n, dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)
    for T in prange(n):
        ia,ib,ic,id = 4*T, 4*T+1, 4*T+2, 4*T+3
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[T] = a*c+b*d
        rowJ[4*T],   colJ[4*T],   valJ[4*T] = T, ia, c
        rowJ[4*T+1], colJ[4*T+1], valJ[4*T+1] = T, ib, d
        rowJ[4*T+2], colJ[4*T+2], valJ[4*T+2] = T, ic, a
        rowJ[4*T+3], colJ[4*T+3], valJ[4*T+3] = T, id, b
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def conformal_connection_noJ(X : np.ndarray, I : np.ndarray, cot : np.ndarray, sepRot, sepScale):
    n = I.shape[0]
    val = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        ie, iT1,iT2 = I[i,:]
        val[i] = 2*atan(X[sepRot + ie]) - cot[ie]*(X[sepScale + iT1] - X[sepScale + iT2])
    return val

@jit(nopython=True, parallel=True, cache=True)
def conformal_connection(X : np.ndarray, I : np.ndarray, cot : np.ndarray, sepRot, sepScale):
    n = I.shape[0]
    val = np.zeros(n, dtype=np.float64)
    valJ = np.zeros(3*n, dtype=np.float64)
    rowJ = np.zeros(3*n, dtype=np.int32)
    colJ = np.zeros(3*n, dtype=np.int32)

    for i in prange(n):
        ie, iT1,iT2 = I[i,:]
        val[i] = 2*atan(X[sepRot + ie]) - cot[ie]*(X[sepScale + iT1] - X[sepScale + iT2])
        rowJ[3*i],   colJ[3*i],   valJ[3*i] = i, sepRot + ie, 2/(1+ X[sepRot + ie]*X[sepRot + ie])
        rowJ[3*i+1], colJ[3*i+1], valJ[3*i+1] = i, sepScale + iT1, -cot[ie]
        rowJ[3*i+2], colJ[3*i+2], valJ[3*i+2] = i, sepScale + iT2,  cot[ie]
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_edge_normal_noJ(X, I, E, PT, sepRot):
    """Edges should match up to the rotation they carry. Version where rotations are considered as variables"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        e, T1, T2 = I[i, :]
        iw = sepRot + e
        w = X[iw]
        a = PT[i]
        pt = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) # parallel transport rotation matrix

        a1,b1,c1,d1 = X[4*T1 : 4*(T1 + 1)]
        a2,b2,c2,d2 = X[4*T2 : 4*(T2 + 1)]

        J1 = np.array([[a1,c1], [b1,d1]])
        J2 = np.array([[a2,c2], [b2,d2]])
        e1, e2 = E[i,0,:], E[i,1,:]
        e1 = np.array([-e1[1], e1[0]])
        e2 = np.array([-e2[1], e2[0]])

        wm = np.array([[1, w], [-w, 1]])
        wp = np.array([[1, -w], [w, 1]])

        en = wm @ J1 @ e1 - wp @ pt @ J2 @ e2
        val[2*i] = en[0]
        val[2*i+1] = en[1]
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_edge_normal(X, I, E, PT, sepRot):
    """Edges should match up to the rotation they carry. Version where rotations are considered as variables"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(18*n, dtype=np.float64)
    rowJ = np.zeros(18*n, dtype=np.int32)
    colJ = np.zeros(18*n, dtype=np.int32)

    dJa = np.array([[1.,0.], [0.,0.]])
    dJb = np.array([[0.,0.], [1.,0.]])
    dJc = np.array([[0.,1.], [0.,0.]])
    dJd = np.array([[0.,0.], [0.,1.]])
    dw = np.array([[0.,1.], [-1.,0.]])

    for i in prange(n):
        e, T1, T2 = I[i, :]
        iw = sepRot + e
        w = X[iw]
        a = PT[i]
        pt = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) # parallel transport rotation matrix

        a1,b1,c1,d1 = X[4*T1 : 4*(T1 + 1)]
        a2,b2,c2,d2 = X[4*T2 : 4*(T2 + 1)]

        J1 = np.array([[a1,c1], [b1,d1]])
        J2 = np.array([[a2,c2], [b2,d2]])
        e1, e2 = E[i,0,:], E[i,1,:]
        e1 = np.array([-e1[1], e1[0]])
        e2 = np.array([-e2[1], e2[0]])

        wm = np.array([[1, w], [-w, 1]])
        wp = np.array([[1, -w], [w, 1]])

        en = wm @ J1 @ e1 - wp @ pt @ J2 @ e2
        val[2*i] = en[0]
        val[2*i+1] = en[1]

        dev_w =  dw @ ( J1 @ e1 + pt @ J2 @ e2)
        rowJ[18*i], colJ[18*i], valJ[18*i] = 2*i, iw, dev_w[0]
        rowJ[18*i+1], colJ[18*i+1], valJ[18*i+1] = 2*i+1, iw, dev_w[1]

        dev_A1 = wm @ dJa @ e1
        rowJ[18*i+2], colJ[18*i+2], valJ[18*i+2] = 2*i, 4*T1, dev_A1[0]
        rowJ[18*i+3], colJ[18*i+3], valJ[18*i+3] = 2*i+1, 4*T1, dev_A1[1]

        dev_A2 = - wp @ pt @ dJa @ e2
        rowJ[18*i+4], colJ[18*i+4], valJ[18*i+4] = 2*i, 4*T2, dev_A2[0]
        rowJ[18*i+5], colJ[18*i+5], valJ[18*i+5] = 2*i+1, 4*T2, dev_A2[1]
        
        dev_B1 = wm @ dJb @ e1
        rowJ[18*i+6], colJ[18*i+6], valJ[18*i+6] = 2*i, 4*T1+1, dev_B1[0]
        rowJ[18*i+7], colJ[18*i+7], valJ[18*i+7] = 2*i+1, 4*T1+1, dev_B1[1]

        dev_B2 = - wp @ pt @ dJb @ e2
        rowJ[18*i+8], colJ[18*i+8], valJ[18*i+8] = 2*i, 4*T2+1, dev_B2[0]
        rowJ[18*i+9], colJ[18*i+9], valJ[18*i+9] = 2*i+1, 4*T2+1, dev_B2[1]
        
        dev_C1 = wm @ dJc @ e1
        rowJ[18*i+28], colJ[18*i+28], valJ[18*i+28] = 2*i, 4*T1+2, dev_C1[0]
        rowJ[18*i+11], colJ[18*i+11], valJ[18*i+11] = 2*i+1, 4*T1+2, dev_C1[1]

        dev_C2 = - wp @ pt @ dJc @ e2
        rowJ[18*i+12], colJ[18*i+12], valJ[18*i+12] = 2*i, 4*T2+2, dev_C2[0]
        rowJ[18*i+13], colJ[18*i+13], valJ[18*i+13] = 2*i+1, 4*T2+2, dev_C2[1]
    
        dev_D1 = wm @ dJd @ e1
        rowJ[18*i+14], colJ[18*i+14], valJ[18*i+14] = 2*i, 4*T1+3, dev_D1[0]
        rowJ[18*i+15], colJ[18*i+15], valJ[18*i+15] = 2*i+1, 4*T1+3, dev_D1[1]

        dev_D2 = - wp @ pt @ dJd @ e2
        rowJ[18*i+16], colJ[18*i+16], valJ[18*i+16] = 2*i, 4*T2+3, dev_D2[0]
        rowJ[18*i+17], colJ[18*i+17], valJ[18*i+17] = 2*i+1, 4*T2+3, dev_D2[1]

    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_rigid_noJ(X, I, E, R, sepRot):
    n = E.shape[0]
    val = np.zeros(4*n, dtype=np.float64)
    for i in prange(n):
        e,T1,T2 = I[i//2,:]
        iw = sepRot + e
        w = X[iw]
        A = R[i,:,:]
        a1,b1,c1,d1 = X[4*T1 : 4*(T1 + 1)]
        a2,b2,c2,d2 = X[4*T2 : 4*(T2 + 1)]

        J1 = np.array([[a1,c1], [b1,d1]])
        J2 = np.array([[a2,c2], [b2,d2]])
        wm = np.array([[1, w], [-w, 1]])
        wp = np.array([[1, -w], [w, 1]])

        for k,(e1,e2) in enumerate([(E[i,0], E[i,1]), (E[i,2], E[i,3])]):
            en =  wm @ J1 @ e1 - wp @ A @ J2 @ e2
            ec = 4*i+2*k
            val[ec] = en[0]
            val[ec+1] = en[1]
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_rigid(X, I, E, R, sepRot):
    n = E.shape[0]
    val = np.zeros(4*n, dtype=np.float64)
    valJ = np.zeros(36*n, dtype=np.float64)
    rowJ = np.zeros(36*n, dtype=np.int32)
    colJ = np.zeros(36*n, dtype=np.int32)
    
    dJa = np.array([[1.,0.], [0.,0.]], dtype=np.float64)
    dJb = np.array([[0.,0.], [1.,0.]], dtype=np.float64)
    dJc = np.array([[0.,1.], [0.,0.]], dtype=np.float64)
    dJd = np.array([[0.,0.], [0.,1.]], dtype=np.float64)
    dw = np.array([[0.,1.], [-1.,0.]], dtype=np.float64)

    for i in prange(n):
        e,T1,T2 = I[i//2,:]
        iw = sepRot + e
        w = X[iw]
        A = R[i,:,:]

        a1,b1,c1,d1 = X[4*T1 : 4*(T1 + 1)]
        a2,b2,c2,d2 = X[4*T2 : 4*(T2 + 1)]

        J1 = np.array([[a1,c1], [b1,d1]])
        J2 = np.array([[a2,c2], [b2,d2]])
        wm = np.array([[1, w], [-w, 1]])
        wp = np.array([[1, -w], [w, 1]])

        for k,(e1,e2) in enumerate([(E[i,0], E[i,1]), (E[i,2], E[i,3])]):
            en =  wm @ J1 @ e1 - wp @ A @ J2 @ e2
            ec = 4*i+2*k
            cc = 36*i + 18*k

            val[ec] = en[0]
            val[ec+1] = en[1]

            dev_w = dw @ ( J1 @ e1 + A @ J2 @ e2)
            rowJ[cc], colJ[cc], valJ[cc] = ec, iw, dev_w[0]
            rowJ[cc+1], colJ[cc+1], valJ[cc+1] = ec+1, iw, dev_w[1]

            dev_A1 = wm @ dJa @ e1
            rowJ[cc+2], colJ[cc+2], valJ[cc+2] = ec, 4*T1, dev_A1[0]
            rowJ[cc+3], colJ[cc+3], valJ[cc+3] = ec+1, 4*T1, dev_A1[1]

            dev_A2 = - wp @ A @ dJa @ e2
            rowJ[cc+4], colJ[cc+4], valJ[cc+4] = ec, 4*T2, dev_A2[0]
            rowJ[cc+5], colJ[cc+5], valJ[cc+5] = ec+1, 4*T2, dev_A2[1]
            
            dev_B1 = wm @ dJb @ e1
            rowJ[cc+6], colJ[cc+6], valJ[cc+6] = ec, 4*T1+1, dev_B1[0]
            rowJ[cc+7], colJ[cc+7], valJ[cc+7] = ec+1, 4*T1+1, dev_B1[1]

            dev_B2 = -wp @ A @ dJb @ e2 
            rowJ[cc+8], colJ[cc+8], valJ[cc+8] = ec, 4*T2+1, dev_B2[0]
            rowJ[cc+9], colJ[cc+9], valJ[cc+9] = ec+1, 4*T2+1, dev_B2[1]
            
            dev_C1 = wm @ dJc @ e1
            rowJ[cc+10], colJ[cc+10], valJ[cc+10] = ec, 4*T1+2, dev_C1[0]
            rowJ[cc+11], colJ[cc+11], valJ[cc+11] = ec+1, 4*T1+2, dev_C1[1]

            dev_C2 = - wp @ A @ dJc @ e2
            rowJ[cc+12], colJ[cc+12], valJ[cc+12] = ec, 4*T2+2, dev_C2[0]
            rowJ[cc+13], colJ[cc+13], valJ[cc+13] = ec+1, 4*T2+2, dev_C2[1]
            
            dev_D1 = wm @ dJd @ e1
            rowJ[cc+14], colJ[cc+14], valJ[cc+14] = ec, 4*T1+3, dev_D1[0]
            rowJ[cc+15], colJ[cc+15], valJ[cc+15] = ec+1, 4*T1+3, dev_D1[1]

            dev_D2 = - wp @ A @ dJd @ e2
            rowJ[cc+16], colJ[cc+16], valJ[cc+16] = ec, 4*T2+3, dev_D2[0]
            rowJ[cc+17], colJ[cc+17], valJ[cc+17] = ec+1, 4*T2+3, dev_D2[1]

    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_norm2_noJ(X, n, sep):
    return X[sep:sep+n].astype(np.float64)

@jit(nopython=True, parallel=True, cache=True)
def distortion_norm2(X, n, sep):
    val  = X[sep:sep+n].astype(np.float64)
    valJ = np.ones(n, dtype=np.float64)
    rowJ = np.arange(0,n).astype(np.int32)
    colJ = np.arange(0,n).astype(np.int32)
    return val, valJ, rowJ, colJ


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def distortion_norm1_noJ(X, n, sep):
    val = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        val[i] += (X[sep+i]**2 + 0.01)**0.25
    return val

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def distortion_norm1(X, n, sep):
    val  = np.zeros(n, dtype=np.float64)
    valJ = np.zeros(n, dtype=np.float64)
    rowJ = np.arange(0,n).astype(np.int32)
    colJ = np.arange(0,n).astype(np.int32)
    for i in prange(n):
        val[i] += (X[sep+i]**2 + 0.01)**0.25
        valJ[i] = 0.5*X[sep+i]*(X[sep+i]**2 + 0.01)**(-0.75)
    return val, valJ, rowJ, colJ